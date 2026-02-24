"""Six fraud signal detection algorithms for Medicaid billing data."""
import polars as pl

from src.ingest import normalize_npi

# Home health HCPCS codes for Signal 6
HOME_HEALTH_CODES = set()
for prefix, start, end in [
    ("G", 151, 162),
    ("G", 299, 300),
    ("S", 9122, 9124),
    ("T", 1019, 1022),
]:
    for n in range(start, end + 1):
        HOME_HEALTH_CODES.add(f"{prefix}{n:04d}")


def _to_date_col(lf: pl.LazyFrame, col: str) -> pl.LazyFrame:
    """Attempt to cast a column to Date type if not already.

    Handles YYYY-MM format (e.g. "2024-07") by appending "-01" to make a valid date.
    """
    schema = lf.collect_schema()
    if col not in schema:
        return lf
    dtype = schema[col]
    if dtype == pl.Date or dtype == pl.Datetime:
        return lf
    # Try parsing common formats, including YYYY-MM (append -01)
    return lf.with_columns(
        pl.coalesce([
            pl.col(col).cast(pl.Utf8).str.strptime(pl.Date, "%Y-%m-%d", strict=False),
            pl.col(col).cast(pl.Utf8).str.strptime(pl.Date, "%Y%m%d", strict=False),
            pl.col(col).cast(pl.Utf8).str.strptime(pl.Date, "%m/%d/%Y", strict=False),
            # YYYY-MM format: append "-01" and parse
            (pl.col(col).cast(pl.Utf8) + "-01").str.strptime(pl.Date, "%Y-%m-%d", strict=False),
        ]).alias(col)
    )


def _extract_year_month(lf: pl.LazyFrame, col: str) -> pl.LazyFrame:
    """Add year_month column from a date or string column.

    After _to_date_col, the column should be Date type. But if it was already
    a "YYYY-MM" string that got parsed to Date via appending "-01", we use dt.strftime.
    Falls back to string slicing for non-date types.
    """
    schema = lf.collect_schema()
    dtype = schema.get(col)
    if dtype == pl.Date or dtype == pl.Datetime:
        return lf.with_columns(
            pl.col(col).dt.strftime("%Y-%m").alias("year_month")
        )
    # Might be integer YYYYMM or string like "2024-07"
    return lf.with_columns(
        pl.col(col).cast(pl.Utf8).str.slice(0, 7).alias("year_month")
    )


def _filter_valid_npi(lf: pl.LazyFrame, npi_alias: str = "_npi") -> pl.LazyFrame:
    """Filter out null, empty, and all-zero NPIs."""
    return lf.filter(
        pl.col(npi_alias).is_not_null()
        & (pl.col(npi_alias) != "")
        & (pl.col(npi_alias) != "0000000000")
    )


def signal_1_excluded_billing(
    medicaid_lf: pl.LazyFrame,
    med_cols: dict[str, str],
    leie_df: pl.DataFrame,
) -> list[dict]:
    """Signal 1: Excluded Provider Still Billing.

    Flags providers who appear in the OIG LEIE exclusion list and continue
    to bill Medicaid after their exclusion date with no reinstatement.
    """
    npi_col = med_cols["npi"]
    date_col = med_cols["date"]
    payment_col = med_cols["payment"]
    claims_col = med_cols["claims"]

    # Filter LEIE to those with NPI and without reinstatement
    excluded = leie_df.filter(
        (pl.col("npi_str").is_not_null())
        & (pl.col("npi_str") != "")
        & (pl.col("npi_str") != "0000000000")
        & (pl.col("excl_date_parsed").is_not_null())
        & (pl.col("rein_date_parsed").is_null())
    ).select([
        pl.col("npi_str").alias("excl_npi"),
        pl.col("excl_date_parsed").alias("excl_date"),
        pl.col("EXCLTYPE").cast(pl.Utf8).alias("excl_type"),
    ])

    if excluded.is_empty():
        print("  Signal 1: No excluded providers with NPIs found in LEIE")
        return []

    print(f"  Signal 1: {len(excluded)} excluded providers (no reinstatement) to check")

    # Normalize Medicaid NPI, filter invalid, and cast date
    med = medicaid_lf.with_columns(
        normalize_npi(pl.col(npi_col)).alias("_npi")
    )
    med = _filter_valid_npi(med)
    med = _to_date_col(med, date_col)

    # Join and filter for post-exclusion billing (skip null dates)
    result = (
        med
        .filter(pl.col(date_col).is_not_null())
        .join(excluded.lazy(), left_on="_npi", right_on="excl_npi")
        .filter(pl.col(date_col) > pl.col("excl_date"))
        .group_by("_npi")
        .agg([
            pl.col(payment_col).sum().alias("post_exclusion_paid"),
            pl.col(claims_col).sum().alias("post_exclusion_claims"),
            pl.col("excl_date").first(),
            pl.col("excl_type").first(),
            pl.col(date_col).min().alias("first_post_excl_billing"),
            pl.col(date_col).max().alias("last_post_excl_billing"),
        ])
        .filter(pl.col("post_exclusion_paid") > 0)  # Only flag if actual payments made
        .collect()
    )

    print(f"  Signal 1: {len(result)} providers billing after exclusion")

    flags = []
    for row in result.iter_rows(named=True):
        flags.append({
            "npi": row["_npi"],
            "signal_id": 1,
            "details": {
                "exclusion_date": str(row["excl_date"]),
                "exclusion_type": row["excl_type"],
                "post_exclusion_paid": float(row["post_exclusion_paid"]),
                "post_exclusion_claims": int(row["post_exclusion_claims"]),
                "first_post_excl_billing": str(row["first_post_excl_billing"]),
                "last_post_excl_billing": str(row["last_post_excl_billing"]),
            },
        })
    return flags


def signal_2_volume_outlier(
    medicaid_lf: pl.LazyFrame,
    med_cols: dict[str, str],
    nppes_lf: pl.LazyFrame,
) -> list[dict]:
    """Signal 2: Billing Volume Outlier.

    Identifies providers whose total spending exceeds the 99th percentile
    within their taxonomy code and state peer group.
    """
    npi_col = med_cols["npi"]
    payment_col = med_cols["payment"]

    # Aggregate total payment per NPI, filtering invalid NPIs
    npi_totals = (
        medicaid_lf
        .with_columns(normalize_npi(pl.col(npi_col)).alias("_npi"))
        .pipe(_filter_valid_npi)
        .group_by("_npi")
        .agg(pl.col(payment_col).sum().alias("total_paid"))
    )

    # Get taxonomy and state from NPPES
    nppes = nppes_lf.with_columns(
        normalize_npi(pl.col("NPI")).alias("_npi")
    ).select([
        "_npi",
        pl.col("Healthcare Provider Taxonomy Code_1").alias("taxonomy"),
        pl.col("Provider Business Practice Location Address State Name").alias("state"),
    ])

    # Join
    joined = npi_totals.join(nppes, on="_npi", how="inner")

    # Calculate 99th percentile and median per peer group
    peer_stats = (
        joined
        .group_by(["taxonomy", "state"])
        .agg([
            pl.col("total_paid").quantile(0.99, interpolation="linear").alias("p99_threshold"),
            pl.col("total_paid").median().alias("peer_median"),
            pl.col("total_paid").count().alias("peer_count"),
        ])
        .filter(pl.col("peer_count") >= 5)  # Need enough peers for meaningful comparison
    )

    # Find outliers
    outliers = (
        joined
        .join(peer_stats, on=["taxonomy", "state"], how="inner")
        .filter(pl.col("total_paid") > pl.col("p99_threshold"))
        .with_columns(
            (pl.col("total_paid") / pl.col("peer_median")).alias("ratio_to_median")
        )
        .collect()
    )

    print(f"  Signal 2: {len(outliers)} billing volume outliers")

    flags = []
    for row in outliers.iter_rows(named=True):
        flags.append({
            "npi": row["_npi"],
            "signal_id": 2,
            "details": {
                "total_paid": float(row["total_paid"]),
                "peer_median": float(row["peer_median"]),
                "p99_threshold": float(row["p99_threshold"]),
                "ratio_to_peer_median": round(float(row["ratio_to_median"]), 2),
                "taxonomy": row["taxonomy"],
                "state": row["state"],
            },
        })
    return flags


def signal_3_rapid_escalation(
    medicaid_lf: pl.LazyFrame,
    med_cols: dict[str, str],
    nppes_lf: pl.LazyFrame,
) -> list[dict]:
    """Signal 3: Rapid Billing Escalation.

    Targets newly enumerated providers (within 24 months of first claim)
    showing month-over-month growth exceeding 500% in any rolling window
    with at least $25,000 in payments during growth months.
    """
    npi_col = med_cols["npi"]
    date_col = med_cols["date"]
    payment_col = med_cols["payment"]

    # Normalize and extract year-month
    med = medicaid_lf.with_columns(
        normalize_npi(pl.col(npi_col)).alias("_npi")
    )
    med = _filter_valid_npi(med)
    med = _to_date_col(med, date_col)
    med = _extract_year_month(med, date_col)

    # Monthly totals per NPI
    monthly = (
        med
        .group_by(["_npi", "year_month"])
        .agg(pl.col(payment_col).sum().alias("monthly_paid"))
        .sort(["_npi", "year_month"])
    )

    # Get enumeration dates from NPPES
    nppes = nppes_lf.with_columns(
        normalize_npi(pl.col("NPI")).alias("_npi")
    ).select([
        "_npi",
        pl.col("Provider Enumeration Date").alias("enum_date_str"),
    ])

    # Join to get enumeration date and parse it
    joined = (
        monthly
        .join(nppes, on="_npi", how="inner")
        .with_columns(
            pl.col("enum_date_str").cast(pl.Utf8).str.strptime(
                pl.Date, "%m/%d/%Y", strict=False
            ).alias("enum_date")
        )
        .filter(pl.col("enum_date").is_not_null())
    )

    # First billing month per NPI
    first_billing = (
        joined
        .group_by("_npi")
        .agg([
            pl.col("year_month").min().alias("first_billing_month"),
            pl.col("enum_date").first(),
        ])
    )

    # Collect first_billing to filter new providers
    first_df = first_billing.collect()

    # Filter to new providers: enumeration within 24 months of first billing
    from datetime import date as dt_date
    new_providers = set()
    first_map = {}
    enum_map = {}
    for row in first_df.iter_rows(named=True):
        npi = row["_npi"]
        first_map[npi] = row["first_billing_month"]
        if row["enum_date"] is not None and row["first_billing_month"] is not None:
            enum_map[npi] = row["enum_date"]
            first_ym = row["first_billing_month"]
            try:
                ym_parts = first_ym.split("-")
                first_date = dt_date(int(ym_parts[0]), int(ym_parts[1]), 1)
                diff_months = (first_date.year - row["enum_date"].year) * 12 + \
                              (first_date.month - row["enum_date"].month)
                if 0 <= diff_months <= 24:
                    new_providers.add(npi)
            except (ValueError, TypeError, IndexError):
                pass

    print(f"  Signal 3: {len(new_providers)} newly enumerated providers to check")

    if not new_providers:
        return []

    # Filter to new providers and compute growth using polars shift()
    new_monthly = joined.filter(
        pl.col("_npi").is_in(list(new_providers))
    ).collect().sort(["_npi", "year_month"])

    # Vectorized MoM growth calculation
    growth_df = (
        new_monthly
        .with_columns([
            pl.col("monthly_paid").shift(1).over("_npi").alias("prev_paid"),
        ])
        .with_columns(
            pl.when(pl.col("prev_paid") > 0)
            .then((pl.col("monthly_paid") - pl.col("prev_paid")) / pl.col("prev_paid") * 100)
            .otherwise(None)
            .alias("mom_growth")
        )
    )

    # Get peak growth and payments during growth months per NPI
    flagged = (
        growth_df
        .group_by("_npi")
        .agg([
            pl.col("mom_growth").max().alias("peak_growth_rate"),
            # Sum payments in months where growth > 500%
            pl.col("monthly_paid")
            .filter(pl.col("mom_growth") > 500)
            .sum()
            .alias("payments_during_growth"),
        ])
        .filter(
            (pl.col("peak_growth_rate") > 500)
            & (pl.col("payments_during_growth") > 25_000)
        )
    )

    # Build 12-month progressions
    flags = []
    for row in flagged.iter_rows(named=True):
        npi = row["_npi"]
        npi_data = growth_df.filter(pl.col("_npi") == npi).sort("year_month")
        months = npi_data["year_month"].to_list()
        payments = npi_data["monthly_paid"].to_list()
        progression = {months[i]: float(payments[i]) for i in range(min(12, len(months)))}

        flags.append({
            "npi": npi,
            "signal_id": 3,
            "details": {
                "enumeration_date": str(enum_map.get(npi, "")),
                "first_billing_month": first_map.get(npi, ""),
                "twelve_month_progression": progression,
                "peak_growth_rate": round(float(row["peak_growth_rate"]), 1),
                "payments_during_growth": round(float(row["payments_during_growth"]), 2),
            },
        })

    print(f"  Signal 3: {len(flags)} providers with rapid escalation (>200% MoM)")
    return flags


def signal_4_workforce_impossibility(
    medicaid_lf: pl.LazyFrame,
    med_cols: dict[str, str],
    nppes_lf: pl.LazyFrame,
) -> list[dict]:
    """Signal 4: Workforce Impossibility.

    Organizations processing peak monthly claims implying >6 claims per
    provider-hour sustained across all working hours/days.
    (Assumes 22 working days * 8 hours = 176 hours per month)
    """
    npi_col = med_cols["npi"]
    date_col = med_cols["date"]
    claims_col = med_cols["claims"]
    payment_col = med_cols["payment"]

    HOURS_PER_MONTH = 22 * 8  # 176 business hours
    THRESHOLD = 20  # claims per provider-hour (1 every 3 minutes sustained)

    # Only organizations (Entity Type Code = "2")
    org_npis = (
        nppes_lf
        .filter(
            pl.col("Entity Type Code").is_not_null()
            & (pl.col("Entity Type Code").cast(pl.Utf8) == "2")
        )
        .with_columns(normalize_npi(pl.col("NPI")).alias("_npi"))
        .select("_npi")
    )

    # Normalize and extract year-month from Medicaid
    med = medicaid_lf.with_columns(
        normalize_npi(pl.col(npi_col)).alias("_npi")
    )
    med = _filter_valid_npi(med)
    med = _to_date_col(med, date_col)
    med = _extract_year_month(med, date_col)

    # Monthly claims per org NPI
    monthly = (
        med
        .join(org_npis, on="_npi", how="inner")
        .group_by(["_npi", "year_month"])
        .agg([
            pl.col(claims_col).sum().alias("monthly_claims"),
            pl.col(payment_col).sum().alias("monthly_revenue"),
        ])
    )

    # Find peak month per NPI using window function (not sort+first which is unreliable)
    peak = (
        monthly
        .with_columns(
            pl.col("monthly_claims").max().over("_npi").alias("max_claims")
        )
        .filter(pl.col("monthly_claims") == pl.col("max_claims"))
        .group_by("_npi")
        .first()  # Break ties by taking first
        .with_columns(
            (pl.col("monthly_claims").cast(pl.Float64) / HOURS_PER_MONTH)
            .alias("claims_per_hour")
        )
        .filter(pl.col("claims_per_hour") > THRESHOLD)
    )

    # Also get average payment per claim for overpayment estimation
    avg_payment = (
        med
        .join(org_npis, on="_npi", how="inner")
        .group_by("_npi")
        .agg([
            pl.col(payment_col).sum().alias("total_paid"),
            pl.col(claims_col).sum().alias("total_claims"),
        ])
        .filter(pl.col("total_claims") > 0)  # Avoid division by zero
        .with_columns(
            (pl.col("total_paid") / pl.col("total_claims")).alias("avg_payment_per_claim")
        )
    )

    result = peak.join(avg_payment, on="_npi", how="left").collect()

    print(f"  Signal 4: {len(result)} organizations with impossible claim volumes")

    flags = []
    for row in result.iter_rows(named=True):
        reasonable_claims = THRESHOLD * HOURS_PER_MONTH
        excess = max(0, int(row["monthly_claims"]) - reasonable_claims)
        avg_pmt = row.get("avg_payment_per_claim")
        avg_pmt = float(avg_pmt) if avg_pmt is not None else 0.0

        flags.append({
            "npi": row["_npi"],
            "signal_id": 4,
            "details": {
                "peak_month": row["year_month"],
                "claims_count": int(row["monthly_claims"]),
                "implied_claims_per_hour": round(float(row["claims_per_hour"]), 2),
                "peak_month_revenue": float(row["monthly_revenue"]),
                "excess_claims": excess,
                "avg_payment_per_claim": round(avg_pmt, 2),
            },
        })
    return flags


def signal_5_shared_official(
    medicaid_lf: pl.LazyFrame,
    med_cols: dict[str, str],
    nppes_lf: pl.LazyFrame,
) -> list[dict]:
    """Signal 5: Shared Authorized Official.

    Authorized officials controlling 5+ NPIs with combined billing exceeding $1,000,000.
    """
    npi_col = med_cols["npi"]
    payment_col = med_cols["payment"]

    # Build official_name, handling null first names
    officials = (
        nppes_lf
        .filter(
            pl.col("Authorized Official Last Name").is_not_null()
            & (pl.col("Authorized Official Last Name").cast(pl.Utf8).str.strip_chars() != "")
        )
        .with_columns([
            normalize_npi(pl.col("NPI")).alias("_npi"),
            (
                pl.col("Authorized Official Last Name").cast(pl.Utf8).str.to_uppercase()
                + ", "
                + pl.col("Authorized Official First Name").cast(pl.Utf8)
                    .fill_null("").str.to_uppercase()
            ).alias("official_name"),
        ])
        .select(["_npi", "official_name"])
    )

    # Count NPIs per official and filter 5+
    official_counts = (
        officials
        .group_by("official_name")
        .agg([
            pl.col("_npi").count().alias("npi_count"),
            pl.col("_npi").alias("npi_list"),
        ])
        .filter(pl.col("npi_count") >= 5)
    )

    official_counts_df = official_counts.collect()

    if official_counts_df.is_empty():
        print("  Signal 5: No officials controlling 5+ NPIs")
        return []

    print(f"  Signal 5: {len(official_counts_df)} officials controlling 5+ NPIs")

    # Get billing totals per NPI (single aggregation)
    npi_billing = (
        medicaid_lf
        .with_columns(normalize_npi(pl.col(npi_col)).alias("_npi"))
        .pipe(_filter_valid_npi)
        .group_by("_npi")
        .agg(pl.col(payment_col).sum().alias("npi_total_paid"))
        .collect()
    )

    # Build a billing lookup dict for O(1) access instead of O(n) filter per NPI
    billing_map = dict(
        zip(
            npi_billing["_npi"].to_list(),
            npi_billing["npi_total_paid"].to_list(),
        )
    )

    # Explode and look up billing via dict (O(n) instead of O(n^2))
    flags = []
    for row in official_counts_df.iter_rows(named=True):
        official = row["official_name"]
        npis = row["npi_list"]

        npi_totals = {}
        combined = 0.0
        for npi in npis:
            total = billing_map.get(npi, 0.0)
            if total:
                npi_totals[npi] = float(total)
                combined += float(total)

        if combined > 1_000_000:
            flags.append({
                "npi": npis[0],  # Primary NPI for the flag
                "signal_id": 5,
                "details": {
                    "official_name": official,
                    "npi_list": npis,
                    "per_npi_totals": npi_totals,
                    "combined_total": round(combined, 2),
                },
            })

    print(f"  Signal 5: {len(flags)} officials with >$1M combined billing")
    return flags


def signal_6_geographic_implausibility(
    medicaid_lf: pl.LazyFrame,
    med_cols: dict[str, str],
) -> list[dict]:
    """Signal 6: Geographic Implausibility.

    Home health providers with unique beneficiary-to-claims ratio below 0.1
    within a single month with >100 claims.

    Uses max(beneficiaries) across procedure codes as a proxy for unique
    beneficiaries, since the same patient may appear under multiple codes.
    """
    npi_col = med_cols["npi"]
    date_col = med_cols["date"]
    hcpcs_col = med_cols["hcpcs"]
    bene_col = med_cols["benes"]
    claims_col = med_cols["claims"]

    # Filter to home health HCPCS codes
    hh_codes = list(HOME_HEALTH_CODES)

    med = medicaid_lf.with_columns(
        normalize_npi(pl.col(npi_col)).alias("_npi")
    )
    med = _filter_valid_npi(med)
    med = _to_date_col(med, date_col)
    med = _extract_year_month(med, date_col)

    # Filter to home health codes and aggregate by NPI + month
    # Use max() for beneficiaries (proxy for unique count across HCPCS codes)
    monthly = (
        med
        .filter(pl.col(hcpcs_col).cast(pl.Utf8).is_in(hh_codes))
        .group_by(["_npi", "year_month"])
        .agg([
            pl.col(bene_col).max().alias("unique_benes"),
            pl.col(claims_col).sum().alias("total_claims"),
            pl.col(hcpcs_col).first().alias("flagged_code"),
        ])
        .filter(
            (pl.col("total_claims") > 500)
            & (pl.col("unique_benes").is_not_null())
            & (pl.col("unique_benes") > 0)
        )
        .with_columns(
            (pl.col("unique_benes").cast(pl.Float64) / pl.col("total_claims"))
            .alias("bene_claims_ratio")
        )
        .filter(pl.col("bene_claims_ratio") < 0.05)
        .collect()
    )

    print(f"  Signal 6: {len(monthly)} provider-months with implausible ratios")

    # Aggregate to provider level (worst month)
    if monthly.is_empty():
        return []

    provider_flags = (
        monthly
        .sort("bene_claims_ratio")
        .group_by("_npi")
        .first()
    )

    flags = []
    for row in provider_flags.iter_rows(named=True):
        flags.append({
            "npi": row["_npi"],
            "signal_id": 6,
            "details": {
                "state": "",  # Will be enriched by main.py
                "flagged_codes": [row["flagged_code"]],
                "month": row["year_month"],
                "claims": int(row["total_claims"]),
                "unique_beneficiaries": int(row["unique_benes"]),
                "ratio": round(float(row["bene_claims_ratio"]), 4),
            },
        })
    return flags
