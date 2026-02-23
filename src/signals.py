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
    """Attempt to cast a column to Date type if not already."""
    schema = lf.collect_schema()
    if col not in schema:
        return lf
    dtype = schema[col]
    if dtype == pl.Date or dtype == pl.Datetime:
        return lf
    # Try parsing common formats
    return lf.with_columns(
        pl.coalesce([
            pl.col(col).cast(pl.Utf8).str.strptime(pl.Date, "%Y-%m-%d", strict=False),
            pl.col(col).cast(pl.Utf8).str.strptime(pl.Date, "%Y%m%d", strict=False),
            pl.col(col).cast(pl.Utf8).str.strptime(pl.Date, "%m/%d/%Y", strict=False),
        ]).alias(col)
    )


def _extract_year_month(lf: pl.LazyFrame, col: str) -> pl.LazyFrame:
    """Add year_month column from a date column."""
    schema = lf.collect_schema()
    dtype = schema.get(col)
    if dtype == pl.Date or dtype == pl.Datetime:
        return lf.with_columns(
            pl.col(col).dt.strftime("%Y-%m").alias("year_month")
        )
    # Might be integer YYYYMM or string
    return lf.with_columns(
        pl.col(col).cast(pl.Utf8).str.slice(0, 7).alias("year_month")
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

    # Normalize Medicaid NPI and cast date
    med = medicaid_lf.with_columns(
        normalize_npi(pl.col(npi_col)).alias("_npi")
    )
    med = _to_date_col(med, date_col)

    # Join and filter for post-exclusion billing
    result = (
        med
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

    # Aggregate total payment per NPI
    npi_totals = (
        medicaid_lf
        .with_columns(normalize_npi(pl.col(npi_col)).alias("_npi"))
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
    showing month-over-month growth exceeding 200% in any rolling 3-month window.
    """
    npi_col = med_cols["npi"]
    date_col = med_cols["date"]
    payment_col = med_cols["payment"]

    # Normalize and extract year-month
    med = medicaid_lf.with_columns(
        normalize_npi(pl.col(npi_col)).alias("_npi")
    )
    med = _to_date_col(med, date_col)
    med = _extract_year_month(med, date_col)

    # Monthly totals per NPI
    monthly = (
        med
        .group_by(["_npi", "year_month"])
        .agg([
            pl.col(payment_col).sum().alias("monthly_paid"),
        ])
        .sort(["_npi", "year_month"])
    )

    # Get enumeration dates from NPPES
    nppes = nppes_lf.with_columns(
        normalize_npi(pl.col("NPI")).alias("_npi")
    ).select([
        "_npi",
        pl.col("Provider Enumeration Date").alias("enum_date_str"),
    ])

    # Join to get enumeration date
    joined = monthly.join(nppes, on="_npi", how="inner")

    # Parse enumeration date and find first billing month
    joined = joined.with_columns(
        pl.col("enum_date_str").cast(pl.Utf8).str.strptime(
            pl.Date, "%m/%d/%Y", strict=False
        ).alias("enum_date")
    )

    # Calculate first billing per NPI and filter new providers
    first_billing = (
        joined
        .group_by("_npi")
        .agg([
            pl.col("year_month").min().alias("first_billing_month"),
            pl.col("enum_date").first(),
        ])
    )

    # Collect to process growth rates
    monthly_df = joined.collect()
    first_df = first_billing.collect()

    # Filter to new providers: enumeration within 24 months of first billing
    new_providers = set()
    first_map = {}
    enum_map = {}
    for row in first_df.iter_rows(named=True):
        npi = row["_npi"]
        first_map[npi] = row["first_billing_month"]
        if row["enum_date"] is not None:
            enum_map[npi] = row["enum_date"]
            # Check if first billing is within 24 months of enumeration
            first_ym = row["first_billing_month"]
            try:
                from datetime import date
                ym_parts = first_ym.split("-")
                first_date = date(int(ym_parts[0]), int(ym_parts[1]), 1)
                diff_months = (first_date.year - row["enum_date"].year) * 12 + \
                              (first_date.month - row["enum_date"].month)
                if 0 <= diff_months <= 24:
                    new_providers.add(npi)
            except (ValueError, TypeError):
                pass

    print(f"  Signal 3: {len(new_providers)} newly enumerated providers to check")

    if not new_providers:
        return []

    # Filter monthly data to new providers and calculate growth
    new_monthly = monthly_df.filter(pl.col("_npi").is_in(list(new_providers)))

    flags = []
    for npi in new_providers:
        npi_data = (
            new_monthly
            .filter(pl.col("_npi") == npi)
            .sort("year_month")
        )
        payments = npi_data["monthly_paid"].to_list()
        months = npi_data["year_month"].to_list()

        if len(payments) < 2:
            continue

        # Calculate MoM growth rates
        max_growth = 0.0
        payments_during_growth = 0.0
        for i in range(1, len(payments)):
            prev = payments[i - 1]
            curr = payments[i]
            if prev > 0:
                growth = ((curr - prev) / prev) * 100
                if growth > max_growth:
                    max_growth = growth
                if growth > 200:
                    payments_during_growth += curr

        if max_growth > 200:
            # Build 12-month progression
            progression = {months[i]: float(payments[i]) for i in range(min(12, len(months)))}

            flags.append({
                "npi": npi,
                "signal_id": 3,
                "details": {
                    "enumeration_date": str(enum_map.get(npi, "")),
                    "first_billing_month": first_map.get(npi, ""),
                    "twelve_month_progression": progression,
                    "peak_growth_rate": round(max_growth, 1),
                    "payments_during_growth": round(payments_during_growth, 2),
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
    THRESHOLD = 6  # claims per provider-hour

    # Only organizations (Entity Type Code = "2")
    org_npis = (
        nppes_lf
        .filter(pl.col("Entity Type Code").cast(pl.Utf8) == "2")
        .with_columns(normalize_npi(pl.col("NPI")).alias("_npi"))
        .select("_npi")
    )

    # Normalize and extract year-month from Medicaid
    med = medicaid_lf.with_columns(
        normalize_npi(pl.col(npi_col)).alias("_npi")
    )
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

    # Peak month per NPI
    peak = (
        monthly
        .sort(["_npi", "monthly_claims"], descending=[False, True])
        .group_by("_npi")
        .first()
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
        avg_pmt = float(row.get("avg_payment_per_claim", 0) or 0)

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

    # Group NPPES by authorized official
    officials = (
        nppes_lf
        .filter(
            pl.col("Authorized Official Last Name").is_not_null()
            & (pl.col("Authorized Official Last Name") != "")
        )
        .with_columns([
            normalize_npi(pl.col("NPI")).alias("_npi"),
            (
                pl.col("Authorized Official Last Name").cast(pl.Utf8).str.to_uppercase()
                + ", "
                + pl.col("Authorized Official First Name").cast(pl.Utf8).str.to_uppercase()
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

    # Get billing totals per NPI
    npi_billing = (
        medicaid_lf
        .with_columns(normalize_npi(pl.col(npi_col)).alias("_npi"))
        .group_by("_npi")
        .agg(pl.col(payment_col).sum().alias("npi_total_paid"))
        .collect()
    )

    flags = []
    for row in official_counts_df.iter_rows(named=True):
        official = row["official_name"]
        npis = row["npi_list"]

        # Look up billing for each NPI
        npi_totals = {}
        combined = 0.0
        for npi in npis:
            billing = npi_billing.filter(pl.col("_npi") == npi)
            if not billing.is_empty():
                total = float(billing["npi_total_paid"][0])
                npi_totals[npi] = total
                combined += total

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
    med = _to_date_col(med, date_col)
    med = _extract_year_month(med, date_col)

    # Filter to home health codes and aggregate by NPI + month
    monthly = (
        med
        .filter(pl.col(hcpcs_col).cast(pl.Utf8).is_in(hh_codes))
        .group_by(["_npi", "year_month"])
        .agg([
            pl.col(bene_col).sum().alias("unique_benes"),
            pl.col(claims_col).sum().alias("total_claims"),
            pl.col(hcpcs_col).first().alias("flagged_code"),
        ])
        .filter(pl.col("total_claims") > 100)
        .with_columns(
            (pl.col("unique_benes").cast(pl.Float64) / pl.col("total_claims"))
            .alias("bene_claims_ratio")
        )
        .filter(pl.col("bene_claims_ratio") < 0.1)
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
