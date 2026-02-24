"""Six fraud signal detection algorithms for Medicaid billing data.

Each signal function takes standardized inputs (Medicaid LazyFrame, column
mapping, and optionally LEIE or NPPES data) and returns a list of flag
dictionaries containing the provider NPI, signal ID, and evidence details.
"""
from __future__ import annotations

from datetime import date as dt_date

import polars as pl

from src.ingest import normalize_npi

# Home health HCPCS codes for Signal 6
HOME_HEALTH_CODES: set[str] = set()
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

    Handles multiple date formats including YYYY-MM-DD, YYYYMMDD, MM/DD/YYYY,
    and YYYY-MM (appends "-01" to make a valid date).

    Args:
        lf: Input LazyFrame.
        col: Name of the column to cast.

    Returns:
        LazyFrame with the specified column cast to Date type, or unchanged
        if the column is already a date type or does not exist.
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
    """Add a year_month column derived from a date or string column.

    Args:
        lf: Input LazyFrame.
        col: Name of the source date/string column.

    Returns:
        LazyFrame with an additional "year_month" column in "YYYY-MM" format.
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
    """Filter out null, empty, and all-zero NPIs.

    Args:
        lf: Input LazyFrame with a normalized NPI column.
        npi_alias: Name of the NPI column to filter on.

    Returns:
        LazyFrame with invalid NPI rows removed.
    """
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

    Flags providers whose NPI matches the OIG LEIE exclusion list where the
    exclusion predates the claim and no reinstatement has occurred. Checks
    both billing and servicing NPI columns when available.

    Args:
        medicaid_lf: Lazy frame of Medicaid provider spending data.
        med_cols: Column name mapping from detect_medicaid_columns().
        leie_df: DataFrame of OIG LEIE exclusion records with parsed dates.

    Returns:
        List of flag dicts, each containing "npi", "signal_id" (1), and
        "details" with exclusion_date, exclusion_type, post_exclusion_paid,
        post_exclusion_claims, and billing date range.
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

    # Build Medicaid frame with both billing and servicing NPI
    med = _to_date_col(medicaid_lf, date_col)

    # Check for SERVICING_PROVIDER_NPI_NUM column
    schema = medicaid_lf.collect_schema()
    has_servicing = "SERVICING_PROVIDER_NPI_NUM" in schema.names()

    # Helper: run exclusion check against a single NPI column
    def _check_npi_col(
        source: pl.LazyFrame, npi_src_col: str
    ) -> pl.DataFrame:
        return (
            source
            .with_columns(normalize_npi(pl.col(npi_src_col)).alias("_npi"))
            .pipe(_filter_valid_npi)
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
            .filter(pl.col("post_exclusion_paid") > 0)
            .collect()
        )

    # Check billing NPI
    billing_result = _check_npi_col(med, npi_col)

    # Also check servicing NPI if available, then merge
    if has_servicing:
        servicing_result = _check_npi_col(med, "SERVICING_PROVIDER_NPI_NUM")
        # Merge: combine results, taking max per NPI
        result = (
            pl.concat([billing_result, servicing_result])
            .group_by("_npi")
            .agg([
                pl.col("post_exclusion_paid").sum().alias("post_exclusion_paid"),
                pl.col("post_exclusion_claims").sum().alias("post_exclusion_claims"),
                pl.col("excl_date").first(),
                pl.col("excl_type").first(),
                pl.col("first_post_excl_billing").min(),
                pl.col("last_post_excl_billing").max(),
            ])
        )
    else:
        result = billing_result

    print(f"  Signal 1: {len(result)} providers billing after exclusion")

    flags: list[dict] = []
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
    within their taxonomy code and state peer group. Peer groups with fewer
    than 5 providers are excluded to avoid false positives.

    Args:
        medicaid_lf: Lazy frame of Medicaid provider spending data.
        med_cols: Column name mapping from detect_medicaid_columns().
        nppes_lf: Lazy frame of NPPES registry data with taxonomy and state.

    Returns:
        List of flag dicts, each containing "npi", "signal_id" (2), and
        "details" with total_paid, peer_median, p99_threshold,
        ratio_to_peer_median, taxonomy, and state.
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
        .filter(pl.col("peer_count") >= 5)
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

    flags: list[dict] = []
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

    Targets newly enumerated providers (within 24 months of first claim).
    Computes month-over-month total paid growth for the first 12 months and
    flags providers where any rolling 3-month average growth rate exceeds 200%.

    Args:
        medicaid_lf: Lazy frame of Medicaid provider spending data.
        med_cols: Column name mapping from detect_medicaid_columns().
        nppes_lf: Lazy frame of NPPES registry data with enumeration dates.

    Returns:
        List of flag dicts, each containing "npi", "signal_id" (3), and
        "details" with enumeration_date, first_billing_month,
        twelve_month_progression, peak_growth_rate, and
        payments_during_growth.
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
    new_providers: set[str] = set()
    first_map: dict[str, str] = {}
    enum_map: dict[str, object] = {}
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

    # Filter to new providers, limit to first 12 months, and compute growth
    new_monthly = joined.filter(
        pl.col("_npi").is_in(list(new_providers))
    ).collect().sort(["_npi", "year_month"])

    # Keep only first 12 months per NPI
    new_monthly = (
        new_monthly
        .with_columns(
            pl.col("year_month").rank("ordinal").over("_npi").alias("_month_rank")
        )
        .filter(pl.col("_month_rank") <= 12)
        .drop("_month_rank")
    )

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

    # Compute rolling 3-month average growth rate per NPI
    growth_df = (
        growth_df
        .with_columns(
            pl.col("mom_growth")
            .rolling_mean(window_size=3, min_samples=3)
            .over("_npi")
            .alias("rolling_3m_avg_growth")
        )
    )

    # Get peak rolling 3-month average growth and payments during growth months
    flagged = (
        growth_df
        .group_by("_npi")
        .agg([
            pl.col("rolling_3m_avg_growth").max().alias("peak_growth_rate"),
            # Sum payments in months where rolling 3-month avg growth > 200%
            pl.col("monthly_paid")
            .filter(pl.col("rolling_3m_avg_growth") > 200)
            .sum()
            .alias("payments_during_growth"),
        ])
        .filter(pl.col("peak_growth_rate") > 200)
    )

    # Build 12-month progressions
    flags: list[dict] = []
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

    print(f"  Signal 3: {len(flags)} providers with rapid escalation (>200% rolling 3-month avg)")
    return flags


def signal_4_workforce_impossibility(
    medicaid_lf: pl.LazyFrame,
    med_cols: dict[str, str],
    nppes_lf: pl.LazyFrame,
) -> list[dict]:
    """Signal 4: Workforce Impossibility.

    Identifies organizations (Entity Type Code 2) with peak monthly claims
    implying more than 6 claims per business hour (claims / 22 working days /
    8 hours > 6), suggesting fabricated or phantom claims.

    Args:
        medicaid_lf: Lazy frame of Medicaid provider spending data.
        med_cols: Column name mapping from detect_medicaid_columns().
        nppes_lf: Lazy frame of NPPES registry data with entity types.

    Returns:
        List of flag dicts, each containing "npi", "signal_id" (4), and
        "details" with peak_month, claims_count, implied_claims_per_hour,
        and peak_month_revenue.
    """
    npi_col = med_cols["npi"]
    date_col = med_cols["date"]
    claims_col = med_cols["claims"]
    payment_col = med_cols["payment"]

    HOURS_PER_MONTH = 22 * 8  # 176 business hours
    THRESHOLD = 6  # claims per provider-hour per spec

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

    # Find peak month per NPI using window function
    peak = (
        monthly
        .with_columns(
            pl.col("monthly_claims").max().over("_npi").alias("max_claims")
        )
        .filter(pl.col("monthly_claims") == pl.col("max_claims"))
        .group_by("_npi")
        .first()
        .with_columns(
            (pl.col("monthly_claims").cast(pl.Float64) / HOURS_PER_MONTH)
            .alias("claims_per_hour")
        )
        .filter(pl.col("claims_per_hour") > THRESHOLD)
    )

    result = peak.collect()

    print(f"  Signal 4: {len(result)} organizations with impossible claim volumes")

    flags: list[dict] = []
    for row in result.iter_rows(named=True):
        flags.append({
            "npi": row["_npi"],
            "signal_id": 4,
            "details": {
                "peak_month": row["year_month"],
                "claims_count": int(row["monthly_claims"]),
                "implied_claims_per_hour": round(float(row["claims_per_hour"]), 2),
                "peak_month_revenue": float(row["monthly_revenue"]),
            },
        })
    return flags


def signal_5_shared_official(
    medicaid_lf: pl.LazyFrame,
    med_cols: dict[str, str],
    nppes_lf: pl.LazyFrame,
) -> list[dict]:
    """Signal 5: Shared Authorized Official.

    Identifies authorized officials controlling 5 or more NPIs with combined
    billing exceeding $1,000,000, which may indicate coordinated billing
    through shell entities.

    Args:
        medicaid_lf: Lazy frame of Medicaid provider spending data.
        med_cols: Column name mapping from detect_medicaid_columns().
        nppes_lf: Lazy frame of NPPES registry data with official names.

    Returns:
        List of flag dicts, each containing "npi" (first NPI in the group),
        "signal_id" (5), and "details" with official_name, npi_list,
        per_npi_totals, and combined_total.
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

    # Build a billing lookup dict for O(1) access
    billing_map: dict[str, float] = dict(
        zip(
            npi_billing["_npi"].to_list(),
            npi_billing["npi_total_paid"].to_list(),
        )
    )

    flags: list[dict] = []
    for row in official_counts_df.iter_rows(named=True):
        official = row["official_name"]
        npis = row["npi_list"]

        npi_totals: dict[str, float] = {}
        combined = 0.0
        for npi in npis:
            total = billing_map.get(npi, 0.0)
            if total:
                npi_totals[npi] = float(total)
                combined += float(total)

        if combined > 1_000_000:
            flags.append({
                "npi": npis[0],
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

    Flags home health providers with more than 100 claims in a single month
    and a unique beneficiaries / claims ratio below 0.1, indicating potential
    phantom services or patient identity reuse.

    Args:
        medicaid_lf: Lazy frame of Medicaid provider spending data.
        med_cols: Column name mapping from detect_medicaid_columns().

    Returns:
        List of flag dicts, each containing "npi", "signal_id" (6), and
        "details" with state, flagged_codes, month, claims,
        unique_beneficiaries, and ratio.
    """
    npi_col = med_cols["npi"]
    date_col = med_cols["date"]
    hcpcs_col = med_cols["hcpcs"]
    bene_col = med_cols["benes"]
    claims_col = med_cols["claims"]

    hh_codes = list(HOME_HEALTH_CODES)

    med = medicaid_lf.with_columns(
        normalize_npi(pl.col(npi_col)).alias("_npi")
    )
    med = _filter_valid_npi(med)
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
        .filter(
            (pl.col("total_claims") > 100)
            & (pl.col("unique_benes").is_not_null())
            & (pl.col("unique_benes") > 0)
        )
        .with_columns(
            (pl.col("unique_benes").cast(pl.Float64) / pl.col("total_claims"))
            .alias("bene_claims_ratio")
        )
        .filter(pl.col("bene_claims_ratio") < 0.1)
        .collect()
    )

    print(f"  Signal 6: {len(monthly)} provider-months with implausible ratios")

    if monthly.is_empty():
        return []

    provider_flags = (
        monthly
        .sort("bene_claims_ratio")
        .group_by("_npi")
        .first()
    )

    flags: list[dict] = []
    for row in provider_flags.iter_rows(named=True):
        flags.append({
            "npi": row["_npi"],
            "signal_id": 6,
            "details": {
                "state": "",
                "flagged_codes": [row["flagged_code"]],
                "month": row["year_month"],
                "claims": int(row["total_claims"]),
                "unique_beneficiaries": int(row["unique_benes"]),
                "ratio": round(float(row["bene_claims_ratio"]), 4),
            },
        })
    return flags
