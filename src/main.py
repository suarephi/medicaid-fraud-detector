"""Main orchestration module - runs the full fraud detection pipeline.

Loads three datasets (Medicaid spending, OIG LEIE, NPPES), runs six fraud
signal detection algorithms, enriches results with provider metadata, and
writes the final JSON report.
"""
from __future__ import annotations

import argparse
import time
import traceback

import polars as pl

from src.ingest import load_medicaid, load_leie, load_nppes, normalize_npi
from src.signals import (
    signal_1_excluded_billing,
    signal_2_volume_outlier,
    signal_3_rapid_escalation,
    signal_4_workforce_impossibility,
    signal_5_shared_official,
    signal_6_geographic_implausibility,
)
from src.output import build_provider_entry, build_report, write_report


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the fraud detection pipeline.

    Returns:
        Parsed arguments with data_dir, output, and no_gpu attributes.
    """
    parser = argparse.ArgumentParser(
        description="Medicaid Provider Fraud Signal Detection Engine"
    )
    parser.add_argument(
        "--data-dir", default="data",
        help="Directory containing downloaded datasets (default: data)",
    )
    parser.add_argument(
        "--output", default="fraud_signals.json",
        help="Output JSON file path (default: fraud_signals.json)",
    )
    parser.add_argument(
        "--no-gpu", action="store_true",
        help="Disable GPU acceleration (no effect - included for compatibility)",
    )
    return parser.parse_args()


def enrich_flags_with_nppes(
    flags: list[dict],
    nppes_lf: pl.LazyFrame,
    medicaid_lf: pl.LazyFrame,
    med_cols: dict[str, str],
) -> list[dict]:
    """Enrich flag entries with provider metadata from NPPES and lifetime billing.

    Looks up provider name, entity type, taxonomy, state, and enumeration date
    from NPPES, and computes lifetime billing totals from Medicaid data. Groups
    flags by NPI and builds complete provider entries.

    Args:
        flags: List of raw signal flag dicts (with npi, signal_id, details).
        nppes_lf: Lazy frame of NPPES registry data.
        medicaid_lf: Lazy frame of Medicaid provider spending data.
        med_cols: Column name mapping from detect_medicaid_columns().

    Returns:
        List of enriched provider entry dicts ready for the final report.
    """
    if not flags:
        return []

    npi_col = med_cols["npi"]
    payment_col = med_cols["payment"]
    claims_col = med_cols["claims"]
    bene_col = med_cols["benes"]

    # Collect unique NPIs from all flags
    flag_npis = list({f["npi"] for f in flags})

    # Get NPPES metadata
    nppes_meta = (
        nppes_lf
        .with_columns(normalize_npi(pl.col("NPI")).alias("_npi"))
        .filter(pl.col("_npi").is_in(flag_npis))
        .collect()
    )

    # Build NPI -> metadata lookup
    meta_map: dict[str, dict[str, str]] = {}
    for row in nppes_meta.iter_rows(named=True):
        npi = row["_npi"]
        entity_type = str(row.get("Entity Type Code", ""))
        if entity_type == "1":
            name = f"{row.get('Provider Last Name (Legal Name)', '')} {row.get('Provider First Name', '')}".strip()
        else:
            name = str(row.get("Provider Organization Name (Legal Business Name)", ""))
        meta_map[npi] = {
            "provider_name": name or "Unknown",
            "entity_type": "individual" if entity_type == "1" else "organization",
            "taxonomy_code": str(row.get("Healthcare Provider Taxonomy Code_1", "")),
            "state": str(row.get("Provider Business Practice Location Address State Name", "")),
            "enumeration_date": str(row.get("Provider Enumeration Date", "")),
        }

    # Get lifetime billing per NPI (including beneficiaries)
    lifetime = (
        medicaid_lf
        .with_columns(normalize_npi(pl.col(npi_col)).alias("_npi"))
        .filter(pl.col("_npi").is_in(flag_npis))
        .group_by("_npi")
        .agg([
            pl.col(payment_col).sum().alias("lifetime_paid"),
            pl.col(claims_col).sum().alias("lifetime_claims"),
            pl.col(bene_col).sum().alias("lifetime_benes"),
        ])
        .collect()
    )

    billing_map: dict[str, dict[str, float | int]] = {}
    for row in lifetime.iter_rows(named=True):
        billing_map[row["_npi"]] = {
            "lifetime_paid": float(row["lifetime_paid"]),
            "lifetime_claims": int(row["lifetime_claims"]),
            "lifetime_benes": int(row["lifetime_benes"]),
        }

    # Merge flags by NPI to build provider entries
    npi_flags: dict[str, list[dict]] = {}
    for f in flags:
        npi_flags.setdefault(f["npi"], []).append(f)

    enriched: list[dict] = []
    for npi, npi_flag_list in npi_flags.items():
        meta = meta_map.get(npi, {
            "provider_name": "Unknown",
            "entity_type": "unknown",
            "taxonomy_code": "",
            "state": "",
            "enumeration_date": "",
        })
        billing = billing_map.get(npi, {
            "lifetime_paid": 0.0,
            "lifetime_claims": 0,
            "lifetime_benes": 0,
        })

        # Enrich Signal 6 state field if missing
        for f in npi_flag_list:
            if f["signal_id"] == 6 and not f["details"].get("state"):
                f["details"]["state"] = meta["state"]

        entry = build_provider_entry(
            npi=npi,
            provider_name=meta["provider_name"],
            entity_type=meta["entity_type"],
            taxonomy_code=meta["taxonomy_code"],
            state=meta["state"],
            enumeration_date=meta["enumeration_date"],
            lifetime_paid=billing["lifetime_paid"],
            lifetime_claims=billing["lifetime_claims"],
            lifetime_benes=billing["lifetime_benes"],
            signals=npi_flag_list,
        )
        enriched.append(entry)

    return enriched


def main() -> None:
    """Run the full fraud detection pipeline.

    Orchestrates data loading, signal detection, result enrichment, and
    report generation. Each signal runs independently with error isolation
    so that a failure in one signal does not prevent others from completing.
    """
    args = parse_args()
    data_dir = args.data_dir
    output_path = args.output

    print("=" * 60)
    print("Medicaid Provider Fraud Signal Detection Engine v1.0.0")
    print("=" * 60)
    start_time = time.time()

    # Load data
    print("\n[1/4] Loading datasets...")
    t = time.time()
    medicaid_lf, med_cols = load_medicaid(data_dir)
    leie_df = load_leie(data_dir)
    nppes_lf = load_nppes(data_dir)
    print(f"  Data loaded in {time.time() - t:.1f}s")

    # Count total providers scanned
    print("\n[2/4] Counting unique providers...")
    t = time.time()
    npi_col = med_cols["npi"]
    scan_count = (
        medicaid_lf
        .select(pl.col(npi_col).n_unique())
        .collect()
        .item()
    )
    print(f"  {scan_count:,} unique providers in {time.time() - t:.1f}s")

    # Run all signals
    print("\n[3/4] Running fraud signal detection...")
    all_flags: list[dict] = []
    signal_tallies: dict[str, int] = {}

    signal_runners = [
        ("signal_1", lambda: signal_1_excluded_billing(medicaid_lf, med_cols, leie_df)),
        ("signal_2", lambda: signal_2_volume_outlier(medicaid_lf, med_cols, nppes_lf)),
        ("signal_3", lambda: signal_3_rapid_escalation(medicaid_lf, med_cols, nppes_lf)),
        ("signal_4", lambda: signal_4_workforce_impossibility(medicaid_lf, med_cols, nppes_lf)),
        ("signal_5", lambda: signal_5_shared_official(medicaid_lf, med_cols, nppes_lf)),
        ("signal_6", lambda: signal_6_geographic_implausibility(medicaid_lf, med_cols)),
    ]

    for signal_name, runner in signal_runners:
        t = time.time()
        print(f"\n  Running {signal_name}...")
        try:
            flags = runner()
            signal_tallies[signal_name] = len(flags)
            all_flags.extend(flags)
            print(f"  {signal_name}: {len(flags)} flags in {time.time() - t:.1f}s")
        except Exception as e:
            print(f"  ERROR in {signal_name}: {e}")
            traceback.print_exc()
            signal_tallies[signal_name] = 0

    # Enrich and build report
    print("\n[4/4] Building report...")
    t = time.time()
    enriched = enrich_flags_with_nppes(all_flags, nppes_lf, medicaid_lf, med_cols)

    report = build_report(
        flagged_providers=enriched,
        scan_count=scan_count,
        signal_tallies=signal_tallies,
    )

    write_report(report, output_path)

    elapsed = time.time() - start_time
    print(f"\nCompleted in {elapsed:.1f}s")
    print(f"Total flags: {len(all_flags)}")
    print(f"Unique providers flagged: {len(enriched)}")


if __name__ == "__main__":
    main()
