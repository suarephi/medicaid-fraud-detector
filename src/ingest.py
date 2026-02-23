"""Data ingestion module - loads and prepares the three required datasets."""
import glob
import os
from typing import Optional

import polars as pl


DATA_DIR = os.environ.get("FRAUD_DATA_DIR", "data")

# Known column name patterns for auto-detection
_NPI_PATTERNS = ["npi", "rndrng_npi", "rendering_npi", "provider_npi"]
_HCPCS_PATTERNS = ["hcpcs_cd", "hcpcs", "hcpcs_code", "procedure_code", "proc_cd"]
_DATE_PATTERNS = ["srvc_dt", "service_date", "srvc_yr_mth", "billing_date",
                   "year_month", "clm_dt", "period"]
_BENE_PATTERNS = ["bene_cnt", "tot_benes", "beneficiary_count", "bene_count",
                   "unique_bene", "bene_unique_cnt"]
_CLM_PATTERNS = ["clm_cnt", "tot_clms", "claim_count", "claims", "tot_claims"]
_PYMT_PATTERNS = ["pymt_amt", "tot_pymt", "payment_amount", "avg_mdcd_pymt_amt",
                   "paid_amt", "mdcd_pymt_amt", "total_payment", "mdcd_paid_amt"]


def _match_column(columns_lower: dict[str, str], patterns: list[str]) -> Optional[str]:
    """Find the first matching column name from a list of patterns."""
    for p in patterns:
        if p in columns_lower:
            return columns_lower[p]
    return None


def detect_medicaid_columns(columns: list[str]) -> dict[str, str]:
    """Auto-detect Medicaid parquet column names and map to standard aliases.

    Returns a dict mapping standard names (npi, hcpcs, date, benes, claims, payment)
    to actual column names in the parquet file.
    """
    cols_lower = {c.lower(): c for c in columns}
    mapping = {}

    for alias, patterns in [
        ("npi", _NPI_PATTERNS),
        ("hcpcs", _HCPCS_PATTERNS),
        ("date", _DATE_PATTERNS),
        ("benes", _BENE_PATTERNS),
        ("claims", _CLM_PATTERNS),
        ("payment", _PYMT_PATTERNS),
    ]:
        match = _match_column(cols_lower, patterns)
        if match:
            mapping[alias] = match

    # Fallback for 7-column files: assume positional order
    if len(mapping) < 6 and len(columns) == 7:
        positional = ["npi", "hcpcs", "date", "benes", "claims", None, "payment"]
        for i, alias in enumerate(positional):
            if alias and alias not in mapping:
                mapping[alias] = columns[i]

    missing = {"npi", "hcpcs", "date", "benes", "claims", "payment"} - set(mapping.keys())
    if missing:
        print(f"WARNING: Could not detect columns for: {missing}")
        print(f"  Available columns: {columns}")
        # Try to assign remaining unmatched columns
        used = set(mapping.values())
        remaining = [c for c in columns if c not in used]
        for alias in sorted(missing):
            if remaining:
                mapping[alias] = remaining.pop(0)
                print(f"  Guessing {alias} -> {mapping[alias]}")

    return mapping


def normalize_npi(expr: pl.Expr) -> pl.Expr:
    """Normalize NPI column to 10-digit zero-padded string."""
    return expr.cast(pl.Utf8).str.strip_chars().str.zfill(10)


def load_medicaid(data_dir: Optional[str] = None) -> tuple[pl.LazyFrame, dict[str, str]]:
    """Load Medicaid provider spending data as a lazy frame.

    Returns:
        Tuple of (LazyFrame, column_mapping dict)
    """
    ddir = data_dir or DATA_DIR
    path = os.path.join(ddir, "medicaid-provider-spending.parquet")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Medicaid data not found at {path}. Run setup.sh first.")

    lf = pl.scan_parquet(path)
    col_map = detect_medicaid_columns(lf.columns)
    print(f"Medicaid data: {len(lf.columns)} columns, mapping: {col_map}")
    return lf, col_map


def load_leie(data_dir: Optional[str] = None) -> pl.DataFrame:
    """Load OIG LEIE exclusion list.

    Key columns: NPI, EXCLDATE (YYYYMMDD), REINDATE, EXCLTYPE, GENERAL
    """
    ddir = data_dir or DATA_DIR
    path = os.path.join(ddir, "UPDATED.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"LEIE data not found at {path}. Run setup.sh first.")

    df = pl.read_csv(path, infer_schema_length=10000, ignore_errors=True)

    # Parse exclusion dates (YYYYMMDD format)
    if "EXCLDATE" in df.columns:
        df = df.with_columns(
            pl.col("EXCLDATE").cast(pl.Utf8).str.strptime(pl.Date, "%Y%m%d", strict=False)
            .alias("excl_date_parsed")
        )

    # Parse reinstatement dates
    if "REINDATE" in df.columns:
        df = df.with_columns(
            pl.col("REINDATE").cast(pl.Utf8).str.strptime(pl.Date, "%Y%m%d", strict=False)
            .alias("rein_date_parsed")
        )

    # Normalize NPI
    if "NPI" in df.columns:
        df = df.with_columns(normalize_npi(pl.col("NPI")).alias("npi_str"))

    print(f"LEIE data: {len(df)} exclusion records, {df.columns}")
    return df


def load_nppes(data_dir: Optional[str] = None) -> pl.LazyFrame:
    """Load NPPES NPI registry with only the 11 required columns."""
    ddir = data_dir or DATA_DIR

    # Find the extracted CSV file
    for pattern in [
        os.path.join(ddir, "npidata_pfile_*.csv"),
        os.path.join(ddir, "NPPES*.csv"),
        os.path.join(ddir, "*.csv"),
    ]:
        files = sorted(glob.glob(pattern))
        # Exclude UPDATED.csv (LEIE data)
        files = [f for f in files if "UPDATED" not in os.path.basename(f)]
        if files:
            break

    if not files:
        raise FileNotFoundError(f"No NPPES data file found in {ddir}. Run setup.sh first.")

    nppes_file = files[0]

    # The 11 columns required by the competition
    needed_cols = [
        "NPI",
        "Entity Type Code",
        "Provider Organization Name (Legal Business Name)",
        "Provider Last Name (Legal Name)",
        "Provider First Name",
        "Provider Business Practice Location Address State Name",
        "Healthcare Provider Taxonomy Code_1",
        "Provider Enumeration Date",
        "Authorized Official Last Name",
        "Authorized Official First Name",
        "Authorized Official Telephone Number",
    ]

    lf = pl.scan_csv(nppes_file, infer_schema_length=10000, ignore_errors=True)

    # Select only columns that exist in the file
    available = set(lf.columns)
    select_cols = [c for c in needed_cols if c in available]

    if not select_cols:
        print(f"WARNING: None of the expected NPPES columns found. Available: {lf.columns[:20]}...")
        return lf

    lf = lf.select(select_cols)
    print(f"NPPES data: selected {len(select_cols)}/{len(needed_cols)} columns from {nppes_file}")
    return lf
