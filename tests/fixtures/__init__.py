"""Synthetic test data generators for each fraud signal."""
from datetime import date, timedelta

import polars as pl


def make_medicaid_df(rows: list[dict]) -> pl.DataFrame:
    """Create a synthetic Medicaid billing DataFrame.

    Each row should have: npi, hcpcs, service_date, benes, claims, payment
    """
    defaults = {
        "Rndrng_NPI": "1234567890",
        "HCPCS_Cd": "99213",
        "Srvc_Dt": date(2023, 6, 15),
        "Bene_Cnt": 10,
        "Clm_Cnt": 20,
        "Srvc_Cnt": 20,
        "Pymt_Amt": 1000.0,
    }
    full_rows = []
    for r in rows:
        row = dict(defaults)
        if "npi" in r:
            row["Rndrng_NPI"] = str(r["npi"])
        if "hcpcs" in r:
            row["HCPCS_Cd"] = r["hcpcs"]
        if "service_date" in r:
            row["Srvc_Dt"] = r["service_date"]
        if "benes" in r:
            row["Bene_Cnt"] = r["benes"]
        if "claims" in r:
            row["Clm_Cnt"] = r["claims"]
        if "payment" in r:
            row["Pymt_Amt"] = r["payment"]
        if "services" in r:
            row["Srvc_Cnt"] = r["services"]
        full_rows.append(row)

    return pl.DataFrame(full_rows)


def make_leie_df(rows: list[dict]) -> pl.DataFrame:
    """Create a synthetic LEIE exclusion DataFrame."""
    full_rows = []
    for r in rows:
        row = {
            "LASTNAME": r.get("lastname", "DOE"),
            "FIRSTNAME": r.get("firstname", "JOHN"),
            "NPI": str(r.get("npi", "")),
            "EXCLDATE": r.get("excldate", "20200101"),
            "EXCLTYPE": r.get("excltype", "1128(a)(1)"),
            "REINDATE": r.get("reindate", None),
        }
        full_rows.append(row)

    df = pl.DataFrame(full_rows)
    # Parse dates like ingest.py does
    df = df.with_columns([
        pl.col("EXCLDATE").cast(pl.Utf8).str.strptime(pl.Date, "%Y%m%d", strict=False)
        .alias("excl_date_parsed"),
        pl.col("REINDATE").cast(pl.Utf8).str.strptime(pl.Date, "%Y%m%d", strict=False)
        .alias("rein_date_parsed"),
        pl.col("NPI").cast(pl.Utf8).str.strip_chars().str.zfill(10).alias("npi_str"),
    ])
    return df


def make_nppes_df(rows: list[dict]) -> pl.DataFrame:
    """Create a synthetic NPPES DataFrame with the 11 required columns."""
    full_rows = []
    for r in rows:
        row = {
            "NPI": str(r.get("npi", "1234567890")),
            "Entity Type Code": str(r.get("entity_type", "1")),
            "Provider Organization Name (Legal Business Name)": r.get("org_name", ""),
            "Provider Last Name (Legal Name)": r.get("last_name", "DOE"),
            "Provider First Name": r.get("first_name", "JOHN"),
            "Provider Business Practice Location Address State Name": r.get("state", "CA"),
            "Healthcare Provider Taxonomy Code_1": r.get("taxonomy", "207Q00000X"),
            "Provider Enumeration Date": r.get("enum_date", "01/01/2020"),
            "Authorized Official Last Name": r.get("auth_last", ""),
            "Authorized Official First Name": r.get("auth_first", ""),
            "Authorized Official Telephone Number": r.get("auth_phone", ""),
        }
        full_rows.append(row)

    return pl.DataFrame(full_rows)


# Standard column mapping for test Medicaid data
TEST_MED_COLS = {
    "npi": "Rndrng_NPI",
    "hcpcs": "HCPCS_Cd",
    "date": "Srvc_Dt",
    "benes": "Bene_Cnt",
    "claims": "Clm_Cnt",
    "payment": "Pymt_Amt",
}
