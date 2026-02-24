"""Tests for the ingest module: column detection, NPI normalization."""
import polars as pl
import pytest

from src.ingest import detect_medicaid_columns, normalize_npi


class TestDetectMedicaidColumns:
    """Tests for detect_medicaid_columns()."""

    def test_detects_standard_column_names(self):
        """Standard CMS-style column names should be detected."""
        columns = ["Rndrng_NPI", "HCPCS_Cd", "Srvc_Dt", "Bene_Cnt", "Clm_Cnt", "Pymt_Amt"]
        mapping = detect_medicaid_columns(columns)
        assert mapping["npi"] == "Rndrng_NPI"
        assert mapping["hcpcs"] == "HCPCS_Cd"
        assert mapping["date"] == "Srvc_Dt"
        assert mapping["benes"] == "Bene_Cnt"
        assert mapping["claims"] == "Clm_Cnt"
        assert mapping["payment"] == "Pymt_Amt"

    def test_detects_alternate_column_names(self):
        """Alternate column name patterns should be detected."""
        columns = ["provider_npi", "hcpcs_code", "service_date", "beneficiary_count",
                    "claim_count", "payment_amount"]
        mapping = detect_medicaid_columns(columns)
        assert mapping["npi"] == "provider_npi"
        assert mapping["hcpcs"] == "hcpcs_code"
        assert mapping["date"] == "service_date"
        assert mapping["benes"] == "beneficiary_count"
        assert mapping["claims"] == "claim_count"
        assert mapping["payment"] == "payment_amount"

    def test_detects_uppercase_columns(self):
        """Case-insensitive detection should work."""
        columns = ["RNDRNG_NPI", "HCPCS_CD", "SRVC_DT", "BENE_CNT", "CLM_CNT", "PYMT_AMT"]
        mapping = detect_medicaid_columns(columns)
        assert mapping["npi"] == "RNDRNG_NPI"
        assert mapping["hcpcs"] == "HCPCS_CD"

    def test_detects_medicaid_specific_patterns(self):
        """Medicaid-specific column names should be detected."""
        columns = ["billing_provider_npi_num", "proc_cd", "clm_dt",
                    "bene_unique_cnt", "tot_clms", "avg_mdcd_pymt_amt"]
        mapping = detect_medicaid_columns(columns)
        assert mapping["npi"] == "billing_provider_npi_num"
        assert mapping["hcpcs"] == "proc_cd"
        assert mapping["date"] == "clm_dt"
        assert mapping["benes"] == "bene_unique_cnt"
        assert mapping["claims"] == "tot_clms"
        assert mapping["payment"] == "avg_mdcd_pymt_amt"

    def test_seven_column_positional_fallback(self):
        """Files with exactly 7 columns and unknown names should use positional fallback."""
        columns = ["col_a", "col_b", "col_c", "col_d", "col_e", "col_f", "col_g"]
        mapping = detect_medicaid_columns(columns)
        # Positional: npi=0, hcpcs=1, date=2, benes=3, claims=4, skip=5, payment=6
        assert mapping["npi"] == "col_a"
        assert mapping["hcpcs"] == "col_b"
        assert mapping["date"] == "col_c"
        assert mapping["benes"] == "col_d"
        assert mapping["claims"] == "col_e"
        assert mapping["payment"] == "col_g"

    def test_partial_detection_fills_remaining(self):
        """Partially matched columns should still produce a full mapping."""
        columns = ["Rndrng_NPI", "unknown_col", "Srvc_Dt", "x", "y", "z"]
        mapping = detect_medicaid_columns(columns)
        assert mapping["npi"] == "Rndrng_NPI"
        assert mapping["date"] == "Srvc_Dt"
        # Remaining should be filled from unmatched columns
        assert len(mapping) == 6


class TestNormalizeNpi:
    """Tests for normalize_npi()."""

    def test_normalizes_string_npi(self):
        """String NPI should be padded to 10 digits."""
        df = pl.DataFrame({"npi": ["1234567890"]})
        result = df.select(normalize_npi(pl.col("npi")).alias("npi"))
        assert result["npi"][0] == "1234567890"

    def test_pads_short_npi(self):
        """Short NPI (less than 10 chars) should be zero-padded."""
        df = pl.DataFrame({"npi": ["123"]})
        result = df.select(normalize_npi(pl.col("npi")).alias("npi"))
        assert result["npi"][0] == "0000000123"

    def test_handles_integer_npi(self):
        """Integer NPI should be cast to string and padded."""
        df = pl.DataFrame({"npi": [1234567890]})
        result = df.select(normalize_npi(pl.col("npi")).alias("npi"))
        assert result["npi"][0] == "1234567890"

    def test_strips_whitespace(self):
        """Whitespace around NPI should be stripped."""
        df = pl.DataFrame({"npi": ["  1234567890  "]})
        result = df.select(normalize_npi(pl.col("npi")).alias("npi"))
        assert result["npi"][0] == "1234567890"

    def test_handles_empty_string(self):
        """Empty string NPI should become zero-padded."""
        df = pl.DataFrame({"npi": [""]})
        result = df.select(normalize_npi(pl.col("npi")).alias("npi"))
        assert result["npi"][0] == "0000000000"

    def test_handles_multiple_rows(self):
        """Multiple NPIs should all be normalized correctly."""
        df = pl.DataFrame({"npi": ["123", "1234567890", "  456  "]})
        result = df.select(normalize_npi(pl.col("npi")).alias("npi"))
        assert result["npi"].to_list() == ["0000000123", "1234567890", "0000000456"]
