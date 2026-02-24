"""Integration tests: run synthetic data through the full signal pipeline."""
from datetime import date

import polars as pl
import pytest

from src.signals import (
    signal_1_excluded_billing,
    signal_2_volume_outlier,
    signal_3_rapid_escalation,
    signal_4_workforce_impossibility,
    signal_5_shared_official,
    signal_6_geographic_implausibility,
)
from src.output import build_provider_entry, build_report, SIGNAL_TYPES
from tests.fixtures import make_medicaid_df, make_leie_df, make_nppes_df, TEST_MED_COLS


def _build_synthetic_datasets():
    """Build synthetic Medicaid, LEIE, and NPPES datasets that trigger all 6 signals."""
    medicaid_rows = []
    nppes_rows = []

    # Signal 1: Excluded provider billing after exclusion
    medicaid_rows.append(
        {"npi": "1000000001", "service_date": date(2023, 6, 1), "payment": 50000.0, "claims": 20, "benes": 15}
    )

    # Signal 2: Volume outlier (need 20+ providers in same peer group, one extreme)
    for i in range(20):
        npi = f"20000000{i:02d}"
        medicaid_rows.append({"npi": npi, "payment": 10000.0, "claims": 50, "benes": 30})
        nppes_rows.append({"npi": npi, "taxonomy": "207Q00000X", "state": "TX"})
    # The outlier
    medicaid_rows.append({"npi": "2099999999", "payment": 2000000.0, "claims": 5000, "benes": 100})
    nppes_rows.append({"npi": "2099999999", "taxonomy": "207Q00000X", "state": "TX"})

    # Signal 3: Rapid escalation from a new provider
    for month in range(1, 5):
        medicaid_rows.append({
            "npi": "3000000001",
            "service_date": date(2023, month, 15),
            "payment": 1000.0 * (5 ** month),
            "claims": 10 * month,
            "benes": 5,
        })
    nppes_rows.append({"npi": "3000000001", "enum_date": "01/01/2023"})

    # Signal 4: Workforce impossibility (org with >6 claims/hour)
    medicaid_rows.append({
        "npi": "4000000001", "service_date": date(2023, 6, 1),
        "payment": 100000.0, "claims": 5000, "benes": 50,
    })
    nppes_rows.append({"npi": "4000000001", "entity_type": "2", "org_name": "Fake Clinic LLC"})

    # Signal 5: Shared authorized official (6 NPIs, >$1M combined)
    for i in range(6):
        npi = f"50000000{i:02d}"
        medicaid_rows.append({"npi": npi, "payment": 300000.0, "claims": 100, "benes": 40})
        nppes_rows.append({
            "npi": npi, "entity_type": "2", "org_name": f"Shell Corp {i}",
            "auth_last": "FRAUDSTER", "auth_first": "FRANK",
        })

    # Signal 6: Geographic implausibility (home health, low bene/claim ratio)
    medicaid_rows.append({
        "npi": "6000000001", "hcpcs": "G0151",
        "service_date": date(2023, 6, 1),
        "benes": 3, "claims": 300, "payment": 20000.0,
    })

    # Also add NPPES entries for providers used in signals 1, 3, 6
    nppes_rows.append({"npi": "1000000001", "entity_type": "1", "last_name": "EXCLUDED", "first_name": "JOHN"})
    nppes_rows.append({"npi": "6000000001", "entity_type": "1", "last_name": "GHOST", "first_name": "NURSE"})

    medicaid = make_medicaid_df(medicaid_rows)
    leie = make_leie_df([
        {"npi": "1000000001", "excldate": "20200101", "reindate": None},
    ])
    nppes = make_nppes_df(nppes_rows)

    return medicaid, leie, nppes


class TestFullPipeline:
    """Run all signals and build a complete report from synthetic data."""

    @pytest.fixture(autouse=True)
    def setup_datasets(self):
        self.medicaid, self.leie, self.nppes = _build_synthetic_datasets()

    def test_signal_1_detects_excluded_provider(self):
        flags = signal_1_excluded_billing(self.medicaid.lazy(), TEST_MED_COLS, self.leie)
        flagged_npis = {f["npi"] for f in flags}
        assert "1000000001" in flagged_npis

    def test_signal_2_detects_outlier(self):
        flags = signal_2_volume_outlier(self.medicaid.lazy(), TEST_MED_COLS, self.nppes.lazy())
        flagged_npis = {f["npi"] for f in flags}
        assert "2099999999" in flagged_npis

    def test_signal_4_detects_workforce_impossibility(self):
        flags = signal_4_workforce_impossibility(self.medicaid.lazy(), TEST_MED_COLS, self.nppes.lazy())
        flagged_npis = {f["npi"] for f in flags}
        assert "4000000001" in flagged_npis

    def test_signal_5_detects_shared_official(self):
        flags = signal_5_shared_official(self.medicaid.lazy(), TEST_MED_COLS, self.nppes.lazy())
        assert len(flags) >= 1
        assert flags[0]["details"]["official_name"] == "FRAUDSTER, FRANK"

    def test_signal_6_detects_geographic_implausibility(self):
        flags = signal_6_geographic_implausibility(self.medicaid.lazy(), TEST_MED_COLS)
        flagged_npis = {f["npi"] for f in flags}
        assert "6000000001" in flagged_npis

    def test_full_report_schema(self):
        """Run all signals, build report, and validate the complete JSON schema."""
        all_flags = []
        signal_tallies = {}

        # Run each signal
        s1 = signal_1_excluded_billing(self.medicaid.lazy(), TEST_MED_COLS, self.leie)
        signal_tallies["signal_1"] = len(s1)
        all_flags.extend(s1)

        s2 = signal_2_volume_outlier(self.medicaid.lazy(), TEST_MED_COLS, self.nppes.lazy())
        signal_tallies["signal_2"] = len(s2)
        all_flags.extend(s2)

        s3 = signal_3_rapid_escalation(self.medicaid.lazy(), TEST_MED_COLS, self.nppes.lazy())
        signal_tallies["signal_3"] = len(s3)
        all_flags.extend(s3)

        s4 = signal_4_workforce_impossibility(self.medicaid.lazy(), TEST_MED_COLS, self.nppes.lazy())
        signal_tallies["signal_4"] = len(s4)
        all_flags.extend(s4)

        s5 = signal_5_shared_official(self.medicaid.lazy(), TEST_MED_COLS, self.nppes.lazy())
        signal_tallies["signal_5"] = len(s5)
        all_flags.extend(s5)

        s6 = signal_6_geographic_implausibility(self.medicaid.lazy(), TEST_MED_COLS)
        signal_tallies["signal_6"] = len(s6)
        all_flags.extend(s6)

        # We should have flags from multiple signals
        assert len(all_flags) >= 4, f"Expected at least 4 flags, got {len(all_flags)}"

        # Group by NPI and build provider entries
        npi_flags = {}
        for f in all_flags:
            npi_flags.setdefault(f["npi"], []).append(f)

        provider_entries = []
        for npi, flags in npi_flags.items():
            entry = build_provider_entry(
                npi=npi,
                provider_name="Test Provider",
                entity_type="individual",
                taxonomy_code="207Q00000X",
                state="TX",
                enumeration_date="01/01/2020",
                lifetime_paid=100000.0,
                lifetime_claims=500,
                lifetime_benes=100,
                signals=flags,
            )
            provider_entries.append(entry)

        report = build_report(
            flagged_providers=provider_entries,
            scan_count=100,
            signal_tallies=signal_tallies,
        )

        # Validate top-level report fields
        assert "generated_at" in report
        assert "tool_version" in report
        assert "total_providers_scanned" in report
        assert report["total_providers_scanned"] == 100
        assert "total_providers_flagged" in report
        assert report["total_providers_flagged"] == len(provider_entries)
        assert "signal_counts" in report
        assert "flagged_providers" in report

        # Validate signal_counts has all six signal types
        for signal_type in SIGNAL_TYPES.values():
            assert signal_type in report["signal_counts"]

        # Validate each provider entry
        for entry in report["flagged_providers"]:
            assert "npi" in entry
            assert "provider_name" in entry
            assert "entity_type" in entry
            assert "signals" in entry
            assert "estimated_overpayment_usd" in entry
            assert "fca_relevance" in entry
            assert isinstance(entry["signals"], list)
            assert len(entry["signals"]) >= 1

            for sig in entry["signals"]:
                assert "signal_type" in sig
                assert sig["signal_type"] in SIGNAL_TYPES.values()
                assert "severity" in sig
                assert sig["severity"] in ("low", "medium", "high", "critical")
                assert "evidence" in sig

    def test_all_flag_entries_have_required_fields(self):
        """Each raw flag entry should have npi, signal_id, and details."""
        all_flags = []
        all_flags.extend(signal_1_excluded_billing(self.medicaid.lazy(), TEST_MED_COLS, self.leie))
        all_flags.extend(signal_2_volume_outlier(self.medicaid.lazy(), TEST_MED_COLS, self.nppes.lazy()))
        all_flags.extend(signal_4_workforce_impossibility(self.medicaid.lazy(), TEST_MED_COLS, self.nppes.lazy()))
        all_flags.extend(signal_5_shared_official(self.medicaid.lazy(), TEST_MED_COLS, self.nppes.lazy()))
        all_flags.extend(signal_6_geographic_implausibility(self.medicaid.lazy(), TEST_MED_COLS))

        for flag in all_flags:
            assert "npi" in flag, f"Flag missing 'npi': {flag}"
            assert "signal_id" in flag, f"Flag missing 'signal_id': {flag}"
            assert "details" in flag, f"Flag missing 'details': {flag}"
            assert isinstance(flag["details"], dict)
            assert flag["signal_id"] in range(1, 7)
