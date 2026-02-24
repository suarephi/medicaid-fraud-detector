"""Unit tests for all six fraud signal detection algorithms."""
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
    HOME_HEALTH_CODES,
)
from tests.fixtures import (
    make_medicaid_df,
    make_leie_df,
    make_nppes_df,
    TEST_MED_COLS,
)


class TestSignal1ExcludedBilling:
    """Tests for Signal 1: Excluded Provider Still Billing."""

    def test_flags_excluded_provider_billing_after_exclusion(self):
        """Provider excluded in 2020 billing in 2023 should be flagged."""
        medicaid = make_medicaid_df([
            {"npi": "1111111111", "service_date": date(2023, 6, 1), "payment": 5000.0, "claims": 10},
            {"npi": "1111111111", "service_date": date(2023, 7, 1), "payment": 3000.0, "claims": 5},
        ])
        leie = make_leie_df([
            {"npi": "1111111111", "excldate": "20200115", "reindate": None},
        ])

        flags = signal_1_excluded_billing(medicaid.lazy(), TEST_MED_COLS, leie)

        assert len(flags) == 1
        assert flags[0]["npi"] == "1111111111"
        assert flags[0]["details"]["post_exclusion_paid"] == 8000.0

    def test_ignores_reinstated_provider(self):
        """Provider who was reinstated should not be flagged."""
        medicaid = make_medicaid_df([
            {"npi": "2222222222", "service_date": date(2023, 6, 1), "payment": 5000.0, "claims": 10},
        ])
        leie = make_leie_df([
            {"npi": "2222222222", "excldate": "20200115", "reindate": "20210601"},
        ])

        flags = signal_1_excluded_billing(medicaid.lazy(), TEST_MED_COLS, leie)
        assert len(flags) == 0

    def test_ignores_billing_before_exclusion(self):
        """Billing before exclusion date should not be flagged."""
        medicaid = make_medicaid_df([
            {"npi": "3333333333", "service_date": date(2019, 6, 1), "payment": 5000.0, "claims": 10},
        ])
        leie = make_leie_df([
            {"npi": "3333333333", "excldate": "20200115", "reindate": None},
        ])

        flags = signal_1_excluded_billing(medicaid.lazy(), TEST_MED_COLS, leie)
        assert len(flags) == 0

    def test_handles_null_npi_gracefully(self):
        """Null or empty NPIs should not cause crashes."""
        medicaid = make_medicaid_df([
            {"npi": "", "service_date": date(2023, 6, 1), "payment": 5000.0, "claims": 10},
            {"npi": "1111111111", "service_date": date(2023, 6, 1), "payment": 3000.0, "claims": 5},
        ])
        leie = make_leie_df([
            {"npi": "", "excldate": "20200115", "reindate": None},
            {"npi": "1111111111", "excldate": "20200115", "reindate": None},
        ])

        flags = signal_1_excluded_billing(medicaid.lazy(), TEST_MED_COLS, leie)
        # Only the valid NPI should be flagged
        assert all(f["npi"] != "" for f in flags)
        assert all(f["npi"] != "0000000000" for f in flags)


class TestSignal2VolumeOutlier:
    """Tests for Signal 2: Billing Volume Outlier."""

    def test_flags_outlier_above_99th_percentile(self):
        """Provider far above peers should be flagged."""
        # Create 20 normal providers + 1 outlier in same taxonomy/state
        rows = []
        for i in range(20):
            rows.append({
                "npi": f"10000000{i:02d}",
                "payment": 10000.0,
                "claims": 50,
            })
        # Outlier with 100x normal billing
        rows.append({"npi": "9999999999", "payment": 1000000.0, "claims": 5000})

        medicaid = make_medicaid_df(rows)

        # All providers in same taxonomy/state
        nppes_rows = []
        for i in range(20):
            nppes_rows.append({"npi": f"10000000{i:02d}", "taxonomy": "207Q00000X", "state": "CA"})
        nppes_rows.append({"npi": "9999999999", "taxonomy": "207Q00000X", "state": "CA"})

        nppes = make_nppes_df(nppes_rows)

        flags = signal_2_volume_outlier(medicaid.lazy(), TEST_MED_COLS, nppes.lazy())

        flagged_npis = {f["npi"] for f in flags}
        assert "9999999999" in flagged_npis

    def test_no_flags_for_uniform_billing(self):
        """All providers billing the same amount should produce no outliers."""
        rows = [{"npi": f"10000000{i:02d}", "payment": 10000.0, "claims": 50} for i in range(10)]
        medicaid = make_medicaid_df(rows)

        nppes_rows = [{"npi": f"10000000{i:02d}", "taxonomy": "207Q00000X", "state": "CA"} for i in range(10)]
        nppes = make_nppes_df(nppes_rows)

        flags = signal_2_volume_outlier(medicaid.lazy(), TEST_MED_COLS, nppes.lazy())
        assert len(flags) == 0

    def test_small_peer_group_ignored(self):
        """Peer groups with fewer than 5 providers should not produce flags."""
        rows = [
            {"npi": "1000000001", "payment": 10000.0, "claims": 50},
            {"npi": "1000000002", "payment": 10000.0, "claims": 50},
            {"npi": "9999999999", "payment": 1000000.0, "claims": 5000},
        ]
        medicaid = make_medicaid_df(rows)

        nppes_rows = [
            {"npi": "1000000001", "taxonomy": "207Q00000X", "state": "CA"},
            {"npi": "1000000002", "taxonomy": "207Q00000X", "state": "CA"},
            {"npi": "9999999999", "taxonomy": "207Q00000X", "state": "CA"},
        ]
        nppes = make_nppes_df(nppes_rows)

        flags = signal_2_volume_outlier(medicaid.lazy(), TEST_MED_COLS, nppes.lazy())
        assert len(flags) == 0  # Peer group too small


class TestSignal3RapidEscalation:
    """Tests for Signal 3: Rapid Billing Escalation."""

    def test_flags_rapid_growth_new_provider(self):
        """New provider with >500% MoM growth and >$25k should be flagged."""
        rows = [
            {"npi": "5555555555", "service_date": date(2023, 1, 15), "payment": 1000.0, "claims": 1},
            {"npi": "5555555555", "service_date": date(2023, 2, 15), "payment": 5000.0, "claims": 5},
            {"npi": "5555555555", "service_date": date(2023, 3, 15), "payment": 50000.0, "claims": 20},
        ]
        medicaid = make_medicaid_df(rows)

        nppes = make_nppes_df([
            {"npi": "5555555555", "enum_date": "01/01/2023"},
        ])

        flags = signal_3_rapid_escalation(medicaid.lazy(), TEST_MED_COLS, nppes.lazy())

        assert len(flags) >= 1
        assert flags[0]["details"]["peak_growth_rate"] > 500

    def test_ignores_established_provider(self):
        """Provider enumerated >24 months before first claim should not be flagged."""
        rows = [
            {"npi": "5555555555", "service_date": date(2023, 1, 15), "payment": 100.0, "claims": 1},
            {"npi": "5555555555", "service_date": date(2023, 2, 15), "payment": 500.0, "claims": 5},
            {"npi": "5555555555", "service_date": date(2023, 3, 15), "payment": 2000.0, "claims": 20},
        ]
        medicaid = make_medicaid_df(rows)

        nppes = make_nppes_df([
            {"npi": "5555555555", "enum_date": "01/01/2018"},  # 5 years before first claim
        ])

        flags = signal_3_rapid_escalation(medicaid.lazy(), TEST_MED_COLS, nppes.lazy())
        assert len(flags) == 0


class TestSignal4WorkforceImpossibility:
    """Tests for Signal 4: Workforce Impossibility."""

    def test_flags_impossible_claim_volume(self):
        """Organization with >20 claims/hour should be flagged."""
        # 176 hours/month * 20 = 3520 claims is the threshold
        rows = [
            {"npi": "6666666666", "service_date": date(2023, 6, 1), "payment": 50000.0, "claims": 5000},
        ]
        medicaid = make_medicaid_df(rows)

        nppes = make_nppes_df([
            {"npi": "6666666666", "entity_type": "2", "org_name": "Big Clinic LLC"},
        ])

        flags = signal_4_workforce_impossibility(medicaid.lazy(), TEST_MED_COLS, nppes.lazy())

        assert len(flags) == 1
        assert flags[0]["details"]["implied_claims_per_hour"] > 20

    def test_ignores_individual_providers(self):
        """Individual providers (Type 1) should not be checked."""
        rows = [
            {"npi": "7777777777", "service_date": date(2023, 6, 1), "payment": 50000.0, "claims": 5000},
        ]
        medicaid = make_medicaid_df(rows)

        nppes = make_nppes_df([
            {"npi": "7777777777", "entity_type": "1"},  # Individual
        ])

        flags = signal_4_workforce_impossibility(medicaid.lazy(), TEST_MED_COLS, nppes.lazy())
        assert len(flags) == 0

    def test_reasonable_volume_not_flagged(self):
        """Organization with reasonable claim volume should not be flagged."""
        # 100 claims/month = 0.57 claims/hour - well under threshold
        rows = [
            {"npi": "6666666666", "service_date": date(2023, 6, 1), "payment": 5000.0, "claims": 100},
        ]
        medicaid = make_medicaid_df(rows)

        nppes = make_nppes_df([
            {"npi": "6666666666", "entity_type": "2", "org_name": "Normal Clinic LLC"},
        ])

        flags = signal_4_workforce_impossibility(medicaid.lazy(), TEST_MED_COLS, nppes.lazy())
        assert len(flags) == 0


class TestSignal5SharedOfficial:
    """Tests for Signal 5: Shared Authorized Official."""

    def test_flags_official_controlling_5_plus_npis(self):
        """Official controlling 5+ NPIs with >$1M combined billing."""
        medicaid_rows = []
        nppes_rows = []
        for i in range(6):
            npi = f"80000000{i:02d}"
            medicaid_rows.append({"npi": npi, "payment": 250000.0, "claims": 100})
            nppes_rows.append({
                "npi": npi,
                "entity_type": "2",
                "org_name": f"Clinic {i}",
                "auth_last": "SMITH",
                "auth_first": "JANE",
            })

        medicaid = make_medicaid_df(medicaid_rows)
        nppes = make_nppes_df(nppes_rows)

        flags = signal_5_shared_official(medicaid.lazy(), TEST_MED_COLS, nppes.lazy())

        assert len(flags) >= 1
        assert flags[0]["details"]["combined_total"] > 1_000_000

    def test_handles_null_first_name(self):
        """Officials with null first name should not crash."""
        medicaid_rows = []
        nppes_rows = []
        for i in range(6):
            npi = f"80000000{i:02d}"
            medicaid_rows.append({"npi": npi, "payment": 250000.0, "claims": 100})
            nppes_rows.append({
                "npi": npi,
                "entity_type": "2",
                "org_name": f"Clinic {i}",
                "auth_last": "SMITH",
                "auth_first": None,  # Null first name
            })

        medicaid = make_medicaid_df(medicaid_rows)

        # Need to handle null in the fixture
        nppes_data = []
        for r in nppes_rows:
            nppes_data.append({
                "NPI": str(r["npi"]),
                "Entity Type Code": str(r.get("entity_type", "1")),
                "Provider Organization Name (Legal Business Name)": r.get("org_name", ""),
                "Provider Last Name (Legal Name)": r.get("last_name", "DOE"),
                "Provider First Name": r.get("first_name", "JOHN"),
                "Provider Business Practice Location Address State Name": r.get("state", "CA"),
                "Healthcare Provider Taxonomy Code_1": r.get("taxonomy", "207Q00000X"),
                "Provider Enumeration Date": r.get("enum_date", "01/01/2020"),
                "Authorized Official Last Name": r.get("auth_last", ""),
                "Authorized Official First Name": None,  # Explicitly null
                "Authorized Official Telephone Number": r.get("auth_phone", ""),
            })
        nppes = pl.DataFrame(nppes_data)

        # Should not crash
        flags = signal_5_shared_official(medicaid.lazy(), TEST_MED_COLS, nppes.lazy())
        assert isinstance(flags, list)


class TestSignal6GeographicImplausibility:
    """Tests for Signal 6: Geographic Implausibility."""

    def test_flags_low_beneficiary_ratio(self):
        """Home health provider with bene/claims ratio < 0.05 and >500 claims."""
        rows = [
            {
                "npi": "9000000001",
                "hcpcs": "G0151",
                "service_date": date(2023, 6, 1),
                "benes": 10,
                "claims": 1000,
                "payment": 50000.0,
            },
        ]
        medicaid = make_medicaid_df(rows)

        flags = signal_6_geographic_implausibility(medicaid.lazy(), TEST_MED_COLS)

        assert len(flags) == 1
        assert flags[0]["details"]["ratio"] < 0.05
        assert flags[0]["details"]["claims"] > 500

    def test_normal_ratio_not_flagged(self):
        """Provider with normal beneficiary ratio should not be flagged."""
        rows = [
            {
                "npi": "9000000001",
                "hcpcs": "G0151",
                "service_date": date(2023, 6, 1),
                "benes": 80,
                "claims": 600,
                "payment": 30000.0,
            },
        ]
        medicaid = make_medicaid_df(rows)

        flags = signal_6_geographic_implausibility(medicaid.lazy(), TEST_MED_COLS)
        assert len(flags) == 0  # Ratio 0.133 is above 0.05 threshold

    def test_home_health_codes_populated(self):
        """Verify the home health code set contains expected codes."""
        assert "G0151" in HOME_HEALTH_CODES
        assert "G0162" in HOME_HEALTH_CODES
        assert "G0299" in HOME_HEALTH_CODES
        assert "T1019" in HOME_HEALTH_CODES
        assert "S9122" in HOME_HEALTH_CODES
        # Non-home-health codes should not be included
        assert "99213" not in HOME_HEALTH_CODES
