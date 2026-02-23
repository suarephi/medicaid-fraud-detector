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


class TestSignal3RapidEscalation:
    """Tests for Signal 3: Rapid Billing Escalation."""

    def test_flags_rapid_growth_new_provider(self):
        """New provider with >200% MoM growth should be flagged."""
        rows = [
            {"npi": "5555555555", "service_date": date(2023, 1, 15), "payment": 100.0, "claims": 1},
            {"npi": "5555555555", "service_date": date(2023, 2, 15), "payment": 500.0, "claims": 5},
            {"npi": "5555555555", "service_date": date(2023, 3, 15), "payment": 2000.0, "claims": 20},
        ]
        medicaid = make_medicaid_df(rows)

        nppes = make_nppes_df([
            {"npi": "5555555555", "enum_date": "01/01/2023"},
        ])

        flags = signal_3_rapid_escalation(medicaid.lazy(), TEST_MED_COLS, nppes.lazy())

        assert len(flags) >= 1
        assert flags[0]["details"]["peak_growth_rate"] > 200


class TestSignal4WorkforceImpossibility:
    """Tests for Signal 4: Workforce Impossibility."""

    def test_flags_impossible_claim_volume(self):
        """Organization with >6 claims/hour should be flagged."""
        # 176 hours/month * 6 = 1056 claims is the threshold
        rows = [
            {"npi": "6666666666", "service_date": date(2023, 6, 1), "payment": 50000.0, "claims": 2000},
        ]
        medicaid = make_medicaid_df(rows)

        nppes = make_nppes_df([
            {"npi": "6666666666", "entity_type": "2", "org_name": "Big Clinic LLC"},
        ])

        flags = signal_4_workforce_impossibility(medicaid.lazy(), TEST_MED_COLS, nppes.lazy())

        assert len(flags) == 1
        assert flags[0]["details"]["implied_claims_per_hour"] > 6

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


class TestSignal6GeographicImplausibility:
    """Tests for Signal 6: Geographic Implausibility."""

    def test_flags_low_beneficiary_ratio(self):
        """Home health provider with bene/claims ratio < 0.1 and >100 claims."""
        rows = [
            {
                "npi": "9000000001",
                "hcpcs": "G0151",
                "service_date": date(2023, 6, 1),
                "benes": 5,
                "claims": 200,
                "payment": 10000.0,
            },
        ]
        medicaid = make_medicaid_df(rows)

        flags = signal_6_geographic_implausibility(medicaid.lazy(), TEST_MED_COLS)

        assert len(flags) == 1
        assert flags[0]["details"]["ratio"] < 0.1
        assert flags[0]["details"]["claims"] > 100

    def test_home_health_codes_populated(self):
        """Verify the home health code set contains expected codes."""
        assert "G0151" in HOME_HEALTH_CODES
        assert "G0162" in HOME_HEALTH_CODES
        assert "G0299" in HOME_HEALTH_CODES
        assert "T1019" in HOME_HEALTH_CODES
        assert "S9122" in HOME_HEALTH_CODES
        # Non-home-health codes should not be included
        assert "99213" not in HOME_HEALTH_CODES
