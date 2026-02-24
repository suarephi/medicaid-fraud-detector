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
        rows = []
        for i in range(20):
            rows.append({
                "npi": f"10000000{i:02d}",
                "payment": 10000.0,
                "claims": 50,
            })
        rows.append({"npi": "9999999999", "payment": 1000000.0, "claims": 5000})

        medicaid = make_medicaid_df(rows)

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
        assert len(flags) == 0


class TestSignal3RapidEscalation:
    """Tests for Signal 3: Rapid Billing Escalation."""

    def test_flags_rapid_growth_new_provider(self):
        """New provider with >200% rolling 3-month avg growth should be flagged."""
        rows = [
            {"npi": "5555555555", "service_date": date(2023, 1, 15), "payment": 1000.0, "claims": 1},
            {"npi": "5555555555", "service_date": date(2023, 2, 15), "payment": 5000.0, "claims": 5},
            {"npi": "5555555555", "service_date": date(2023, 3, 15), "payment": 25000.0, "claims": 20},
            {"npi": "5555555555", "service_date": date(2023, 4, 15), "payment": 80000.0, "claims": 50},
        ]
        medicaid = make_medicaid_df(rows)

        nppes = make_nppes_df([
            {"npi": "5555555555", "enum_date": "01/01/2023"},
        ])

        flags = signal_3_rapid_escalation(medicaid.lazy(), TEST_MED_COLS, nppes.lazy())

        assert len(flags) >= 1
        assert flags[0]["details"]["peak_growth_rate"] > 200

    def test_ignores_established_provider(self):
        """Provider enumerated >24 months before first claim should not be flagged."""
        rows = [
            {"npi": "5555555555", "service_date": date(2023, 1, 15), "payment": 100.0, "claims": 1},
            {"npi": "5555555555", "service_date": date(2023, 2, 15), "payment": 500.0, "claims": 5},
            {"npi": "5555555555", "service_date": date(2023, 3, 15), "payment": 2000.0, "claims": 20},
            {"npi": "5555555555", "service_date": date(2023, 4, 15), "payment": 10000.0, "claims": 50},
        ]
        medicaid = make_medicaid_df(rows)

        nppes = make_nppes_df([
            {"npi": "5555555555", "enum_date": "01/01/2018"},
        ])

        flags = signal_3_rapid_escalation(medicaid.lazy(), TEST_MED_COLS, nppes.lazy())
        assert len(flags) == 0


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
            {"npi": "7777777777", "entity_type": "1"},
        ])

        flags = signal_4_workforce_impossibility(medicaid.lazy(), TEST_MED_COLS, nppes.lazy())
        assert len(flags) == 0

    def test_reasonable_volume_not_flagged(self):
        """Organization with reasonable claim volume should not be flagged."""
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
                "auth_first": None,
            })

        medicaid = make_medicaid_df(medicaid_rows)

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
                "Authorized Official First Name": None,
                "Authorized Official Telephone Number": r.get("auth_phone", ""),
            })
        nppes = pl.DataFrame(nppes_data)

        flags = signal_5_shared_official(medicaid.lazy(), TEST_MED_COLS, nppes.lazy())
        assert isinstance(flags, list)


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

    def test_normal_ratio_not_flagged(self):
        """Provider with normal beneficiary ratio should not be flagged."""
        rows = [
            {
                "npi": "9000000001",
                "hcpcs": "G0151",
                "service_date": date(2023, 6, 1),
                "benes": 80,
                "claims": 200,
                "payment": 10000.0,
            },
        ]
        medicaid = make_medicaid_df(rows)

        flags = signal_6_geographic_implausibility(medicaid.lazy(), TEST_MED_COLS)
        assert len(flags) == 0  # Ratio 0.4 is above 0.1 threshold

    def test_home_health_codes_populated(self):
        """Verify the home health code set contains expected codes."""
        assert "G0151" in HOME_HEALTH_CODES
        assert "G0162" in HOME_HEALTH_CODES
        assert "G0299" in HOME_HEALTH_CODES
        assert "T1019" in HOME_HEALTH_CODES
        assert "S9122" in HOME_HEALTH_CODES
        assert "99213" not in HOME_HEALTH_CODES


# ──────────────────────────────────────────────────────────────
# Edge-case and boundary-value tests
# ──────────────────────────────────────────────────────────────

def _empty_medicaid():
    """Create an empty Medicaid DataFrame with correct schema (0 rows)."""
    df = make_medicaid_df([{"npi": "0000000000"}])
    return df.filter(pl.col("Rndrng_NPI") == "__impossible__")


class TestSignal1EdgeCases:
    """Edge case tests for Signal 1."""

    def test_empty_medicaid_dataframe(self):
        """Empty Medicaid data should produce no flags."""
        medicaid = _empty_medicaid()
        leie = make_leie_df([
            {"npi": "1111111111", "excldate": "20200115", "reindate": None},
        ])
        flags = signal_1_excluded_billing(medicaid.lazy(), TEST_MED_COLS, leie)
        assert flags == []

    def test_empty_leie_dataframe(self):
        """LEIE with only reinstated providers should produce no flags."""
        medicaid = make_medicaid_df([
            {"npi": "1111111111", "service_date": date(2023, 6, 1), "payment": 5000.0, "claims": 10},
        ])
        # All providers reinstated => effectively empty exclusion set
        leie = make_leie_df([
            {"npi": "9999999999", "excldate": "20200115", "reindate": "20210101"},
        ])
        flags = signal_1_excluded_billing(medicaid.lazy(), TEST_MED_COLS, leie)
        assert flags == []

    def test_billing_on_exclusion_date_not_flagged(self):
        """Billing exactly on the exclusion date should NOT be flagged (needs to be after)."""
        medicaid = make_medicaid_df([
            {"npi": "1111111111", "service_date": date(2020, 1, 15), "payment": 5000.0, "claims": 10},
        ])
        leie = make_leie_df([
            {"npi": "1111111111", "excldate": "20200115", "reindate": None},
        ])
        flags = signal_1_excluded_billing(medicaid.lazy(), TEST_MED_COLS, leie)
        assert len(flags) == 0

    def test_multiple_excluded_providers(self):
        """Multiple excluded providers billing should each be flagged."""
        medicaid = make_medicaid_df([
            {"npi": "1111111111", "service_date": date(2023, 6, 1), "payment": 5000.0, "claims": 10},
            {"npi": "2222222222", "service_date": date(2023, 6, 1), "payment": 3000.0, "claims": 5},
        ])
        leie = make_leie_df([
            {"npi": "1111111111", "excldate": "20200115", "reindate": None},
            {"npi": "2222222222", "excldate": "20190101", "reindate": None},
        ])
        flags = signal_1_excluded_billing(medicaid.lazy(), TEST_MED_COLS, leie)
        flagged_npis = {f["npi"] for f in flags}
        assert "1111111111" in flagged_npis
        assert "2222222222" in flagged_npis

    def test_very_large_payment_values(self):
        """Very large payment values should be handled correctly."""
        medicaid = make_medicaid_df([
            {"npi": "1111111111", "service_date": date(2023, 6, 1), "payment": 99999999.99, "claims": 1},
        ])
        leie = make_leie_df([
            {"npi": "1111111111", "excldate": "20200115", "reindate": None},
        ])
        flags = signal_1_excluded_billing(medicaid.lazy(), TEST_MED_COLS, leie)
        assert len(flags) == 1
        assert flags[0]["details"]["post_exclusion_paid"] == 99999999.99


class TestSignal2EdgeCases:
    """Edge case tests for Signal 2."""

    def test_empty_medicaid_dataframe(self):
        """Empty Medicaid data should produce no flags."""
        medicaid = _empty_medicaid()
        nppes = make_nppes_df([{"npi": "1000000001", "taxonomy": "207Q00000X", "state": "CA"}])
        flags = signal_2_volume_outlier(medicaid.lazy(), TEST_MED_COLS, nppes.lazy())
        assert flags == []

    def test_no_nppes_match_produces_no_flags(self):
        """If no Medicaid NPIs match NPPES, no flags should be produced."""
        rows = [{"npi": f"10000000{i:02d}", "payment": 10000.0, "claims": 50} for i in range(10)]
        medicaid = make_medicaid_df(rows)
        nppes = make_nppes_df([{"npi": "9999999999", "taxonomy": "207Q00000X", "state": "CA"}])
        flags = signal_2_volume_outlier(medicaid.lazy(), TEST_MED_COLS, nppes.lazy())
        assert flags == []

    def test_very_large_payment_outlier(self):
        """Extremely large payment should be flagged."""
        rows = [{"npi": f"10000000{i:02d}", "payment": 10000.0, "claims": 50} for i in range(20)]
        rows.append({"npi": "9999999999", "payment": 50000000.0, "claims": 5000})
        medicaid = make_medicaid_df(rows)

        nppes_rows = [{"npi": f"10000000{i:02d}", "taxonomy": "207Q00000X", "state": "CA"} for i in range(20)]
        nppes_rows.append({"npi": "9999999999", "taxonomy": "207Q00000X", "state": "CA"})
        nppes = make_nppes_df(nppes_rows)

        flags = signal_2_volume_outlier(medicaid.lazy(), TEST_MED_COLS, nppes.lazy())
        flagged_npis = {f["npi"] for f in flags}
        assert "9999999999" in flagged_npis
        assert flags[0]["details"]["total_paid"] == 50000000.0


class TestSignal3EdgeCases:
    """Edge case tests for Signal 3."""

    def test_empty_medicaid_dataframe(self):
        """Empty Medicaid data should produce no flags."""
        medicaid = _empty_medicaid()
        nppes = make_nppes_df([{"npi": "5555555555", "enum_date": "01/01/2023"}])
        flags = signal_3_rapid_escalation(medicaid.lazy(), TEST_MED_COLS, nppes.lazy())
        assert flags == []

    def test_single_month_no_growth(self):
        """A provider with only one month of billing cannot have growth."""
        rows = [
            {"npi": "5555555555", "service_date": date(2023, 1, 15), "payment": 100000.0, "claims": 50},
        ]
        medicaid = make_medicaid_df(rows)
        nppes = make_nppes_df([{"npi": "5555555555", "enum_date": "01/01/2023"}])
        flags = signal_3_rapid_escalation(medicaid.lazy(), TEST_MED_COLS, nppes.lazy())
        assert flags == []

    def test_stable_growth_not_flagged(self):
        """Steady small growth should not be flagged."""
        rows = [
            {"npi": "5555555555", "service_date": date(2023, 1, 15), "payment": 1000.0, "claims": 1},
            {"npi": "5555555555", "service_date": date(2023, 2, 15), "payment": 1100.0, "claims": 1},
            {"npi": "5555555555", "service_date": date(2023, 3, 15), "payment": 1200.0, "claims": 1},
            {"npi": "5555555555", "service_date": date(2023, 4, 15), "payment": 1300.0, "claims": 2},
        ]
        medicaid = make_medicaid_df(rows)
        nppes = make_nppes_df([{"npi": "5555555555", "enum_date": "01/01/2023"}])
        flags = signal_3_rapid_escalation(medicaid.lazy(), TEST_MED_COLS, nppes.lazy())
        assert flags == []


class TestSignal4EdgeCases:
    """Edge case tests for Signal 4."""

    def test_empty_medicaid_dataframe(self):
        """Empty Medicaid data should produce no flags."""
        medicaid = _empty_medicaid()
        nppes = make_nppes_df([{"npi": "6666666666", "entity_type": "2", "org_name": "Clinic"}])
        flags = signal_4_workforce_impossibility(medicaid.lazy(), TEST_MED_COLS, nppes.lazy())
        assert flags == []

    def test_exactly_at_threshold_not_flagged(self):
        """Exactly 6 claims/hour (1056 claims/month) should NOT be flagged (needs >6)."""
        # 6 claims/hour * 176 hours = 1056 claims exactly
        rows = [
            {"npi": "6666666666", "service_date": date(2023, 6, 1), "payment": 50000.0, "claims": 1056},
        ]
        medicaid = make_medicaid_df(rows)
        nppes = make_nppes_df([{"npi": "6666666666", "entity_type": "2", "org_name": "Clinic"}])
        flags = signal_4_workforce_impossibility(medicaid.lazy(), TEST_MED_COLS, nppes.lazy())
        assert len(flags) == 0

    def test_just_above_threshold_flagged(self):
        """1057 claims (just above 6 claims/hour threshold) should be flagged."""
        rows = [
            {"npi": "6666666666", "service_date": date(2023, 6, 1), "payment": 50000.0, "claims": 1057},
        ]
        medicaid = make_medicaid_df(rows)
        nppes = make_nppes_df([{"npi": "6666666666", "entity_type": "2", "org_name": "Clinic"}])
        flags = signal_4_workforce_impossibility(medicaid.lazy(), TEST_MED_COLS, nppes.lazy())
        assert len(flags) == 1
        assert flags[0]["details"]["implied_claims_per_hour"] > 6

    def test_very_large_claim_volume(self):
        """Extremely large claim volume should be handled."""
        rows = [
            {"npi": "6666666666", "service_date": date(2023, 6, 1), "payment": 5000000.0, "claims": 100000},
        ]
        medicaid = make_medicaid_df(rows)
        nppes = make_nppes_df([{"npi": "6666666666", "entity_type": "2", "org_name": "Mega Clinic"}])
        flags = signal_4_workforce_impossibility(medicaid.lazy(), TEST_MED_COLS, nppes.lazy())
        assert len(flags) == 1
        assert flags[0]["details"]["implied_claims_per_hour"] > 100

    def test_multiple_months_peak_month_used(self):
        """Signal should flag based on peak month, not average."""
        rows = [
            {"npi": "6666666666", "service_date": date(2023, 5, 1), "payment": 1000.0, "claims": 50},
            {"npi": "6666666666", "service_date": date(2023, 6, 1), "payment": 50000.0, "claims": 2000},
            {"npi": "6666666666", "service_date": date(2023, 7, 1), "payment": 1000.0, "claims": 50},
        ]
        medicaid = make_medicaid_df(rows)
        nppes = make_nppes_df([{"npi": "6666666666", "entity_type": "2", "org_name": "Clinic"}])
        flags = signal_4_workforce_impossibility(medicaid.lazy(), TEST_MED_COLS, nppes.lazy())
        assert len(flags) == 1


class TestSignal5EdgeCases:
    """Edge case tests for Signal 5."""

    def test_empty_medicaid_dataframe(self):
        """Empty Medicaid data should produce no flags."""
        medicaid = _empty_medicaid()
        nppes_rows = []
        for i in range(6):
            nppes_rows.append({
                "npi": f"80000000{i:02d}", "entity_type": "2",
                "org_name": f"Clinic {i}", "auth_last": "SMITH", "auth_first": "JANE",
            })
        nppes = make_nppes_df(nppes_rows)
        flags = signal_5_shared_official(medicaid.lazy(), TEST_MED_COLS, nppes.lazy())
        assert flags == []

    def test_four_npis_not_flagged(self):
        """Official controlling only 4 NPIs should not be flagged (needs 5+)."""
        medicaid_rows = []
        nppes_rows = []
        for i in range(4):
            npi = f"80000000{i:02d}"
            medicaid_rows.append({"npi": npi, "payment": 500000.0, "claims": 100})
            nppes_rows.append({
                "npi": npi, "entity_type": "2", "org_name": f"Clinic {i}",
                "auth_last": "SMITH", "auth_first": "JANE",
            })
        medicaid = make_medicaid_df(medicaid_rows)
        nppes = make_nppes_df(nppes_rows)
        flags = signal_5_shared_official(medicaid.lazy(), TEST_MED_COLS, nppes.lazy())
        assert flags == []

    def test_five_npis_under_1m_not_flagged(self):
        """Official with 5 NPIs but under $1M combined should not be flagged."""
        medicaid_rows = []
        nppes_rows = []
        for i in range(5):
            npi = f"80000000{i:02d}"
            medicaid_rows.append({"npi": npi, "payment": 100000.0, "claims": 50})
            nppes_rows.append({
                "npi": npi, "entity_type": "2", "org_name": f"Clinic {i}",
                "auth_last": "DOE", "auth_first": "JOHN",
            })
        medicaid = make_medicaid_df(medicaid_rows)
        nppes = make_nppes_df(nppes_rows)
        flags = signal_5_shared_official(medicaid.lazy(), TEST_MED_COLS, nppes.lazy())
        assert flags == []


class TestSignal6EdgeCases:
    """Edge case tests for Signal 6."""

    def test_empty_medicaid_dataframe(self):
        """Empty Medicaid data should produce no flags."""
        medicaid = _empty_medicaid()
        flags = signal_6_geographic_implausibility(medicaid.lazy(), TEST_MED_COLS)
        assert flags == []

    def test_exactly_at_ratio_threshold_not_flagged(self):
        """Ratio of exactly 0.1 should NOT be flagged (needs < 0.1)."""
        rows = [
            {
                "npi": "9000000001", "hcpcs": "G0151",
                "service_date": date(2023, 6, 1),
                "benes": 20, "claims": 200, "payment": 10000.0,
            },
        ]
        medicaid = make_medicaid_df(rows)
        flags = signal_6_geographic_implausibility(medicaid.lazy(), TEST_MED_COLS)
        assert len(flags) == 0  # 20/200 = 0.1, not < 0.1

    def test_100_claims_not_flagged(self):
        """Exactly 100 claims should NOT be flagged (needs >100)."""
        rows = [
            {
                "npi": "9000000001", "hcpcs": "G0151",
                "service_date": date(2023, 6, 1),
                "benes": 5, "claims": 100, "payment": 10000.0,
            },
        ]
        medicaid = make_medicaid_df(rows)
        flags = signal_6_geographic_implausibility(medicaid.lazy(), TEST_MED_COLS)
        assert len(flags) == 0

    def test_non_home_health_code_not_flagged(self):
        """Non-home-health HCPCS code should not be flagged."""
        rows = [
            {
                "npi": "9000000001", "hcpcs": "99213",
                "service_date": date(2023, 6, 1),
                "benes": 5, "claims": 200, "payment": 10000.0,
            },
        ]
        medicaid = make_medicaid_df(rows)
        flags = signal_6_geographic_implausibility(medicaid.lazy(), TEST_MED_COLS)
        assert len(flags) == 0

    def test_multiple_providers_flagged(self):
        """Multiple providers with implausible ratios should each be flagged."""
        rows = [
            {
                "npi": "9000000001", "hcpcs": "G0151",
                "service_date": date(2023, 6, 1),
                "benes": 3, "claims": 200, "payment": 10000.0,
            },
            {
                "npi": "9000000002", "hcpcs": "T1019",
                "service_date": date(2023, 6, 1),
                "benes": 2, "claims": 300, "payment": 15000.0,
            },
        ]
        medicaid = make_medicaid_df(rows)
        flags = signal_6_geographic_implausibility(medicaid.lazy(), TEST_MED_COLS)
        flagged_npis = {f["npi"] for f in flags}
        assert "9000000001" in flagged_npis
        assert "9000000002" in flagged_npis
