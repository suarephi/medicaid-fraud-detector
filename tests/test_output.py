"""Tests for the output module: severity classification, overpayment estimates, report building."""
import json
from datetime import date, datetime

import pytest

from src.output import (
    classify_severity,
    estimate_overpayment,
    build_provider_entry,
    build_report,
    _json_serializer,
    SIGNAL_TYPES,
    FCA_STATUTES,
    FCA_CLAIM_TYPES,
    NEXT_STEPS,
)


class TestClassifySeverity:
    """Tests for classify_severity()."""

    def test_signal_1_always_critical(self):
        """Signal 1 (excluded provider) is always critical regardless of details."""
        assert classify_severity(1, {}) == "critical"
        assert classify_severity(1, {"post_exclusion_paid": 100}) == "critical"

    def test_signal_2_high_when_ratio_above_5(self):
        """Signal 2 is high severity when ratio_to_peer_median > 5."""
        assert classify_severity(2, {"ratio_to_peer_median": 6}) == "high"
        assert classify_severity(2, {"ratio_to_peer_median": 100}) == "high"

    def test_signal_2_medium_when_ratio_at_or_below_5(self):
        """Signal 2 is medium severity when ratio_to_peer_median <= 5."""
        assert classify_severity(2, {"ratio_to_peer_median": 5}) == "medium"
        assert classify_severity(2, {"ratio_to_peer_median": 3}) == "medium"
        assert classify_severity(2, {}) == "medium"

    def test_signal_3_high_when_growth_above_500(self):
        """Signal 3 is high severity when peak_growth_rate > 500."""
        assert classify_severity(3, {"peak_growth_rate": 501}) == "high"
        assert classify_severity(3, {"peak_growth_rate": 1000}) == "high"

    def test_signal_3_medium_when_growth_at_or_below_500(self):
        """Signal 3 is medium severity when peak_growth_rate <= 500."""
        assert classify_severity(3, {"peak_growth_rate": 500}) == "medium"
        assert classify_severity(3, {"peak_growth_rate": 250}) == "medium"
        assert classify_severity(3, {}) == "medium"

    def test_signal_4_always_high(self):
        """Signal 4 (workforce impossibility) is always high."""
        assert classify_severity(4, {}) == "high"
        assert classify_severity(4, {"claims_count": 99999}) == "high"

    def test_signal_5_high_when_combined_above_5m(self):
        """Signal 5 is high severity when combined_total > $5M."""
        assert classify_severity(5, {"combined_total": 5_000_001}) == "high"

    def test_signal_5_medium_when_combined_at_or_below_5m(self):
        """Signal 5 is medium severity when combined_total <= $5M."""
        assert classify_severity(5, {"combined_total": 5_000_000}) == "medium"
        assert classify_severity(5, {"combined_total": 1_500_000}) == "medium"

    def test_signal_6_high_when_ratio_below_005(self):
        """Signal 6 is high severity when ratio < 0.05."""
        assert classify_severity(6, {"ratio": 0.01}) == "high"
        assert classify_severity(6, {"ratio": 0.04}) == "high"

    def test_signal_6_medium_when_ratio_at_or_above_005(self):
        """Signal 6 is medium severity when ratio >= 0.05."""
        assert classify_severity(6, {"ratio": 0.05}) == "medium"
        assert classify_severity(6, {"ratio": 0.09}) == "medium"
        assert classify_severity(6, {}) == "medium"


class TestEstimateOverpayment:
    """Tests for estimate_overpayment()."""

    def test_signal_1_returns_post_exclusion_paid(self):
        """Signal 1 overpayment is the full post-exclusion amount."""
        assert estimate_overpayment(1, {"post_exclusion_paid": 50000.0}) == 50000.0
        assert estimate_overpayment(1, {}) == 0.0

    def test_signal_2_returns_excess_above_threshold(self):
        """Signal 2 overpayment = total_paid - p99_threshold, floored at 0."""
        details = {"total_paid": 100000, "p99_threshold": 80000}
        assert estimate_overpayment(2, details) == 20000.0

    def test_signal_2_returns_zero_when_below_threshold(self):
        """Signal 2 overpayment is 0 if total < threshold."""
        details = {"total_paid": 50000, "p99_threshold": 80000}
        assert estimate_overpayment(2, details) == 0.0

    def test_signal_3_returns_payments_during_growth(self):
        """Signal 3 overpayment is payments_during_growth."""
        assert estimate_overpayment(3, {"payments_during_growth": 75000.0}) == 75000.0
        assert estimate_overpayment(3, {}) == 0.0

    def test_signal_4_formula(self):
        """Signal 4: excess = max(0, claims - 1056) * (revenue / claims)."""
        details = {"claims_count": 2000, "peak_month_revenue": 100000.0}
        # excess = 2000 - 1056 = 944
        # cost_per_claim = 100000 / 2000 = 50
        # overpayment = 944 * 50 = 47200
        assert estimate_overpayment(4, details) == 47200.0

    def test_signal_4_zero_when_below_1056(self):
        """Signal 4 overpayment is 0 when claims <= 1056."""
        details = {"claims_count": 1056, "peak_month_revenue": 100000.0}
        assert estimate_overpayment(4, details) == 0.0

    def test_signal_4_zero_claims(self):
        """Signal 4 handles zero claims without division error."""
        details = {"claims_count": 0, "peak_month_revenue": 0.0}
        assert estimate_overpayment(4, details) == 0.0

    def test_signal_5_returns_excess_above_1m(self):
        """Signal 5 overpayment is excess above $1M threshold."""
        assert estimate_overpayment(5, {"combined_total": 5_000_000}) == 4_000_000.0
        assert estimate_overpayment(5, {"combined_total": 1_500_000}) == 500_000.0

    def test_signal_5_returns_zero_when_under_1m(self):
        """Signal 5 overpayment is 0 when combined is at or below $1M."""
        assert estimate_overpayment(5, {"combined_total": 1_000_000}) == 0.0
        assert estimate_overpayment(5, {"combined_total": 500_000}) == 0.0
        assert estimate_overpayment(5, {}) == 0.0

    def test_signal_6_returns_zero(self):
        """Signal 6 overpayment is not estimated (returns 0)."""
        assert estimate_overpayment(6, {"claims": 500, "ratio": 0.02}) == 0.0


class TestBuildProviderEntry:
    """Tests for build_provider_entry()."""

    def test_produces_correct_schema(self):
        """Entry should have all required top-level fields."""
        signals = [{"signal_id": 1, "details": {"post_exclusion_paid": 5000.0}}]
        entry = build_provider_entry(
            npi="1234567890",
            provider_name="Test Provider",
            entity_type="individual",
            taxonomy_code="207Q00000X",
            state="CA",
            enumeration_date="01/01/2020",
            lifetime_paid=100000.0,
            lifetime_claims=500,
            lifetime_benes=200,
            signals=signals,
        )
        required_fields = [
            "npi", "provider_name", "entity_type", "taxonomy_code",
            "state", "enumeration_date", "total_paid_all_time",
            "total_claims_all_time", "total_unique_beneficiaries_all_time",
            "signals", "estimated_overpayment_usd", "fca_relevance",
        ]
        for field in required_fields:
            assert field in entry, f"Missing field: {field}"

    def test_signal_entries_contain_type_severity_evidence(self):
        """Each signal entry should have signal_type, severity, and evidence."""
        signals = [{"signal_id": 2, "details": {"ratio_to_peer_median": 10, "total_paid": 500000, "p99_threshold": 100000}}]
        entry = build_provider_entry(
            npi="1234567890", provider_name="Test", entity_type="individual",
            taxonomy_code="207Q00000X", state="CA", enumeration_date="01/01/2020",
            lifetime_paid=500000.0, lifetime_claims=100, lifetime_benes=50,
            signals=signals,
        )
        sig = entry["signals"][0]
        assert sig["signal_type"] == "billing_outlier"
        assert sig["severity"] in ("low", "medium", "high", "critical")
        assert "evidence" in sig

    def test_fca_relevance_populated(self):
        """FCA relevance block should contain claim_type, statute, and next_steps."""
        signals = [{"signal_id": 4, "details": {}}]
        entry = build_provider_entry(
            npi="1234567890", provider_name="Test", entity_type="organization",
            taxonomy_code="207Q00000X", state="CA", enumeration_date="01/01/2020",
            lifetime_paid=100000.0, lifetime_claims=100, lifetime_benes=50,
            signals=signals,
        )
        fca = entry["fca_relevance"]
        assert "claim_type" in fca
        assert "statute_reference" in fca
        assert "suggested_next_steps" in fca
        assert isinstance(fca["suggested_next_steps"], list)

    def test_overpayment_sums_across_signals(self):
        """Estimated overpayment should sum across all signals on a provider."""
        signals = [
            {"signal_id": 1, "details": {"post_exclusion_paid": 10000.0}},
            {"signal_id": 4, "details": {"claims_count": 2000, "peak_month_revenue": 100000.0}},
        ]
        entry = build_provider_entry(
            npi="1234567890", provider_name="Test", entity_type="organization",
            taxonomy_code="207Q00000X", state="CA", enumeration_date="01/01/2020",
            lifetime_paid=200000.0, lifetime_claims=3000, lifetime_benes=100,
            signals=signals,
        )
        # Signal 1: 10000.0, Signal 4: (2000-1056)*(100000/2000) = 944*50 = 47200
        assert entry["estimated_overpayment_usd"] == 57200.0

    def test_empty_signals_list(self):
        """Provider entry with no signals should have empty signals and zero overpayment."""
        entry = build_provider_entry(
            npi="1234567890", provider_name="Test", entity_type="individual",
            taxonomy_code="207Q00000X", state="CA", enumeration_date="01/01/2020",
            lifetime_paid=50000.0, lifetime_claims=100, lifetime_benes=50,
            signals=[],
        )
        assert entry["signals"] == []
        assert entry["estimated_overpayment_usd"] == 0.0


class TestBuildReport:
    """Tests for build_report()."""

    def test_report_has_required_top_level_fields(self):
        """Report should contain all required top-level keys."""
        report = build_report(
            flagged_providers=[],
            scan_count=1000,
            signal_tallies={"signal_1": 5, "signal_2": 3},
        )
        required_fields = [
            "generated_at", "tool_version", "total_providers_scanned",
            "total_providers_flagged", "signal_counts", "flagged_providers",
        ]
        for field in required_fields:
            assert field in report, f"Missing field: {field}"

    def test_signal_counts_has_all_six_signals(self):
        """Signal counts should include all six signal types."""
        report = build_report(
            flagged_providers=[], scan_count=100, signal_tallies={},
        )
        expected_keys = [
            "excluded_provider", "billing_outlier", "rapid_escalation",
            "workforce_impossibility", "shared_official", "geographic_implausibility",
        ]
        for key in expected_keys:
            assert key in report["signal_counts"]

    def test_flagged_count_matches_provider_list(self):
        """total_providers_flagged should equal len(flagged_providers)."""
        providers = [{"npi": "111"}, {"npi": "222"}]
        report = build_report(
            flagged_providers=providers, scan_count=100,
            signal_tallies={"signal_1": 2},
        )
        assert report["total_providers_flagged"] == 2

    def test_generated_at_is_iso_format(self):
        """generated_at should be a valid ISO 8601 datetime string."""
        report = build_report(flagged_providers=[], scan_count=0, signal_tallies={})
        datetime.fromisoformat(report["generated_at"])


class TestJsonSerializer:
    """Tests for _json_serializer()."""

    def test_handles_date_objects(self):
        """Dates should be serialized with isoformat."""
        result = _json_serializer(date(2023, 6, 15))
        assert result == "2023-06-15"

    def test_handles_datetime_objects(self):
        """Datetimes should be serialized with isoformat."""
        dt = datetime(2023, 6, 15, 12, 30, 0)
        result = _json_serializer(dt)
        assert "2023-06-15" in result

    def test_raises_for_unsupported_types(self):
        """Non-serializable types should raise TypeError."""
        with pytest.raises(TypeError, match="not JSON serializable"):
            _json_serializer(set([1, 2, 3]))

    def test_handles_numpy_like_scalars(self):
        """Objects with .item() method (numpy/polars scalars) should be serialized."""
        class FakeScalar:
            def item(self):
                return 42
        assert _json_serializer(FakeScalar()) == 42


class TestSignalMappingConstants:
    """Tests for output module constants."""

    def test_all_six_signals_have_types(self):
        """SIGNAL_TYPES should have entries for signals 1-6."""
        for i in range(1, 7):
            assert i in SIGNAL_TYPES

    def test_all_six_signals_have_fca_statutes(self):
        """FCA_STATUTES should have entries for signals 1-6."""
        for i in range(1, 7):
            assert i in FCA_STATUTES

    def test_all_six_signals_have_claim_types(self):
        """FCA_CLAIM_TYPES should have entries for signals 1-6."""
        for i in range(1, 7):
            assert i in FCA_CLAIM_TYPES

    def test_all_six_signals_have_next_steps(self):
        """NEXT_STEPS should have entries for signals 1-6 with at least one step each."""
        for i in range(1, 7):
            assert i in NEXT_STEPS
            assert len(NEXT_STEPS[i]) >= 1
