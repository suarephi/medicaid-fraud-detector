"""JSON report generation module.

Builds the structured fraud_signals.json output matching the competition schema,
including severity classification, overpayment estimation, and False Claims Act
relevance mappings for each flagged provider.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from src import __version__


# Signal type string identifiers per spec
SIGNAL_TYPES: dict[int, str] = {
    1: "excluded_provider",
    2: "billing_outlier",
    3: "rapid_escalation",
    4: "workforce_impossibility",
    5: "shared_official",
    6: "geographic_implausibility",
}

# FCA statute mappings per signal -- most specific applicable subsection
FCA_STATUTES: dict[int, str] = {
    1: "31 U.S.C. \u00a73729(a)(1)(A); 42 U.S.C. \u00a71320a-7b(f)",
    2: "31 U.S.C. \u00a73729(a)(1)(A); 42 U.S.C. \u00a71320a-7a(a)(1)(A)",
    3: "31 U.S.C. \u00a73729(a)(1)(A); 42 CFR \u00a7424.530",
    4: "31 U.S.C. \u00a73729(a)(1)(B); 42 U.S.C. \u00a71320a-7b(a)(3)",
    5: "31 U.S.C. \u00a73729(a)(1)(C); 42 U.S.C. \u00a71320a-7b(b)",
    6: "31 U.S.C. \u00a73729(a)(1)(G); 42 U.S.C. \u00a71395nn(a)",
}

# FCA claim type descriptions -- legally precise per statute
FCA_CLAIM_TYPES: dict[int, str] = {
    1: (
        "Presenting false or fraudulent claims in violation of 31 U.S.C. \u00a73729(a)(1)(A) -- "
        "provider excluded from Federal healthcare programs under 42 U.S.C. \u00a71320a-7b is "
        "prohibited from billing Medicaid and all claims submitted post-exclusion constitute "
        "per se false claims under the FCA"
    ),
    2: (
        "Presenting false or fraudulent claims through billing volume that materially exceeds "
        "peer benchmarks, raising an inference of upcoding, unbundling, or services lacking "
        "medical necessity in violation of 42 U.S.C. \u00a71320a-7a(a)(1)(A); may also implicate "
        "anti-kickback violations under 42 U.S.C. \u00a71320a-7b(b) if referral relationships "
        "drove inflated volume"
    ),
    3: (
        "Presenting false or fraudulent claims consistent with a bust-out fraud pattern -- "
        "newly enumerated provider exhibits rapid billing escalation characteristic of schemes "
        "to extract maximum reimbursement before detection; implicates 42 CFR \u00a7424.530 "
        "enrollment integrity requirements and potential fraudulent inducement of enrollment"
    ),
    4: (
        "Making or using false records or statements material to a false claim under "
        "31 U.S.C. \u00a73729(a)(1)(B) -- claim volume exceeds physically possible service "
        "delivery capacity, constituting fabricated records in violation of "
        "42 U.S.C. \u00a71320a-7b(a)(3) prohibition on false statements material to claims"
    ),
    5: (
        "Conspiracy to submit false or fraudulent claims under 31 U.S.C. \u00a73729(a)(1)(C) -- "
        "multiple NPIs under shared authorized official control exhibit coordinated billing "
        "patterns indicative of organizational fraud through shell entities, potentially "
        "violating anti-kickback statute 42 U.S.C. \u00a71320a-7b(b) through patient steering "
        "and self-referrals among controlled entities"
    ),
    6: (
        "Concealing or improperly avoiding an obligation to repay under "
        "31 U.S.C. \u00a73729(a)(1)(G) (reverse false claims) -- home health provider billing "
        "pattern with disproportionately low beneficiary-to-claim ratio indicates phantom "
        "services or repeated billing on same patients, violating geographic service area "
        "requirements and 42 U.S.C. \u00a71395nn(a) referral limitations"
    ),
}

# Next steps templates per signal -- actionable with regulatory references
NEXT_STEPS: dict[int, list[str]] = {
    1: [
        "Verify current exclusion status via OIG LEIE database (https://exclusions.oig.hhs.gov/) "
        "and SAM.gov; confirm no reinstatement has been granted",
        "Calculate total Federal healthcare program payments made post-exclusion date for "
        "treble damages assessment under 31 U.S.C. \u00a73729(a)(1)(G) (current penalty: "
        "$13,946-$27,894 per claim plus treble damages)",
        "Refer to OIG Hotline (1-800-HHS-TIPS / oig.hhs.gov/fraud/report-fraud/) and "
        "state Medicaid Fraud Control Unit (MFCU) for coordinated investigation",
        "Review billing entity's compliance program for evidence of knowing participation "
        "or deliberate ignorance per 31 U.S.C. \u00a73729(b)(1) scienter standard",
    ],
    2: [
        "Request Recovery Audit Contractor (RAC) or Zone Program Integrity Contractor (ZPIC) "
        "review of provider's claims for targeted medical record audit",
        "Pull stratified random sample of high-volume claims and request medical records to "
        "verify medical necessity under 42 CFR \u00a7440.230(d)",
        "Compare procedure code distribution against specialty norms to detect upcoding "
        "(e.g., E&M level inflation) or unbundling violations per NCCI edits",
        "Assess referral patterns for potential anti-kickback statute (AKS) violations under "
        "42 U.S.C. \u00a71320a-7b(b); refer to OIG if kickback indicators found",
    ],
    3: [
        "Refer to Medicare Administrative Contractor (MAC) for enrollment verification and "
        "potential revocation under 42 CFR \u00a7424.535 for abuse of billing privileges",
        "Request ZPIC investigation of provider enrollment timeline, practice legitimacy, "
        "and beneficiary access-to-care documentation",
        "Audit patient records during rapid growth period for phantom billing indicators; "
        "cross-reference beneficiary identities against OIG identity fraud databases",
        "File referral with state MFCU and OIG Hotline (1-800-HHS-TIPS) citing "
        "bust-out fraud pattern per OIG Work Plan priorities",
    ],
    4: [
        "Conduct unannounced site visit to verify staffing levels, operational capacity, "
        "and physical infrastructure per 42 CFR \u00a7447.45 provider participation standards",
        "Subpoena payroll records (W-2s, 1099s) and compare documented workforce against "
        "claimed service volume; calculate maximum physically deliverable services",
        "Review claims data for impossible scheduling patterns (overlapping services, "
        "overnight billing, >24 service-hours/day) per MAC audit protocols",
        "Refer to OIG Hotline and state MFCU; request ZPIC data analysis of "
        "time-of-service patterns across all billing dates",
    ],
    5: [
        "Investigate corporate structure via Secretary of State filings, IRS EIN records, "
        "and beneficial ownership disclosures for all controlled NPIs",
        "Request ZPIC cross-entity analysis of billing patterns, shared addresses, phone "
        "numbers, bank accounts, and beneficiary overlap across controlled NPIs",
        "Assess whether organizational structure was designed to circumvent program billing "
        "limits or avoid per-provider utilization review under 42 CFR \u00a7456",
        "Refer to OIG for potential criminal conspiracy investigation under "
        "18 U.S.C. \u00a71347 (healthcare fraud) and 18 U.S.C. \u00a7371 (conspiracy); "
        "notify state MFCU for parallel state action",
    ],
    6: [
        "Request RAC or ZPIC audit of home health claims; verify service delivery through "
        "beneficiary interviews, caregiver logs, and GPS-enabled electronic visit verification (EVV) "
        "data per 21st Century Cures Act \u00a712006 requirements",
        "Review plans of care (POC) and physician orders for medical necessity under "
        "42 CFR \u00a7440.70; verify ordering physician is not excluded or sanctioned",
        "Cross-reference claimed visit dates against beneficiary hospitalization, SNF admission, "
        "or out-of-state travel records to identify phantom services",
        "Refer to state MFCU and OIG Hotline (1-800-HHS-TIPS) for investigation of "
        "potential ghost patient schemes; request MAC claims suspension if warranted "
        "under 42 CFR \u00a7455.23",
    ],
}


# Severity ranking for FCA materiality: critical > high > medium > low
_SEVERITY_RANK: dict[str, int] = {"critical": 4, "high": 3, "medium": 2, "low": 1}


def classify_severity(signal_id: int, details: dict) -> str:
    """Classify signal severity aligned with FCA materiality thresholds.

    Severity levels reflect FCA materiality analysis:
    - critical: Per se FCA violations (excluded provider billing constitutes automatic
      liability; no materiality defense available per Universal Health Services v. Escobar)
    - high: Strong FCA indicators meeting the materiality standard -- evidence would likely
      influence government payment decision (e.g., >5x peer billing, >500% growth rate,
      physically impossible service delivery, >$5M coordinated billing)
    - medium: Moderate FCA indicators warranting investigation -- patterns consistent with
      fraudulent conduct but requiring additional corroboration to establish materiality

    Args:
        signal_id: The numeric signal identifier (1-6).
        details: Evidence details dict from the signal detection function.

    Returns:
        Severity level string: "critical", "high", or "medium".
    """
    if signal_id == 1:
        # Excluded provider billing is a per se FCA violation -- no materiality
        # defense is available; any post-exclusion claim is automatically false
        return "critical"

    if signal_id == 2:
        ratio = details.get("ratio_to_peer_median", 0)
        # >5x peer median is strong materiality indicator per CMS program integrity guidance
        return "high" if ratio > 5 else "medium"

    if signal_id == 3:
        growth = details.get("peak_growth_rate", 0)
        # >500% growth in bust-out pattern is highly material; CMS enrollment
        # revocation threshold under 42 CFR 424.535(a)(8) for abuse of billing
        return "high" if growth > 500 else "medium"

    if signal_id == 4:
        # Physically impossible claim volumes constitute strong evidence of
        # fabricated records -- inherently material under Escobar standard
        return "high"

    if signal_id == 5:
        combined = details.get("combined_total", 0)
        # >$5M combined across shell entities meets DOJ civil fraud threshold
        # for priority investigation per DOJ Civil Fraud Initiative guidelines
        return "high" if combined > 5_000_000 else "medium"

    if signal_id == 6:
        ratio = details.get("ratio", 1.0)
        # Extremely low beneficiary ratio (<0.05) is strong phantom billing indicator
        return "high" if ratio < 0.05 else "medium"

    return "medium"


def estimate_overpayment(signal_id: int, details: dict) -> float:
    """Calculate estimated overpayment amount based on signal-specific formulas.

    Args:
        signal_id: The numeric signal identifier (1-6).
        details: Evidence details dict from the signal detection function.

    Returns:
        Estimated overpayment in USD. Returns 0.0 for signals where
        overpayment is not directly estimable from the available data.
    """
    if signal_id == 1:
        return float(details.get("post_exclusion_paid", 0))

    if signal_id == 2:
        total = details.get("total_paid", 0)
        threshold = details.get("p99_threshold", 0)
        return max(0.0, float(total - threshold))

    if signal_id == 3:
        return float(details.get("payments_during_growth", 0))

    if signal_id == 4:
        # (peak_claims - 1056) * (peak_paid / peak_claims), floored at 0
        peak_claims = details.get("claims_count", 0)
        peak_paid = details.get("peak_month_revenue", 0)
        excess = max(0, peak_claims - (6 * 8 * 22))
        if peak_claims > 0:
            return max(0.0, float(excess * (peak_paid / peak_claims)))
        return 0.0

    if signal_id == 5:
        # Conservative estimate: excess above $1M threshold across controlled entities
        combined = details.get("combined_total", 0)
        return max(0.0, float(combined - 1_000_000))

    if signal_id == 6:
        # Conservative estimate based on excess claims beyond expected ratio
        claims = details.get("claims", 0)
        unique_benes = details.get("unique_beneficiaries", 0)
        ratio = details.get("ratio", 1.0)
        if claims > 0 and ratio < 0.1:
            # Expected claims at 0.1 ratio = unique_benes / 0.1
            expected_claims = unique_benes * 10
            excess_claims = max(0, claims - expected_claims)
            # Estimate per-claim cost from total (rough average)
            return 0.0  # Cannot reliably estimate without per-claim cost data

    return 0.0


def build_provider_entry(
    npi: str,
    provider_name: str,
    entity_type: str,
    taxonomy_code: str,
    state: str,
    enumeration_date: str,
    lifetime_paid: float,
    lifetime_claims: int,
    lifetime_benes: int,
    signals: list[dict],
) -> dict:
    """Build a single flagged provider entry matching the competition JSON schema.

    Aggregates all signals for a provider, computes severity and overpayment
    for each, and attaches FCA relevance based on the most severe signal.

    Args:
        npi: 10-digit National Provider Identifier.
        provider_name: Provider or organization name from NPPES.
        entity_type: "individual" or "organization".
        taxonomy_code: Healthcare Provider Taxonomy Code from NPPES.
        state: Provider's practice location state.
        enumeration_date: Date the NPI was enumerated (from NPPES).
        lifetime_paid: Total payments across all claims for this NPI.
        lifetime_claims: Total claim count across all periods.
        lifetime_benes: Total unique beneficiary count across all periods.
        signals: List of signal flag dicts for this provider.

    Returns:
        A dict matching the competition JSON schema with provider metadata,
        signal entries, estimated overpayment, and FCA relevance.
    """
    signal_entries: list[dict] = []
    total_overpayment = 0.0

    for sig in signals:
        sig_id = sig["signal_id"]
        details = sig.get("details", {})
        severity = classify_severity(sig_id, details)
        overpayment = estimate_overpayment(sig_id, details)
        total_overpayment += overpayment

        signal_entries.append({
            "signal_type": SIGNAL_TYPES[sig_id],
            "severity": severity,
            "evidence": details,
        })

    return {
        "npi": str(npi),
        "provider_name": str(provider_name),
        "entity_type": str(entity_type).lower(),
        "taxonomy_code": str(taxonomy_code),
        "state": str(state),
        "enumeration_date": str(enumeration_date),
        "total_paid_all_time": round(float(lifetime_paid), 2),
        "total_claims_all_time": int(lifetime_claims),
        "total_unique_beneficiaries_all_time": int(lifetime_benes),
        "signals": signal_entries,
        "estimated_overpayment_usd": round(total_overpayment, 2),
        "fca_relevance": _build_fca_relevance(signals),
    }


def _build_fca_relevance(signals: list[dict]) -> dict:
    """Build FCA relevance block from the MOST SEVERE signal.

    Selects the signal with the highest severity classification to drive
    the FCA analysis, ensuring the strongest legal theory is presented.
    Severity ranking: critical > high > medium > low.

    Args:
        signals: List of signal flag dicts for a provider.

    Returns:
        A dict with claim_type, statute_reference, and suggested_next_steps,
        or an empty dict if no signals are provided.
    """
    if not signals:
        return {}

    # Find the most severe signal by classifying each and ranking
    best_sig = signals[0]
    best_rank = _SEVERITY_RANK.get(
        classify_severity(best_sig["signal_id"], best_sig.get("details", {})), 0
    )

    for sig in signals[1:]:
        rank = _SEVERITY_RANK.get(
            classify_severity(sig["signal_id"], sig.get("details", {})), 0
        )
        if rank > best_rank:
            best_rank = rank
            best_sig = sig

    sig_id = best_sig["signal_id"]
    return {
        "claim_type": FCA_CLAIM_TYPES[sig_id],
        "statute_reference": FCA_STATUTES[sig_id],
        "suggested_next_steps": NEXT_STEPS[sig_id],
    }


def build_report(
    flagged_providers: list[dict],
    scan_count: int,
    signal_tallies: dict[str, int],
) -> dict:
    """Assemble the complete fraud_signals.json report matching competition schema.

    Args:
        flagged_providers: List of enriched provider entry dicts from
            build_provider_entry().
        scan_count: Total number of unique providers scanned.
        signal_tallies: Mapping of signal names (e.g. "signal_1") to flag counts.

    Returns:
        The complete report dict ready for JSON serialization, containing
        metadata, signal counts, and the flagged providers array.
    """
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "tool_version": __version__,
        "total_providers_scanned": scan_count,
        "total_providers_flagged": len(flagged_providers),
        "signal_counts": {
            "excluded_provider": signal_tallies.get("signal_1", 0),
            "billing_outlier": signal_tallies.get("signal_2", 0),
            "rapid_escalation": signal_tallies.get("signal_3", 0),
            "workforce_impossibility": signal_tallies.get("signal_4", 0),
            "shared_official": signal_tallies.get("signal_5", 0),
            "geographic_implausibility": signal_tallies.get("signal_6", 0),
        },
        "flagged_providers": flagged_providers,
    }


def write_report(report: dict, path: str) -> None:
    """Write the report dict to a JSON file.

    Args:
        report: The complete report dict from build_report().
        path: File path to write the JSON output to.
    """
    with open(path, "w") as f:
        json.dump(report, f, indent=2, default=_json_serializer)
    print(f"Report written to {path}")
    print(f"  Providers flagged: {report['total_providers_flagged']}")
    print(f"  Signal counts: {report['signal_counts']}")


def _json_serializer(obj: Any) -> Any:
    """Handle non-JSON-serializable types during report serialization.

    Converts date/datetime objects to ISO format strings and numpy/polars
    scalar types to native Python types.

    Args:
        obj: The object that failed default JSON serialization.

    Returns:
        A JSON-serializable representation of the object.

    Raises:
        TypeError: If the object type is not recognized.
    """
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    if hasattr(obj, "item"):  # numpy/polars scalars
        return obj.item()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
