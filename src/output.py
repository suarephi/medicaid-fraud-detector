"""JSON report generation module."""
import json
from datetime import datetime, timezone
from typing import Any

from src import __version__


# FCA statute mappings per signal
FCA_STATUTES = {
    1: {
        "statute": "31 U.S.C. \u00a73729(a)(1)(A)",
        "description": "Knowingly presenting false claims for payment",
    },
    2: {
        "statute": "31 U.S.C. \u00a73729(a)(1)(A)",
        "description": "Overbilling through inflated volume",
    },
    3: {
        "statute": "31 U.S.C. \u00a73729(a)(1)(A)",
        "description": "Bust-out billing schemes",
    },
    4: {
        "statute": "31 U.S.C. \u00a73729(a)(1)(B)",
        "description": "Making or using false records or statements",
    },
    5: {
        "statute": "31 U.S.C. \u00a73729(a)(1)(C)",
        "description": "Conspiracy to commit fraud",
    },
    6: {
        "statute": "31 U.S.C. \u00a73729(a)(1)(G)",
        "description": "Reverse false claims - obligation avoidance",
    },
}

# Signal names
SIGNAL_NAMES = {
    1: "Excluded Provider Still Billing",
    2: "Billing Volume Outlier",
    3: "Rapid Billing Escalation",
    4: "Workforce Impossibility",
    5: "Shared Authorized Official",
    6: "Geographic Implausibility",
}

# Next steps templates per signal
NEXT_STEPS = {
    1: [
        "Verify current exclusion status via OIG LEIE database and confirm no reinstatement",
        "Calculate total Federal healthcare program payments made post-exclusion for damages assessment",
        "Review billing patterns for potential knowing participation by billing entities",
    ],
    2: [
        "Conduct peer comparison audit of billed services against specialty benchmarks",
        "Review medical records for a sample of high-volume claims to verify medical necessity",
        "Assess whether billing patterns indicate upcoding or unbundling violations",
    ],
    3: [
        "Investigate provider enrollment timeline and verify legitimacy of practice establishment",
        "Audit patient records during rapid growth period for phantom billing indicators",
        "Cross-reference beneficiary lists for signs of patient recruitment or identity fraud",
    ],
    4: [
        "Verify actual staffing levels and provider capacity through site visit or payroll records",
        "Compare documented service hours against physically possible delivery capacity",
        "Review time-of-service data for impossible scheduling patterns (overnight, weekends)",
    ],
    5: [
        "Investigate corporate structure and beneficial ownership across all controlled NPIs",
        "Review for coordinated billing patterns or patient steering between controlled entities",
        "Assess whether organizational structure was designed to circumvent billing limits",
    ],
    6: [
        "Verify home health service delivery through beneficiary interviews or GPS visit logs",
        "Review plan-of-care documentation for medical necessity and visit frequency",
        "Cross-reference claimed visits with beneficiary hospitalization or facility admission records",
    ],
}


def classify_severity(signal_id: int, details: dict) -> str:
    """Classify signal severity per competition rules."""
    if signal_id == 1:
        return "critical"

    if signal_id == 2:
        ratio = details.get("ratio_to_peer_median", 0)
        return "high" if ratio > 5 else "medium"

    if signal_id == 3:
        growth = details.get("peak_growth_rate", 0)
        return "high" if growth > 500 else "medium"

    if signal_id == 4:
        return "high"

    if signal_id == 5:
        combined = details.get("combined_total", 0)
        return "high" if combined > 5_000_000 else "medium"

    return "medium"


def estimate_overpayment(signal_id: int, details: dict) -> float:
    """Calculate estimated overpayment per signal formulas."""
    if signal_id == 1:
        return float(details.get("post_exclusion_paid", 0))

    if signal_id == 2:
        total = details.get("total_paid", 0)
        threshold = details.get("p99_threshold", 0)
        return max(0.0, float(total - threshold))

    if signal_id == 3:
        return float(details.get("payments_during_growth", 0))

    if signal_id == 4:
        excess = details.get("excess_claims", 0)
        avg_payment = details.get("avg_payment_per_claim", 0)
        return float(excess * avg_payment)

    # Signals 5 and 6: not estimated per rules
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
    signals: list[dict],
) -> dict:
    """Build a single flagged provider entry."""
    signal_entries = []
    for sig in signals:
        sig_id = sig["signal_id"]
        details = sig.get("details", {})
        severity = classify_severity(sig_id, details)
        overpayment = estimate_overpayment(sig_id, details)

        signal_entries.append({
            "signal_id": sig_id,
            "signal_name": SIGNAL_NAMES[sig_id],
            "severity": severity,
            "details": details,
            "estimated_overpayment": round(overpayment, 2),
            "fca_statute": FCA_STATUTES[sig_id]["statute"],
            "fca_description": FCA_STATUTES[sig_id]["description"],
            "next_steps": NEXT_STEPS[sig_id],
        })

    return {
        "npi": str(npi),
        "provider_name": str(provider_name),
        "entity_type": str(entity_type),
        "taxonomy_code": str(taxonomy_code),
        "state": str(state),
        "enumeration_date": str(enumeration_date),
        "lifetime_paid": round(float(lifetime_paid), 2),
        "lifetime_claims": int(lifetime_claims),
        "signals": signal_entries,
    }


def build_report(
    flagged_providers: list[dict],
    scan_count: int,
    signal_tallies: dict[str, int],
) -> dict:
    """Assemble the complete fraud_signals.json report."""
    return {
        "metadata": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "tool_version": __version__,
            "scan_count": scan_count,
            "flag_count": len(flagged_providers),
        },
        "signal_tallies": signal_tallies,
        "flagged_providers": flagged_providers,
    }


def write_report(report: dict, path: str) -> None:
    """Write the report to a JSON file."""
    with open(path, "w") as f:
        json.dump(report, f, indent=2, default=_json_serializer)
    print(f"Report written to {path}")
    print(f"  Providers flagged: {report['metadata']['flag_count']}")
    print(f"  Signal tallies: {report['signal_tallies']}")


def _json_serializer(obj: Any) -> Any:
    """Handle non-serializable types."""
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    if hasattr(obj, "item"):  # numpy/polars scalars
        return obj.item()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
