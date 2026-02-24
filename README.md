# Medicaid Provider Fraud Signal Detection Engine

A command-line tool that ingests three public healthcare datasets and detects six types of fraud signals in Medicaid billing data, producing structured JSON reports for legal professionals pursuing False Claims Act (FCA) cases.

Built for the Medicaid Fraud Detection competition. Analyzes 227M+ billing rows to identify providers exhibiting patterns consistent with fraudulent billing, and maps each finding to specific FCA statutes with actionable next steps.

## Architecture

```
                    +-----------------+
                    |   setup.sh      |  Download & prepare datasets
                    +--------+--------+
                             |
                    +--------v--------+
                    |   ingest.py     |  Load parquet/CSV, auto-detect columns,
                    |                 |  normalize NPIs
                    +--------+--------+
                             |
              +--------------+--------------+
              |              |              |
     Medicaid LF        LEIE DF       NPPES LF
              |              |              |
              +--------------+--------------+
                             |
                    +--------v--------+
                    |   signals.py    |  Run 6 detection algorithms
                    |  (1) Excluded   |  Each returns list[dict] of flags
                    |  (2) Volume     |
                    |  (3) Escalation |
                    |  (4) Workforce  |
                    |  (5) Official   |
                    |  (6) Geographic |
                    +--------+--------+
                             |
                    +--------v--------+
                    |   main.py       |  Enrich flags with NPPES metadata,
                    |                 |  aggregate by provider NPI
                    +--------+--------+
                             |
                    +--------v--------+
                    |   output.py     |  Classify severity, estimate overpayment,
                    |                 |  map FCA statutes, write JSON
                    +--------+--------+
                             |
                    +--------v--------+
                    | fraud_signals   |  Final structured JSON report
                    |    .json        |
                    +-----------------+
```

## Quick Start

### Prerequisites

- Python 3.11+
- ~5 GB disk space for datasets
- 16 GB+ RAM recommended

### Setup

```bash
# Create virtual environment and install dependencies
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Download datasets
bash setup.sh
```

This downloads:
1. **HHS Medicaid Provider Spending** (~2.9 GB parquet, 227M rows)
2. **OIG LEIE Exclusion List** (CSV, ~50K exclusion records)
3. **NPPES NPI Registry** (~1 GB zip, 8M+ provider records)

### Run

```bash
python -m src.main --data-dir data --output fraud_signals.json
```

Or use the convenience script:
```bash
bash run.sh
```

Options:
```bash
python -m src.main --data-dir /path/to/data --output results.json --no-gpu
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--data-dir` | `data` | Directory containing downloaded datasets |
| `--output` | `fraud_signals.json` | Output JSON file path |
| `--no-gpu` | off | Compatibility flag (no effect) |

## Output Schema

The output `fraud_signals.json` follows this structure:

```json
{
  "generated_at": "2024-01-01T00:00:00+00:00",
  "tool_version": "1.0.0",
  "total_providers_scanned": 500000,
  "total_providers_flagged": 150,
  "signal_counts": {
    "excluded_provider": 10,
    "billing_outlier": 50,
    "rapid_escalation": 20,
    "workforce_impossibility": 15,
    "shared_official": 25,
    "geographic_implausibility": 30
  },
  "flagged_providers": [
    {
      "npi": "1234567890",
      "provider_name": "Example Provider LLC",
      "entity_type": "organization",
      "taxonomy_code": "207Q00000X",
      "state": "FL",
      "enumeration_date": "01/15/2020",
      "total_paid_all_time": 500000.00,
      "total_claims_all_time": 2000,
      "total_unique_beneficiaries_all_time": 150,
      "signals": [
        {
          "signal_type": "excluded_provider",
          "severity": "critical",
          "evidence": { "..." : "..." }
        }
      ],
      "estimated_overpayment_usd": 500000.00,
      "fca_relevance": {
        "claim_type": "Presenting false or fraudulent claims...",
        "statute_reference": "31 U.S.C. 3729(a)(1)(A); 42 U.S.C. 1320a-7b(f)",
        "suggested_next_steps": ["..."]
      }
    }
  ]
}
```

## Fraud Signals Detected

| # | Signal | Description | Severity | FCA Statute |
|---|--------|-------------|----------|-------------|
| 1 | Excluded Provider Still Billing | NPI on OIG LEIE exclusion list with post-exclusion claims | Critical | 31 U.S.C. 3729(a)(1)(A) |
| 2 | Billing Volume Outlier | Total spending > 99th percentile within taxonomy/state peer group (min 5 peers) | Medium/High | 31 U.S.C. 3729(a)(1)(A) |
| 3 | Rapid Billing Escalation | Newly enumerated provider with >200% rolling 3-month avg growth rate | Medium/High | 31 U.S.C. 3729(a)(1)(A) |
| 4 | Workforce Impossibility | Organization with peak monthly claims > 6/hour (176 business hours/month) | High | 31 U.S.C. 3729(a)(1)(B) |
| 5 | Shared Authorized Official | Official controlling 5+ NPIs with >$1M combined billing | Medium/High | 31 U.S.C. 3729(a)(1)(C) |
| 6 | Geographic Implausibility | Home health provider with >100 claims/month and beneficiary/claims ratio < 0.1 | Medium/High | 31 U.S.C. 3729(a)(1)(G) |

## Testing

Run the full test suite (97 tests covering all modules):

```bash
# Using the project virtual environment
.venv/bin/python -m pytest tests/ -v

# Or if the venv is activated
pytest tests/ -v
```

Tests include:
- **Signal unit tests** (41): Core detection logic for all 6 signals with edge cases
- **Output tests** (24): Severity classification, overpayment estimation, JSON schema, FCA mapping
- **Ingest tests** (12): Column auto-detection, NPI normalization
- **Integration tests** (7): End-to-end pipeline with synthetic data, schema validation
- **Edge cases**: Empty datasets, boundary values at exact thresholds, null handling

## Project Structure

```
medicaid-fraud-detector/
  README.md              # This file
  requirements.txt       # Python dependencies (polars, pyarrow, pytest)
  setup.sh               # Dataset download script
  run.sh                 # Convenience run script
  src/
    __init__.py          # Package init, version string
    ingest.py            # Data loading, column auto-detection, NPI normalization
    signals.py           # Six fraud signal detection algorithms
    output.py            # JSON report generation, severity, FCA mapping
    main.py              # Pipeline orchestration and CLI entry point
  tests/
    __init__.py
    test_signals.py      # Signal detection unit tests + edge cases
    test_output.py       # Output module tests (severity, overpayment, schema)
    test_ingest.py       # Ingest module tests (column detection, NPI normalization)
    test_integration.py  # End-to-end pipeline integration tests
    fixtures/
      __init__.py        # Synthetic data generators (Medicaid, LEIE, NPPES)
  METHODOLOGY.md         # Detailed legal and algorithmic methodology
  fraud_signals.json     # Output report (generated by pipeline)
```

## Requirements

- **polars** >= 0.20.0 -- High-performance DataFrame library with lazy evaluation
- **pyarrow** >= 14.0.0 -- Parquet file format support
- **requests** >= 2.31.0 -- HTTP client for dataset downloads
- **tqdm** >= 4.66.0 -- Progress bars for data processing
- **pytest** >= 7.4.0 -- Test framework

## Performance

- Uses Polars lazy evaluation for memory-efficient processing of 227M+ rows
- Streams parquet data without loading entire dataset into memory
- Signals run sequentially to minimize peak memory usage
- Target runtime: < 4 hours on 16 GB MacBook, < 60 minutes on 64 GB Linux
