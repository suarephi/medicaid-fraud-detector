# Medicaid Provider Fraud Signal Detection Engine

A command-line tool that ingests three public healthcare datasets and detects six types of fraud signals in Medicaid billing data, producing structured reports for legal professionals pursuing False Claims Act cases.

## Quick Start

### Prerequisites

- Python 3.11+
- ~5 GB disk space for datasets
- 16 GB+ RAM recommended

### Setup

```bash
# Install dependencies and download datasets
bash setup.sh
```

This downloads:
1. **HHS Medicaid Provider Spending** (~2.9 GB parquet, 227M rows)
2. **OIG LEIE Exclusion List** (CSV)
3. **NPPES NPI Registry** (~1 GB zip)

### Run

```bash
bash run.sh
```

Options:
```bash
bash run.sh --data-dir /path/to/data --output results.json --no-gpu
```

### Output

Produces `fraud_signals.json` with:
- Metadata (timestamp, version, scan/flag counts)
- Signal tallies by type
- Flagged provider array with NPI, provider info, signal details, severity, estimated overpayment, and FCA statute references

## Fraud Signals Detected

| # | Signal | Severity | FCA Statute |
|---|--------|----------|-------------|
| 1 | Excluded Provider Still Billing | Critical | 31 U.S.C. 3729(a)(1)(A) |
| 2 | Billing Volume Outlier | Medium/High | 31 U.S.C. 3729(a)(1)(A) |
| 3 | Rapid Billing Escalation | Medium/High | 31 U.S.C. 3729(a)(1)(A) |
| 4 | Workforce Impossibility | High | 31 U.S.C. 3729(a)(1)(B) |
| 5 | Shared Authorized Official | Medium/High | 31 U.S.C. 3729(a)(1)(C) |
| 6 | Geographic Implausibility | Medium | 31 U.S.C. 3729(a)(1)(G) |

## Testing

```bash
pytest tests/ -v
```

## Project Structure

```
submission/
  README.md
  requirements.txt
  setup.sh
  run.sh
  src/
    __init__.py
    ingest.py        # Data loading and column detection
    signals.py       # Six signal implementations
    output.py        # JSON report generation
    main.py          # Pipeline orchestration
  tests/
    __init__.py
    test_signals.py  # Unit tests with synthetic data
    fixtures/
      __init__.py    # Synthetic data generators
  fraud_signals.json # Sample output
```

## Performance

- Uses Polars for memory-efficient lazy evaluation
- Processes 227M rows on 16GB MacBook via streaming
- Signals run sequentially to minimize peak memory
- Target: <4 hours on MacBook, <60 minutes on 64GB Linux
