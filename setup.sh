#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${1:-data}"
mkdir -p "$DATA_DIR"

echo "=== Setting up Python virtual environment ==="
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi
source .venv/bin/activate

echo "=== Installing dependencies ==="
pip install -r requirements.txt

echo "=== Downloading Medicaid Provider Spending (~2.9 GB) ==="
if [ ! -f "$DATA_DIR/medicaid-provider-spending.parquet" ]; then
    curl -L --progress-bar -o "$DATA_DIR/medicaid-provider-spending.parquet" \
        "https://stopendataprod.blob.core.windows.net/datasets/medicaid-provider-spending/2026-02-09/medicaid-provider-spending.parquet"
else
    echo "  Already downloaded, skipping."
fi

echo "=== Downloading OIG LEIE Exclusion List ==="
curl -L --progress-bar -o "$DATA_DIR/UPDATED.csv" \
    "https://oig.hhs.gov/exclusions/downloadables/UPDATED.csv"

echo "=== Downloading NPPES NPI Registry (~1 GB) ==="
if [ ! -f "$DATA_DIR/NPPES.zip" ]; then
    curl -L --progress-bar -o "$DATA_DIR/NPPES.zip" \
        "https://download.cms.gov/nppes/NPPES_Data_Dissemination_February_2026_V2.zip"
else
    echo "  Already downloaded, skipping."
fi

echo "=== Extracting NPPES data ==="
if ! ls "$DATA_DIR"/npidata_pfile_*.csv 1>/dev/null 2>&1; then
    cd "$DATA_DIR"
    unzip -o NPPES.zip "npidata_pfile_*.csv" || unzip -o NPPES.zip
    cd - > /dev/null
else
    echo "  Already extracted, skipping."
fi

echo "=== Setup complete ==="
echo "Data directory: $DATA_DIR"
ls -lh "$DATA_DIR"
