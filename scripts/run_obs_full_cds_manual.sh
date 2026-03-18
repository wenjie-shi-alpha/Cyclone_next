#!/usr/bin/env bash
set -euo pipefail

# Full run for ASCAT + Recon on CDS JupyterLab terminal.
# Usage:
#   bash scripts/run_obs_full_cds_manual.sh [YEAR_START] [YEAR_END] [RUN_SAMPLE_BUILD]
# Example:
#   bash scripts/run_obs_full_cds_manual.sh 2016 2025 1
#
# Notes:
# - The yearly controlled scripts currently iterate 2016..2025 internally.
# - RUN_SAMPLE_BUILD=1 will run build_dataset_sample_preview_v0_1.py at the end.

YEAR_START="${1:-2016}"
YEAR_END="${2:-2025}"
RUN_SAMPLE_BUILD="${3:-1}"
BASE_DIR="$(cd "$(dirname "$0")/.." && pwd)"

cd "$BASE_DIR"
source .venv/bin/activate

ASCAT_FULL_MANIFEST="data/interim/ascat/ascat_request_manifest_full.csv"
RECON_FULL_MANIFEST="data/interim/recon/recon_request_manifest_full.csv"

echo "[INFO] build full manifests: years=${YEAR_START}-${YEAR_END}"
python scripts/build_obs_request_manifest.py \
  --year-start "$YEAR_START" \
  --year-end "$YEAR_END" \
  --ascat-out-csv "$ASCAT_FULL_MANIFEST" \
  --recon-out-csv "$RECON_FULL_MANIFEST"

echo "[INFO] run ASCAT full controlled pipeline..."
bash scripts/run_ascat_full_controlled.sh

echo "[INFO] run Recon full controlled pipeline..."
bash scripts/run_recon_full_controlled.sh

echo "[INFO] link full outputs to default build-sample input paths..."
ln -sf ascat_observation_features_full.csv data/interim/ascat/ascat_observation_features.csv
ln -sf recon_observation_features_full.csv data/interim/recon/recon_observation_features.csv

if [[ "$RUN_SAMPLE_BUILD" == "1" ]]; then
  echo "[INFO] build sample preview..."
  python scripts/build_dataset_sample_preview_v0_1.py
fi

echo "[INFO] full run completed:"
echo "  data/interim/ascat/ascat_observation_features_full.csv"
echo "  data/interim/ascat/ascat_observation_features_full_summary.json"
echo "  data/interim/recon/recon_observation_features_full.csv"
echo "  data/interim/recon/recon_observation_features_full_summary.json"
