#!/usr/bin/env bash
set -euo pipefail

# Smoke run for ASCAT + Recon on CDS JupyterLab terminal.
# Usage:
#   bash scripts/run_obs_smoke_cds_manual.sh [YEAR] [LIMIT]
# Example:
#   bash scripts/run_obs_smoke_cds_manual.sh 2020 50

YEAR="${1:-2020}"
LIMIT="${2:-50}"
BASE_DIR="$(cd "$(dirname "$0")/.." && pwd)"

cd "$BASE_DIR"
source .venv/bin/activate

ASCAT_MANIFEST="data/interim/ascat/ascat_request_manifest_smoke.csv"
RECON_MANIFEST="data/interim/recon/recon_request_manifest_smoke.csv"
ASCAT_OUT="data/interim/ascat/ascat_observation_features_smoke.csv"
ASCAT_SUMMARY="data/interim/ascat/ascat_observation_features_smoke_summary.json"
RECON_OUT="data/interim/recon/recon_observation_features_smoke.csv"
RECON_SUMMARY="data/interim/recon/recon_observation_features_smoke_summary.json"

echo "[INFO] build smoke manifests: year=${YEAR}, limit=${LIMIT}"
python scripts/build_obs_request_manifest.py \
  --year-start "$YEAR" \
  --year-end "$YEAR" \
  --ascat-out-csv "$ASCAT_MANIFEST" \
  --recon-out-csv "$RECON_MANIFEST" \
  --limit "$LIMIT"

echo "[INFO] run ASCAT smoke extraction..."
python scripts/extract_ascat_features_remote.py \
  --manifest-csv "$ASCAT_MANIFEST" \
  --out-csv "$ASCAT_OUT" \
  --summary-json "$ASCAT_SUMMARY" \
  --only-with-storm-id \
  --max-retries 3 \
  --sleep-sec 0.05

echo "[INFO] run Recon smoke extraction..."
python scripts/extract_recon_features_remote.py \
  --manifest-csv "$RECON_MANIFEST" \
  --out-csv "$RECON_OUT" \
  --summary-json "$RECON_SUMMARY" \
  --only-with-storm-id \
  --max-retries 3 \
  --sleep-sec 0.05

echo "[INFO] smoke run completed:"
echo "  $ASCAT_OUT"
echo "  $ASCAT_SUMMARY"
echo "  $RECON_OUT"
echo "  $RECON_SUMMARY"
