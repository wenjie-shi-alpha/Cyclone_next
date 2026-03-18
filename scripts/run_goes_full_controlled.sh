#!/usr/bin/env bash
set -euo pipefail

# Full GOES extraction with conservative throttling and yearly checkpointing.
# Usage:
#   bash scripts/run_goes_full_controlled.sh [PROJECT_ID] [--force]

PROJECT_ID="eminent-glider-467006-r0"
FORCE_RERUN=0
PROJECT_SET=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --force)
      FORCE_RERUN=1
      shift
      ;;
    -h|--help)
      echo "Usage: bash scripts/run_goes_full_controlled.sh [PROJECT_ID] [--force]"
      echo "  PROJECT_ID: optional, default eminent-glider-467006-r0"
      echo "  --force: rerun all years even when yearly outputs already exist"
      exit 0
      ;;
    *)
      if [[ "$PROJECT_SET" -eq 0 ]]; then
        PROJECT_ID="$1"
        PROJECT_SET=1
        shift
      else
        echo "[ERROR] unknown argument: $1" >&2
        exit 2
      fi
      ;;
  esac
done

BASE_DIR="$(cd "$(dirname "$0")/.." && pwd)"

cd "$BASE_DIR"
source .venv/bin/activate

FULL_MANIFEST="data/interim/goes/goes_request_manifest_full.csv"
YEAR_MANIFEST_DIR="data/interim/goes/full_by_year/manifests"
YEAR_OUT_DIR="data/interim/goes/full_by_year/features"
YEAR_SUMMARY_DIR="data/interim/goes/full_by_year/summaries"
FINAL_OUT="data/interim/goes/goes_observation_features_full.csv"
FINAL_SUMMARY="data/interim/goes/goes_observation_features_full_summary.json"
CANONICAL_OUT="data/interim/goes/goes_observation_features.csv"
CANONICAL_SUMMARY="data/interim/goes/goes_observation_features_summary.json"

mkdir -p "$YEAR_MANIFEST_DIR" "$YEAR_OUT_DIR" "$YEAR_SUMMARY_DIR"
echo "[INFO] project=${PROJECT_ID} force_rerun=${FORCE_RERUN}"

if [[ ! -f "$FULL_MANIFEST" ]]; then
  echo "[INFO] full manifest missing, generating..."
  python scripts/build_goes_request_manifest.py \
    --year-start 2016 \
    --year-end 2025 \
    --out-csv "$FULL_MANIFEST"
fi

echo "[INFO] splitting full manifest by year..."
python - <<'PY'
import csv
from pathlib import Path

full = Path("data/interim/goes/goes_request_manifest_full.csv")
out_dir = Path("data/interim/goes/full_by_year/manifests")
out_dir.mkdir(parents=True, exist_ok=True)

with full.open("r", encoding="utf-8", newline="") as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames or []
    by_year = {}
    rows_by_year = {}
    for row in reader:
        y = (row.get("issue_time_utc") or "")[:4]
        if not y:
            continue
        rows_by_year.setdefault(y, []).append(row)

for y, rows in sorted(rows_by_year.items()):
    fp = out_dir / f"goes_request_manifest_{y}.csv"
    with fp.open("w", encoding="utf-8", newline="") as fw:
        w = csv.DictWriter(fw, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"{y}: {len(rows)}")
PY

echo "[INFO] start yearly extraction with throttling..."
for year in 2016 2017 2018 2019 2020 2021 2022 2023 2024 2025; do
  manifest="$YEAR_MANIFEST_DIR/goes_request_manifest_${year}.csv"
  out_csv="$YEAR_OUT_DIR/goes_observation_features_${year}.csv"
  out_json="$YEAR_SUMMARY_DIR/goes_observation_features_${year}_summary.json"

  if [[ ! -f "$manifest" ]]; then
    echo "[WARN] missing manifest for ${year}, skip"
    continue
  fi

  if [[ "$FORCE_RERUN" -eq 0 && -f "$out_csv" && -f "$out_json" ]]; then
    echo "[INFO] year=${year} already completed, skip"
    continue
  fi

  if [[ "$FORCE_RERUN" -eq 1 && ( -f "$out_csv" || -f "$out_json" ) ]]; then
    echo "[INFO] year=${year} force rerun: overwrite existing yearly outputs"
  fi

  echo "[INFO] year=${year} running..."
  python scripts/extract_goes_features_gee.py \
    --manifest-csv "$manifest" \
    --out-csv "$out_csv" \
    --summary-json "$out_json" \
    --project "$PROJECT_ID" \
    --only-with-storm-id \
    --batch-size 200 \
    --sleep-sec 1.2 \
    --max-retries 6

  echo "[INFO] year=${year} done. cooldown 8s..."
  sleep 8
done

echo "[INFO] merging yearly outputs..."
python - <<'PY'
import csv
import json
from pathlib import Path

year_csv_dir = Path("data/interim/goes/full_by_year/features")
year_sum_dir = Path("data/interim/goes/full_by_year/summaries")
final_csv = Path("data/interim/goes/goes_observation_features_full.csv")
final_summary = Path("data/interim/goes/goes_observation_features_full_summary.json")

csv_files = sorted(year_csv_dir.glob("goes_observation_features_*.csv"))
summary_files = sorted(year_sum_dir.glob("goes_observation_features_*_summary.json"))
if not csv_files:
    raise SystemExit("no yearly csv outputs found")

fieldnames = None
rows_written = 0
available_rows = 0
missing_rows = 0
by_year = {}

with final_csv.open("w", encoding="utf-8", newline="") as fw:
    writer = None
    for fp in csv_files:
        year = fp.stem.split("_")[-1]
        with fp.open("r", encoding="utf-8", newline="") as fr:
            reader = csv.DictReader(fr)
            if fieldnames is None:
                fieldnames = reader.fieldnames
                writer = csv.DictWriter(fw, fieldnames=fieldnames)
                writer.writeheader()
            for row in reader:
                writer.writerow(row)
                rows_written += 1
                status = (row.get("goes_status") or "").strip()
                if status == "available":
                    available_rows += 1
                    by_year.setdefault(year, {"total": 0, "available": 0, "missing": 0})
                    by_year[year]["available"] += 1
                else:
                    missing_rows += 1
                    by_year.setdefault(year, {"total": 0, "available": 0, "missing": 0})
                    by_year[year]["missing"] += 1
                by_year.setdefault(year, {"total": 0, "available": 0, "missing": 0})
                by_year[year]["total"] += 1

dataset_ids_used = []
for fp in summary_files:
    obj = json.loads(fp.read_text(encoding="utf-8"))
    for ds in obj.get("dataset_ids_used", []):
        if ds not in dataset_ids_used:
            dataset_ids_used.append(ds)

for y in sorted(by_year):
    t = by_year[y]["total"]
    by_year[y]["coverage_rate"] = round((by_year[y]["available"] / t) if t else 0.0, 6)

final = {
    "generated_from": "yearly_controlled_runs",
    "requests_total": rows_written,
    "rows_written": rows_written,
    "available_rows": available_rows,
    "missing_rows": missing_rows,
    "dataset_ids_used": dataset_ids_used,
    "coverage_by_year": by_year,
    "yearly_summary_files": [str(p) for p in summary_files],
}
final_summary.write_text(json.dumps(final, ensure_ascii=False, indent=2), encoding="utf-8")
print(final_csv)
print(final_summary)
print("rows_written:", rows_written)
print("available_rows:", available_rows)
print("missing_rows:", missing_rows)
PY

echo "[INFO] full controlled run completed."
cp "$FINAL_OUT" "$CANONICAL_OUT"
cp "$FINAL_SUMMARY" "$CANONICAL_SUMMARY"
echo "[INFO] canonical GOES files synced:"
echo "  $CANONICAL_OUT"
echo "  $CANONICAL_SUMMARY"
