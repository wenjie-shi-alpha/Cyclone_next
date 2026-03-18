#!/usr/bin/env bash
set -euo pipefail

# Full ASCAT extraction with yearly checkpoints.
# Usage:
#   bash scripts/run_ascat_full_controlled.sh

BASE_DIR="$(cd "$(dirname "$0")/.." && pwd)"

cd "$BASE_DIR"
source .venv/bin/activate

FULL_MANIFEST="data/interim/ascat/ascat_request_manifest_full.csv"
RECON_MANIFEST_DUMMY="data/interim/recon/recon_request_manifest_full.csv"
YEAR_MANIFEST_DIR="data/interim/ascat/full_by_year/manifests"
YEAR_OUT_DIR="data/interim/ascat/full_by_year/features"
YEAR_SUMMARY_DIR="data/interim/ascat/full_by_year/summaries"
FINAL_OUT="data/interim/ascat/ascat_observation_features_full.csv"
FINAL_SUMMARY="data/interim/ascat/ascat_observation_features_full_summary.json"

mkdir -p "$YEAR_MANIFEST_DIR" "$YEAR_OUT_DIR" "$YEAR_SUMMARY_DIR"

if [[ ! -f "$FULL_MANIFEST" ]]; then
  echo "[INFO] ASCAT full manifest missing, generating..."
  python scripts/build_obs_request_manifest.py \
    --year-start 2016 \
    --year-end 2025 \
    --ascat-out-csv "$FULL_MANIFEST" \
    --recon-out-csv "$RECON_MANIFEST_DUMMY"
fi

echo "[INFO] splitting ASCAT full manifest by year..."
python - <<'PY'
import csv
from pathlib import Path

full = Path("data/interim/ascat/ascat_request_manifest_full.csv")
out_dir = Path("data/interim/ascat/full_by_year/manifests")
out_dir.mkdir(parents=True, exist_ok=True)

with full.open("r", encoding="utf-8", newline="") as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames or []
    rows_by_year = {}
    for row in reader:
        y = (row.get("issue_time_utc") or "")[:4]
        if not y:
            continue
        rows_by_year.setdefault(y, []).append(row)

for y, rows in sorted(rows_by_year.items()):
    fp = out_dir / f"ascat_request_manifest_{y}.csv"
    with fp.open("w", encoding="utf-8", newline="") as fw:
        w = csv.DictWriter(fw, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"{y}: {len(rows)}")
PY

echo "[INFO] start yearly ASCAT extraction..."
for year in 2016 2017 2018 2019 2020 2021 2022 2023 2024 2025; do
  manifest="$YEAR_MANIFEST_DIR/ascat_request_manifest_${year}.csv"
  out_csv="$YEAR_OUT_DIR/ascat_observation_features_${year}.csv"
  out_json="$YEAR_SUMMARY_DIR/ascat_observation_features_${year}_summary.json"

  if [[ ! -f "$manifest" ]]; then
    echo "[WARN] missing manifest for ${year}, skip"
    continue
  fi

  if [[ -f "$out_csv" && -f "$out_json" ]]; then
    echo "[INFO] year=${year} already completed, skip"
    continue
  fi

  echo "[INFO] year=${year} running..."
  python scripts/extract_ascat_features_remote.py \
    --manifest-csv "$manifest" \
    --out-csv "$out_csv" \
    --summary-json "$out_json" \
    --only-with-storm-id \
    --max-retries 3 \
    --sleep-sec 0.05

  echo "[INFO] year=${year} done. cooldown 2s..."
  sleep 2
done

echo "[INFO] merging yearly ASCAT outputs..."
python - <<'PY'
import csv
import json
from pathlib import Path

year_csv_dir = Path("data/interim/ascat/full_by_year/features")
year_sum_dir = Path("data/interim/ascat/full_by_year/summaries")
final_csv = Path("data/interim/ascat/ascat_observation_features_full.csv")
final_summary = Path("data/interim/ascat/ascat_observation_features_full_summary.json")

csv_files = sorted(year_csv_dir.glob("ascat_observation_features_*.csv"))
summary_files = sorted(year_sum_dir.glob("ascat_observation_features_*_summary.json"))
if not csv_files:
    raise SystemExit("no yearly ASCAT csv outputs found")

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
                status = (row.get("ascat_status") or "").strip()
                by_year.setdefault(year, {"total": 0, "available": 0, "missing": 0})
                by_year[year]["total"] += 1
                if status == "available":
                    available_rows += 1
                    by_year[year]["available"] += 1
                else:
                    missing_rows += 1
                    by_year[year]["missing"] += 1

for y in sorted(by_year):
    t = by_year[y]["total"]
    by_year[y]["coverage_rate"] = round((by_year[y]["available"] / t) if t else 0.0, 6)

dataset_ids_used = []
for fp in summary_files:
    obj = json.loads(fp.read_text(encoding="utf-8"))
    for ds in obj.get("dataset_ids_used", []):
        if ds not in dataset_ids_used:
            dataset_ids_used.append(ds)

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

echo "[INFO] ASCAT full controlled run completed."
