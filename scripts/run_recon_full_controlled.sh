#!/usr/bin/env bash
set -euo pipefail

# Full Recon extraction with yearly checkpoints.
# Usage:
#   bash scripts/run_recon_full_controlled.sh

BASE_DIR="$(cd "$(dirname "$0")/.." && pwd)"

cd "$BASE_DIR"
source .venv/bin/activate

FULL_MANIFEST="data/interim/recon/recon_request_manifest_full.csv"
ASCAT_MANIFEST_DUMMY="data/interim/ascat/ascat_request_manifest_full.csv"
YEAR_MANIFEST_DIR="data/interim/recon/full_by_year/manifests"
YEAR_OUT_DIR="data/interim/recon/full_by_year/features"
YEAR_SUMMARY_DIR="data/interim/recon/full_by_year/summaries"
FINAL_OUT="data/interim/recon/recon_observation_features_full.csv"
FINAL_SUMMARY="data/interim/recon/recon_observation_features_full_summary.json"
RECON_SLEEP_SEC="${RECON_SLEEP_SEC:-0.20}"
RECON_HTTP_SLEEP_SEC="${RECON_HTTP_SLEEP_SEC:-0.08}"
RECON_YEAR_COOLDOWN_SEC="${RECON_YEAR_COOLDOWN_SEC:-3}"
RECON_MAX_RETRIES="${RECON_MAX_RETRIES:-3}"
RECON_CACHE_DIR="${RECON_CACHE_DIR:-data/interim/recon/cache}"
RECON_PREFETCH_FIRST="${RECON_PREFETCH_FIRST:-1}"
RECON_SUBDIRS="${RECON_SUBDIRS:-REPNT2 REPNT3}"
RECON_YEAR_START="${RECON_YEAR_START:-2016}"
RECON_YEAR_END="${RECON_YEAR_END:-2025}"
RECON_CACHE_ONLY="${RECON_CACHE_ONLY:-0}"
RECON_HTTP_TIMEOUT_SEC="${RECON_HTTP_TIMEOUT_SEC:-20}"
RECON_MAX_CANDIDATES="${RECON_MAX_CANDIDATES:-120}"

mkdir -p "$YEAR_MANIFEST_DIR" "$YEAR_OUT_DIR" "$YEAR_SUMMARY_DIR" "$RECON_CACHE_DIR"

if [[ ! -f "$FULL_MANIFEST" ]]; then
  if [[ -f "recon_request_manifest.csv" ]]; then
    echo "[INFO] using existing local recon manifest: recon_request_manifest.csv"
    cp -f "recon_request_manifest.csv" "$FULL_MANIFEST"
  else
    echo "[INFO] Recon full manifest missing, generating..."
    python scripts/build_obs_request_manifest.py \
      --year-start "$RECON_YEAR_START" \
      --year-end "$RECON_YEAR_END" \
      --ascat-out-csv "$ASCAT_MANIFEST_DUMMY" \
      --recon-out-csv "$FULL_MANIFEST"
  fi
fi

echo "[INFO] local throttling: row_sleep=${RECON_SLEEP_SEC}s http_sleep=${RECON_HTTP_SLEEP_SEC}s year_cooldown=${RECON_YEAR_COOLDOWN_SEC}s retries=${RECON_MAX_RETRIES}"
echo "[INFO] recon subdirs: ${RECON_SUBDIRS}"
echo "[INFO] year range: ${RECON_YEAR_START}-${RECON_YEAR_END}"
echo "[INFO] cache_only: ${RECON_CACHE_ONLY}"
echo "[INFO] http_timeout_sec: ${RECON_HTTP_TIMEOUT_SEC}"
echo "[INFO] max_candidates_per_request: ${RECON_MAX_CANDIDATES}"

read -r -a RECON_SUBDIR_ARRAY <<< "$RECON_SUBDIRS"
RECON_SUBDIR_ARGS=()
for subdir in "${RECON_SUBDIR_ARRAY[@]}"; do
  RECON_SUBDIR_ARGS+=(--subdir "$subdir")
done

RECON_CACHE_ONLY_ARGS=()
if [[ "$RECON_CACHE_ONLY" == "1" ]]; then
  RECON_CACHE_ONLY_ARGS+=(--cache-only)
fi

if [[ "$RECON_PREFETCH_FIRST" == "1" ]]; then
  echo "[INFO] prefetching Recon cache (shell curl path)..."
  RECON_CACHE_DIR="$RECON_CACHE_DIR" \
  RECON_HTTP_SLEEP_SEC="$RECON_HTTP_SLEEP_SEC" \
  RECON_SUBDIRS="$RECON_SUBDIRS" \
  bash scripts/prefetch_recon_cache_local.sh "$RECON_YEAR_START" "$RECON_YEAR_END"
fi

echo "[INFO] splitting Recon full manifest by year..."
python - <<'PY'
import csv
from pathlib import Path

full = Path("data/interim/recon/recon_request_manifest_full.csv")
out_dir = Path("data/interim/recon/full_by_year/manifests")
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
    fp = out_dir / f"recon_request_manifest_{y}.csv"
    with fp.open("w", encoding="utf-8", newline="") as fw:
        w = csv.DictWriter(fw, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"{y}: {len(rows)}")
PY

echo "[INFO] start yearly Recon extraction..."
for year in $(seq "$RECON_YEAR_START" "$RECON_YEAR_END"); do
  manifest="$YEAR_MANIFEST_DIR/recon_request_manifest_${year}.csv"
  out_csv="$YEAR_OUT_DIR/recon_observation_features_${year}.csv"
  out_json="$YEAR_SUMMARY_DIR/recon_observation_features_${year}_summary.json"

  if [[ ! -f "$manifest" ]]; then
    echo "[WARN] missing manifest for ${year}, skip"
    continue
  fi

  if [[ -f "$out_csv" && -f "$out_json" ]]; then
    if python - <<PY
import json
from pathlib import Path
fp = Path("$out_json")
data = json.loads(fp.read_text(encoding="utf-8"))
catalog_total = int(data.get("catalog_entries_total", 0) or 0)
rows_written = int(data.get("rows_written", 0) or 0)
raise SystemExit(0 if catalog_total > 0 and rows_written > 0 else 1)
PY
    then
      echo "[INFO] year=${year} already completed with non-empty catalog, skip"
      continue
    else
      echo "[WARN] year=${year} existing output looks invalid (empty catalog), rerun"
    fi
  fi

  echo "[INFO] year=${year} running..."
  python scripts/extract_recon_features_remote.py \
    --manifest-csv "$manifest" \
    --out-csv "$out_csv" \
    --summary-json "$out_json" \
    --only-with-storm-id \
    "${RECON_SUBDIR_ARGS[@]}" \
    "${RECON_CACHE_ONLY_ARGS[@]}" \
    --catalog-cache-dir "$RECON_CACHE_DIR" \
    --max-retries "$RECON_MAX_RETRIES" \
    --sleep-sec "$RECON_SLEEP_SEC" \
    --http-sleep-sec "$RECON_HTTP_SLEEP_SEC" \
    --http-timeout-sec "$RECON_HTTP_TIMEOUT_SEC" \
    --max-candidates-per-request "$RECON_MAX_CANDIDATES"

  echo "[INFO] year=${year} done. cooldown ${RECON_YEAR_COOLDOWN_SEC}s..."
  sleep "$RECON_YEAR_COOLDOWN_SEC"
done

echo "[INFO] merging yearly Recon outputs..."
python - <<'PY'
import csv
import json
from pathlib import Path

year_csv_dir = Path("data/interim/recon/full_by_year/features")
year_sum_dir = Path("data/interim/recon/full_by_year/summaries")
final_csv = Path("data/interim/recon/recon_observation_features_full.csv")
final_summary = Path("data/interim/recon/recon_observation_features_full_summary.json")

csv_files = sorted(year_csv_dir.glob("recon_observation_features_*.csv"))
summary_files = sorted(year_sum_dir.glob("recon_observation_features_*_summary.json"))
if not csv_files:
    raise SystemExit("no yearly Recon csv outputs found")

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
                status = (row.get("recon_status") or "").strip()
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

final = {
    "generated_from": "yearly_controlled_runs",
    "requests_total": rows_written,
    "rows_written": rows_written,
    "available_rows": available_rows,
    "missing_rows": missing_rows,
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

echo "[INFO] Recon full controlled run completed."
