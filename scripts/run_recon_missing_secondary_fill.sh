#!/usr/bin/env bash
set -euo pipefail

# Targeted Recon fill:
# 1. keep the current REPNT2/REPNT3 baseline outputs
# 2. build manifests only for request_ids that are still missing
# 3. run AHONT1/AHOPN1 on those missing rows
# 4. merge "missing -> available" promotions into supplemented yearly/full outputs

BASE_DIR="$(cd "$(dirname "$0")/.." && pwd)"

cd "$BASE_DIR"
source .venv/bin/activate

FULL_MANIFEST="data/interim/recon/recon_request_manifest_full.csv"
ASCAT_MANIFEST_DUMMY="data/interim/ascat/ascat_request_manifest_full.csv"
BASE_YEAR_MANIFEST_DIR="data/interim/recon/full_by_year/manifests"
BASE_YEAR_FEATURE_DIR="data/interim/recon/full_by_year/features"
BASE_YEAR_SUMMARY_DIR="data/interim/recon/full_by_year/summaries"

SUPP_ROOT="data/interim/recon/supplement_secondary_fill"
SUPP_YEAR_MANIFEST_DIR="${SUPP_ROOT}/manifests"
SUPP_YEAR_MANIFEST_SUMMARY_DIR="${SUPP_ROOT}/manifest_summaries"
SUPP_YEAR_FEATURE_DIR="${SUPP_ROOT}/features"
SUPP_YEAR_SUMMARY_DIR="${SUPP_ROOT}/summaries"
MERGED_YEAR_FEATURE_DIR="${SUPP_ROOT}/merged_features"
MERGED_YEAR_SUMMARY_DIR="${SUPP_ROOT}/merged_summaries"

FINAL_OUT="${FINAL_OUT:-data/interim/recon/recon_observation_features_full_supplemented.csv}"
FINAL_SUMMARY="${FINAL_SUMMARY:-data/interim/recon/recon_observation_features_full_supplemented_summary.json}"
PROMOTE_TO_CANONICAL="${PROMOTE_TO_CANONICAL:-0}"
FORCE_RERUN="${FORCE_RERUN:-0}"
export FINAL_OUT FINAL_SUMMARY

CANONICAL_OUT="data/interim/recon/recon_observation_features.csv"
CANONICAL_SUMMARY="data/interim/recon/recon_observation_features_summary.json"
FULL_CANONICAL_OUT="data/interim/recon/recon_observation_features_full.csv"
FULL_CANONICAL_SUMMARY="data/interim/recon/recon_observation_features_full_summary.json"

RECON_SLEEP_SEC="${RECON_SLEEP_SEC:-0.10}"
RECON_HTTP_SLEEP_SEC="${RECON_HTTP_SLEEP_SEC:-0.08}"
RECON_YEAR_COOLDOWN_SEC="${RECON_YEAR_COOLDOWN_SEC:-2}"
RECON_MAX_RETRIES="${RECON_MAX_RETRIES:-2}"
RECON_CACHE_DIR="${RECON_CACHE_DIR:-data/interim/recon/cache}"
RECON_YEAR_START="${RECON_YEAR_START:-2016}"
RECON_YEAR_END="${RECON_YEAR_END:-2025}"
RECON_CACHE_ONLY="${RECON_CACHE_ONLY:-0}"
RECON_HTTP_TIMEOUT_SEC="${RECON_HTTP_TIMEOUT_SEC:-12}"
RECON_MAX_CANDIDATES="${RECON_MAX_CANDIDATES:-80}"
RECON_SECONDARY_SUBDIRS="${RECON_SECONDARY_SUBDIRS:-AHONT1 AHOPN1}"

mkdir -p \
  "$BASE_YEAR_MANIFEST_DIR" \
  "$SUPP_YEAR_MANIFEST_DIR" \
  "$SUPP_YEAR_MANIFEST_SUMMARY_DIR" \
  "$SUPP_YEAR_FEATURE_DIR" \
  "$SUPP_YEAR_SUMMARY_DIR" \
  "$MERGED_YEAR_FEATURE_DIR" \
  "$MERGED_YEAR_SUMMARY_DIR" \
  "$RECON_CACHE_DIR"

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

echo "[INFO] targeted Recon secondary fill"
echo "[INFO] secondary subdirs: ${RECON_SECONDARY_SUBDIRS}"
echo "[INFO] year range: ${RECON_YEAR_START}-${RECON_YEAR_END}"
echo "[INFO] cache_only: ${RECON_CACHE_ONLY}"
echo "[INFO] http_timeout_sec: ${RECON_HTTP_TIMEOUT_SEC}"
echo "[INFO] max_candidates_per_request: ${RECON_MAX_CANDIDATES}"
echo "[INFO] force_rerun: ${FORCE_RERUN}"

read -r -a RECON_SECONDARY_SUBDIR_ARRAY <<< "$RECON_SECONDARY_SUBDIRS"
RECON_SECONDARY_SUBDIR_ARGS=()
for subdir in "${RECON_SECONDARY_SUBDIR_ARRAY[@]}"; do
  RECON_SECONDARY_SUBDIR_ARGS+=(--subdir "$subdir")
done

RECON_CACHE_ONLY_ARGS=()
if [[ "$RECON_CACHE_ONLY" == "1" ]]; then
  RECON_CACHE_ONLY_ARGS+=(--cache-only)
fi

echo "[INFO] ensuring yearly Recon manifests exist..."
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
    if fp.exists():
        continue
    with fp.open("w", encoding="utf-8", newline="") as fw:
        w = csv.DictWriter(fw, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"{y}: {len(rows)}")
PY

for year in $(seq "$RECON_YEAR_START" "$RECON_YEAR_END"); do
  base_manifest="${BASE_YEAR_MANIFEST_DIR}/recon_request_manifest_${year}.csv"
  base_features="${BASE_YEAR_FEATURE_DIR}/recon_observation_features_${year}.csv"
  base_summary="${BASE_YEAR_SUMMARY_DIR}/recon_observation_features_${year}_summary.json"
  supp_manifest="${SUPP_YEAR_MANIFEST_DIR}/recon_missing_manifest_${year}.csv"
  supp_manifest_summary="${SUPP_YEAR_MANIFEST_SUMMARY_DIR}/recon_missing_manifest_${year}_summary.json"
  supp_features="${SUPP_YEAR_FEATURE_DIR}/recon_secondary_fill_${year}.csv"
  supp_summary="${SUPP_YEAR_SUMMARY_DIR}/recon_secondary_fill_${year}_summary.json"
  merged_features="${MERGED_YEAR_FEATURE_DIR}/recon_observation_features_${year}.csv"
  merged_summary="${MERGED_YEAR_SUMMARY_DIR}/recon_observation_features_${year}_summary.json"

  if [[ ! -f "$base_manifest" ]]; then
    echo "[WARN] missing base manifest for ${year}, skip"
    continue
  fi
  if [[ ! -f "$base_features" ]]; then
    echo "[WARN] missing base features for ${year}, skip"
    continue
  fi
  if [[ ! -f "$base_summary" ]]; then
    echo "[WARN] missing base summary for ${year}, skip"
    continue
  fi

  if [[ "$FORCE_RERUN" != "1" && -f "$merged_features" && -f "$merged_summary" ]]; then
    echo "[INFO] year=${year} supplemented output already exists, skip"
    continue
  fi

  echo "[INFO] year=${year} building missing-only manifest..."
  python scripts/build_recon_missing_manifest.py \
    --manifest-csv "$base_manifest" \
    --base-features-csv "$base_features" \
    --out-csv "$supp_manifest" \
    --summary-json "$supp_manifest_summary"

  missing_rows="$(python - <<PY
import json
from pathlib import Path
fp = Path("$supp_manifest_summary")
data = json.loads(fp.read_text(encoding="utf-8"))
print(int(data.get("rows_written", 0) or 0))
PY
)"
  echo "[INFO] year=${year} missing rows to supplement: ${missing_rows}"

  if [[ "$missing_rows" -gt 0 ]]; then
    echo "[INFO] year=${year} running AHONT1/AHOPN1 supplement..."
    python scripts/extract_recon_features_remote.py \
      --manifest-csv "$supp_manifest" \
      --out-csv "$supp_features" \
      --summary-json "$supp_summary" \
      --only-with-storm-id \
      "${RECON_SECONDARY_SUBDIR_ARGS[@]}" \
      "${RECON_CACHE_ONLY_ARGS[@]}" \
      --catalog-cache-dir "$RECON_CACHE_DIR" \
      --max-retries "$RECON_MAX_RETRIES" \
      --sleep-sec "$RECON_SLEEP_SEC" \
      --http-sleep-sec "$RECON_HTTP_SLEEP_SEC" \
      --http-timeout-sec "$RECON_HTTP_TIMEOUT_SEC" \
      --max-candidates-per-request "$RECON_MAX_CANDIDATES"
  else
    rm -f "$supp_features" "$supp_summary"
  fi

  python scripts/merge_recon_supplement.py \
    --base-features-csv "$base_features" \
    --supplement-features-csv "$supp_features" \
    --out-csv "$merged_features" \
    --summary-json "$merged_summary"

  echo "[INFO] year=${year} supplemented merge done. cooldown ${RECON_YEAR_COOLDOWN_SEC}s..."
  sleep "$RECON_YEAR_COOLDOWN_SEC"
done

echo "[INFO] merging supplemented yearly Recon outputs..."
python - <<'PY'
import csv
import json
import os
from pathlib import Path

year_csv_dir = Path("data/interim/recon/supplement_secondary_fill/merged_features")
year_sum_dir = Path("data/interim/recon/supplement_secondary_fill/merged_summaries")
final_csv = Path(os.environ["FINAL_OUT"])
final_summary = Path(os.environ["FINAL_SUMMARY"])

csv_files = sorted(year_csv_dir.glob("recon_observation_features_*.csv"))
summary_files = sorted(year_sum_dir.glob("recon_observation_features_*_summary.json"))
if not csv_files:
    raise SystemExit("no supplemented yearly Recon csv outputs found")

fieldnames = None
rows_written = 0
available_rows = 0
missing_rows = 0
coverage_by_year = {}
promoted_rows_total = 0

for summary_fp in summary_files:
    try:
        obj = json.loads(summary_fp.read_text(encoding="utf-8"))
    except Exception:
        continue
    promoted_rows_total += int(obj.get("promoted_rows", 0) or 0)

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
                bucket = coverage_by_year.setdefault(year, {"total": 0, "available": 0, "missing": 0})
                bucket["total"] += 1
                if (row.get("recon_status") or "").strip() == "available":
                    available_rows += 1
                    bucket["available"] += 1
                else:
                    missing_rows += 1
                    bucket["missing"] += 1

for year, bucket in coverage_by_year.items():
    total = bucket["total"]
    bucket["coverage_rate"] = round((bucket["available"] / total), 6) if total else 0.0

final = {
    "generated_from": "secondary_fill_by_year",
    "requests_total": rows_written,
    "rows_written": rows_written,
    "available_rows": available_rows,
    "missing_rows": missing_rows,
    "promoted_rows_total": promoted_rows_total,
    "coverage_by_year": coverage_by_year,
    "yearly_summary_files": [str(p) for p in summary_files],
}
final_summary.write_text(json.dumps(final, ensure_ascii=False, indent=2), encoding="utf-8")
print(final_csv)
print(final_summary)
print("rows_written:", rows_written)
print("available_rows:", available_rows)
print("missing_rows:", missing_rows)
print("promoted_rows_total:", promoted_rows_total)
PY

if [[ "$PROMOTE_TO_CANONICAL" == "1" ]]; then
  cp "$FINAL_OUT" "$FULL_CANONICAL_OUT"
  cp "$FINAL_SUMMARY" "$FULL_CANONICAL_SUMMARY"
  cp "$FINAL_OUT" "$CANONICAL_OUT"
  cp "$FINAL_SUMMARY" "$CANONICAL_SUMMARY"
  echo "[INFO] promoted supplemented Recon outputs to canonical files"
  echo "  $FULL_CANONICAL_OUT"
  echo "  $FULL_CANONICAL_SUMMARY"
  echo "  $CANONICAL_OUT"
  echo "  $CANONICAL_SUMMARY"
fi

echo "[INFO] Recon secondary fill completed."
