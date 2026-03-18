#!/usr/bin/env bash
set -euo pipefail

# Prefetch NHC Recon archive listings + message texts to local cache.
# This keeps extraction in "local parse, low-frequency requests" mode.
#
# Usage:
#   bash scripts/prefetch_recon_cache_local.sh [YEAR_START] [YEAR_END]

BASE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$BASE_DIR"

YEAR_START="${1:-2016}"
YEAR_END="${2:-2025}"
RECON_BASE_URL="${RECON_BASE_URL:-https://www.nhc.noaa.gov/archive/recon}"
RECON_CACHE_DIR="${RECON_CACHE_DIR:-data/interim/recon/cache}"
RECON_HTTP_SLEEP_SEC="${RECON_HTTP_SLEEP_SEC:-0.08}"
RECON_SUBDIRS="${RECON_SUBDIRS:-REPNT2 REPNT3}"
RECON_CURL_MAX_TIME="${RECON_CURL_MAX_TIME:-25}"

read -r -a SUBDIRS <<< "$RECON_SUBDIRS"

mkdir -p "$RECON_CACHE_DIR"

echo "[INFO] recon cache prefetch start: years=${YEAR_START}-${YEAR_END} sleep=${RECON_HTTP_SLEEP_SEC}s curl_max_time=${RECON_CURL_MAX_TIME}s cache=${RECON_CACHE_DIR}"

for year in $(seq "$YEAR_START" "$YEAR_END"); do
  for subdir in "${SUBDIRS[@]}"; do
    listing_url="${RECON_BASE_URL%/}/${year}/${subdir}/"
    listing_fp="${RECON_CACHE_DIR}/listing_${year}_${subdir}.html"

    if [[ ! -s "$listing_fp" ]]; then
      tmp_fp="${listing_fp}.tmp"
      if curl -s --max-time "$RECON_CURL_MAX_TIME" --retry 1 --retry-delay 1 "$listing_url" > "$tmp_fp" && [[ -s "$tmp_fp" ]]; then
        mv "$tmp_fp" "$listing_fp"
        echo "[INFO] fetched listing: ${year}/${subdir}"
      else
        rm -f "$tmp_fp"
        echo "[WARN] listing fetch failed: ${listing_url}"
        continue
      fi
      sleep "$RECON_HTTP_SLEEP_SEC"
    else
      echo "[INFO] listing cached: ${year}/${subdir}"
    fi

    mapfile -t files < <(rg -o --no-line-number '[A-Z0-9_-]+\.[0-9]{12}\.txt' "$listing_fp" | sort -u)
    total="${#files[@]}"
    msg_dir="${RECON_CACHE_DIR}/messages/${year}/${subdir}"
    mkdir -p "$msg_dir"
    echo "[INFO] ${year}/${subdir} catalog files=${total}"

    if (( total == 0 )); then
      continue
    fi

    idx=0
    downloaded=0
    skipped=0
    failed=0
    for fn in "${files[@]}"; do
      idx=$((idx + 1))
      out_fp="${msg_dir}/${fn}"
      if [[ -s "$out_fp" ]]; then
        skipped=$((skipped + 1))
      else
        tmp_fp="${out_fp}.tmp"
        if curl -s --max-time "$RECON_CURL_MAX_TIME" --retry 1 --retry-delay 1 "${listing_url}${fn}" > "$tmp_fp" && [[ -s "$tmp_fp" ]]; then
          mv "$tmp_fp" "$out_fp"
          downloaded=$((downloaded + 1))
        else
          rm -f "$tmp_fp"
          failed=$((failed + 1))
        fi
        sleep "$RECON_HTTP_SLEEP_SEC"
      fi

      if (( idx % 500 == 0 || idx == total )); then
        echo "[INFO] ${year}/${subdir} progress ${idx}/${total} downloaded=${downloaded} cached=${skipped} failed=${failed}"
      fi
    done
  done
done

echo "[INFO] recon cache prefetch complete."
