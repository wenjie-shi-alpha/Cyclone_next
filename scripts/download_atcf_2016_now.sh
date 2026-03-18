#!/usr/bin/env bash
set -euo pipefail

START_YEAR="${1:-2016}"
CURRENT_YEAR="$(date +%Y)"
ROOT="${2:-/root/Cyclone_next/data/raw/atcf/by_category}"
END_YEAR="${3:-$CURRENT_YEAR}"

if (( END_YEAR > CURRENT_YEAR )); then
  END_YEAR="$CURRENT_YEAR"
fi

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

attempts=0
failures=0
downloaded=0
skipped=0

fetch_index() {
  local url="$1"
  local out="$2"
  curl -fsSL "$url" -o "$out"
}

extract_hrefs() {
  local file="$1"
  grep -oE 'href="[^"]+"' "$file" | sed -E 's/^href="//; s/"$//'
}

download_from_index() {
  local url="$1"
  local deck="$2"
  local index_file="$TMP_DIR/index_$(echo "$url" | tr '/:' '__').html"

  if ! fetch_index "$url" "$index_file"; then
    echo "[WARN] skip index: $url"
    return 0
  fi

  while IFS= read -r href; do
    [[ -z "$href" ]] && continue
    [[ "$href" == */ ]] && continue

    local fname="${href##*/}"
    local year
    if [[ "$fname" =~ ^${deck}[a-z]{2}[a-z0-9]{2}([0-9]{4})\.dat(\.gz)?$ ]]; then
      year="${BASH_REMATCH[1]}"
      if (( year < START_YEAR || year > CURRENT_YEAR )); then
        continue
      fi

      local category_dir
      if [[ "$deck" == "a" ]]; then
        category_dir="$ROOT/a_deck/$year"
      else
        category_dir="$ROOT/b_deck/$year"
      fi
      mkdir -p "$category_dir"

      attempts=$((attempts + 1))

      local target_file="$category_dir/$fname"
      local before_mtime=""
      if [[ -f "$target_file" ]]; then
        before_mtime="$(stat -c %Y "$target_file" 2>/dev/null || true)"
      fi

      if ! wget -q -N --timeout=30 --tries=3 --waitretry=2 -P "$category_dir" "${url}${fname}"; then
        failures=$((failures + 1))
        echo "[WARN] download failed: ${url}${fname}"
      else
        local after_mtime=""
        if [[ -f "$target_file" ]]; then
          after_mtime="$(stat -c %Y "$target_file" 2>/dev/null || true)"
        fi

        if [[ -z "$before_mtime" || "$before_mtime" != "$after_mtime" ]]; then
          downloaded=$((downloaded + 1))
        else
          skipped=$((skipped + 1))
        fi
      fi

      if (( attempts % 100 == 0 )); then
        echo "[PROGRESS] attempts=$attempts downloaded=$downloaded skipped=$skipped failures=$failures"
      fi
    fi
  done < <(extract_hrefs "$index_file")
}

echo "[INFO] START_YEAR=$START_YEAR END_YEAR=$END_YEAR CURRENT_YEAR=$CURRENT_YEAR ROOT=$ROOT"

for y in $(seq "$START_YEAR" "$END_YEAR"); do
  echo "[YEAR] archive $y ..."
  download_from_index "https://ftp.nhc.noaa.gov/atcf/archive/${y}/" a
  download_from_index "https://ftp.nhc.noaa.gov/atcf/archive/${y}/" b
  echo "[YEAR] archive $y done"
done

echo "[SOURCE] realtime aid_public ..."
download_from_index "https://ftp.nhc.noaa.gov/atcf/aid_public/" a
echo "[SOURCE] realtime btk ..."
download_from_index "https://ftp.nhc.noaa.gov/atcf/btk/" b
echo "[SOURCE] realtime btk/cphc ..."
download_from_index "https://ftp.nhc.noaa.gov/atcf/btk/cphc/" b || true
echo "[SOURCE] realtime done"

A_COUNT=$(find "$ROOT/a_deck" -type f | wc -l)
B_COUNT=$(find "$ROOT/b_deck" -type f | wc -l)
A_BYTES=$(du -sb "$ROOT/a_deck" | awk '{print $1}')
B_BYTES=$(du -sb "$ROOT/b_deck" | awk '{print $1}')
TOTAL_BYTES=$((A_BYTES + B_BYTES))

printf '\n[SUMMARY]\n'
printf 'attempts=%s downloaded=%s skipped=%s failures=%s\n' "$attempts" "$downloaded" "$skipped" "$failures"
printf 'a_deck_files=%s b_deck_files=%s\n' "$A_COUNT" "$B_COUNT"
printf 'a_deck_bytes=%s b_deck_bytes=%s total_bytes=%s\n' "$A_BYTES" "$B_BYTES" "$TOTAL_BYTES"

printf '\n[A_DECK_YEARS]\n'
find "$ROOT/a_deck" -mindepth 1 -maxdepth 1 -type d | sed 's#.*/##' | sort -n | tr '\n' ' '
printf '\n\n[B_DECK_YEARS]\n'
find "$ROOT/b_deck" -mindepth 1 -maxdepth 1 -type d | sed 's#.*/##' | sort -n | tr '\n' ' '
printf '\n'
