#!/usr/bin/env bash
set -euo pipefail

SRC_ROOT="${1:-/root/Cyclone_next/data/raw/atcf/by_category}"
DST_ROOT="${2:-/root/Cyclone_next/data/processed/atcf/plaintext}"

MANIFEST="$DST_ROOT/manifest.csv"
TMP_MANIFEST="$DST_ROOT/manifest.csv.tmp"

mkdir -p "$DST_ROOT/a_deck" "$DST_ROOT/b_deck"

processed=0
decompressed=0
copied=0
skipped=0
errors=0

printf 'category,year,file_name,source_path,target_path,source_bytes,target_bytes,line_count,action\n' > "$TMP_MANIFEST"

process_file() {
  local src_file="$1"

  local rel_path="${src_file#$SRC_ROOT/}"
  local category="${rel_path%%/*}"
  local rest="${rel_path#*/}"
  local year="${rest%%/*}"
  local file_name="${rest#*/}"

  local dst_dir="$DST_ROOT/$category/$year"
  mkdir -p "$dst_dir"

  local dst_file=""
  local action=""

  if [[ "$file_name" == *.dat.gz ]]; then
    dst_file="$dst_dir/${file_name%.gz}"

    if [[ -f "$dst_file" && "$dst_file" -nt "$src_file" ]]; then
      action="skip"
      skipped=$((skipped + 1))
    else
      if gzip -cd "$src_file" > "$dst_file.tmp"; then
        mv "$dst_file.tmp" "$dst_file"
        action="decompress"
        decompressed=$((decompressed + 1))
      else
        rm -f "$dst_file.tmp"
        action="error"
        errors=$((errors + 1))
      fi
    fi
  elif [[ "$file_name" == *.dat ]]; then
    dst_file="$dst_dir/$file_name"

    if [[ -f "$dst_file" && "$dst_file" -nt "$src_file" ]]; then
      action="skip"
      skipped=$((skipped + 1))
    else
      cp -p "$src_file" "$dst_file"
      action="copy"
      copied=$((copied + 1))
    fi
  else
    return 0
  fi

  local src_bytes=0
  local dst_bytes=0
  local line_count=0

  if [[ -f "$src_file" ]]; then
    src_bytes="$(stat -c %s "$src_file" 2>/dev/null || echo 0)"
  fi

  if [[ -n "$dst_file" && -f "$dst_file" ]]; then
    dst_bytes="$(stat -c %s "$dst_file" 2>/dev/null || echo 0)"
    line_count="$(wc -l < "$dst_file" | tr -d ' ')"
  fi

  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "$category" \
    "$year" \
    "$file_name" \
    "$src_file" \
    "$dst_file" \
    "$src_bytes" \
    "$dst_bytes" \
    "$line_count" \
    "$action" >> "$TMP_MANIFEST"

  processed=$((processed + 1))
  if (( processed % 100 == 0 )); then
    echo "[PROGRESS] processed=$processed decompressed=$decompressed copied=$copied skipped=$skipped errors=$errors"
  fi
}

while IFS= read -r src_file; do
  process_file "$src_file"
done < <(find "$SRC_ROOT" -type f \( -name '*.dat.gz' -o -name '*.dat' \) | sort)

mv "$TMP_MANIFEST" "$MANIFEST"

A_COUNT=$(find "$DST_ROOT/a_deck" -type f -name '*.dat' | wc -l)
B_COUNT=$(find "$DST_ROOT/b_deck" -type f -name '*.dat' | wc -l)
A_BYTES=$(du -sb "$DST_ROOT/a_deck" | awk '{print $1}')
B_BYTES=$(du -sb "$DST_ROOT/b_deck" | awk '{print $1}')

printf '\n[SUMMARY]\n'
printf 'processed=%s decompressed=%s copied=%s skipped=%s errors=%s\n' "$processed" "$decompressed" "$copied" "$skipped" "$errors"
printf 'a_dat_files=%s b_dat_files=%s\n' "$A_COUNT" "$B_COUNT"
printf 'a_dat_bytes=%s b_dat_bytes=%s total_bytes=%s\n' "$A_BYTES" "$B_BYTES" "$((A_BYTES + B_BYTES))"
printf 'manifest=%s\n' "$MANIFEST"
