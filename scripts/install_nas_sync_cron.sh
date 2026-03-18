#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SYNC_SCRIPT="$SCRIPT_DIR/sync_projects_to_nas.sh"
CRON_EXPR="${1:-0 */6 * * *}"
LOG_FILE="$PROJECT_ROOT/.state/nas-sync/cron.log"
MARKER="# cyclone_nas_sync_job"

if [[ ! -x "$SYNC_SCRIPT" ]]; then
  chmod +x "$SYNC_SCRIPT"
fi

mkdir -p "$(dirname "$LOG_FILE")"

tmp_file="$(mktemp)"
if crontab -l >/dev/null 2>&1; then
  crontab -l | grep -Fv "$MARKER" > "$tmp_file" || true
else
  : > "$tmp_file"
fi

printf "%s bash -lc '%s >> %s 2>&1' %s\n" \
  "$CRON_EXPR" \
  "$SYNC_SCRIPT" \
  "$LOG_FILE" \
  "$MARKER" >> "$tmp_file"

crontab "$tmp_file"
rm -f "$tmp_file"

echo "Installed cron job: $CRON_EXPR"
echo "Sync script: $SYNC_SCRIPT"
echo "Log file: $LOG_FILE"
