#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_CONFIG="$SCRIPT_DIR/nas_sync_projects.conf"
DEFAULT_DEST_BASE="/mnt/c/Users/swj55/nas/nas_personal/SynologyDrive/github projects"
DEFAULT_STATE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)/.state/nas-sync"
STATE_DIR="${NAS_SYNC_STATE_DIR:-$DEFAULT_STATE_DIR}"
LOCK_FILE="${STATE_DIR}/sync.lock"

DEST_BASE="${NAS_SYNC_DEST_BASE:-$DEFAULT_DEST_BASE}"
CONFIG_FILE="${NAS_SYNC_CONFIG_FILE:-$DEFAULT_CONFIG}"
PROJECT_FILTER=""
DRY_RUN=0
USE_DELETE=1
ALLOW_MISSING_GITIGNORE=0

usage() {
  cat <<'EOF'
Usage: sync_projects_to_nas.sh [options]

Options:
  --dry-run                  Show planned changes only, do not write.
  --no-delete                Do not delete files in destination.
  --project NAME             Sync one project from config by target name.
  --dest PATH                Override destination base folder.
  --config PATH              Override config file path.
  --allow-missing-gitignore  Sync even if source has no .gitignore.
  -h, --help                 Show this help.
EOF
}

timestamp() {
  date '+%Y-%m-%d %H:%M:%S'
}

log() {
  printf '[%s] %s\n' "$(timestamp)" "$*"
}

trim() {
  local s="$1"
  s="${s#"${s%%[![:space:]]*}"}"
  s="${s%"${s##*[![:space:]]}"}"
  printf '%s' "$s"
}

while (($# > 0)); do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --no-delete)
      USE_DELETE=0
      shift
      ;;
    --project)
      PROJECT_FILTER="${2:-}"
      if [[ -z "$PROJECT_FILTER" ]]; then
        echo "ERROR: --project needs a value" >&2
        exit 2
      fi
      shift 2
      ;;
    --dest)
      DEST_BASE="${2:-}"
      if [[ -z "$DEST_BASE" ]]; then
        echo "ERROR: --dest needs a value" >&2
        exit 2
      fi
      shift 2
      ;;
    --config)
      CONFIG_FILE="${2:-}"
      if [[ -z "$CONFIG_FILE" ]]; then
        echo "ERROR: --config needs a value" >&2
        exit 2
      fi
      shift 2
      ;;
    --allow-missing-gitignore)
      ALLOW_MISSING_GITIGNORE=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "ERROR: unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if [[ ! -f "$CONFIG_FILE" ]]; then
  echo "ERROR: config not found: $CONFIG_FILE" >&2
  exit 2
fi

mkdir -p "$STATE_DIR"
if command -v flock >/dev/null 2>&1; then
  exec 9>"$LOCK_FILE"
  if ! flock -n 9; then
    log "Another sync is running, exit."
    exit 0
  fi
fi

mkdir -p "$DEST_BASE"

log "Sync started. dest_base=$DEST_BASE dry_run=$DRY_RUN delete=$USE_DELETE"

synced_count=0
skipped_count=0
failed_count=0
matched_project=0

while IFS= read -r raw_line || [[ -n "$raw_line" ]]; do
  line="$(trim "$raw_line")"
  [[ -z "$line" ]] && continue
  [[ "$line" == \#* ]] && continue

  if [[ "$line" != *"|"* ]]; then
    log "WARN config line ignored (missing '|'): $line"
    skipped_count=$((skipped_count + 1))
    continue
  fi

  src_raw="${line%%|*}"
  name_raw="${line#*|}"
  src="$(trim "$src_raw")"
  name="$(trim "$name_raw")"

  if [[ -n "$PROJECT_FILTER" && "$name" != "$PROJECT_FILTER" ]]; then
    continue
  fi
  matched_project=1

  if [[ ! -d "$src" ]]; then
    log "WARN source missing, skip: $name ($src)"
    skipped_count=$((skipped_count + 1))
    continue
  fi

  if [[ ! -f "$src/.gitignore" && "$ALLOW_MISSING_GITIGNORE" -eq 0 ]]; then
    log "WARN no .gitignore, skip: $name ($src)"
    skipped_count=$((skipped_count + 1))
    continue
  fi

  dest="$DEST_BASE/$name"
  mkdir -p "$dest"

  rsync_args=(
    -a
    --exclude=.git/
    --exclude=.gitmodules
    --filter
    ":- .gitignore"
  )

  if [[ "$USE_DELETE" -eq 1 ]]; then
    rsync_args+=(--delete)
  fi
  if [[ "$DRY_RUN" -eq 1 ]]; then
    rsync_args+=(-n --itemize-changes)
  fi

  log "Syncing: $name"
  if rsync "${rsync_args[@]}" "$src/" "$dest/"; then
    size="$(du -sh "$dest" | awk '{print $1}')"
    log "Done: $name (size=$size)"
    synced_count=$((synced_count + 1))
  else
    log "ERROR sync failed: $name"
    failed_count=$((failed_count + 1))
  fi
done < "$CONFIG_FILE"

if [[ -n "$PROJECT_FILTER" && "$matched_project" -eq 0 ]]; then
  log "ERROR project not found in config: $PROJECT_FILTER"
  exit 3
fi

log "Sync finished. synced=$synced_count skipped=$skipped_count failed=$failed_count"
if [[ "$failed_count" -gt 0 ]]; then
  exit 1
fi
