#!/bin/bash
#SBATCH --job-name=ascat_cmems_full
#SBATCH --output=logs/ascat_full_%j.out
#SBATCH --error=logs/ascat_full_%j.err
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=200G
#SBATCH --time=36:00:00
#SBATCH --qos=normal
#SBATCH --mail-type=END,FAIL

set -euo pipefail

# Single-entry Sherlock ASCAT pipeline.
# This Sherlock path expects a prebuilt ASCAT full manifest synced from local.
# Each submission writes into its own run-tagged subdirectory under
#   data/interim/ascat/runs/<RUN_TAG>/
# unless explicitly overridden.
# Submit directly with:
#   sbatch --export=ALL,PROJECT_DIR=/scratch/users/$USER/Cyclone_next,ENV_PREFIX=/scratch/users/$USER/conda/envs/cyclone,ASCAT_CREDENTIALS_FILE=$HOME/.config/copernicusmarine/credentials,FULL_MANIFEST=/scratch/users/$USER/Cyclone_next/data/interim/ascat/ascat_request_manifest_full.csv slurm/run_ascat_sherlock_array.slurm

DEFAULT_PROJECT_DIR="/scratch/users/${USER}/Cyclone_next"
DEFAULT_ENV_PREFIX="/scratch/users/${USER}/conda/envs/cyclone"
PROJECT_DIR="${PROJECT_DIR:-$DEFAULT_PROJECT_DIR}"
DEFAULT_RUN_TAG="$(date -u +"%Y%m%dT%H%M%SZ")_job${SLURM_JOB_ID:-manual}"
RUN_TAG="${RUN_TAG:-$DEFAULT_RUN_TAG}"
RUN_BASE_DIR="${RUN_BASE_DIR:-$PROJECT_DIR/data/interim/ascat/runs}"
RUN_ROOT="${RUN_ROOT:-$RUN_BASE_DIR/$RUN_TAG}"

FULL_MANIFEST="${FULL_MANIFEST:-$PROJECT_DIR/data/interim/ascat/ascat_request_manifest_full.csv}"
YEAR_MANIFEST_DIR="${YEAR_MANIFEST_DIR:-$RUN_ROOT/manifests}"
YEAR_OUT_DIR="${YEAR_OUT_DIR:-$RUN_ROOT/features}"
YEAR_SUMMARY_DIR="${YEAR_SUMMARY_DIR:-$RUN_ROOT/summaries}"
FINAL_OUT="${FINAL_OUT:-$RUN_ROOT/ascat_observation_features_full.csv}"
FINAL_SUMMARY="${FINAL_SUMMARY:-$RUN_ROOT/ascat_observation_features_full_summary.json}"
CANONICAL_OUT="${CANONICAL_OUT:-$RUN_ROOT/ascat_observation_features.csv}"
CANONICAL_SUMMARY="${CANONICAL_SUMMARY:-$RUN_ROOT/ascat_observation_features_summary.json}"
SHARED_CANONICAL_OUT="${SHARED_CANONICAL_OUT:-$PROJECT_DIR/data/interim/ascat/ascat_observation_features.csv}"
SHARED_CANONICAL_SUMMARY="${SHARED_CANONICAL_SUMMARY:-$PROJECT_DIR/data/interim/ascat/ascat_observation_features_summary.json}"
PROMOTE_TO_CANONICAL="${PROMOTE_TO_CANONICAL:-0}"

ENV_PREFIX="${ENV_PREFIX:-$DEFAULT_ENV_PREFIX}"
VENV_DIR="${VENV_DIR:-$PROJECT_DIR/.venv}"
PYTHON_BIN="${PYTHON_BIN:-}"
ASCAT_CREDENTIALS_FILE="${ASCAT_CREDENTIALS_FILE:-}"
PROJECT_LOCAL_AUTH_ENV="${PROJECT_LOCAL_AUTH_ENV:-$PROJECT_DIR/.secrets/copernicusmarine.env}"

YEAR_START="${YEAR_START:-2016}"
YEAR_END="${YEAR_END:-2025}"
ASCAT_YEAR_LIST="${ASCAT_YEAR_LIST:-}"
FORCE_RERUN="${FORCE_RERUN:-0}"
FORCE_REBUILD_MANIFEST="${FORCE_REBUILD_MANIFEST:-0}"

ASCAT_DRY_RUN="${ASCAT_DRY_RUN:-0}"
ASCAT_ONLY_WITH_STORM_ID="${ASCAT_ONLY_WITH_STORM_ID:-0}"
ASCAT_KEEP_TEMP_FILES="${ASCAT_KEEP_TEMP_FILES:-0}"
ASCAT_MAX_ROWS="${ASCAT_MAX_ROWS:-0}"
ASCAT_MAX_RETRIES="${ASCAT_MAX_RETRIES:-3}"
ASCAT_REQUEST_PAUSE_SEC="${ASCAT_REQUEST_PAUSE_SEC:-0.20}"
ASCAT_SUBSET_TIMEOUT_SEC="${ASCAT_SUBSET_TIMEOUT_SEC:-300}"
ASCAT_WINDOW_BEFORE_MIN="${ASCAT_WINDOW_BEFORE_MIN:-180}"
ASCAT_WINDOW_AFTER_MIN="${ASCAT_WINDOW_AFTER_MIN:-180}"
ASCAT_INNER_RADIUS_KM="${ASCAT_INNER_RADIUS_KM:-200.0}"
ASCAT_OUTER_RADIUS_KM="${ASCAT_OUTER_RADIUS_KM:-500.0}"
ASCAT_BBOX_MARGIN_DEG="${ASCAT_BBOX_MARGIN_DEG:-0.5}"
ASCAT_ASSUMED_CELL_KM="${ASCAT_ASSUMED_CELL_KM:-25.0}"
ASCAT_DATASET_IDS="${ASCAT_DATASET_IDS:-}"
ASCAT_VARIABLES="${ASCAT_VARIABLES:-}"
ASCAT_TMP_ROOT="${ASCAT_TMP_ROOT:-${SLURM_TMPDIR:-/tmp}}"

if [[ ! "$YEAR_START" =~ ^[0-9]{4}$ || ! "$YEAR_END" =~ ^[0-9]{4}$ ]]; then
  echo "❌ YEAR_START/YEAR_END must be 4-digit years" >&2
  exit 2
fi
if (( YEAR_END < YEAR_START )); then
  echo "❌ YEAR_END must be >= YEAR_START" >&2
  exit 2
fi
if [[ -z "$RUN_TAG" || "$RUN_TAG" == */* || "$RUN_TAG" =~ [[:space:]] ]]; then
  echo "❌ RUN_TAG must be non-empty and cannot contain '/' or whitespace: $RUN_TAG" >&2
  exit 2
fi

if ! command -v module >/dev/null 2>&1; then
  if [[ -f /etc/profile.d/modules.sh ]]; then
    source /etc/profile.d/modules.sh
  fi
fi

if command -v module >/dev/null 2>&1; then
  for mod in mamba micromamba miniconda3 anaconda3 miniforge3 mambaforge; do
    if module --ignore_cache load "$mod" >/dev/null 2>&1 || module load "$mod" >/dev/null 2>&1; then
      echo "Loaded module: $mod"
      break
    fi
  done
fi

resolve_python_runner() {
  if [[ -n "$PYTHON_BIN" ]]; then
    PY_RUN=("$PYTHON_BIN")
    return
  fi
  if [[ ! -d "$ENV_PREFIX" ]]; then
    echo "❌ ENV_PREFIX not found: $ENV_PREFIX" >&2
    exit 2
  fi
  if [[ -n "$ENV_PREFIX" ]]; then
    if [[ -x "$ENV_PREFIX/bin/python3" ]]; then
      PY_RUN=("$ENV_PREFIX/bin/python3")
      return
    fi
    if [[ -x "$ENV_PREFIX/bin/python" ]]; then
      PY_RUN=("$ENV_PREFIX/bin/python")
      return
    fi
    if command -v conda >/dev/null 2>&1; then
      PY_RUN=(conda run --no-capture-output -p "$ENV_PREFIX" python3)
      return
    fi
    if command -v mamba >/dev/null 2>&1; then
      PY_RUN=(mamba run --no-capture-output -p "$ENV_PREFIX" python3)
      return
    fi
    if command -v micromamba >/dev/null 2>&1; then
      PY_RUN=(micromamba run -p "$ENV_PREFIX" python3)
      return
    fi
    echo "❌ Need conda/mamba/micromamba or a direct python binary under ENV_PREFIX: $ENV_PREFIX" >&2
    exit 2
  fi
  if [[ -n "$VENV_DIR" && -x "$VENV_DIR/bin/python" ]]; then
    PY_RUN=("$VENV_DIR/bin/python")
    return
  fi
  if [[ -x "$PROJECT_DIR/.venv/bin/python" ]]; then
    PY_RUN=("$PROJECT_DIR/.venv/bin/python")
    return
  fi
  echo "❌ no usable python interpreter found" >&2
  exit 2
}

validate_python_runner() {
  local version_output
  version_output="$("${PY_RUN[@]}" - <<'PY'
import sys
print("sys_executable=" + sys.executable)
print("sys_version=" + sys.version.replace("\n", " "))
if sys.version_info < (3, 8):
    raise SystemExit(86)
PY
)" || {
    local exit_code=$?
    if [[ "$exit_code" -eq 86 ]]; then
      echo "❌ resolved python is too old; require >= 3.8" >&2
    else
      echo "❌ failed to execute resolved python: ${PY_RUN[*]}" >&2
    fi
    exit "$exit_code"
  }
  echo "$version_output"
}

validate_ascat_runtime_deps() {
  local dep_output
  dep_output="$("${PY_RUN[@]}" - <<'PY'
import importlib

required = ["numpy", "xarray", "copernicusmarine"]
missing = []
for name in required:
    try:
        importlib.import_module(name)
    except Exception as exc:
        missing.append(f"{name}:{type(exc).__name__}")

if missing:
    print("missing_python_deps=" + ",".join(missing))
    raise SystemExit(87)

print("python_deps_ok=" + ",".join(required))
PY
)" || {
    local exit_code=$?
    if [[ "$exit_code" -eq 87 ]]; then
      echo "❌ ASCAT runtime dependencies are incomplete in ENV_PREFIX=$ENV_PREFIX" >&2
      echo "   Install missing packages into the cyclone environment before resubmitting." >&2
    else
      echo "❌ failed to validate ASCAT runtime dependencies" >&2
    fi
    exit "$exit_code"
  }
  echo "$dep_output"
}

load_project_local_auth_env() {
  local env_file="$1"
  local line=""
  local key=""
  local value=""

  [[ -f "$env_file" ]] || return 0

  while IFS= read -r line || [[ -n "$line" ]]; do
    line="${line%%#*}"
    line="${line#"${line%%[![:space:]]*}"}"
    line="${line%"${line##*[![:space:]]}"}"
    [[ -z "$line" ]] && continue

    if [[ "$line" == export[[:space:]]* ]]; then
      line="${line#export }"
      line="${line#"${line%%[![:space:]]*}"}"
    fi
    [[ "$line" == *=* ]] || continue

    key="${line%%=*}"
    value="${line#*=}"
    key="${key#"${key%%[![:space:]]*}"}"
    key="${key%"${key##*[![:space:]]}"}"
    value="${value#"${value%%[![:space:]]*}"}"
    value="${value%"${value##*[![:space:]]}"}"

    if [[ "$value" == \"*\" && "$value" == *\" ]]; then
      value="${value:1:${#value}-2}"
    elif [[ "$value" == \'*\' && "$value" == *\' ]]; then
      value="${value:1:${#value}-2}"
    fi

    case "$key" in
      ASCAT_USERNAME|ASCAT_PASSWORD|ASCAT_CREDENTIALS_FILE|ASCAT_SERVICE|CMEMS_USERNAME|CMEMS_PASSWORD|CMEMS_CREDENTIALS_FILE|CMEMS_SERVICE|COPERNICUSMARINE_USERNAME|COPERNICUSMARINE_PASSWORD|COPERNICUSMARINE_CREDENTIALS_FILE|COPERNICUSMARINE_SERVICE|COPERNICUSMARINE_SERVICE_USERNAME|COPERNICUSMARINE_SERVICE_PASSWORD)
        export "$key=$value"
        ;;
    esac
  done < "$env_file"
}

resolve_copernicus_auth() {
  local candidate=""
  local auth_user=""
  local auth_pass=""

  load_project_local_auth_env "$PROJECT_LOCAL_AUTH_ENV"

  if [[ -z "$ASCAT_CREDENTIALS_FILE" ]]; then
    for candidate in \
      "$HOME/.config/copernicusmarine/credentials" \
      "$HOME/.copernicusmarine/.copernicusmarine-credentials"
    do
      if [[ -f "$candidate" ]]; then
        ASCAT_CREDENTIALS_FILE="$candidate"
        break
      fi
    done
  fi

  if [[ -n "$ASCAT_CREDENTIALS_FILE" ]]; then
    if [[ ! -f "$ASCAT_CREDENTIALS_FILE" ]]; then
      echo "❌ ASCAT_CREDENTIALS_FILE not found: $ASCAT_CREDENTIALS_FILE" >&2
      exit 2
    fi
    export ASCAT_CREDENTIALS_FILE
    if [[ "$ASCAT_CREDENTIALS_FILE" == "$PROJECT_DIR/"* ]]; then
      AUTH_SOURCE="project_credentials_file:$ASCAT_CREDENTIALS_FILE"
    else
      AUTH_SOURCE="credentials_file:$ASCAT_CREDENTIALS_FILE"
    fi
    return
  fi

  auth_user="${ASCAT_USERNAME:-${CMEMS_USERNAME:-${COPERNICUSMARINE_USERNAME:-${COPERNICUSMARINE_SERVICE_USERNAME:-}}}}"
  auth_pass="${ASCAT_PASSWORD:-${CMEMS_PASSWORD:-${COPERNICUSMARINE_PASSWORD:-${COPERNICUSMARINE_SERVICE_PASSWORD:-}}}}"
  if [[ -n "$auth_user" && -n "$auth_pass" ]]; then
    if [[ -f "$PROJECT_LOCAL_AUTH_ENV" ]]; then
      AUTH_SOURCE="project_env_file:$PROJECT_LOCAL_AUTH_ENV"
    else
      AUTH_SOURCE="env_username_password"
    fi
    return
  fi

  echo "❌ no Copernicus Marine credentials configured" >&2
  echo "   Provide $PROJECT_LOCAL_AUTH_ENV, ASCAT_CREDENTIALS_FILE, or ASCAT_USERNAME / ASCAT_PASSWORD." >&2
  exit 2
}

parse_year_list() {
  if [[ -n "$ASCAT_YEAR_LIST" ]]; then
    YEAR_LIST=()
    for token in ${ASCAT_YEAR_LIST//,/ }; do
      token="$(echo "$token" | xargs)"
      [[ -z "$token" ]] && continue
      if [[ ! "$token" =~ ^[0-9]{4}$ ]]; then
        echo "❌ invalid year in ASCAT_YEAR_LIST: $token" >&2
        exit 2
      fi
      YEAR_LIST+=("$token")
    done
    if [[ "${#YEAR_LIST[@]}" -eq 0 ]]; then
      echo "❌ ASCAT_YEAR_LIST resolved to empty" >&2
      exit 2
    fi
    return
  fi

  YEAR_LIST=()
  for ((year=YEAR_START; year<=YEAR_END; year++)); do
    YEAR_LIST+=("$year")
  done
}

manifest_has_rows() {
  local manifest_path="$1"
  "${PY_RUN[@]}" - <<'PY' "$manifest_path"
import sys
from pathlib import Path
fp = Path(sys.argv[1])
with fp.open("r", encoding="utf-8") as f:
    row_count = sum(1 for _ in f)
raise SystemExit(0 if row_count > 1 else 1)
PY
}

year_outputs_indicate_failed_runtime() {
  local out_csv="$1"
  local out_json="$2"
  "${PY_RUN[@]}" - <<'PY' "$out_csv" "$out_json"
import csv
import json
import sys
from pathlib import Path

csv_path = Path(sys.argv[1])
json_path = Path(sys.argv[2])
if not csv_path.exists() or not json_path.exists():
    raise SystemExit(1)

try:
    summary = json.loads(json_path.read_text(encoding="utf-8"))
except Exception:
    raise SystemExit(0)

rows_written = int(summary.get("rows_written") or 0)
available_rows = int(summary.get("available_rows") or 0)
missing_rows = int(summary.get("missing_rows") or 0)

if rows_written <= 0:
    raise SystemExit(0)
if available_rows != 0 or missing_rows != rows_written:
    raise SystemExit(1)

with csv_path.open("r", encoding="utf-8", newline="") as f:
    reader = csv.DictReader(f)
    saw_row = False
    for row in reader:
        saw_row = True
        status = (row.get("ascat_status") or "").strip()
        reason = (row.get("missing_reason") or "").strip()
        if status != "missing_real_data" or not reason.startswith("runtime_dependency_error:"):
            raise SystemExit(1)

if not saw_row:
    raise SystemExit(0)

raise SystemExit(0)
PY
}

build_extractor_cmd() {
  local manifest="$1"
  local out_csv="$2"
  local out_json="$3"
  local tmp_dir="$4"

  CMD=(
    "${PY_RUN[@]}"
    scripts/extract_ascat_features_remote.py
    --manifest-csv "$manifest"
    --out-csv "$out_csv"
    --summary-json "$out_json"
    --window-before-min "$ASCAT_WINDOW_BEFORE_MIN"
    --window-after-min "$ASCAT_WINDOW_AFTER_MIN"
    --inner-radius-km "$ASCAT_INNER_RADIUS_KM"
    --outer-radius-km "$ASCAT_OUTER_RADIUS_KM"
    --bbox-margin-deg "$ASCAT_BBOX_MARGIN_DEG"
    --assumed-cell-km "$ASCAT_ASSUMED_CELL_KM"
    --tmp-dir "$tmp_dir"
    --max-retries "$ASCAT_MAX_RETRIES"
    --subset-timeout-sec "$ASCAT_SUBSET_TIMEOUT_SEC"
  )

  if [[ "$ASCAT_ONLY_WITH_STORM_ID" == "1" ]]; then
    CMD+=(--only-with-storm-id)
  fi
  if [[ "$ASCAT_KEEP_TEMP_FILES" == "1" ]]; then
    CMD+=(--keep-temp-files)
  fi
  if [[ "$ASCAT_MAX_ROWS" =~ ^[0-9]+$ && "$ASCAT_MAX_ROWS" -gt 0 ]]; then
    CMD+=(--max-rows "$ASCAT_MAX_ROWS")
  fi
  if [[ -n "$ASCAT_DATASET_IDS" ]]; then
    IFS=',' read -r -a DATASET_ARRAY <<< "$ASCAT_DATASET_IDS"
    for ds in "${DATASET_ARRAY[@]}"; do
      ds="$(echo "$ds" | xargs)"
      [[ -z "$ds" ]] && continue
      CMD+=(--dataset-id "$ds")
    done
  fi
  if [[ -n "$ASCAT_VARIABLES" ]]; then
    IFS=',' read -r -a VARIABLE_ARRAY <<< "$ASCAT_VARIABLES"
    for var in "${VARIABLE_ARRAY[@]}"; do
      var="$(echo "$var" | xargs)"
      [[ -z "$var" ]] && continue
      CMD+=(--variable "$var")
    done
  fi
  if [[ "$ASCAT_DRY_RUN" == "1" ]]; then
    CMD+=(--dry-run)
  fi
}

mkdir -p "$PROJECT_DIR/logs" "$RUN_ROOT" "$YEAR_MANIFEST_DIR" "$YEAR_OUT_DIR" "$YEAR_SUMMARY_DIR"
resolve_python_runner
parse_year_list
AUTH_SOURCE=""
resolve_copernicus_auth

STATUS_FILE="${STATUS_FILE:-$RUN_ROOT/job_${SLURM_JOB_ID:-manual}.status}"
mkdir -p "$(dirname "$STATUS_FILE")"
COMPLETED_YEARS=()
SKIPPED_YEARS=()
RUN_YEARS=()
ACTIVE_YEAR=""

write_status() {
  cat > "$STATUS_FILE" <<EOF
state=$1
updated_utc=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
job_id=${SLURM_JOB_ID:-}
project_dir=$PROJECT_DIR
run_tag=$RUN_TAG
run_root=$RUN_ROOT
year_start=$YEAR_START
year_end=$YEAR_END
active_year=${ACTIVE_YEAR:-}
completed_years=$(IFS=,; echo "${COMPLETED_YEARS[*]:-}")
skipped_years=$(IFS=,; echo "${SKIPPED_YEARS[*]:-}")
run_years=$(IFS=,; echo "${RUN_YEARS[*]:-}")
full_manifest=$FULL_MANIFEST
final_out=$FINAL_OUT
final_summary=$FINAL_SUMMARY
canonical_out=$CANONICAL_OUT
canonical_summary=$CANONICAL_SUMMARY
promote_to_canonical=$PROMOTE_TO_CANONICAL
shared_canonical_out=$SHARED_CANONICAL_OUT
shared_canonical_summary=$SHARED_CANONICAL_SUMMARY
EOF
}

cleanup() {
  local exit_code=$?
  if [[ "$exit_code" -eq 0 ]]; then
    write_status "finished"
  else
    write_status "failed($exit_code)"
  fi
}
trap cleanup EXIT
write_status "running"

ulimit -n 4096 || true
export PYTHONUNBUFFERED=1
export ASCAT_REQUEST_PAUSE_SEC

echo "=== ASCAT Sherlock Full Run ==="
echo "HOST=$(hostname)"
echo "PROJECT_DIR=$PROJECT_DIR"
echo "RUN_TAG=$RUN_TAG"
echo "RUN_ROOT=$RUN_ROOT"
echo "FULL_MANIFEST=$FULL_MANIFEST"
echo "PROJECT_LOCAL_AUTH_ENV=$PROJECT_LOCAL_AUTH_ENV"
echo "ENV_PREFIX=$ENV_PREFIX"
echo "PYTHON=${PY_RUN[*]}"
validate_python_runner
validate_ascat_runtime_deps
echo "COPERNICUS_AUTH_SOURCE=$AUTH_SOURCE"
echo "YEAR_LIST=$(IFS=,; echo "${YEAR_LIST[*]}")"
echo "DRY_RUN=$ASCAT_DRY_RUN"
echo "FORCE_RERUN=$FORCE_RERUN"
echo "FORCE_REBUILD_MANIFEST=$FORCE_REBUILD_MANIFEST"
echo "ASCAT_ONLY_WITH_STORM_ID=$ASCAT_ONLY_WITH_STORM_ID"
echo "ASCAT_REQUEST_PAUSE_SEC=$ASCAT_REQUEST_PAUSE_SEC"
echo "ASCAT_SUBSET_TIMEOUT_SEC=$ASCAT_SUBSET_TIMEOUT_SEC"
echo "PROMOTE_TO_CANONICAL=$PROMOTE_TO_CANONICAL"

cd "$PROJECT_DIR"

if [[ "$FORCE_REBUILD_MANIFEST" == "1" ]]; then
  echo "❌ Remote manifest rebuild is disabled on Sherlock." >&2
  echo "   Commit and push data/interim/ascat/ascat_request_manifest_full.csv first," >&2
  echo "   then clone or refresh the Sherlock checkout and resubmit with:" >&2
  echo "   FULL_MANIFEST=$FULL_MANIFEST" >&2
  exit 4
fi

if [[ ! -f "$FULL_MANIFEST" ]]; then
  echo "❌ FULL_MANIFEST not found: $FULL_MANIFEST" >&2
  echo "   Clone only fetches git-tracked files." >&2
  echo "   Ensure data/interim/ascat/ascat_request_manifest_full.csv is committed and pushed," >&2
  echo "   then refresh the Sherlock checkout before resubmitting." >&2
  exit 4
fi

if ! manifest_has_rows "$FULL_MANIFEST"; then
  echo "❌ FULL_MANIFEST exists but has no data rows: $FULL_MANIFEST" >&2
  echo "   Rebuild the manifest locally, resync it to Sherlock, then resubmit." >&2
  exit 4
fi

echo "[INFO] splitting ASCAT full manifest by year..."
"${PY_RUN[@]}" scripts/prepare_ascat_yearly_manifests.py \
  --full-manifest-csv "$FULL_MANIFEST" \
  --year-manifest-dir "$YEAR_MANIFEST_DIR" \
  --year-start "$YEAR_START" \
  --year-end "$YEAR_END"

for year in "${YEAR_LIST[@]}"; do
  ACTIVE_YEAR="$year"
  MANIFEST="$YEAR_MANIFEST_DIR/ascat_request_manifest_${year}.csv"
  OUT_CSV="$YEAR_OUT_DIR/ascat_observation_features_${year}.csv"
  OUT_JSON="$YEAR_SUMMARY_DIR/ascat_observation_features_${year}_summary.json"
  TMP_DIR="$ASCAT_TMP_ROOT/ascat_${SLURM_JOB_ID:-manual}_${year}"

  if [[ ! -f "$MANIFEST" ]]; then
    echo "[WARN] year=$year manifest missing, skip"
    SKIPPED_YEARS+=("$year")
    write_status "running"
    continue
  fi

  if ! manifest_has_rows "$MANIFEST"; then
    echo "[WARN] year=$year manifest empty, skip"
    SKIPPED_YEARS+=("$year")
    write_status "running"
    continue
  fi

  RUN_YEARS+=("$year")

  if [[ "$FORCE_RERUN" != "1" && -f "$OUT_CSV" && -f "$OUT_JSON" ]]; then
    if year_outputs_indicate_failed_runtime "$OUT_CSV" "$OUT_JSON"; then
      echo "[INFO] year=$year previous output is a failed runtime placeholder; rerun"
      rm -f "$OUT_CSV" "$OUT_JSON"
    else
      echo "[INFO] year=$year already completed, skip"
      SKIPPED_YEARS+=("$year")
      write_status "running"
      continue
    fi
  fi

  rm -rf "$TMP_DIR"
  mkdir -p "$TMP_DIR"

  echo "[INFO] year=$year running..."
  build_extractor_cmd "$MANIFEST" "$OUT_CSV" "$OUT_JSON" "$TMP_DIR"
  "${CMD[@]}"

  COMPLETED_YEARS+=("$year")
  write_status "running"

  if [[ "$ASCAT_DRY_RUN" != "1" ]]; then
    rm -rf "$TMP_DIR" || true
  fi
  echo "[INFO] year=$year done."
done

ACTIVE_YEAR=""

if [[ "$ASCAT_DRY_RUN" == "1" ]]; then
  echo "[INFO] dry-run mode enabled; skip merge step"
  exit 0
fi

if [[ "${#RUN_YEARS[@]}" -eq 0 ]]; then
  echo "❌ no non-empty yearly ASCAT manifests found to run or merge" >&2
  exit 3
fi

REQUIRE_YEARS_CSV="$(IFS=,; echo "${RUN_YEARS[*]}")"

echo "[INFO] merging yearly ASCAT outputs..."
"${PY_RUN[@]}" scripts/merge_ascat_yearly_outputs.py \
  --year-csv-dir "$YEAR_OUT_DIR" \
  --year-summary-dir "$YEAR_SUMMARY_DIR" \
  --out-csv "$FINAL_OUT" \
  --summary-json "$FINAL_SUMMARY" \
  --canonical-out "$CANONICAL_OUT" \
  --canonical-summary "$CANONICAL_SUMMARY" \
  --require-years-csv "$REQUIRE_YEARS_CSV" \
  --generated-from "sherlock_single_slurm_job"

if [[ "$PROMOTE_TO_CANONICAL" == "1" ]]; then
  mkdir -p "$(dirname "$SHARED_CANONICAL_OUT")" "$(dirname "$SHARED_CANONICAL_SUMMARY")"
  cp "$FINAL_OUT" "$SHARED_CANONICAL_OUT"
  cp "$FINAL_SUMMARY" "$SHARED_CANONICAL_SUMMARY"
  echo "[INFO] promoted run outputs to shared canonical:"
  echo "  $SHARED_CANONICAL_OUT"
  echo "  $SHARED_CANONICAL_SUMMARY"
fi

echo "[INFO] ASCAT Sherlock full run completed:"
echo "  $FINAL_OUT"
echo "  $FINAL_SUMMARY"
echo "  $CANONICAL_OUT"
echo "  $CANONICAL_SUMMARY"
