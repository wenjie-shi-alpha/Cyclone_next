#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEFAULT_GRPO_CONFIG="$ROOT_DIR/configs/training/grpo_gemma_4_e4b_unsloth_forecast_v2_round1.yaml"
PYTHON_BIN="$ROOT_DIR/.venv/bin/python"

GRPO_CONFIG="${1:-$DEFAULT_GRPO_CONFIG}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="${RUN_ROOT:-$ROOT_DIR/runs/grpo_only_${TIMESTAMP}}"
LOG_DIR="$RUN_ROOT/logs"
LOG_FILE="$LOG_DIR/train.log"
PID_FILE="$RUN_ROOT/train.pid"
CMD_FILE="$RUN_ROOT/launch_command.sh"
SESSION_FILE="$RUN_ROOT/tmux_session.txt"
SESSION_NAME=""
BACKEND="nohup"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Missing Python executable: $PYTHON_BIN" >&2
  exit 1
fi

if [[ ! -f "$GRPO_CONFIG" ]]; then
  echo "Missing GRPO config: $GRPO_CONFIG" >&2
  exit 1
fi

mkdir -p "$LOG_DIR"

cmd=(
  env
  PYTHONUNBUFFERED=1
  TOKENIZERS_PARALLELISM=false
  "$PYTHON_BIN"
  "$ROOT_DIR/scripts/train_grpo.py"
  --config
  "$GRPO_CONFIG"
  --verbose
)

printf '#!/usr/bin/env bash\n' > "$CMD_FILE"
printf 'set -euo pipefail\n' >> "$CMD_FILE"
printf '%q ' "${cmd[@]}" >> "$CMD_FILE"
printf '\n' >> "$CMD_FILE"
chmod +x "$CMD_FILE"

if [[ "${DISABLE_TMUX:-0}" != "1" ]] && command -v tmux >/dev/null 2>&1; then
  BACKEND="tmux"
  SESSION_NAME="${TMUX_SESSION_NAME:-cyclone_grpo_${TIMESTAMP}}"
  printf '%s\n' "$SESSION_NAME" > "$SESSION_FILE"
  printf -v TMUX_LAUNCH_CMD 'cd %q && exec %q > %q 2>&1' "$ROOT_DIR" "$CMD_FILE" "$LOG_FILE"
  tmux new-session -d -s "$SESSION_NAME" "$TMUX_LAUNCH_CMD"
  PID="$(tmux display-message -p -t "$SESSION_NAME" '#{pane_pid}')"
  echo "$PID" > "$PID_FILE"
  sleep 2
  if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    STATUS="running"
  else
    STATUS="exited"
  fi
else
  nohup "$CMD_FILE" > "$LOG_FILE" 2>&1 &
  PID=$!
  echo "$PID" > "$PID_FILE"
  sleep 2
  if kill -0 "$PID" 2>/dev/null; then
    STATUS="running"
  else
    STATUS="exited"
  fi
fi

echo "BACKEND=$BACKEND"
echo "RUN_ROOT=$RUN_ROOT"
echo "PID=$PID"
echo "LOG_FILE=$LOG_FILE"
echo "PID_FILE=$PID_FILE"
echo "COMMAND_FILE=$CMD_FILE"
if [[ -n "$SESSION_NAME" ]]; then
  echo "SESSION_NAME=$SESSION_NAME"
  echo "SESSION_FILE=$SESSION_FILE"
fi
echo "STATUS=$STATUS"
