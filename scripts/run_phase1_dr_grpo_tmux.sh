#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
CONFIG_PATH="${1:-$ROOT_DIR/configs/training/dr_grpo_gemma_4_e4b_unsloth_forecast_v2_phase1.yaml}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="${RUN_ROOT:-$ROOT_DIR/runs/dr_grpo_phase1_${TIMESTAMP}}"
LOG_DIR="$RUN_ROOT/logs"
LOG_FILE="$LOG_DIR/train.log"
CMD_FILE="$RUN_ROOT/launch_command.sh"
PID_FILE="$RUN_ROOT/train.pid"
SESSION_FILE="$RUN_ROOT/tmux_session.txt"
SESSION_NAME="${TMUX_SESSION_NAME:-dr_grpo_phase1_${TIMESTAMP}}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Missing Python executable: $PYTHON_BIN" >&2
  exit 1
fi

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Missing config: $CONFIG_PATH" >&2
  exit 1
fi

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux is required for this launcher." >&2
  exit 1
fi

mkdir -p "$LOG_DIR"

cat > "$CMD_FILE" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "$ROOT_DIR"
env PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false "$PYTHON_BIN" "$ROOT_DIR/scripts/train_grpo.py" --config "$CONFIG_PATH" --verbose
EOF
chmod +x "$CMD_FILE"

printf '%s\n' "$SESSION_NAME" > "$SESSION_FILE"
printf -v TMUX_LAUNCH_CMD 'cd %q && stdbuf -oL -eL %q 2>&1 | tee -a %q' "$ROOT_DIR" "$CMD_FILE" "$LOG_FILE"
tmux new-session -d -s "$SESSION_NAME" "$TMUX_LAUNCH_CMD"
PID="$(tmux display-message -p -t "$SESSION_NAME" '#{pane_pid}')"
printf '%s\n' "$PID" > "$PID_FILE"
sleep 2

echo "SESSION_NAME=$SESSION_NAME"
echo "RUN_ROOT=$RUN_ROOT"
echo "LOG_FILE=$LOG_FILE"
echo "PID=$PID"
echo "PID_FILE=$PID_FILE"
echo "COMMAND_FILE=$CMD_FILE"
echo "STATUS=running"
