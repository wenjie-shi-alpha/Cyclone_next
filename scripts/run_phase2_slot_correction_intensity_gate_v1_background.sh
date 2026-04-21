#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="${RUN_ROOT:-$ROOT_DIR/runs/phase2_slot_correction_intensity_gate_v1_${TIMESTAMP}}"
LOG_DIR="$RUN_ROOT/logs"
LOG_FILE="$LOG_DIR/phase2.log"
PID_FILE="$RUN_ROOT/train.pid"
CMD_FILE="$RUN_ROOT/launch_command.sh"
SESSION_FILE="$RUN_ROOT/tmux_session.txt"
SESSION_NAME=""
BACKEND="nohup"

mkdir -p "$LOG_DIR"

cmd=(
  env
  PYTHONUNBUFFERED=1
  TOKENIZERS_PARALLELISM=false
  RUN_ROOT="$RUN_ROOT"
  FORECAST_CONFIG="${FORECAST_CONFIG:-$ROOT_DIR/configs/training/sft_gemma_4_e4b_unsloth_forecast_phase1_baseline_v2.yaml}"
  REWARD_CONFIG="${REWARD_CONFIG:-$ROOT_DIR/configs/training/grpo_gemma_4_e4b_unsloth_forecast_phase1_baseline_v2_reward.yaml}"
  FORECAST_ADAPTER="${FORECAST_ADAPTER:-$ROOT_DIR/runs/phase1_baseline_v2_formal_20260415_013403/sft/final_adapter}"
  DIAGNOSTIC_CONFIG="${DIAGNOSTIC_CONFIG:-$ROOT_DIR/configs/training/sft_gemma_4_e4b_unsloth_diagnostic_v2_slot_correction_stage_v1.yaml}"
  DIAGNOSTIC_ADAPTER="${DIAGNOSTIC_ADAPTER:-$ROOT_DIR/runs/phase2_diagnostic_slot_correction_v1_20260418_184506/sft/final_adapter}"
  ORACLE_REPORT="${ORACLE_REPORT:-$ROOT_DIR/runs/phase2_diagnostic_slot_correction_oracle_v0_20260418_154645/evals/forecast_integration_compare_sample200_seed3407.json}"
  BASELINE_INTENSITY_OFFSET_SCALE="${BASELINE_INTENSITY_OFFSET_SCALE:-1.20}"
  DATASET_ROOT="${DATASET_ROOT:-$ROOT_DIR/data/training_rebuilt_v2_20260414_guidancefix}"
  CANONICAL_TRAIN="${CANONICAL_TRAIN:-$ROOT_DIR/data/training_rebuilt_v2_20260414_guidancefix/canonical_v2/train.jsonl}"
  CANONICAL_TEST="${CANONICAL_TEST:-$ROOT_DIR/data/training_rebuilt_v2_20260414_guidancefix/canonical_v2/test.jsonl}"
  DIAGNOSTIC_TRAIN_DATASET="${DIAGNOSTIC_TRAIN_DATASET:-$ROOT_DIR/data/training_rebuilt_v2_20260414_guidancefix/views/diagnostic_slot_correction_only/train.jsonl}"
  DIAGNOSTIC_TEST_DATASET="${DIAGNOSTIC_TEST_DATASET:-$ROOT_DIR/data/training_rebuilt_v2_20260414_guidancefix/views/diagnostic_slot_correction_only/test.jsonl}"
  FORECAST_SFT_TEST_DATASET="${FORECAST_SFT_TEST_DATASET:-$ROOT_DIR/data/training_rebuilt_v2_20260414_guidancefix/views/forecast_only/test.jsonl}"
  FORECAST_RL_TEST_DATASET="${FORECAST_RL_TEST_DATASET:-$ROOT_DIR/data/training_rebuilt_v2_20260414_guidancefix/views/forecast_only/rl_test.jsonl}"
  SAMPLE_COUNT="${SAMPLE_COUNT:-200}"
  SAMPLE_SEED="${SAMPLE_SEED:-3407}"
  DIAGNOSTIC_BATCH_SIZE="${DIAGNOSTIC_BATCH_SIZE:-4}"
  FORECAST_BATCH_SIZE="${FORECAST_BATCH_SIZE:-4}"
  MAX_PROMPT_TOKENS="${MAX_PROMPT_TOKENS:-1792}"
  DIAGNOSTIC_MAX_NEW_TOKENS="${DIAGNOSTIC_MAX_NEW_TOKENS:-256}"
  FORECAST_MAX_NEW_TOKENS="${FORECAST_MAX_NEW_TOKENS:-160}"
  bash
  "$ROOT_DIR/scripts/run_phase2_slot_correction_intensity_gate_v1_formal.sh"
)

printf '#!/usr/bin/env bash\n' > "$CMD_FILE"
printf 'set -euo pipefail\n' >> "$CMD_FILE"
printf '%q ' "${cmd[@]}" >> "$CMD_FILE"
printf '\n' >> "$CMD_FILE"
chmod +x "$CMD_FILE"

if [[ "${DISABLE_TMUX:-0}" != "1" ]] && command -v tmux >/dev/null 2>&1; then
  BACKEND="tmux"
  SESSION_NAME="${TMUX_SESSION_NAME:-phase2_slot_correction_intensity_gate_v1_${TIMESTAMP}}"
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
