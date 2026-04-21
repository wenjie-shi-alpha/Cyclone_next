#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="${RUN_ROOT:-$ROOT_DIR/runs/phase2_slot_correction_scale_sweep_v0_${TIMESTAMP}}"
LOG_DIR="$RUN_ROOT/logs"
LOG_FILE="$LOG_DIR/phase2.log"
PID_FILE="$RUN_ROOT/train.pid"
CMD_FILE="$RUN_ROOT/launch_command.sh"

mkdir -p "$LOG_DIR"

cmd=(
  env
  PYTHONUNBUFFERED=1
  TOKENIZERS_PARALLELISM=false
  RUN_ROOT="$RUN_ROOT"
  SOURCE_RUN="${SOURCE_RUN:-$ROOT_DIR/runs/phase2_slot_correction_intensity_gate_v0_20260419_025200}"
  REWARD_CONFIG="${REWARD_CONFIG:-$ROOT_DIR/configs/training/grpo_gemma_4_e4b_unsloth_forecast_phase1_baseline_v2_reward.yaml}"
  FORECAST_RL_TEST_DATASET="${FORECAST_RL_TEST_DATASET:-$ROOT_DIR/data/training_rebuilt_v2_20260414_guidancefix/views/forecast_only/rl_test.jsonl}"
  FORECAST_SFT_TEST_DATASET="${FORECAST_SFT_TEST_DATASET:-$ROOT_DIR/data/training_rebuilt_v2_20260414_guidancefix/views/forecast_only/test.jsonl}"
  CANONICAL_TEST="${CANONICAL_TEST:-$ROOT_DIR/data/training_rebuilt_v2_20260414_guidancefix/canonical_v2/test.jsonl}"
  BASELINE_REPORT="${BASELINE_REPORT:-$ROOT_DIR/runs/phase2_slot_correction_intensity_gate_v0_20260419_025200/evals/forecast_integration_compare_sample200_seed3407.models/01_baseline_forecast_sft_v2.json}"
  OFFICIAL_REPORT="${OFFICIAL_REPORT:-$ROOT_DIR/runs/phase2_slot_correction_intensity_gate_v0_20260419_025200/evals/forecast_integration_compare_sample200_seed3407.models/07_expert_official.json}"
  PREDICTED_PAYLOAD="${PREDICTED_PAYLOAD:-$ROOT_DIR/runs/phase2_slot_correction_intensity_gate_v0_20260419_025200/evals/forecast_integration_compare_sample200_seed3407.artifacts/predicted_slot_correction_payload.json}"
  CALIBRATION="${CALIBRATION:-$ROOT_DIR/runs/phase2_slot_correction_intensity_gate_v0_20260419_025200/evals/forecast_integration_compare_sample200_seed3407.artifacts/slot_correction_calibration.json}"
  SCALES="${SCALES:-0.85,0.90,0.95,1.00,1.05,1.10,1.15,1.20}"
  bash
  "$ROOT_DIR/scripts/run_phase2_slot_correction_scale_sweep_v0_formal.sh"
)

printf '#!/usr/bin/env bash\n' > "$CMD_FILE"
printf 'set -euo pipefail\n' >> "$CMD_FILE"
printf '%q ' "${cmd[@]}" >> "$CMD_FILE"
printf '\n' >> "$CMD_FILE"
chmod +x "$CMD_FILE"

nohup "$CMD_FILE" > "$LOG_FILE" 2>&1 &
PID=$!
echo "$PID" > "$PID_FILE"
sleep 2
if kill -0 "$PID" 2>/dev/null; then
  STATUS="running"
else
  STATUS="exited"
fi

echo "BACKEND=nohup"
echo "RUN_ROOT=$RUN_ROOT"
echo "PID=$PID"
echo "LOG_FILE=$LOG_FILE"
echo "PID_FILE=$PID_FILE"
echo "COMMAND_FILE=$CMD_FILE"
echo "STATUS=$STATUS"
