#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="$ROOT_DIR/.venv/bin/python"

SOURCE_RUN="${SOURCE_RUN:-$ROOT_DIR/runs/phase2_slot_correction_intensity_gate_v0_20260419_025200}"
REWARD_CONFIG="${REWARD_CONFIG:-$ROOT_DIR/configs/training/grpo_gemma_4_e4b_unsloth_forecast_phase1_baseline_v2_reward.yaml}"
FORECAST_RL_TEST_DATASET="${FORECAST_RL_TEST_DATASET:-$ROOT_DIR/data/training_rebuilt_v2_20260414_guidancefix/views/forecast_only/rl_test.jsonl}"
FORECAST_SFT_TEST_DATASET="${FORECAST_SFT_TEST_DATASET:-$ROOT_DIR/data/training_rebuilt_v2_20260414_guidancefix/views/forecast_only/test.jsonl}"
CANONICAL_TEST="${CANONICAL_TEST:-$ROOT_DIR/data/training_rebuilt_v2_20260414_guidancefix/canonical_v2/test.jsonl}"
BASELINE_REPORT="${BASELINE_REPORT:-$SOURCE_RUN/evals/forecast_integration_compare_sample200_seed3407.models/01_baseline_forecast_sft_v2.json}"
PREDICTED_PAYLOAD="${PREDICTED_PAYLOAD:-$SOURCE_RUN/evals/forecast_integration_compare_sample200_seed3407.artifacts/predicted_slot_correction_payload.json}"
CALIBRATION="${CALIBRATION:-$SOURCE_RUN/evals/forecast_integration_compare_sample200_seed3407.artifacts/slot_correction_calibration.json}"
OFFICIAL_REPORT="${OFFICIAL_REPORT:-$SOURCE_RUN/evals/forecast_integration_compare_sample200_seed3407.models/07_expert_official.json}"
SCALES="${SCALES:-0.85,0.90,0.95,1.00,1.05,1.10,1.15,1.20}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="${RUN_ROOT:-$ROOT_DIR/runs/phase2_slot_correction_scale_sweep_v0_${TIMESTAMP}}"
EVAL_ROOT="${EVAL_ROOT:-$RUN_ROOT/evals}"
OUTPUT="${OUTPUT:-$EVAL_ROOT/forecast_integration_scale_sweep_sample200_seed3407.json}"

mkdir -p "$EVAL_ROOT"

echo "RUN_ROOT=$RUN_ROOT"
echo "SOURCE_RUN=$SOURCE_RUN"
echo "SCALES=$SCALES"

"$PYTHON_BIN" "$ROOT_DIR/scripts/compare_slot_correction_scale_sweep.py" \
  --reward-config "$REWARD_CONFIG" \
  --forecast-rl-dataset "$FORECAST_RL_TEST_DATASET" \
  --forecast-sft-dataset "$FORECAST_SFT_TEST_DATASET" \
  --canonical-test "$CANONICAL_TEST" \
  --baseline-report "$BASELINE_REPORT" \
  --official-report "$OFFICIAL_REPORT" \
  --predicted-payload "$PREDICTED_PAYLOAD" \
  --calibration "$CALIBRATION" \
  --scales "$SCALES" \
  --output "$OUTPUT"

echo "[phase2_slot_correction_scale_sweep_v0] completed"
