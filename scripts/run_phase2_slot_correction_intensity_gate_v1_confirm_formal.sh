#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="$ROOT_DIR/.venv/bin/python"

FORECAST_CONFIG="${FORECAST_CONFIG:-$ROOT_DIR/configs/training/sft_gemma_4_e4b_unsloth_forecast_phase1_baseline_v2.yaml}"
REWARD_CONFIG="${REWARD_CONFIG:-$ROOT_DIR/configs/training/grpo_gemma_4_e4b_unsloth_forecast_phase1_baseline_v2_reward.yaml}"
FORECAST_ADAPTER="${FORECAST_ADAPTER:-$ROOT_DIR/runs/phase1_baseline_v2_formal_20260415_013403/sft/final_adapter}"
DIAGNOSTIC_CONFIG="${DIAGNOSTIC_CONFIG:-$ROOT_DIR/configs/training/sft_gemma_4_e4b_unsloth_diagnostic_v2_slot_correction_stage_v1.yaml}"
DIAGNOSTIC_ADAPTER="${DIAGNOSTIC_ADAPTER:-$ROOT_DIR/runs/phase2_diagnostic_slot_correction_v1_20260418_184506/sft/final_adapter}"
ORACLE_REPORT="${ORACLE_REPORT:-$ROOT_DIR/runs/phase2_diagnostic_slot_correction_oracle_v0_20260418_154645/evals/forecast_integration_compare_sample200_seed3407.json}"
BASELINE_INTENSITY_OFFSET_SCALE="${BASELINE_INTENSITY_OFFSET_SCALE:-1.20}"

DATASET_ROOT="${DATASET_ROOT:-$ROOT_DIR/data/training_rebuilt_v2_20260414_guidancefix}"
CANONICAL_TRAIN="${CANONICAL_TRAIN:-$DATASET_ROOT/canonical_v2/train.jsonl}"
CANONICAL_TEST="${CANONICAL_TEST:-$DATASET_ROOT/canonical_v2/test.jsonl}"
DIAGNOSTIC_TRAIN_DATASET="${DIAGNOSTIC_TRAIN_DATASET:-$DATASET_ROOT/views/diagnostic_slot_correction_only/train.jsonl}"
DIAGNOSTIC_TEST_DATASET="${DIAGNOSTIC_TEST_DATASET:-$DATASET_ROOT/views/diagnostic_slot_correction_only/test.jsonl}"
FORECAST_SFT_TEST_DATASET="${FORECAST_SFT_TEST_DATASET:-$DATASET_ROOT/views/forecast_only/test.jsonl}"
FORECAST_RL_TEST_DATASET="${FORECAST_RL_TEST_DATASET:-$DATASET_ROOT/views/forecast_only/rl_test.jsonl}"

SAMPLE_COUNT="${SAMPLE_COUNT:-1613}"
SAMPLE_SEED="${SAMPLE_SEED:-3407}"
DIAGNOSTIC_BATCH_SIZE="${DIAGNOSTIC_BATCH_SIZE:-4}"
FORECAST_BATCH_SIZE="${FORECAST_BATCH_SIZE:-4}"
MAX_PROMPT_TOKENS="${MAX_PROMPT_TOKENS:-1792}"
DIAGNOSTIC_MAX_NEW_TOKENS="${DIAGNOSTIC_MAX_NEW_TOKENS:-256}"
FORECAST_MAX_NEW_TOKENS="${FORECAST_MAX_NEW_TOKENS:-160}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="${RUN_ROOT:-$ROOT_DIR/runs/phase2_slot_correction_intensity_gate_v1_confirm_${TIMESTAMP}}"
EVAL_ROOT="${EVAL_ROOT:-$RUN_ROOT/evals}"
OUTPUT="${OUTPUT:-$EVAL_ROOT/forecast_integration_compare_sample${SAMPLE_COUNT}_seed${SAMPLE_SEED}.json}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Missing Python executable: $PYTHON_BIN" >&2
  exit 1
fi

"$PYTHON_BIN" "$ROOT_DIR/scripts/check_phase2_diagnostic_slot_correction.py" \
  --diagnostic-config "$DIAGNOSTIC_CONFIG" \
  --forecast-config "$FORECAST_CONFIG" \
  --reward-config "$REWARD_CONFIG" \
  --forecast-adapter "$FORECAST_ADAPTER" \
  --oracle-report "$ORACLE_REPORT"

mkdir -p "$EVAL_ROOT"

echo "RUN_ROOT=$RUN_ROOT"
echo "OUTPUT=$OUTPUT"
echo "DIAGNOSTIC_ADAPTER=$DIAGNOSTIC_ADAPTER"
echo "BASELINE_INTENSITY_OFFSET_SCALE=$BASELINE_INTENSITY_OFFSET_SCALE"
echo "SAMPLE_COUNT=$SAMPLE_COUNT"

"$PYTHON_BIN" "$ROOT_DIR/scripts/compare_slot_correction_forecast_integration.py" \
  --forecast-config "$FORECAST_CONFIG" \
  --reward-config "$REWARD_CONFIG" \
  --forecast-adapter "$FORECAST_ADAPTER" \
  --forecast-rl-dataset "$FORECAST_RL_TEST_DATASET" \
  --forecast-sft-dataset "$FORECAST_SFT_TEST_DATASET" \
  --diagnostic-dataset "$DIAGNOSTIC_TEST_DATASET" \
  --diagnostic-config "$DIAGNOSTIC_CONFIG" \
  --diagnostic-adapter "$DIAGNOSTIC_ADAPTER" \
  --diagnostic-train-dataset "$DIAGNOSTIC_TRAIN_DATASET" \
  --diagnostic-prediction-mode adapter \
  --canonical-train "$CANONICAL_TRAIN" \
  --canonical-test "$CANONICAL_TEST" \
  --sample-count "$SAMPLE_COUNT" \
  --sample-seed "$SAMPLE_SEED" \
  --forecast-batch-size "$FORECAST_BATCH_SIZE" \
  --diagnostic-batch-size "$DIAGNOSTIC_BATCH_SIZE" \
  --max-prompt-tokens "$MAX_PROMPT_TOKENS" \
  --max-new-tokens "$FORECAST_MAX_NEW_TOKENS" \
  --diagnostic-max-new-tokens "$DIAGNOSTIC_MAX_NEW_TOKENS" \
  --baseline-label "baseline_forecast_sft_v2" \
  --predicted-label "predicted_slot_locked_forecast_correction_v1" \
  --predicted-baseline-intensity-label "predicted_slot_locked_track_plus_baseline_intensity_scale_1p20_v1" \
  --baseline-intensity-offset-scale "$BASELINE_INTENSITY_OFFSET_SCALE" \
  --official-label "expert_official" \
  --skip-visible-consensus \
  --skip-oracle \
  --skip-oracle-baseline-intensity \
  --output "$OUTPUT"

echo "[phase2_slot_correction_intensity_gate_v1_confirm] completed"
