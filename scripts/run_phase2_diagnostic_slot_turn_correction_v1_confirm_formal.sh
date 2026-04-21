#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="$ROOT_DIR/.venv/bin/python"

DIAGNOSTIC_CONFIG="${DIAGNOSTIC_CONFIG:-$ROOT_DIR/configs/training/sft_gemma_4_e4b_unsloth_diagnostic_v2_slot_turn_correction_stage_v1.yaml}"
FORECAST_CONFIG="${FORECAST_CONFIG:-$ROOT_DIR/configs/training/sft_gemma_4_e4b_unsloth_forecast_phase1_baseline_v2.yaml}"
REWARD_CONFIG="${REWARD_CONFIG:-$ROOT_DIR/configs/training/grpo_gemma_4_e4b_unsloth_forecast_phase1_baseline_v2_reward.yaml}"
FORECAST_ADAPTER="${FORECAST_ADAPTER:-$ROOT_DIR/runs/phase1_baseline_v2_formal_20260415_013403/sft/final_adapter}"
MAINLINE_REPORT="${MAINLINE_REPORT:-$ROOT_DIR/runs/phase2_slot_correction_intensity_gate_v1_confirm_20260419_112630/evals/forecast_integration_compare_sample1613_seed3407.json}"

SOURCE_RUN_ROOT="${SOURCE_RUN_ROOT:-$ROOT_DIR/runs/phase2_diagnostic_slot_turn_correction_v1_20260420_095215}"
DIAGNOSTIC_ADAPTER="${DIAGNOSTIC_ADAPTER:-$SOURCE_RUN_ROOT/sft/final_adapter}"

DATASET_ROOT="${DATASET_ROOT:-$ROOT_DIR/data/training_rebuilt_v2_20260414_guidancefix}"
CANONICAL_TRAIN="${CANONICAL_TRAIN:-$DATASET_ROOT/canonical_v2/train.jsonl}"
CANONICAL_TEST="${CANONICAL_TEST:-$DATASET_ROOT/canonical_v2/test.jsonl}"
DIAGNOSTIC_TRAIN_DATASET="${DIAGNOSTIC_TRAIN_DATASET:-$DATASET_ROOT/views/diagnostic_slot_turn_correction_only/train.jsonl}"
DIAGNOSTIC_TEST_DATASET="${DIAGNOSTIC_TEST_DATASET:-$DATASET_ROOT/views/diagnostic_slot_turn_correction_only/test.jsonl}"
FORECAST_SFT_TEST_DATASET="${FORECAST_SFT_TEST_DATASET:-$DATASET_ROOT/views/forecast_only/test.jsonl}"
FORECAST_RL_TEST_DATASET="${FORECAST_RL_TEST_DATASET:-$DATASET_ROOT/views/forecast_only/rl_test.jsonl}"

SAMPLE_COUNT="${SAMPLE_COUNT:-1613}"
SAMPLE_SEED="${SAMPLE_SEED:-3407}"
DIAGNOSTIC_BATCH_SIZE="${DIAGNOSTIC_BATCH_SIZE:-4}"
FORECAST_BATCH_SIZE="${FORECAST_BATCH_SIZE:-4}"
MAX_PROMPT_TOKENS="${MAX_PROMPT_TOKENS:-1792}"
DIAGNOSTIC_MAX_NEW_TOKENS="${DIAGNOSTIC_MAX_NEW_TOKENS:-384}"
FORECAST_MAX_NEW_TOKENS="${FORECAST_MAX_NEW_TOKENS:-160}"
BASELINE_INTENSITY_OFFSET_SCALE="${BASELINE_INTENSITY_OFFSET_SCALE:-1.20}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="${RUN_ROOT:-$ROOT_DIR/runs/phase2_diagnostic_slot_turn_correction_v1_confirm_${TIMESTAMP}}"
EVAL_ROOT="${EVAL_ROOT:-$RUN_ROOT/evals}"
DIAGNOSTIC_OUTPUT="${DIAGNOSTIC_OUTPUT:-$EVAL_ROOT/diagnostic_eval_summary_sample${SAMPLE_COUNT}_seed${SAMPLE_SEED}.json}"
FORECAST_OUTPUT="${FORECAST_OUTPUT:-$EVAL_ROOT/forecast_integration_compare_sample${SAMPLE_COUNT}_seed${SAMPLE_SEED}.json}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Missing Python executable: $PYTHON_BIN" >&2
  exit 1
fi

"$PYTHON_BIN" "$ROOT_DIR/scripts/check_phase2_diagnostic_slot_turn_correction.py" \
  --diagnostic-config "$DIAGNOSTIC_CONFIG" \
  --forecast-config "$FORECAST_CONFIG" \
  --reward-config "$REWARD_CONFIG" \
  --forecast-adapter "$FORECAST_ADAPTER" \
  --mainline-report "$MAINLINE_REPORT"

if [[ ! -d "$DIAGNOSTIC_ADAPTER" ]]; then
  echo "Missing diagnostic adapter for confirmatory compare: $DIAGNOSTIC_ADAPTER" >&2
  exit 1
fi

mkdir -p "$EVAL_ROOT"

echo "RUN_ROOT=$RUN_ROOT"
echo "SOURCE_RUN_ROOT=$SOURCE_RUN_ROOT"
echo "DIAGNOSTIC_ADAPTER=$DIAGNOSTIC_ADAPTER"
echo "DIAGNOSTIC_OUTPUT=$DIAGNOSTIC_OUTPUT"
echo "FORECAST_OUTPUT=$FORECAST_OUTPUT"
echo "SAMPLE_COUNT=$SAMPLE_COUNT"
echo "MAX_PROMPT_TOKENS=$MAX_PROMPT_TOKENS"

echo "[phase2c_slot_turn_v1_confirm] running standalone diagnostic held-out compare"
"$PYTHON_BIN" "$ROOT_DIR/scripts/compare_diagnostic_models.py" \
  --config "$DIAGNOSTIC_CONFIG" \
  --dataset "$DIAGNOSTIC_TEST_DATASET" \
  --train-dataset "$DIAGNOSTIC_TRAIN_DATASET" \
  --synthetic-baseline majority_label \
  --synthetic-baseline rule_echo \
  --model "diagnostic_adapter_slot_turn_correction_v1=$DIAGNOSTIC_ADAPTER" \
  --sample-count "$SAMPLE_COUNT" \
  --sample-seed "$SAMPLE_SEED" \
  --batch-size "$DIAGNOSTIC_BATCH_SIZE" \
  --max-prompt-tokens "$MAX_PROMPT_TOKENS" \
  --max-new-tokens "$DIAGNOSTIC_MAX_NEW_TOKENS" \
  --output "$DIAGNOSTIC_OUTPUT"

echo "[phase2c_slot_turn_v1_confirm] running forecast integration compare"
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
  --visible-consensus-label "visible_atcf_consensus_passthrough_v0" \
  --oracle-label "oracle_slot_turn_locked_forecast_correction_v1" \
  --predicted-label "predicted_slot_turn_locked_forecast_correction_v1" \
  --oracle-baseline-intensity-label "oracle_slot_turn_track_plus_baseline_intensity_scale_1p20_v1" \
  --predicted-baseline-intensity-label "predicted_slot_turn_track_plus_baseline_intensity_scale_1p20_v1" \
  --official-label "expert_official" \
  --baseline-intensity-offset-scale "$BASELINE_INTENSITY_OFFSET_SCALE" \
  --output "$FORECAST_OUTPUT"

echo "[phase2c_slot_turn_v1_confirm] completed"
