#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="$ROOT_DIR/.venv/bin/python"

DIAGNOSTIC_CONFIG="${DIAGNOSTIC_CONFIG:-$ROOT_DIR/configs/training/sft_gemma_4_e4b_unsloth_diagnostic_v2_track_core_v0.yaml}"
FORECAST_CONFIG="${FORECAST_CONFIG:-$ROOT_DIR/configs/training/sft_gemma_4_e4b_unsloth_forecast_phase1_baseline_v2.yaml}"
REWARD_CONFIG="${REWARD_CONFIG:-$ROOT_DIR/configs/training/grpo_gemma_4_e4b_unsloth_forecast_phase1_baseline_v2_reward.yaml}"
FORECAST_ADAPTER="${FORECAST_ADAPTER:-$ROOT_DIR/runs/phase1_baseline_v2_formal_20260415_013403/sft/final_adapter}"

DATASET_ROOT="${DATASET_ROOT:-$ROOT_DIR/data/training_rebuilt_v2_20260414_guidancefix}"
DIAGNOSTIC_TRAIN_DATASET="${DIAGNOSTIC_TRAIN_DATASET:-$DATASET_ROOT/views/diagnostic_track_core_only/train.jsonl}"
DIAGNOSTIC_TEST_DATASET="${DIAGNOSTIC_TEST_DATASET:-$DATASET_ROOT/views/diagnostic_track_core_only/test.jsonl}"
FORECAST_SFT_TEST_DATASET="${FORECAST_SFT_TEST_DATASET:-$DATASET_ROOT/views/forecast_only/test.jsonl}"
FORECAST_RL_TEST_DATASET="${FORECAST_RL_TEST_DATASET:-$DATASET_ROOT/views/forecast_only/rl_test.jsonl}"

DIAGNOSTIC_SAMPLE_COUNT="${DIAGNOSTIC_SAMPLE_COUNT:-200}"
DIAGNOSTIC_SAMPLE_SEED="${DIAGNOSTIC_SAMPLE_SEED:-3407}"
FORECAST_SAMPLE_COUNT="${FORECAST_SAMPLE_COUNT:-200}"
FORECAST_SAMPLE_SEED="${FORECAST_SAMPLE_SEED:-3407}"
DIAGNOSTIC_BATCH_SIZE="${DIAGNOSTIC_BATCH_SIZE:-4}"
FORECAST_BATCH_SIZE="${FORECAST_BATCH_SIZE:-4}"
MAX_PROMPT_TOKENS="${MAX_PROMPT_TOKENS:-1024}"
DIAGNOSTIC_MAX_NEW_TOKENS="${DIAGNOSTIC_MAX_NEW_TOKENS:-256}"
FORECAST_MAX_NEW_TOKENS="${FORECAST_MAX_NEW_TOKENS:-160}"

SKIP_TRAIN="${SKIP_TRAIN:-0}"
SKIP_STANDALONE_COMPARE="${SKIP_STANDALONE_COMPARE:-0}"
SKIP_FORECAST_COMPARE="${SKIP_FORECAST_COMPARE:-0}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="${RUN_ROOT:-$ROOT_DIR/runs/phase2_diagnostic_track_core_v0_${TIMESTAMP}}"
EVAL_ROOT="${EVAL_ROOT:-$RUN_ROOT/evals}"
DIAGNOSTIC_OUTPUT="${DIAGNOSTIC_OUTPUT:-$EVAL_ROOT/diagnostic_eval_summary_sample${DIAGNOSTIC_SAMPLE_COUNT}_seed${DIAGNOSTIC_SAMPLE_SEED}.json}"
FORECAST_OUTPUT="${FORECAST_OUTPUT:-$EVAL_ROOT/forecast_integration_compare_sample${FORECAST_SAMPLE_COUNT}_seed${FORECAST_SAMPLE_SEED}.json}"
DIAGNOSTIC_ADAPTER="${DIAGNOSTIC_ADAPTER:-$RUN_ROOT/sft/final_adapter}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Missing Python executable: $PYTHON_BIN" >&2
  exit 1
fi

"$PYTHON_BIN" "$ROOT_DIR/scripts/check_phase2_diagnostic_v0.py" \
  --mode formal \
  --diagnostic-config "$DIAGNOSTIC_CONFIG" \
  --forecast-config "$FORECAST_CONFIG" \
  --reward-config "$REWARD_CONFIG" \
  --forecast-adapter "$FORECAST_ADAPTER"

mkdir -p "$EVAL_ROOT"

echo "RUN_ROOT=$RUN_ROOT"
echo "DIAGNOSTIC_OUTPUT=$DIAGNOSTIC_OUTPUT"
echo "FORECAST_OUTPUT=$FORECAST_OUTPUT"
echo "FORECAST_ADAPTER=$FORECAST_ADAPTER"

if [[ "$SKIP_TRAIN" != "1" ]]; then
  echo "[phase2] training diagnostic_adapter_track_core"
  "$PYTHON_BIN" "$ROOT_DIR/scripts/train_sft.py" \
    --config "$DIAGNOSTIC_CONFIG" \
    --run-root "$RUN_ROOT" \
    --verbose
else
  echo "[phase2] skipping training"
fi

if [[ "$SKIP_STANDALONE_COMPARE" != "1" || "$SKIP_FORECAST_COMPARE" != "1" ]]; then
  if [[ ! -d "$DIAGNOSTIC_ADAPTER" ]]; then
    echo "Missing diagnostic adapter for compare steps: $DIAGNOSTIC_ADAPTER" >&2
    exit 1
  fi
fi

if [[ "$SKIP_STANDALONE_COMPARE" != "1" ]]; then
  echo "[phase2] running standalone diagnostic held-out compare"
  "$PYTHON_BIN" "$ROOT_DIR/scripts/compare_diagnostic_models.py" \
    --config "$DIAGNOSTIC_CONFIG" \
    --dataset "$DIAGNOSTIC_TEST_DATASET" \
    --train-dataset "$DIAGNOSTIC_TRAIN_DATASET" \
    --synthetic-baseline majority_label \
    --synthetic-baseline rule_echo \
    --model "diagnostic_adapter_track_core=$DIAGNOSTIC_ADAPTER" \
    --sample-count "$DIAGNOSTIC_SAMPLE_COUNT" \
    --sample-seed "$DIAGNOSTIC_SAMPLE_SEED" \
    --batch-size "$DIAGNOSTIC_BATCH_SIZE" \
    --max-prompt-tokens "$MAX_PROMPT_TOKENS" \
    --max-new-tokens "$DIAGNOSTIC_MAX_NEW_TOKENS" \
    --output "$DIAGNOSTIC_OUTPUT"
else
  echo "[phase2] skipping standalone diagnostic compare"
fi

if [[ "$SKIP_FORECAST_COMPARE" != "1" ]]; then
  echo "[phase2] running oracle/predicted diagnostic forecast integration compare"
  "$PYTHON_BIN" "$ROOT_DIR/scripts/compare_diagnostic_forecast_integration.py" \
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
    --sample-count "$FORECAST_SAMPLE_COUNT" \
    --sample-seed "$FORECAST_SAMPLE_SEED" \
    --forecast-batch-size "$FORECAST_BATCH_SIZE" \
    --diagnostic-batch-size "$DIAGNOSTIC_BATCH_SIZE" \
    --max-prompt-tokens "$MAX_PROMPT_TOKENS" \
    --max-new-tokens "$FORECAST_MAX_NEW_TOKENS" \
    --diagnostic-max-new-tokens "$DIAGNOSTIC_MAX_NEW_TOKENS" \
    --section-title "Track-Core Structured Diagnostic Assessment" \
    --baseline-label "baseline_forecast_sft_v2" \
    --oracle-label "oracle_track_core_diagnostics_plus_forecast" \
    --predicted-label "predicted_track_core_diagnostics_plus_forecast" \
    --official-label "expert_official" \
    --output "$FORECAST_OUTPUT"
else
  echo "[phase2] skipping forecast integration compare"
fi

echo "[phase2] completed"
