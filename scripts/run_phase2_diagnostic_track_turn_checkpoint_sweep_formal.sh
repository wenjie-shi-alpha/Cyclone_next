#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="$ROOT_DIR/.venv/bin/python"

SOURCE_RUN_ROOT="${SOURCE_RUN_ROOT:-$ROOT_DIR/runs/phase2_diagnostic_track_turn_v0_20260416_131804}"
DIAGNOSTIC_CONFIG="${DIAGNOSTIC_CONFIG:-$ROOT_DIR/configs/training/sft_gemma_4_e4b_unsloth_diagnostic_v2_track_turn_stage_v0.yaml}"
FORECAST_CONFIG="${FORECAST_CONFIG:-$ROOT_DIR/configs/training/sft_gemma_4_e4b_unsloth_forecast_phase1_baseline_v2.yaml}"
REWARD_CONFIG="${REWARD_CONFIG:-$ROOT_DIR/configs/training/grpo_gemma_4_e4b_unsloth_forecast_phase1_baseline_v2_reward.yaml}"
FORECAST_ADAPTER="${FORECAST_ADAPTER:-$ROOT_DIR/runs/phase1_baseline_v2_formal_20260415_013403/sft/final_adapter}"

DATASET_ROOT="${DATASET_ROOT:-$ROOT_DIR/data/training_rebuilt_v2_20260414_guidancefix}"
DIAGNOSTIC_TRAIN_DATASET="${DIAGNOSTIC_TRAIN_DATASET:-$DATASET_ROOT/views/diagnostic_track_turn_only/train.jsonl}"
DIAGNOSTIC_TEST_DATASET="${DIAGNOSTIC_TEST_DATASET:-$DATASET_ROOT/views/diagnostic_track_turn_only/test.jsonl}"
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

SWEEP_OUTPUT="${SWEEP_OUTPUT:-$SOURCE_RUN_ROOT/evals/checkpoint_sweep_sample${DIAGNOSTIC_SAMPLE_COUNT}_seed${DIAGNOSTIC_SAMPLE_SEED}.json}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Missing Python executable: $PYTHON_BIN" >&2
  exit 1
fi

"$PYTHON_BIN" "$ROOT_DIR/scripts/sweep_phase2_diagnostic_track_turn_checkpoints.py" \
  --source-run-root "$SOURCE_RUN_ROOT" \
  --diagnostic-config "$DIAGNOSTIC_CONFIG" \
  --diagnostic-dataset "$DIAGNOSTIC_TEST_DATASET" \
  --diagnostic-train-dataset "$DIAGNOSTIC_TRAIN_DATASET" \
  --forecast-config "$FORECAST_CONFIG" \
  --reward-config "$REWARD_CONFIG" \
  --forecast-adapter "$FORECAST_ADAPTER" \
  --forecast-rl-dataset "$FORECAST_RL_TEST_DATASET" \
  --forecast-sft-dataset "$FORECAST_SFT_TEST_DATASET" \
  --diagnostic-sample-count "$DIAGNOSTIC_SAMPLE_COUNT" \
  --diagnostic-sample-seed "$DIAGNOSTIC_SAMPLE_SEED" \
  --forecast-sample-count "$FORECAST_SAMPLE_COUNT" \
  --forecast-sample-seed "$FORECAST_SAMPLE_SEED" \
  --diagnostic-batch-size "$DIAGNOSTIC_BATCH_SIZE" \
  --forecast-batch-size "$FORECAST_BATCH_SIZE" \
  --max-prompt-tokens "$MAX_PROMPT_TOKENS" \
  --diagnostic-max-new-tokens "$DIAGNOSTIC_MAX_NEW_TOKENS" \
  --forecast-max-new-tokens "$FORECAST_MAX_NEW_TOKENS" \
  --section-title "Track-Turn Structured Diagnostic Assessment" \
  --output "$SWEEP_OUTPUT"
