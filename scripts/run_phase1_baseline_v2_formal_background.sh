#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
SFT_CONFIG="${SFT_CONFIG:-$ROOT_DIR/configs/training/sft_gemma_4_e4b_unsloth_forecast_phase1_baseline_v2.yaml}"
GRPO_CONFIG="${GRPO_CONFIG:-$ROOT_DIR/configs/training/grpo_gemma_4_e4b_unsloth_forecast_phase1_baseline_v2_reward.yaml}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Missing Python executable: $PYTHON_BIN" >&2
  exit 1
fi

"$PYTHON_BIN" "$ROOT_DIR/scripts/check_phase1_baseline_v2.py" \
  --mode formal \
  --sft-config "$SFT_CONFIG" \
  --grpo-config "$GRPO_CONFIG"

export RUN_ROOT="${RUN_ROOT:-$ROOT_DIR/runs/phase1_baseline_v2_formal_${TIMESTAMP}}"
export TMUX_SESSION_NAME="${TMUX_SESSION_NAME:-phase1_baseline_v2_formal_${TIMESTAMP}}"

exec bash "$ROOT_DIR/scripts/run_sft_then_grpo_background.sh" "$SFT_CONFIG" "$GRPO_CONFIG"
