#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SFT_CONFIG="$ROOT_DIR/configs/training/sft_gemma_4_e4b_unsloth_forecast_v2_round1.yaml"
GRPO_CONFIG="$ROOT_DIR/configs/training/grpo_gemma_4_e4b_unsloth_forecast_v2_round1.yaml"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

export RUN_ROOT="${RUN_ROOT:-$ROOT_DIR/runs/forecast_v2_round1_${TIMESTAMP}}"
export TMUX_SESSION_NAME="${TMUX_SESSION_NAME:-forecast_v2_round1_${TIMESTAMP}}"

exec "$ROOT_DIR/scripts/run_sft_then_grpo_background.sh" "$SFT_CONFIG" "$GRPO_CONFIG"
