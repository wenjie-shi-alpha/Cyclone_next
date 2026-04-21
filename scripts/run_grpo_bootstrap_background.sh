#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BOOTSTRAP_CONFIG="$ROOT_DIR/configs/training/grpo_gemma_4_e4b_unsloth_forecast_v2_round1_bootstrap.yaml"

export DISABLE_TMUX=1
exec bash "$ROOT_DIR/scripts/run_grpo_background.sh" "$BOOTSTRAP_CONFIG"
