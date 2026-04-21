#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_PATH="$ROOT_DIR/configs/training/grpo_gemma_4_e4b_unsloth_forecast_v2_phase1_v6_100step.yaml"

export RUN_ROOT="${RUN_ROOT:-$ROOT_DIR/runs/grpo_phase1_v6_100step_20260413}"
export TMUX_SESSION_NAME="${TMUX_SESSION_NAME:-grpo_phase1_v6_100step_20260413}"

exec bash "$ROOT_DIR/scripts/run_grpo_background.sh" "$CONFIG_PATH"
