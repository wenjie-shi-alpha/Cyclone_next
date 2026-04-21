#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

export RUN_ROOT="${RUN_ROOT:-$ROOT_DIR/runs/phase2_diagnostic_track_turn_v0_1_${TIMESTAMP}}"
export DIAGNOSTIC_CONFIG="${DIAGNOSTIC_CONFIG:-$ROOT_DIR/configs/training/sft_gemma_4_e4b_unsloth_diagnostic_v2_track_turn_stage_v0_1.yaml}"

exec bash "$ROOT_DIR/scripts/run_phase2_diagnostic_track_turn_formal.sh"
