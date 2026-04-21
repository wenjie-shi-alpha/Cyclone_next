#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_ID="${1:-google/gemma-4-E4B-it}"
LOCAL_DIR="${2:-$ROOT_DIR/models/google/gemma-4-E4B-it}"

mkdir -p "$(dirname "$LOCAL_DIR")"
source "$ROOT_DIR/.venv/bin/activate"
modelscope download --model "$MODEL_ID" --local_dir "$LOCAL_DIR"
