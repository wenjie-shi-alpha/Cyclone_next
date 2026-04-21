#!/usr/bin/env python3
"""Preflight validation for the rebuilt Phase 1 forecast-only baseline v2."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cyclone_training.config import load_grpo_config, load_sft_config


DATASET_ROOT = (ROOT / "data/training_rebuilt_v2_20260414_guidancefix").resolve()
FORECAST_VIEW_ROOT = (DATASET_ROOT / "views" / "forecast_only").resolve()
EXPECTED_PATHS = {
    "sft_train": (FORECAST_VIEW_ROOT / "train.jsonl").resolve(),
    "sft_eval": (FORECAST_VIEW_ROOT / "val.jsonl").resolve(),
    "rl_train": (FORECAST_VIEW_ROOT / "rl_train.jsonl").resolve(),
    "rl_eval": (FORECAST_VIEW_ROOT / "rl_val.jsonl").resolve(),
}


def _same_path(left: Path | None, right: Path | None) -> bool:
    if left is None or right is None:
        return left is None and right is None
    return left.resolve() == right.resolve()


def _require_file(path: Path, label: str, errors: list[str]) -> None:
    if not path.exists():
        errors.append(f"{label} does not exist: {path}")


def _require(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def _validate_sft(smoke_mode: bool, config_path: Path, errors: list[str]) -> None:
    config = load_sft_config(config_path)
    _require(
        _same_path(config.data.dataset_root, DATASET_ROOT),
        (
            "SFT dataset_root must be "
            f"{DATASET_ROOT}, got {config.data.dataset_root}"
        ),
        errors,
    )
    _require(
        _same_path(config.data.sft_train_file, EXPECTED_PATHS["sft_train"]),
        (
            "SFT train file must be the frozen forecast_only view: "
            f"{EXPECTED_PATHS['sft_train']}"
        ),
        errors,
    )
    _require(
        _same_path(config.data.sft_eval_file, EXPECTED_PATHS["sft_eval"]),
        (
            "SFT eval file must be the frozen forecast_only view: "
            f"{EXPECTED_PATHS['sft_eval']}"
        ),
        errors,
    )
    _require(config.trainer.resume_from_checkpoint is None, "SFT resume_from_checkpoint must be null.", errors)
    if smoke_mode:
        _require(
            (config.data.max_train_samples or 0) > 0,
            "Smoke SFT config must bound max_train_samples.",
            errors,
        )
        _require(config.trainer.max_steps > 0, "Smoke SFT config must bound max_steps.", errors)
    else:
        _require(
            config.data.max_train_samples is None,
            "Formal SFT config must not cap max_train_samples.",
            errors,
        )
        _require(
            config.data.max_eval_samples is None,
            "Formal SFT config must not cap max_eval_samples.",
            errors,
        )
    _require_file(EXPECTED_PATHS["sft_train"], "SFT train file", errors)
    _require_file(EXPECTED_PATHS["sft_eval"], "SFT eval file", errors)


def _validate_grpo(smoke_mode: bool, config_path: Path, errors: list[str]) -> None:
    config = load_grpo_config(config_path)
    _require(
        _same_path(config.data.dataset_root, DATASET_ROOT),
        (
            "GRPO dataset_root must be "
            f"{DATASET_ROOT}, got {config.data.dataset_root}"
        ),
        errors,
    )
    _require(
        _same_path(config.data.rl_train_file, EXPECTED_PATHS["rl_train"]),
        (
            "GRPO train file must be the frozen forecast_only view: "
            f"{EXPECTED_PATHS['rl_train']}"
        ),
        errors,
    )
    _require(
        config.data.rl_eval_file is None or _same_path(config.data.rl_eval_file, EXPECTED_PATHS["rl_eval"]),
        (
            "GRPO eval file, when present, must be the frozen forecast_only view: "
            f"{EXPECTED_PATHS['rl_eval']}"
        ),
        errors,
    )
    _require(
        config.trainer.adapter_init_path is None,
        "GRPO adapter_init_path must stay null so the new SFT adapter is injected by the pipeline.",
        errors,
    )
    _require(
        config.trainer.reference_adapter_path is None,
        "GRPO reference_adapter_path must stay null so the new SFT adapter is reused as the reference adapter.",
        errors,
    )
    _require(config.trainer.resume_from_checkpoint is None, "GRPO resume_from_checkpoint must be null.", errors)
    _require(
        config.trainer.reward_save_threshold is not None
        and config.trainer.reward_save_threshold >= 0.4,
        "GRPO reward_save_threshold must be at least 0.4.",
        errors,
    )
    if smoke_mode:
        _require(
            (config.data.max_train_samples or 0) > 0,
            "Smoke GRPO config must bound max_train_samples.",
            errors,
        )
        _require(config.trainer.max_steps > 0, "Smoke GRPO config must bound max_steps.", errors)
    else:
        _require(config.data.max_train_samples is None, "Formal GRPO config must not cap max_train_samples.", errors)
        _require(config.data.max_eval_samples is None, "Formal GRPO config must not cap max_eval_samples.", errors)
        _require(config.trainer.max_steps == 100, "Formal GRPO config must run 100 steps.", errors)
        _require(config.trainer.save_steps == 10, "Formal GRPO config must save every 10 steps.", errors)
    _require_file(EXPECTED_PATHS["rl_train"], "GRPO train file", errors)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate Phase 1 baseline v2 configs.")
    parser.add_argument("--sft-config", required=True, help="Path to the SFT YAML config.")
    parser.add_argument("--grpo-config", required=True, help="Path to the GRPO YAML config.")
    parser.add_argument(
        "--mode",
        choices=["smoke", "formal"],
        required=True,
        help="Validation mode. Formal mode enforces the frozen production settings.",
    )
    args = parser.parse_args(argv)

    errors: list[str] = []
    smoke_mode = args.mode == "smoke"
    _validate_sft(smoke_mode=smoke_mode, config_path=Path(args.sft_config).resolve(), errors=errors)
    _validate_grpo(smoke_mode=smoke_mode, config_path=Path(args.grpo_config).resolve(), errors=errors)

    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1

    print(f"OK: dataset_root={DATASET_ROOT}")
    print(f"OK: forecast_only_view={FORECAST_VIEW_ROOT}")
    print(f"OK: mode={args.mode}")
    print(f"OK: sft_config={Path(args.sft_config).resolve()}")
    print(f"OK: grpo_config={Path(args.grpo_config).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
