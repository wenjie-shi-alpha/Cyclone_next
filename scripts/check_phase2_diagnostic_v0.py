#!/usr/bin/env python3
"""Preflight validation for Phase 2 v0 track-core diagnostic supervision."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cyclone_training.config import load_grpo_config, load_sft_config


DATASET_ROOT = (ROOT / "data/training_rebuilt_v2_20260414_guidancefix").resolve()
DATASET_READY_REPORT = (DATASET_ROOT / "dataset_ready_report.json").resolve()
DIAGNOSTIC_VIEW_ROOT = (DATASET_ROOT / "views" / "diagnostic_track_core_only").resolve()
FORECAST_VIEW_ROOT = (DATASET_ROOT / "views" / "forecast_only").resolve()
EXPECTED_DIAGNOSTIC_PATHS = {
    "train": (DIAGNOSTIC_VIEW_ROOT / "train.jsonl").resolve(),
    "val": (DIAGNOSTIC_VIEW_ROOT / "val.jsonl").resolve(),
    "test": (DIAGNOSTIC_VIEW_ROOT / "test.jsonl").resolve(),
}
EXPECTED_FORECAST_PATHS = {
    "sft_train": (FORECAST_VIEW_ROOT / "train.jsonl").resolve(),
    "sft_val": (FORECAST_VIEW_ROOT / "val.jsonl").resolve(),
    "sft_test": (FORECAST_VIEW_ROOT / "test.jsonl").resolve(),
    "rl_train": (FORECAST_VIEW_ROOT / "rl_train.jsonl").resolve(),
    "rl_test": (FORECAST_VIEW_ROOT / "rl_test.jsonl").resolve(),
}
EXPECTED_BASE_MODEL = "models/google/gemma-4-E4B-it"


def _same_path(left: Path | None, right: Path | None) -> bool:
    if left is None or right is None:
        return left is None and right is None
    return left.resolve() == right.resolve()


def _require(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def _require_file(path: Path, label: str, errors: list[str]) -> None:
    if not path.exists():
        errors.append(f"{label} does not exist: {path}")


def _validate_dataset_ready(errors: list[str]) -> None:
    _require_file(DATASET_READY_REPORT, "dataset_ready_report", errors)
    if errors:
        return
    payload = json.loads(DATASET_READY_REPORT.read_text(encoding="utf-8"))
    _require(
        bool(payload.get("forecast_root_compatible")),
        f"dataset_ready_report must mark forecast_root_compatible=true: {DATASET_READY_REPORT}",
        errors,
    )
    _require(
        bool(payload.get("diagnostic_track_core_view_ready")),
        f"dataset_ready_report must mark diagnostic_track_core_view_ready=true: {DATASET_READY_REPORT}",
        errors,
    )


def _validate_diagnostic_sft(smoke_mode: bool, config_path: Path, errors: list[str]) -> str | None:
    config = load_sft_config(config_path)
    _require(
        _same_path(config.data.dataset_root, DATASET_ROOT),
        f"Diagnostic dataset_root must be {DATASET_ROOT}, got {config.data.dataset_root}",
        errors,
    )
    _require(
        _same_path(config.data.sft_train_file, EXPECTED_DIAGNOSTIC_PATHS["train"]),
        (
            "Diagnostic train file must use the frozen diagnostic_track_core_only train split: "
            f"{EXPECTED_DIAGNOSTIC_PATHS['train']}"
        ),
        errors,
    )
    _require(
        _same_path(config.data.sft_eval_file, EXPECTED_DIAGNOSTIC_PATHS["val"]),
        (
            "Diagnostic eval file must use the frozen diagnostic_track_core_only val split: "
            f"{EXPECTED_DIAGNOSTIC_PATHS['val']}"
        ),
        errors,
    )
    _require(
        config.trainer.resume_from_checkpoint is None,
        "Diagnostic SFT resume_from_checkpoint must be null.",
        errors,
    )
    _require(
        config.trainer.completion_only_loss is True,
        "Diagnostic SFT completion_only_loss must stay true.",
        errors,
    )
    if smoke_mode:
        _require(
            (config.data.max_train_samples or 0) > 0,
            "Smoke diagnostic SFT config must bound max_train_samples.",
            errors,
        )
        _require(
            config.trainer.max_steps > 0,
            "Smoke diagnostic SFT config must bound max_steps.",
            errors,
        )
    else:
        _require(
            config.data.max_train_samples is None,
            "Formal diagnostic SFT config must not cap max_train_samples.",
            errors,
        )
        _require(
            config.data.max_eval_samples is None,
            "Formal diagnostic SFT config must not cap max_eval_samples.",
            errors,
        )
    _require_file(EXPECTED_DIAGNOSTIC_PATHS["train"], "Diagnostic train split", errors)
    _require_file(EXPECTED_DIAGNOSTIC_PATHS["val"], "Diagnostic val split", errors)
    _require_file(EXPECTED_DIAGNOSTIC_PATHS["test"], "Diagnostic test split", errors)
    return config.model.name_or_path


def _validate_forecast_sft(config_path: Path, errors: list[str]) -> str | None:
    config = load_sft_config(config_path)
    _require(
        _same_path(config.data.dataset_root, DATASET_ROOT),
        f"Forecast dataset_root must be {DATASET_ROOT}, got {config.data.dataset_root}",
        errors,
    )
    _require(
        _same_path(config.data.sft_train_file, EXPECTED_FORECAST_PATHS["sft_train"]),
        (
            "Forecast SFT train file must use the frozen forecast_only train split: "
            f"{EXPECTED_FORECAST_PATHS['sft_train']}"
        ),
        errors,
    )
    _require(
        _same_path(config.data.sft_eval_file, EXPECTED_FORECAST_PATHS["sft_val"]),
        (
            "Forecast SFT eval file must use the frozen forecast_only val split: "
            f"{EXPECTED_FORECAST_PATHS['sft_val']}"
        ),
        errors,
    )
    _require_file(EXPECTED_FORECAST_PATHS["sft_test"], "Forecast SFT test split", errors)
    return config.model.name_or_path


def _validate_reward_config(config_path: Path, errors: list[str]) -> None:
    config = load_grpo_config(config_path)
    _require(
        _same_path(config.data.dataset_root, DATASET_ROOT),
        f"Reward config dataset_root must be {DATASET_ROOT}, got {config.data.dataset_root}",
        errors,
    )
    _require(
        _same_path(config.data.rl_train_file, EXPECTED_FORECAST_PATHS["rl_train"]),
        (
            "Reward config RL train file must use the frozen forecast_only rl_train split: "
            f"{EXPECTED_FORECAST_PATHS['rl_train']}"
        ),
        errors,
    )
    _require_file(EXPECTED_FORECAST_PATHS["rl_test"], "Forecast RL test split", errors)


def _validate_forecast_adapter(adapter_path: Path, errors: list[str]) -> None:
    _require_file(adapter_path, "Forecast baseline adapter directory", errors)
    if errors:
        return
    _require(
        adapter_path.is_dir(),
        f"Forecast baseline adapter must be a directory: {adapter_path}",
        errors,
    )
    _require_file(adapter_path / "adapter_model.safetensors", "Forecast adapter weights", errors)
    _require_file(adapter_path / "adapter_config.json", "Forecast adapter config", errors)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate Phase 2 v0 track-core configs and artifacts.")
    parser.add_argument("--diagnostic-config", required=True, help="Path to diagnostic SFT YAML config.")
    parser.add_argument("--forecast-config", required=True, help="Path to forecast SFT YAML config.")
    parser.add_argument("--reward-config", required=True, help="Path to reward YAML config.")
    parser.add_argument("--forecast-adapter", required=True, help="Path to the frozen forecast baseline adapter.")
    parser.add_argument(
        "--mode",
        choices=["smoke", "formal"],
        required=True,
        help="Validation mode. Formal mode enforces the frozen production settings.",
    )
    args = parser.parse_args(argv)

    errors: list[str] = []
    smoke_mode = args.mode == "smoke"
    _validate_dataset_ready(errors)
    diagnostic_model_name = _validate_diagnostic_sft(
        smoke_mode=smoke_mode,
        config_path=Path(args.diagnostic_config).resolve(),
        errors=errors,
    )
    forecast_model_name = _validate_forecast_sft(
        config_path=Path(args.forecast_config).resolve(),
        errors=errors,
    )
    _validate_reward_config(Path(args.reward_config).resolve(), errors)
    _validate_forecast_adapter(Path(args.forecast_adapter).resolve(), errors)

    if diagnostic_model_name is not None:
        _require(
            diagnostic_model_name == EXPECTED_BASE_MODEL,
            (
                "Diagnostic SFT must start from the frozen shared base model "
                f"{EXPECTED_BASE_MODEL}, got {diagnostic_model_name}"
            ),
            errors,
        )
    if forecast_model_name is not None:
        _require(
            forecast_model_name == EXPECTED_BASE_MODEL,
            (
                "Forecast SFT config must reference the frozen shared base model "
                f"{EXPECTED_BASE_MODEL}, got {forecast_model_name}"
            ),
            errors,
        )
    if diagnostic_model_name is not None and forecast_model_name is not None:
        _require(
            diagnostic_model_name == forecast_model_name,
            "Diagnostic and forecast configs must share the same base model initialization.",
            errors,
        )

    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1

    print(f"OK: dataset_root={DATASET_ROOT}")
    print(f"OK: diagnostic_track_core_view={DIAGNOSTIC_VIEW_ROOT}")
    print(f"OK: forecast_view={FORECAST_VIEW_ROOT}")
    print(f"OK: mode={args.mode}")
    print(f"OK: diagnostic_config={Path(args.diagnostic_config).resolve()}")
    print(f"OK: forecast_config={Path(args.forecast_config).resolve()}")
    print(f"OK: reward_config={Path(args.reward_config).resolve()}")
    print(f"OK: forecast_adapter={Path(args.forecast_adapter).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
