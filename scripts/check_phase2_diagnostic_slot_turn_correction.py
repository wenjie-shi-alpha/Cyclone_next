#!/usr/bin/env python3
"""Preflight validation for Phase 2C slot+turn correction training and closure."""

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
DIAGNOSTIC_VIEW_ROOT = (DATASET_ROOT / "views" / "diagnostic_slot_turn_correction_only").resolve()
FORECAST_VIEW_ROOT = (DATASET_ROOT / "views" / "forecast_only").resolve()
CANONICAL_ROOT = (DATASET_ROOT / "canonical_v2").resolve()
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
EXPECTED_CANONICAL_PATHS = {
    "train": (CANONICAL_ROOT / "train.jsonl").resolve(),
    "test": (CANONICAL_ROOT / "test.jsonl").resolve(),
}
EXPECTED_BASE_MODEL = "models/google/gemma-4-E4B-it"
DEFAULT_SUCCESSFUL_MAINLINE_REPORT = (
    ROOT
    / "runs/phase2_slot_correction_intensity_gate_v1_confirm_20260419_112630/evals/forecast_integration_compare_sample1613_seed3407.json"
).resolve()
MAINLINE_LABEL_CANDIDATES = (
    "predicted_slot_turn_track_plus_baseline_intensity_scale_1p20_v1",
    "predicted_slot_locked_track_plus_baseline_intensity_scale_1p20_v1",
)


def _same_path(left: Path | None, right: Path | None) -> bool:
    if left is None or right is None:
        return left is None and right is None
    return left.resolve() == right.resolve()


def _require(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def _require_file(path: Path | None, label: str, errors: list[str]) -> None:
    if path is None or not path.exists():
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
        bool(payload.get("diagnostic_slot_turn_correction_view_ready")),
        (
            "dataset_ready_report must mark diagnostic_slot_turn_correction_view_ready=true: "
            f"{DATASET_READY_REPORT}"
        ),
        errors,
    )


def _validate_current_mainline(report_path: Path, errors: list[str]) -> None:
    _require_file(report_path, "passed slot-correction mainline report", errors)
    if errors:
        return
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    summary_rows = {str(row.get("label")): row for row in payload.get("summary", []) or []}
    baseline = summary_rows.get("baseline_forecast_sft_v2")
    mainline = None
    matched_mainline_label = None
    for candidate in MAINLINE_LABEL_CANDIDATES:
        if candidate in summary_rows:
            mainline = summary_rows[candidate]
            matched_mainline_label = candidate
            break
    _require(
        baseline is not None,
        f"Mainline report missing baseline_forecast_sft_v2 summary row: {report_path}",
        errors,
    )
    _require(
        mainline is not None,
        (
            "Mainline report missing a supported predicted baseline-intensity row "
            f"({', '.join(MAINLINE_LABEL_CANDIDATES)}): {report_path}"
        ),
        errors,
    )
    if baseline is None or mainline is None:
        return
    _require(
        float(mainline["track_error_km"]) < float(baseline["track_error_km"]),
        f"Mainline report must improve baseline track_error_km: {report_path}",
        errors,
    )
    _require(
        float(mainline["mean_track_diff_vs_official_km"]) < float(baseline["mean_track_diff_vs_official_km"]),
        f"Mainline report must improve baseline mean_track_diff_vs_official_km: {report_path}",
        errors,
    )


def _validate_diagnostic_sft(config_path: Path, errors: list[str]) -> str | None:
    config = load_sft_config(config_path)
    _require(
        _same_path(config.data.dataset_root, DATASET_ROOT),
        f"Diagnostic dataset_root must be {DATASET_ROOT}, got {config.data.dataset_root}",
        errors,
    )
    _require(
        _same_path(config.data.sft_train_file, EXPECTED_DIAGNOSTIC_PATHS["train"]),
        (
            "Diagnostic train file must use the diagnostic_slot_turn_correction_only train split: "
            f"{EXPECTED_DIAGNOSTIC_PATHS['train']}"
        ),
        errors,
    )
    _require(
        _same_path(config.data.sft_eval_file, EXPECTED_DIAGNOSTIC_PATHS["val"]),
        (
            "Diagnostic eval file must use the diagnostic_slot_turn_correction_only val split: "
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
    _require(
        config.trainer.adapter_init_path is not None,
        "Slot+turn correction SFT should use staged init from the best prior slot-based adapter.",
        errors,
    )
    if config.trainer.adapter_init_path is not None:
        _require_file(config.trainer.adapter_init_path, "Diagnostic init adapter", errors)
        _require_file(
            config.trainer.adapter_init_path / "adapter_model.safetensors",
            "Diagnostic init adapter weights",
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
    _require(adapter_path.is_dir(), f"Forecast baseline adapter must be a directory: {adapter_path}", errors)
    _require_file(adapter_path / "adapter_model.safetensors", "Forecast adapter weights", errors)
    _require_file(adapter_path / "adapter_config.json", "Forecast adapter config", errors)


def _validate_canonical_paths(errors: list[str]) -> None:
    for split_name, path in EXPECTED_CANONICAL_PATHS.items():
        _require_file(path, f"Canonical {split_name} split", errors)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate Phase 2C slot+turn correction training configs and artifacts.")
    parser.add_argument("--diagnostic-config", required=True, help="Path to the slot+turn correction diagnostic SFT YAML config.")
    parser.add_argument("--forecast-config", required=True, help="Path to the forecast SFT YAML config.")
    parser.add_argument("--reward-config", required=True, help="Path to the reward YAML config.")
    parser.add_argument("--forecast-adapter", required=True, help="Path to the frozen forecast baseline adapter.")
    parser.add_argument(
        "--mainline-report",
        default=str(DEFAULT_SUCCESSFUL_MAINLINE_REPORT),
        help="Successful passed Phase 2B report used as the staging gate.",
    )
    args = parser.parse_args(argv)

    errors: list[str] = []
    _validate_dataset_ready(errors)
    _validate_canonical_paths(errors)
    _validate_current_mainline(Path(args.mainline_report).resolve(), errors)
    diagnostic_model_name = _validate_diagnostic_sft(Path(args.diagnostic_config).resolve(), errors)
    forecast_model_name = _validate_forecast_sft(Path(args.forecast_config).resolve(), errors)
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
                "Forecast SFT must start from the frozen shared base model "
                f"{EXPECTED_BASE_MODEL}, got {forecast_model_name}"
            ),
            errors,
        )

    if errors:
        for error in errors:
            print(f"[FAIL] {error}")
        return 1

    print("[OK] Phase 2C slot+turn correction preflight passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
