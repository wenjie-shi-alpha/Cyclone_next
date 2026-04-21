#!/usr/bin/env python3
"""Sweep completed track-turn checkpoints on standalone and forecast-integration gates."""

from __future__ import annotations

import argparse
import gc
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import build_forecast_prompt_overrides as prompt_builder
import eval_diagnostic_heldout as diagnostic_eval
import eval_strict_forecast_heldout as forecast_eval


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return slug or "checkpoint"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl_sample_ids(path: Path) -> list[str]:
    sample_ids: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            sample_id = str(row.get("sample_id", ""))
            if not sample_id:
                raise ValueError(f"Missing sample_id in {path}")
            sample_ids.append(sample_id)
    return sample_ids


def _select_sample_ids(
    dataset_path: Path,
    sample_count: int | None,
    sample_seed: int,
) -> list[str]:
    sample_ids = _read_jsonl_sample_ids(dataset_path)
    if sample_count is None or sample_count >= len(sample_ids):
        return sample_ids

    import random

    rng = random.Random(sample_seed)
    selected = set(rng.sample(sample_ids, sample_count))
    return [sample_id for sample_id in sample_ids if sample_id in selected]


def _parse_checkpoint_step(path: Path) -> int | None:
    match = re.fullmatch(r"checkpoint-(\d+)", path.name)
    if match is None:
        return None
    return int(match.group(1))


def _discover_adapters(run_root: Path) -> list[dict[str, Any]]:
    sft_root = run_root / "sft"
    if not sft_root.exists():
        raise FileNotFoundError(f"Missing SFT output directory: {sft_root}")

    adapters: list[dict[str, Any]] = []
    for path in sorted(sft_root.glob("checkpoint-*")):
        if not path.is_dir():
            continue
        step = _parse_checkpoint_step(path)
        if step is None:
            continue
        adapters.append(
            {
                "label": path.name,
                "adapter_path": path,
                "checkpoint_step": step,
                "kind": "checkpoint",
                "trainer_state_path": path / "trainer_state.json",
            }
        )

    final_adapter = sft_root / "final_adapter"
    if final_adapter.exists():
        adapters.append(
            {
                "label": "final_adapter",
                "adapter_path": final_adapter,
                "checkpoint_step": None,
                "kind": "final_adapter",
                "trainer_state_path": None,
            }
        )

    if not adapters:
        raise ValueError(f"No checkpoints discovered under {sft_root}")
    return adapters


def _cleanup_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _higher_is_better_gap_closed(
    baseline_value: float | None,
    new_value: float | None,
    expert_value: float | None,
) -> float | None:
    if baseline_value is None or new_value is None or expert_value is None:
        return None
    denominator = expert_value - baseline_value
    if abs(denominator) < 1e-12:
        return None
    return (new_value - baseline_value) / denominator


def _lower_is_better_gap_closed(
    baseline_value: float | None,
    new_value: float | None,
    expert_value: float | None,
) -> float | None:
    if baseline_value is None or new_value is None or expert_value is None:
        return None
    denominator = baseline_value - expert_value
    if abs(denominator) < 1e-12:
        return None
    return (baseline_value - new_value) / denominator


def _summary_row(report: dict[str, Any]) -> dict[str, Any]:
    reward = report["reward"]
    matches = report["matches"]
    errors = report["errors"]
    vs_official = report.get("vs_official") or {}
    return {
        "label": report["variant_label"],
        "variant_kind": report["variant_kind"],
        "forecast_adapter_path": report.get("forecast_adapter_path"),
        "diagnostic_adapter_path": report.get("diagnostic_adapter_path"),
        "diagnostic_prediction_mode": report.get("diagnostic_prediction_mode"),
        "sample_count": report["sample_count"],
        "reward_mean": reward["mean"],
        "coverage": matches["mean_match_coverage_vs_target"],
        "track_error_km": errors["mean_track_error_km"],
        "intensity_error_kt": errors["mean_intensity_error_kt"],
        "mean_track_diff_vs_official_km": vs_official.get("mean_track_diff_km"),
        "mean_intensity_diff_vs_official_kt": vs_official.get("mean_intensity_diff_kt"),
        "slot_time_match_rate_vs_official": vs_official.get("slot_time_match_rate"),
        "strict_parseable_rate": report["schema_rates"]["strict_forecast_parseable"],
        "prompt_override_count": report.get("prompt_override_count", 0),
    }


def _delta_or_none(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return left - right


def _augment_forecast_rows(
    summary_rows: list[dict[str, Any]],
    *,
    baseline_label: str,
    expert_label: str,
) -> list[dict[str, Any]]:
    row_by_label = {row["label"]: row for row in summary_rows}
    baseline = row_by_label.get(baseline_label)
    expert = row_by_label.get(expert_label)
    if baseline is None or expert is None:
        return summary_rows

    augmented: list[dict[str, Any]] = []
    for row in summary_rows:
        updated = dict(row)
        updated["gap_closed"] = {
            "reward_mean": _higher_is_better_gap_closed(
                baseline["reward_mean"],
                row["reward_mean"],
                expert["reward_mean"],
            ),
            "coverage": _higher_is_better_gap_closed(
                baseline["coverage"],
                row["coverage"],
                expert["coverage"],
            ),
            "track_error_km": _lower_is_better_gap_closed(
                baseline["track_error_km"],
                row["track_error_km"],
                expert["track_error_km"],
            ),
            "intensity_error_kt": _lower_is_better_gap_closed(
                baseline["intensity_error_kt"],
                row["intensity_error_kt"],
                expert["intensity_error_kt"],
            ),
            "mean_track_diff_vs_official_km": _lower_is_better_gap_closed(
                baseline["mean_track_diff_vs_official_km"],
                row["mean_track_diff_vs_official_km"],
                expert["mean_track_diff_vs_official_km"],
            ),
            "mean_intensity_diff_vs_official_kt": _lower_is_better_gap_closed(
                baseline["mean_intensity_diff_vs_official_kt"],
                row["mean_intensity_diff_vs_official_kt"],
                expert["mean_intensity_diff_vs_official_kt"],
            ),
            "slot_time_match_rate_vs_official": _higher_is_better_gap_closed(
                baseline["slot_time_match_rate_vs_official"],
                row["slot_time_match_rate_vs_official"],
                expert["slot_time_match_rate_vs_official"],
            ),
        }
        updated["delta_vs_baseline"] = {
            "reward_mean": _delta_or_none(row["reward_mean"], baseline["reward_mean"]),
            "coverage": _delta_or_none(row["coverage"], baseline["coverage"]),
            "track_error_km": _delta_or_none(row["track_error_km"], baseline["track_error_km"]),
            "intensity_error_kt": _delta_or_none(
                row["intensity_error_kt"], baseline["intensity_error_kt"]
            ),
            "mean_track_diff_vs_official_km": _delta_or_none(
                row["mean_track_diff_vs_official_km"],
                baseline["mean_track_diff_vs_official_km"],
            ),
            "mean_intensity_diff_vs_official_kt": _delta_or_none(
                row["mean_intensity_diff_vs_official_kt"],
                baseline["mean_intensity_diff_vs_official_kt"],
            ),
            "slot_time_match_rate_vs_official": _delta_or_none(
                row["slot_time_match_rate_vs_official"],
                baseline["slot_time_match_rate_vs_official"],
            ),
        }
        augmented.append(updated)
    return augmented


def _load_trainer_state_summary(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    payload = _read_json(path)
    eval_logs = [row for row in payload.get("log_history", []) if "eval_loss" in row]
    best_eval = min(eval_logs, key=lambda row: row["eval_loss"]) if eval_logs else None
    last_eval = eval_logs[-1] if eval_logs else None
    return {
        "trainer_state_path": str(path),
        "global_step": payload.get("global_step"),
        "epoch": payload.get("epoch"),
        "best_model_checkpoint": payload.get("best_model_checkpoint"),
        "best_metric": payload.get("best_metric"),
        "best_eval_loss": best_eval.get("eval_loss") if best_eval is not None else None,
        "best_eval_step": best_eval.get("step") if best_eval is not None else None,
        "best_eval_epoch": best_eval.get("epoch") if best_eval is not None else None,
        "last_eval_loss": last_eval.get("eval_loss") if last_eval is not None else None,
        "last_eval_step": last_eval.get("step") if last_eval is not None else None,
        "last_eval_epoch": last_eval.get("epoch") if last_eval is not None else None,
    }


def _standalone_summary(report: dict[str, Any]) -> dict[str, Any]:
    field_reports = report.get("field_reports", {})
    track_control = field_reports.get("track_control_signal", {})
    turning = field_reports.get("turning_signal", {})
    return {
        "json_parseable_rate": report["json_parseable_rate"],
        "joint_exact_match_rate": report["joint_exact_match_rate"],
        "mean_field_exact_accuracy": report["mean_field_exact_accuracy"],
        "mean_field_macro_f1": report["mean_field_macro_f1"],
        "track_control_signal_exact_accuracy": track_control.get("exact_accuracy"),
        "track_control_signal_macro_f1": track_control.get("macro_f1"),
        "turning_signal_exact_accuracy": turning.get("exact_accuracy"),
        "turning_signal_macro_f1": turning.get("macro_f1"),
        "prediction_distribution": {
            field_name: (field_payload.get("prediction_distribution") or {})
            for field_name, field_payload in field_reports.items()
        },
        "report_path": report.get("report_path"),
    }


def _checkpoint_sort_key(candidate: dict[str, Any]) -> tuple[int, int]:
    step = candidate.get("checkpoint_step")
    if step is None:
        return (1, 10**12)
    return (0, int(step))


def _rank_labels(
    rows: list[dict[str, Any]],
    key: str,
    *,
    higher_is_better: bool,
) -> list[str]:
    sentinel = float("-inf") if higher_is_better else float("inf")
    return [
        row["label"]
        for row in sorted(
            rows,
            key=lambda row: row[key] if row.get(key) is not None else sentinel,
            reverse=higher_is_better,
        )
    ]


def _candidate_is_guard_ok(candidate: dict[str, Any]) -> bool:
    delta = candidate["forecast"].get("delta_vs_baseline") or {}
    reward_delta = delta.get("reward_mean")
    coverage_delta = delta.get("coverage")
    return (
        reward_delta is not None
        and reward_delta >= -0.02
        and coverage_delta is not None
        and coverage_delta >= -0.02
    )


def _candidate_has_track_non_regression(candidate: dict[str, Any]) -> bool:
    delta = candidate["forecast"].get("delta_vs_baseline") or {}
    track_delta = delta.get("track_error_km")
    track_official_delta = delta.get("mean_track_diff_vs_official_km")
    return (track_delta is not None and track_delta <= 0.0) or (
        track_official_delta is not None and track_official_delta <= 0.0
    )


def _recommend_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not candidates:
        return None

    guard_ok = [candidate for candidate in candidates if _candidate_is_guard_ok(candidate)]
    preferred = [candidate for candidate in guard_ok if _candidate_has_track_non_regression(candidate)]
    pool = preferred or guard_ok or candidates
    selected = sorted(
        pool,
        key=lambda candidate: (
            -(candidate["forecast"]["delta_vs_baseline"].get("slot_time_match_rate_vs_official") or 0.0),
            candidate["forecast"].get("track_error_km")
            if candidate["forecast"].get("track_error_km") is not None
            else float("inf"),
            candidate["forecast"].get("mean_track_diff_vs_official_km")
            if candidate["forecast"].get("mean_track_diff_vs_official_km") is not None
            else float("inf"),
            -(candidate["forecast"].get("reward_mean") or 0.0),
            -(candidate["standalone"].get("track_control_signal_macro_f1") or 0.0),
            -(candidate["standalone"].get("turning_signal_macro_f1") or 0.0),
            _checkpoint_sort_key(candidate["meta"]),
        ),
    )[0]
    reason = (
        "selected from truth-guarded checkpoints with non-regressing track metrics"
        if selected in preferred
        else "selected from truth-guarded checkpoints by downstream slot-match and reward"
        if selected in guard_ok
        else "selected from all checkpoints because no candidate satisfied the truth-side guard"
    )
    return {
        "label": selected["label"],
        "adapter_path": selected["adapter_path"],
        "checkpoint_step": selected["meta"]["checkpoint_step"],
        "reason": reason,
    }


def _summary_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Track-Turn Checkpoint Sweep Summary",
        "",
        f"- generated_at_utc: `{report['generated_at_utc']}`",
        f"- source_run_root: `{report['source_run_root']}`",
        f"- diagnostic_sample_count: `{report['diagnostic_sample_count']}`",
        f"- forecast_sample_count: `{report['forecast_sample_count']}`",
        f"- recommended_checkpoint: `{(report.get('recommended_checkpoint') or {}).get('label')}`",
        "",
        "## Forecast Reference",
        "",
        "| label | reward | coverage | track err km | track diff vs official km | slot-match vs official |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report.get("reference_forecast_summary", []):
        lines.append(
            "| {label} | {reward:.4f} | {coverage:.4f} | {track:.2f} | {track_official:.2f} | {slot:.4f} |".format(
                label=row["label"],
                reward=row["reward_mean"] or 0.0,
                coverage=row["coverage"] or 0.0,
                track=row["track_error_km"] or 0.0,
                track_official=row["mean_track_diff_vs_official_km"] or 0.0,
                slot=row["slot_time_match_rate_vs_official"] or 0.0,
            )
        )
    lines.extend(
        [
            "",
            "## Candidate Ranking",
            "",
            "| label | step | standalone macro-F1 | track-control F1 | turning F1 | reward | coverage | track err km | track-official km | slot-match | reward delta | track err delta | track-official delta | slot-match delta |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for candidate in report.get("candidates", []):
        forecast = candidate["forecast"]
        standalone = candidate["standalone"]
        delta = forecast.get("delta_vs_baseline") or {}
        lines.append(
            "| {label} | {step} | {standalone_macro:.4f} | {track_f1:.4f} | {turn_f1:.4f} | {reward:.4f} | {coverage:.4f} | {track:.2f} | {track_official:.2f} | {slot:.4f} | {reward_delta:.4f} | {track_delta:.2f} | {track_official_delta:.2f} | {slot_delta:.4f} |".format(
                label=candidate["label"],
                step=candidate["meta"]["checkpoint_step"] if candidate["meta"]["checkpoint_step"] is not None else "final",
                standalone_macro=standalone.get("mean_field_macro_f1") or 0.0,
                track_f1=standalone.get("track_control_signal_macro_f1") or 0.0,
                turn_f1=standalone.get("turning_signal_macro_f1") or 0.0,
                reward=forecast.get("reward_mean") or 0.0,
                coverage=forecast.get("coverage") or 0.0,
                track=forecast.get("track_error_km") or 0.0,
                track_official=forecast.get("mean_track_diff_vs_official_km") or 0.0,
                slot=forecast.get("slot_time_match_rate_vs_official") or 0.0,
                reward_delta=delta.get("reward_mean") or 0.0,
                track_delta=delta.get("track_error_km") or 0.0,
                track_official_delta=delta.get("mean_track_diff_vs_official_km") or 0.0,
                slot_delta=delta.get("slot_time_match_rate_vs_official") or 0.0,
            )
        )
    return "\n".join(lines) + "\n"


def _report_dir(output_path: Path) -> Path:
    return output_path.parent / f"{output_path.stem}.artifacts"


def _manifest_path(output_path: Path) -> Path:
    return output_path.parent / f"{output_path.stem}.manifest.json"


def _ensure_manifest(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    if not path.exists():
        _write_json(path, payload)
        return payload

    existing = _read_json(path)
    comparable_keys = [
        "source_run_root",
        "diagnostic_config_path",
        "diagnostic_dataset_path",
        "diagnostic_train_dataset_path",
        "forecast_config_path",
        "reward_config_path",
        "forecast_adapter_path",
        "forecast_rl_dataset_path",
        "forecast_sft_dataset_path",
        "diagnostic_sample_count",
        "diagnostic_sample_seed",
        "diagnostic_sample_ids",
        "forecast_sample_count",
        "forecast_sample_seed",
        "forecast_sample_ids",
        "candidate_labels",
    ]
    mismatches = [
        key
        for key in comparable_keys
        if existing.get(key) != payload.get(key)
    ]
    if mismatches:
        mismatch_summary = ", ".join(mismatches)
        raise ValueError(
            f"Existing sweep manifest at {path} is incompatible with the current run: {mismatch_summary}"
        )
    return existing


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Sweep completed track-turn checkpoints on standalone and forecast-integration gates."
    )
    parser.add_argument("--source-run-root", required=True, help="Completed phase2 track-turn run root.")
    parser.add_argument("--diagnostic-config", required=True, help="Diagnostic SFT YAML config.")
    parser.add_argument("--diagnostic-dataset", required=True, help="Diagnostic test JSONL.")
    parser.add_argument("--diagnostic-train-dataset", required=True, help="Diagnostic train JSONL.")
    parser.add_argument("--forecast-config", required=True, help="Forecast SFT YAML config.")
    parser.add_argument("--reward-config", required=True, help="Reward YAML config.")
    parser.add_argument("--forecast-adapter", required=True, help="Forecast adapter directory.")
    parser.add_argument("--forecast-rl-dataset", required=True, help="Forecast RL test JSONL.")
    parser.add_argument("--forecast-sft-dataset", required=True, help="Forecast SFT test JSONL.")
    parser.add_argument("--diagnostic-sample-count", type=int, default=200)
    parser.add_argument("--diagnostic-sample-seed", type=int, default=3407)
    parser.add_argument("--forecast-sample-count", type=int, default=200)
    parser.add_argument("--forecast-sample-seed", type=int, default=3407)
    parser.add_argument("--diagnostic-batch-size", type=int, default=4)
    parser.add_argument("--forecast-batch-size", type=int, default=4)
    parser.add_argument("--max-prompt-tokens", type=int, default=1024)
    parser.add_argument("--diagnostic-max-new-tokens", type=int, default=256)
    parser.add_argument("--forecast-max-new-tokens", type=int, default=160)
    parser.add_argument(
        "--section-title",
        default="Track-Turn Structured Diagnostic Assessment",
        help="Heading used for injected diagnostic prompt blocks.",
    )
    parser.add_argument("--output", required=True, help="Aggregate checkpoint sweep JSON output.")
    args = parser.parse_args(argv)

    source_run_root = Path(args.source_run_root).resolve()
    diagnostic_config_path = Path(args.diagnostic_config).resolve()
    diagnostic_dataset_path = Path(args.diagnostic_dataset).resolve()
    diagnostic_train_dataset_path = Path(args.diagnostic_train_dataset).resolve()
    forecast_config_path = Path(args.forecast_config).resolve()
    reward_config_path = Path(args.reward_config).resolve()
    forecast_adapter_path = Path(args.forecast_adapter).resolve()
    forecast_rl_dataset_path = Path(args.forecast_rl_dataset).resolve()
    forecast_sft_dataset_path = Path(args.forecast_sft_dataset).resolve()
    output_path = Path(args.output).resolve()
    artifact_dir = _report_dir(output_path)
    prompt_dir = artifact_dir / "prompts"
    standalone_dir = artifact_dir / "standalone"
    forecast_dir = artifact_dir / "forecast"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    prompt_dir.mkdir(parents=True, exist_ok=True)
    standalone_dir.mkdir(parents=True, exist_ok=True)
    forecast_dir.mkdir(parents=True, exist_ok=True)

    candidates_meta = _discover_adapters(source_run_root)
    diagnostic_sample_ids = _select_sample_ids(
        diagnostic_dataset_path,
        args.diagnostic_sample_count,
        args.diagnostic_sample_seed,
    )
    forecast_sample_ids = _select_sample_ids(
        forecast_rl_dataset_path,
        args.forecast_sample_count,
        args.forecast_sample_seed,
    )
    manifest = _ensure_manifest(
        _manifest_path(output_path),
        {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_run_root": str(source_run_root),
            "diagnostic_config_path": str(diagnostic_config_path),
            "diagnostic_dataset_path": str(diagnostic_dataset_path),
            "diagnostic_train_dataset_path": str(diagnostic_train_dataset_path),
            "forecast_config_path": str(forecast_config_path),
            "reward_config_path": str(reward_config_path),
            "forecast_adapter_path": str(forecast_adapter_path),
            "forecast_rl_dataset_path": str(forecast_rl_dataset_path),
            "forecast_sft_dataset_path": str(forecast_sft_dataset_path),
            "diagnostic_sample_count": len(diagnostic_sample_ids),
            "diagnostic_sample_seed": args.diagnostic_sample_seed,
            "diagnostic_sample_ids": diagnostic_sample_ids,
            "forecast_sample_count": len(forecast_sample_ids),
            "forecast_sample_seed": args.forecast_sample_seed,
            "forecast_sample_ids": forecast_sample_ids,
            "candidate_labels": [candidate["label"] for candidate in candidates_meta],
        },
    )

    baseline_report_path = forecast_dir / "baseline_forecast_sft_v2.json"
    if baseline_report_path.exists():
        baseline_report = _read_json(baseline_report_path)
    else:
        baseline_report = forecast_eval.evaluate(
            config_path=forecast_config_path,
            reward_config_path=reward_config_path,
            adapter_path=forecast_adapter_path,
            rl_dataset_path=forecast_rl_dataset_path,
            sft_dataset_path=forecast_sft_dataset_path,
            max_samples=None,
            batch_size=args.forecast_batch_size,
            max_prompt_tokens=args.max_prompt_tokens,
            max_new_tokens=args.forecast_max_new_tokens,
            do_sample=False,
            temperature=0.7,
            top_p=0.9,
            prompt_overrides=None,
            sample_ids=forecast_sample_ids,
            include_samples=False,
        )
        baseline_report["variant_label"] = "baseline_forecast_sft_v2"
        baseline_report["variant_kind"] = "baseline"
        baseline_report["forecast_adapter_path"] = str(forecast_adapter_path)
        baseline_report["diagnostic_adapter_path"] = None
        baseline_report["diagnostic_prediction_mode"] = None
        _write_json(baseline_report_path, baseline_report)
        _cleanup_cuda()

    oracle_prompt_path = prompt_dir / "oracle_track_turn_diagnostics_plus_forecast.json"
    if oracle_prompt_path.exists():
        oracle_prompt_payload = _read_json(oracle_prompt_path)
    else:
        oracle_prompt_payload = prompt_builder.build_prompt_overrides(
            forecast_rl_dataset_path=forecast_rl_dataset_path,
            diagnostic_dataset_path=diagnostic_dataset_path,
            injection_mode="oracle",
            sample_count=args.forecast_sample_count,
            sample_seed=args.forecast_sample_seed,
            diagnostic_config_path=None,
            diagnostic_adapter_path=None,
            diagnostic_prediction_mode="rule_echo",
            diagnostic_train_dataset_path=diagnostic_train_dataset_path,
            batch_size=args.diagnostic_batch_size,
            max_prompt_tokens=args.max_prompt_tokens,
            max_new_tokens=args.diagnostic_max_new_tokens,
            do_sample=False,
            temperature=0.7,
            top_p=0.9,
            section_title=args.section_title,
        )
        _write_json(oracle_prompt_path, oracle_prompt_payload)
        _cleanup_cuda()

    oracle_report_path = forecast_dir / "oracle_track_turn_diagnostics_plus_forecast.json"
    if oracle_report_path.exists():
        oracle_report = _read_json(oracle_report_path)
    else:
        oracle_report = forecast_eval.evaluate(
            config_path=forecast_config_path,
            reward_config_path=reward_config_path,
            adapter_path=forecast_adapter_path,
            rl_dataset_path=forecast_rl_dataset_path,
            sft_dataset_path=forecast_sft_dataset_path,
            max_samples=None,
            batch_size=args.forecast_batch_size,
            max_prompt_tokens=args.max_prompt_tokens,
            max_new_tokens=args.forecast_max_new_tokens,
            do_sample=False,
            temperature=0.7,
            top_p=0.9,
            prompt_overrides=oracle_prompt_payload["overrides"],
            sample_ids=forecast_sample_ids,
            include_samples=False,
        )
        oracle_report["variant_label"] = "oracle_track_turn_diagnostics_plus_forecast"
        oracle_report["variant_kind"] = "oracle"
        oracle_report["forecast_adapter_path"] = str(forecast_adapter_path)
        oracle_report["diagnostic_adapter_path"] = None
        oracle_report["diagnostic_prediction_mode"] = "oracle"
        oracle_report["prompt_override_path"] = str(oracle_prompt_path)
        _write_json(oracle_report_path, oracle_report)
        _cleanup_cuda()

    expert_report_path = forecast_dir / "expert_official.json"
    if expert_report_path.exists():
        expert_report = _read_json(expert_report_path)
    else:
        official_samples = forecast_eval.load_eval_samples(
            rl_dataset_path=forecast_rl_dataset_path,
            sft_dataset_path=forecast_sft_dataset_path,
            max_samples=None,
            sample_ids=forecast_sample_ids,
            prompt_overrides=None,
        )
        official_outputs = [(sample["target_text"] or "") for sample in official_samples]
        expert_report = forecast_eval.evaluate_outputs(
            config_path=None,
            reward_config_path=reward_config_path,
            adapter_path=None,
            rl_dataset_path=forecast_rl_dataset_path,
            sft_dataset_path=forecast_sft_dataset_path,
            prepared_samples=official_samples,
            outputs=official_outputs,
            batch_size=0,
            max_prompt_tokens=args.max_prompt_tokens,
            max_new_tokens=0,
            do_sample=False,
            temperature=0.7,
            top_p=0.9,
            include_samples=False,
        )
        expert_report["variant_label"] = "expert_official"
        expert_report["variant_kind"] = "expert_official"
        expert_report["forecast_adapter_path"] = str(forecast_adapter_path)
        expert_report["diagnostic_adapter_path"] = None
        expert_report["diagnostic_prediction_mode"] = None
        _write_json(expert_report_path, expert_report)
        _cleanup_cuda()

    forecast_rows = _augment_forecast_rows(
        [
            _summary_row(baseline_report),
            _summary_row(oracle_report),
            _summary_row(expert_report),
        ],
        baseline_label="baseline_forecast_sft_v2",
        expert_label="expert_official",
    )
    forecast_row_by_label = {row["label"]: row for row in forecast_rows}

    candidates: list[dict[str, Any]] = []
    for candidate_meta in sorted(candidates_meta, key=_checkpoint_sort_key):
        label = candidate_meta["label"]
        adapter_path = candidate_meta["adapter_path"]
        slug = _slugify(label)
        standalone_report_path = standalone_dir / f"{slug}.json"
        if standalone_report_path.exists():
            standalone_report = _read_json(standalone_report_path)
        else:
            standalone_report = diagnostic_eval.evaluate(
                config_path=diagnostic_config_path,
                adapter_path=adapter_path,
                dataset_path=diagnostic_dataset_path,
                max_samples=None,
                batch_size=args.diagnostic_batch_size,
                max_prompt_tokens=args.max_prompt_tokens,
                max_new_tokens=args.diagnostic_max_new_tokens,
                do_sample=False,
                temperature=0.7,
                top_p=0.9,
                prediction_mode="adapter",
                train_dataset_path=diagnostic_train_dataset_path,
                sample_ids=diagnostic_sample_ids,
                include_samples=False,
            )
            standalone_report["model_label"] = label
            standalone_report["prediction_mode"] = "adapter"
            standalone_report["adapter_path"] = str(adapter_path)
            standalone_report["model_kind"] = "adapter"
            standalone_report["report_path"] = str(standalone_report_path)
            _write_json(standalone_report_path, standalone_report)
            _cleanup_cuda()

        predicted_prompt_path = prompt_dir / f"{slug}.json"
        if predicted_prompt_path.exists():
            predicted_prompt_payload = _read_json(predicted_prompt_path)
        else:
            predicted_prompt_payload = prompt_builder.build_prompt_overrides(
                forecast_rl_dataset_path=forecast_rl_dataset_path,
                diagnostic_dataset_path=diagnostic_dataset_path,
                injection_mode="predicted",
                sample_count=args.forecast_sample_count,
                sample_seed=args.forecast_sample_seed,
                diagnostic_config_path=diagnostic_config_path,
                diagnostic_adapter_path=adapter_path,
                diagnostic_prediction_mode="adapter",
                diagnostic_train_dataset_path=diagnostic_train_dataset_path,
                batch_size=args.diagnostic_batch_size,
                max_prompt_tokens=args.max_prompt_tokens,
                max_new_tokens=args.diagnostic_max_new_tokens,
                do_sample=False,
                temperature=0.7,
                top_p=0.9,
                section_title=args.section_title,
            )
            _write_json(predicted_prompt_path, predicted_prompt_payload)
            _cleanup_cuda()

        predicted_report_path = forecast_dir / f"{slug}.json"
        if predicted_report_path.exists():
            predicted_report = _read_json(predicted_report_path)
        else:
            predicted_report = forecast_eval.evaluate(
                config_path=forecast_config_path,
                reward_config_path=reward_config_path,
                adapter_path=forecast_adapter_path,
                rl_dataset_path=forecast_rl_dataset_path,
                sft_dataset_path=forecast_sft_dataset_path,
                max_samples=None,
                batch_size=args.forecast_batch_size,
                max_prompt_tokens=args.max_prompt_tokens,
                max_new_tokens=args.forecast_max_new_tokens,
                do_sample=False,
                temperature=0.7,
                top_p=0.9,
                prompt_overrides=predicted_prompt_payload["overrides"],
                sample_ids=forecast_sample_ids,
                include_samples=False,
            )
            predicted_report["variant_label"] = f"predicted_{label}"
            predicted_report["variant_kind"] = "predicted"
            predicted_report["forecast_adapter_path"] = str(forecast_adapter_path)
            predicted_report["diagnostic_adapter_path"] = str(adapter_path)
            predicted_report["diagnostic_prediction_mode"] = "adapter"
            predicted_report["prompt_override_path"] = str(predicted_prompt_path)
            _write_json(predicted_report_path, predicted_report)
            _cleanup_cuda()

        augmented_predicted_row = _augment_forecast_rows(
            [
                forecast_row_by_label["baseline_forecast_sft_v2"],
                _summary_row(predicted_report),
                forecast_row_by_label["expert_official"],
            ],
            baseline_label="baseline_forecast_sft_v2",
            expert_label="expert_official",
        )[1]

        candidates.append(
            {
                "label": label,
                "adapter_path": str(adapter_path),
                "meta": {
                    "kind": candidate_meta["kind"],
                    "checkpoint_step": candidate_meta["checkpoint_step"],
                    "trainer_state": _load_trainer_state_summary(candidate_meta["trainer_state_path"]),
                },
                "standalone": _standalone_summary(standalone_report),
                "forecast": augmented_predicted_row,
            }
        )
        partial_report = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_run_root": str(source_run_root),
            "diagnostic_sample_count": manifest["diagnostic_sample_count"],
            "forecast_sample_count": manifest["forecast_sample_count"],
            "reference_forecast_summary": forecast_rows,
            "candidates": candidates,
            "recommended_checkpoint": _recommend_candidate(candidates),
        }
        _write_json(output_path, partial_report)
        _write_text(output_path.with_suffix(".summary.md"), _summary_markdown(partial_report))

    candidate_standalone_rows = [
        {
            "label": candidate["label"],
            "mean_field_macro_f1": candidate["standalone"]["mean_field_macro_f1"],
            "track_control_signal_macro_f1": candidate["standalone"]["track_control_signal_macro_f1"],
            "turning_signal_macro_f1": candidate["standalone"]["turning_signal_macro_f1"],
        }
        for candidate in candidates
    ]
    candidate_forecast_rows = [candidate["forecast"] for candidate in candidates]
    final_report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_run_root": str(source_run_root),
        "diagnostic_config_path": str(diagnostic_config_path),
        "diagnostic_dataset_path": str(diagnostic_dataset_path),
        "diagnostic_train_dataset_path": str(diagnostic_train_dataset_path),
        "forecast_config_path": str(forecast_config_path),
        "reward_config_path": str(reward_config_path),
        "forecast_adapter_path": str(forecast_adapter_path),
        "forecast_rl_dataset_path": str(forecast_rl_dataset_path),
        "forecast_sft_dataset_path": str(forecast_sft_dataset_path),
        "diagnostic_sample_count": manifest["diagnostic_sample_count"],
        "diagnostic_sample_seed": manifest["diagnostic_sample_seed"],
        "diagnostic_sample_ids": manifest["diagnostic_sample_ids"],
        "forecast_sample_count": manifest["forecast_sample_count"],
        "forecast_sample_seed": manifest["forecast_sample_seed"],
        "forecast_sample_ids": manifest["forecast_sample_ids"],
        "reference_forecast_summary": forecast_rows,
        "candidates": candidates,
        "rankings": {
            "standalone_macro_f1_desc": _rank_labels(
                candidate_standalone_rows,
                "mean_field_macro_f1",
                higher_is_better=True,
            ),
            "track_control_macro_f1_desc": _rank_labels(
                candidate_standalone_rows,
                "track_control_signal_macro_f1",
                higher_is_better=True,
            ),
            "turning_macro_f1_desc": _rank_labels(
                candidate_standalone_rows,
                "turning_signal_macro_f1",
                higher_is_better=True,
            ),
            "predicted_reward_desc": _rank_labels(
                candidate_forecast_rows,
                "reward_mean",
                higher_is_better=True,
            ),
            "predicted_coverage_desc": _rank_labels(
                candidate_forecast_rows,
                "coverage",
                higher_is_better=True,
            ),
            "predicted_track_error_asc": _rank_labels(
                candidate_forecast_rows,
                "track_error_km",
                higher_is_better=False,
            ),
            "predicted_track_official_asc": _rank_labels(
                candidate_forecast_rows,
                "mean_track_diff_vs_official_km",
                higher_is_better=False,
            ),
            "predicted_slot_match_desc": _rank_labels(
                candidate_forecast_rows,
                "slot_time_match_rate_vs_official",
                higher_is_better=True,
            ),
        },
        "recommended_checkpoint": _recommend_candidate(candidates),
    }
    _write_json(output_path, final_report)
    _write_text(output_path.with_suffix(".summary.md"), _summary_markdown(final_report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
