#!/usr/bin/env python3
"""Compare baseline/oracle/predicted diagnostic injections on strict-forecast evaluation."""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import build_forecast_prompt_overrides as prompt_builder
import eval_strict_forecast_heldout as forecast_eval


BASE_MODEL_SENTINELS = {"", "base", "none", "no_adapter", "null"}


def _slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return slug or "variant"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


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
    rl_dataset_path: Path,
    sample_count: int | None,
    sample_seed: int,
) -> list[str]:
    sample_ids = _read_jsonl_sample_ids(rl_dataset_path)
    if sample_count is None or sample_count >= len(sample_ids):
        return sample_ids

    rng = random.Random(sample_seed)
    selected = set(rng.sample(sample_ids, sample_count))
    return [sample_id for sample_id in sample_ids if sample_id in selected]


def _parse_adapter_path(raw_adapter: str | None) -> Path | None:
    if raw_adapter is None or raw_adapter.strip().lower() in BASE_MODEL_SENTINELS:
        return None
    adapter_path = Path(raw_adapter)
    if not adapter_path.is_absolute():
        adapter_path = (ROOT / adapter_path).resolve()
    if not adapter_path.exists():
        raise FileNotFoundError(f"Adapter path does not exist: {adapter_path}")
    return adapter_path


def _model_dir(output_path: Path) -> Path:
    return output_path.parent / f"{output_path.stem}.models"


def _prompt_dir(output_path: Path) -> Path:
    return output_path.parent / f"{output_path.stem}.prompts"


def _build_variant_report_path(model_dir: Path, index: int, label: str) -> Path:
    return model_dir / f"{index:02d}_{_slugify(label)}.json"


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


def _rank_models(
    summary_rows: list[dict[str, Any]],
    key: str,
    *,
    higher_is_better: bool,
) -> list[str]:
    sentinel = float("-inf") if higher_is_better else float("inf")
    return [
        row["label"]
        for row in sorted(
            summary_rows,
            key=lambda row: row[key] if row.get(key) is not None else sentinel,
            reverse=higher_is_better,
        )
    ]


def _delta_or_none(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return left - right


def _augment_summary_with_gap_closing(
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


def _summary_markdown(comparison_report: dict[str, Any]) -> str:
    lines = [
        "# Forecast Diagnostic Integration Summary",
        "",
        f"- generated_at_utc: `{comparison_report['generated_at_utc']}`",
        f"- forecast_rl_dataset: `{comparison_report['forecast_rl_dataset_path']}`",
        f"- forecast_sft_dataset: `{comparison_report['forecast_sft_dataset_path']}`",
        f"- diagnostic_dataset: `{comparison_report['diagnostic_dataset_path']}`",
        f"- sample_count: `{comparison_report['sample_count']}`",
        f"- baseline_label: `{comparison_report['baseline_label']}`",
        f"- expert_label: `{comparison_report['expert_label']}`",
        "",
        "| label | kind | reward | coverage | track err km | intensity err kt | track diff vs official km | intensity diff vs official kt | slot-time match vs official |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in comparison_report.get("summary", []):
        lines.append(
            "| {label} | {kind} | {reward:.4f} | {coverage:.4f} | {track:.2f} | {intensity:.2f} | {track_official:.2f} | {intensity_official:.2f} | {slot:.4f} |".format(
                label=row["label"],
                kind=row["variant_kind"],
                reward=row["reward_mean"] or 0.0,
                coverage=row["coverage"] or 0.0,
                track=row["track_error_km"] or 0.0,
                intensity=row["intensity_error_kt"] or 0.0,
                track_official=row["mean_track_diff_vs_official_km"] or 0.0,
                intensity_official=row["mean_intensity_diff_vs_official_kt"] or 0.0,
                slot=row["slot_time_match_rate_vs_official"] or 0.0,
            )
        )
    lines.extend(
        [
            "",
            "| label | reward gap closed | coverage gap closed | track gap closed | intensity gap closed | track-official gap closed | intensity-official gap closed | slot-match gap closed |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in comparison_report.get("summary", []):
        gap = row.get("gap_closed") or {}
        lines.append(
            "| {label} | {reward:.4f} | {coverage:.4f} | {track:.4f} | {intensity:.4f} | {track_official:.4f} | {intensity_official:.4f} | {slot:.4f} |".format(
                label=row["label"],
                reward=gap.get("reward_mean") or 0.0,
                coverage=gap.get("coverage") or 0.0,
                track=gap.get("track_error_km") or 0.0,
                intensity=gap.get("intensity_error_kt") or 0.0,
                track_official=gap.get("mean_track_diff_vs_official_km") or 0.0,
                intensity_official=gap.get("mean_intensity_diff_vs_official_kt") or 0.0,
                slot=gap.get("slot_time_match_rate_vs_official") or 0.0,
            )
        )
    return "\n".join(lines) + "\n"


def _build_report(
    *,
    output_path: Path,
    prompt_paths: dict[str, str],
    args: argparse.Namespace,
    model_reports: list[dict[str, Any]],
) -> dict[str, Any]:
    summary_rows = [_summary_row(report) for report in model_reports]
    summary_rows = _augment_summary_with_gap_closing(
        summary_rows,
        baseline_label=args.baseline_label,
        expert_label=args.official_label,
    )
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_path": str(output_path),
        "forecast_config_path": str(Path(args.forecast_config).resolve()),
        "reward_config_path": str(Path(args.reward_config).resolve()),
        "forecast_adapter_path": (
            str(_parse_adapter_path(args.forecast_adapter))
            if _parse_adapter_path(args.forecast_adapter) is not None
            else None
        ),
        "forecast_rl_dataset_path": str(Path(args.forecast_rl_dataset).resolve()),
        "forecast_sft_dataset_path": str(Path(args.forecast_sft_dataset).resolve()),
        "diagnostic_dataset_path": str(Path(args.diagnostic_dataset).resolve()),
        "diagnostic_config_path": (
            str(Path(args.diagnostic_config).resolve()) if args.diagnostic_config else None
        ),
        "diagnostic_adapter_path": (
            str(Path(args.diagnostic_adapter).resolve()) if args.diagnostic_adapter else None
        ),
        "diagnostic_prediction_mode": args.diagnostic_prediction_mode,
        "sample_count": len(model_reports[0]["samples"]) if model_reports and model_reports[0].get("samples") else model_reports[0]["sample_count"] if model_reports else 0,
        "requested_sample_count": args.sample_count,
        "sample_seed": args.sample_seed,
        "baseline_label": args.baseline_label,
        "expert_label": args.official_label,
        "prompt_override_paths": prompt_paths,
        "summary": summary_rows,
        "rankings": {
            "reward_mean_desc": _rank_models(summary_rows, "reward_mean", higher_is_better=True),
            "coverage_desc": _rank_models(summary_rows, "coverage", higher_is_better=True),
            "track_error_asc": _rank_models(summary_rows, "track_error_km", higher_is_better=False),
            "intensity_error_asc": _rank_models(
                summary_rows, "intensity_error_kt", higher_is_better=False
            ),
            "track_diff_vs_official_asc": _rank_models(
                summary_rows,
                "mean_track_diff_vs_official_km",
                higher_is_better=False,
            ),
            "intensity_diff_vs_official_asc": _rank_models(
                summary_rows,
                "mean_intensity_diff_vs_official_kt",
                higher_is_better=False,
            ),
            "slot_time_match_vs_official_desc": _rank_models(
                summary_rows,
                "slot_time_match_rate_vs_official",
                higher_is_better=True,
            ),
        },
        "models": model_reports,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare baseline/oracle/predicted diagnostic injections on strict-forecast evaluation."
    )
    parser.add_argument("--forecast-config", required=True, help="Forecast SFT YAML config.")
    parser.add_argument(
        "--reward-config",
        required=True,
        help="Reward config used by forecast held-out evaluation.",
    )
    parser.add_argument(
        "--forecast-adapter",
        required=True,
        help="Forecast adapter path, or 'none' to use the base model.",
    )
    parser.add_argument("--forecast-rl-dataset", required=True, help="Forecast RL JSONL split.")
    parser.add_argument("--forecast-sft-dataset", required=True, help="Forecast SFT JSONL split.")
    parser.add_argument("--diagnostic-dataset", required=True, help="Diagnostic SFT JSONL split.")
    parser.add_argument("--diagnostic-config", help="Diagnostic SFT YAML config.")
    parser.add_argument("--diagnostic-adapter", help="Diagnostic adapter path.")
    parser.add_argument(
        "--diagnostic-train-dataset",
        help="Optional train split used for majority-label diagnostic prediction.",
    )
    parser.add_argument(
        "--diagnostic-prediction-mode",
        choices=["adapter", "base_model", "majority_label", "rule_echo"],
        default="adapter",
        help="How predicted diagnostics should be produced.",
    )
    parser.add_argument("--sample-count", type=int, default=256)
    parser.add_argument("--sample-seed", type=int, default=3407)
    parser.add_argument("--forecast-batch-size", type=int, default=4)
    parser.add_argument("--diagnostic-batch-size", type=int, default=4)
    parser.add_argument("--max-prompt-tokens", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--diagnostic-max-new-tokens", type=int, default=256)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument(
        "--section-title",
        default="Structured Diagnostic Assessment",
        help="Heading used for injected diagnostic prompt blocks.",
    )
    parser.add_argument("--baseline-label", default="baseline_forecast")
    parser.add_argument("--oracle-label", default="oracle_diagnostics_plus_forecast")
    parser.add_argument("--predicted-label", default="predicted_diagnostics_plus_forecast")
    parser.add_argument("--official-label", default="expert_official")
    parser.add_argument("--skip-oracle", action="store_true")
    parser.add_argument("--skip-predicted", action="store_true")
    parser.add_argument("--skip-official", action="store_true")
    parser.add_argument("--output", required=True, help="JSON comparison report path.")
    args = parser.parse_args(argv)

    if not args.skip_predicted:
        if args.diagnostic_prediction_mode in {"adapter", "base_model"} and not args.diagnostic_config:
            raise ValueError("--diagnostic-config is required for model-based predicted diagnostics.")
        if args.diagnostic_prediction_mode == "adapter" and not args.diagnostic_adapter:
            raise ValueError("--diagnostic-adapter is required for adapter-based predicted diagnostics.")

    forecast_adapter_path = _parse_adapter_path(args.forecast_adapter)
    forecast_config_path = Path(args.forecast_config).resolve()
    reward_config_path = Path(args.reward_config).resolve()
    forecast_rl_dataset_path = Path(args.forecast_rl_dataset).resolve()
    forecast_sft_dataset_path = Path(args.forecast_sft_dataset).resolve()
    diagnostic_dataset_path = Path(args.diagnostic_dataset).resolve()
    diagnostic_config_path = Path(args.diagnostic_config).resolve() if args.diagnostic_config else None
    diagnostic_adapter_path = (
        Path(args.diagnostic_adapter).resolve() if args.diagnostic_adapter else None
    )
    diagnostic_train_dataset_path = (
        Path(args.diagnostic_train_dataset).resolve()
        if args.diagnostic_train_dataset
        else None
    )

    sample_ids = _select_sample_ids(
        forecast_rl_dataset_path,
        args.sample_count,
        args.sample_seed,
    )
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model_dir = _model_dir(output_path)
    prompt_dir = _prompt_dir(output_path)
    model_dir.mkdir(parents=True, exist_ok=True)
    prompt_dir.mkdir(parents=True, exist_ok=True)

    print(
        "[compare_diagnostic_forecast_integration] "
        f"sample_count={len(sample_ids)} output={output_path}",
        flush=True,
    )

    prompt_paths: dict[str, str] = {}
    model_reports: list[dict[str, Any]] = []

    print(
        "[compare_diagnostic_forecast_integration] "
        f"starting baseline variant={args.baseline_label}",
        flush=True,
    )
    baseline_report = forecast_eval.evaluate(
        config_path=forecast_config_path,
        reward_config_path=reward_config_path,
        adapter_path=forecast_adapter_path,
        rl_dataset_path=forecast_rl_dataset_path,
        sft_dataset_path=forecast_sft_dataset_path,
        max_samples=None,
        batch_size=args.forecast_batch_size,
        max_prompt_tokens=args.max_prompt_tokens,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
        prompt_overrides=None,
        sample_ids=sample_ids,
        include_samples=True,
    )
    baseline_report["variant_label"] = args.baseline_label
    baseline_report["variant_kind"] = "baseline"
    baseline_report["forecast_adapter_path"] = (
        str(forecast_adapter_path) if forecast_adapter_path is not None else None
    )
    baseline_report["diagnostic_adapter_path"] = None
    baseline_report["diagnostic_prediction_mode"] = None
    baseline_report_path = _build_variant_report_path(model_dir, 1, args.baseline_label)
    _write_json(baseline_report_path, baseline_report)
    model_reports.append(baseline_report)
    print(
        "[compare_diagnostic_forecast_integration] "
        f"completed baseline variant={args.baseline_label} report={baseline_report_path}",
        flush=True,
    )

    next_index = 2
    if not args.skip_oracle:
        print(
            "[compare_diagnostic_forecast_integration] "
            f"building oracle prompt overrides label={args.oracle_label}",
            flush=True,
        )
        oracle_prompt_payload = prompt_builder.build_prompt_overrides(
            forecast_rl_dataset_path=forecast_rl_dataset_path,
            diagnostic_dataset_path=diagnostic_dataset_path,
            injection_mode="oracle",
            sample_count=args.sample_count,
            sample_seed=args.sample_seed,
            diagnostic_config_path=None,
            diagnostic_adapter_path=None,
            diagnostic_prediction_mode="rule_echo",
            diagnostic_train_dataset_path=diagnostic_train_dataset_path,
            batch_size=args.diagnostic_batch_size,
            max_prompt_tokens=args.max_prompt_tokens,
            max_new_tokens=args.diagnostic_max_new_tokens,
            do_sample=args.do_sample,
            temperature=args.temperature,
            top_p=args.top_p,
            section_title=args.section_title,
        )
        oracle_prompt_path = prompt_dir / f"{_slugify(args.oracle_label)}.json"
        _write_json(oracle_prompt_path, oracle_prompt_payload)
        prompt_paths[args.oracle_label] = str(oracle_prompt_path)

        print(
            "[compare_diagnostic_forecast_integration] "
            f"starting oracle variant={args.oracle_label}",
            flush=True,
        )
        oracle_report = forecast_eval.evaluate(
            config_path=forecast_config_path,
            reward_config_path=reward_config_path,
            adapter_path=forecast_adapter_path,
            rl_dataset_path=forecast_rl_dataset_path,
            sft_dataset_path=forecast_sft_dataset_path,
            max_samples=None,
            batch_size=args.forecast_batch_size,
            max_prompt_tokens=args.max_prompt_tokens,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.do_sample,
            temperature=args.temperature,
            top_p=args.top_p,
            prompt_overrides=oracle_prompt_payload["overrides"],
            sample_ids=sample_ids,
            include_samples=True,
        )
        oracle_report["variant_label"] = args.oracle_label
        oracle_report["variant_kind"] = "oracle"
        oracle_report["forecast_adapter_path"] = (
            str(forecast_adapter_path) if forecast_adapter_path is not None else None
        )
        oracle_report["diagnostic_adapter_path"] = None
        oracle_report["diagnostic_prediction_mode"] = "oracle"
        oracle_report["prompt_override_path"] = str(oracle_prompt_path)
        oracle_report_path = _build_variant_report_path(model_dir, next_index, args.oracle_label)
        _write_json(oracle_report_path, oracle_report)
        model_reports.append(oracle_report)
        next_index += 1
        print(
            "[compare_diagnostic_forecast_integration] "
            f"completed oracle variant={args.oracle_label} report={oracle_report_path}",
            flush=True,
        )

    if not args.skip_predicted:
        print(
            "[compare_diagnostic_forecast_integration] "
            f"building predicted prompt overrides label={args.predicted_label}",
            flush=True,
        )
        predicted_prompt_payload = prompt_builder.build_prompt_overrides(
            forecast_rl_dataset_path=forecast_rl_dataset_path,
            diagnostic_dataset_path=diagnostic_dataset_path,
            injection_mode="predicted",
            sample_count=args.sample_count,
            sample_seed=args.sample_seed,
            diagnostic_config_path=diagnostic_config_path,
            diagnostic_adapter_path=diagnostic_adapter_path,
            diagnostic_prediction_mode=args.diagnostic_prediction_mode,
            diagnostic_train_dataset_path=diagnostic_train_dataset_path,
            batch_size=args.diagnostic_batch_size,
            max_prompt_tokens=args.max_prompt_tokens,
            max_new_tokens=args.diagnostic_max_new_tokens,
            do_sample=args.do_sample,
            temperature=args.temperature,
            top_p=args.top_p,
            section_title=args.section_title,
        )
        predicted_prompt_path = prompt_dir / f"{_slugify(args.predicted_label)}.json"
        _write_json(predicted_prompt_path, predicted_prompt_payload)
        prompt_paths[args.predicted_label] = str(predicted_prompt_path)

        print(
            "[compare_diagnostic_forecast_integration] "
            f"starting predicted variant={args.predicted_label}",
            flush=True,
        )
        predicted_report = forecast_eval.evaluate(
            config_path=forecast_config_path,
            reward_config_path=reward_config_path,
            adapter_path=forecast_adapter_path,
            rl_dataset_path=forecast_rl_dataset_path,
            sft_dataset_path=forecast_sft_dataset_path,
            max_samples=None,
            batch_size=args.forecast_batch_size,
            max_prompt_tokens=args.max_prompt_tokens,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.do_sample,
            temperature=args.temperature,
            top_p=args.top_p,
            prompt_overrides=predicted_prompt_payload["overrides"],
            sample_ids=sample_ids,
            include_samples=True,
        )
        predicted_report["variant_label"] = args.predicted_label
        predicted_report["variant_kind"] = "predicted"
        predicted_report["forecast_adapter_path"] = (
            str(forecast_adapter_path) if forecast_adapter_path is not None else None
        )
        predicted_report["diagnostic_adapter_path"] = (
            str(diagnostic_adapter_path) if diagnostic_adapter_path is not None else None
        )
        predicted_report["diagnostic_prediction_mode"] = args.diagnostic_prediction_mode
        predicted_report["prompt_override_path"] = str(predicted_prompt_path)
        predicted_report_path = _build_variant_report_path(
            model_dir,
            next_index,
            args.predicted_label,
        )
        _write_json(predicted_report_path, predicted_report)
        model_reports.append(predicted_report)
        next_index += 1
        print(
            "[compare_diagnostic_forecast_integration] "
            f"completed predicted variant={args.predicted_label} report={predicted_report_path}",
            flush=True,
        )

    if not args.skip_official:
        print(
            "[compare_diagnostic_forecast_integration] "
            f"starting official reference variant={args.official_label}",
            flush=True,
        )
        official_samples = forecast_eval.load_eval_samples(
            rl_dataset_path=forecast_rl_dataset_path,
            sft_dataset_path=forecast_sft_dataset_path,
            max_samples=None,
            sample_ids=sample_ids,
            prompt_overrides=None,
        )
        official_outputs = [(sample["target_text"] or "") for sample in official_samples]
        official_report = forecast_eval.evaluate_outputs(
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
            temperature=0.0,
            top_p=1.0,
            include_samples=True,
        )
        official_report["variant_label"] = args.official_label
        official_report["variant_kind"] = "expert_official"
        official_report["forecast_adapter_path"] = None
        official_report["diagnostic_adapter_path"] = None
        official_report["diagnostic_prediction_mode"] = None
        official_report_path = _build_variant_report_path(model_dir, next_index, args.official_label)
        _write_json(official_report_path, official_report)
        model_reports.append(official_report)
        print(
            "[compare_diagnostic_forecast_integration] "
            f"completed official reference variant={args.official_label} report={official_report_path}",
            flush=True,
        )

    comparison_report = _build_report(
        output_path=output_path,
        prompt_paths=prompt_paths,
        args=args,
        model_reports=model_reports,
    )
    _write_json(output_path, comparison_report)
    _write_text(output_path.with_suffix(".summary.md"), _summary_markdown(comparison_report))
    print(
        "[compare_diagnostic_forecast_integration] "
        f"completed comparison output={output_path}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
