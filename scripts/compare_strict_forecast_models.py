#!/usr/bin/env python3
"""Compare multiple strict-forecast models on one held-out subset."""

from __future__ import annotations

import argparse
import gc
import json
import random
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

from eval_strict_forecast_heldout import evaluate


BASE_MODEL_SENTINELS = {"", "base", "none", "no_adapter", "null"}


def _slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return slug or "model"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def _model_output_dir(output_path: Path) -> Path:
    return output_path.parent / f"{output_path.stem}.models"


def _model_report_path(model_dir: Path, index: int, label: str) -> Path:
    return model_dir / f"{index:02d}_{_slugify(label)}.json"


def _build_manifest(
    *,
    config_path: Path,
    reward_config_path: Path,
    rl_dataset_path: Path,
    sft_dataset_path: Path | None,
    sample_count: int | None,
    sample_seed: int,
    sample_ids: list[str],
    max_prompt_tokens: int,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    model_specs: list[dict[str, Any]],
    batch_size: int,
) -> dict[str, Any]:
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path),
        "reward_config_path": str(reward_config_path),
        "rl_dataset_path": str(rl_dataset_path),
        "sft_dataset_path": str(sft_dataset_path) if sft_dataset_path is not None else None,
        "sample_count": len(sample_ids),
        "requested_sample_count": sample_count,
        "sample_seed": sample_seed,
        "sample_ids": sample_ids,
        "max_prompt_tokens": max_prompt_tokens,
        "max_new_tokens": max_new_tokens,
        "decode": {
            "do_sample": do_sample,
            "temperature": temperature,
            "top_p": top_p,
        },
        "models": [
            {
                "label": spec["label"],
                "adapter_path": (
                    str(spec["adapter_path"]) if spec["adapter_path"] is not None else None
                ),
            }
            for spec in model_specs
        ],
        "runtime": {
            "requested_batch_size": batch_size,
        },
    }


def _ensure_manifest(manifest_path: Path, current_manifest: dict[str, Any]) -> dict[str, Any]:
    if not manifest_path.exists():
        _write_json(manifest_path, current_manifest)
        return current_manifest

    existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    comparable_keys = [
        "config_path",
        "reward_config_path",
        "rl_dataset_path",
        "sft_dataset_path",
        "sample_count",
        "sample_seed",
        "sample_ids",
        "max_prompt_tokens",
        "max_new_tokens",
        "decode",
        "models",
    ]
    mismatches = [
        key
        for key in comparable_keys
        if existing_manifest.get(key) != current_manifest.get(key)
    ]
    if mismatches:
        mismatch_summary = ", ".join(mismatches)
        raise ValueError(
            f"Existing manifest at {manifest_path} is incompatible with the current run: "
            f"{mismatch_summary}"
        )

    if existing_manifest.get("runtime", {}).get("requested_batch_size") != current_manifest.get(
        "runtime", {}
    ).get("requested_batch_size"):
        existing_manifest["runtime"] = current_manifest["runtime"]
        existing_manifest["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
        _write_json(manifest_path, existing_manifest)

    return existing_manifest


def _load_cached_model_report(
    report_path: Path,
    *,
    model_spec: dict[str, Any],
    manifest: dict[str, Any],
) -> dict[str, Any] | None:
    if not report_path.exists():
        return None

    report = json.loads(report_path.read_text(encoding="utf-8"))
    expected_adapter_path = (
        str(model_spec["adapter_path"]) if model_spec["adapter_path"] is not None else None
    )
    expected_sample_ids = manifest["sample_ids"]
    cached_sample_ids = [
        str(sample.get("sample_id", "")) for sample in report.get("samples", [])
    ]
    checks = {
        "model_label": model_spec["label"],
        "adapter_path": expected_adapter_path,
        "config_path": manifest["config_path"],
        "reward_config_path": manifest["reward_config_path"],
        "rl_dataset_path": manifest["rl_dataset_path"],
        "sft_dataset_path": manifest["sft_dataset_path"],
        "sample_count": manifest["sample_count"],
        "max_prompt_tokens": manifest["max_prompt_tokens"],
        "max_new_tokens": manifest["max_new_tokens"],
        "decode": manifest["decode"],
    }
    mismatches = [
        key
        for key, expected in checks.items()
        if report.get(key) != expected
    ]
    if cached_sample_ids != expected_sample_ids:
        mismatches.append("sample_ids")
    if mismatches:
        mismatch_summary = ", ".join(mismatches)
        raise ValueError(
            f"Cached model report at {report_path} is incompatible with the current run: "
            f"{mismatch_summary}"
        )

    report["resume_source"] = "cache"
    report["model_report_path"] = str(report_path)
    return report


def _build_comparison_report(
    *,
    output_path: Path,
    model_dir: Path,
    manifest: dict[str, Any],
    model_specs: list[dict[str, Any]],
    model_reports: list[dict[str, Any]],
    reference_label: str | None,
) -> dict[str, Any]:
    summary_rows = [_model_summary(report) for report in model_reports]
    reward_maps = _build_reward_maps(model_reports)
    pairwise_reward = _pairwise_reward_matrix(summary_rows, reward_maps)
    rankings = {
        "reward_mean_desc": _rank_models(summary_rows, "reward_mean", higher_is_better=True),
        "strict_parseable_desc": _rank_models(
            summary_rows, "strict_parseable_rate", higher_is_better=True
        ),
        "coverage_desc": _rank_models(
            summary_rows, "mean_match_coverage_vs_target", higher_is_better=True
        ),
        "track_error_asc": _rank_models(
            summary_rows, "mean_track_error_km", higher_is_better=False
        ),
        "intensity_error_asc": _rank_models(
            summary_rows, "mean_intensity_error_kt", higher_is_better=False
        ),
    }
    completed_labels = [report["model_label"] for report in model_reports]
    completed_label_set = set(completed_labels)
    pending_labels = [
        spec["label"] for spec in model_specs if spec["label"] not in completed_label_set
    ]
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_path": str(output_path),
        "model_report_dir": str(model_dir),
        "config_path": manifest["config_path"],
        "reward_config_path": manifest["reward_config_path"],
        "rl_dataset_path": manifest["rl_dataset_path"],
        "sft_dataset_path": manifest["sft_dataset_path"],
        "sample_count": manifest["sample_count"],
        "sample_seed": manifest["sample_seed"],
        "sample_ids": manifest["sample_ids"],
        "decode": manifest["decode"],
        "max_prompt_tokens": manifest["max_prompt_tokens"],
        "max_new_tokens": manifest["max_new_tokens"],
        "runtime": manifest.get("runtime", {}),
        "completed_model_labels": completed_labels,
        "pending_model_labels": pending_labels,
        "summary": summary_rows,
        "rankings": rankings,
        "pairwise_reward": pairwise_reward,
        "vs_reference": _reference_deltas(
            summary_rows, pairwise_reward, reference_label=reference_label
        ),
        "models": model_reports,
    }


def _write_aggregate_report(
    *,
    output_path: Path,
    model_dir: Path,
    manifest: dict[str, Any],
    model_specs: list[dict[str, Any]],
    model_reports: list[dict[str, Any]],
    reference_label: str | None,
) -> None:
    comparison_report = _build_comparison_report(
        output_path=output_path,
        model_dir=model_dir,
        manifest=manifest,
        model_specs=model_specs,
        model_reports=model_reports,
        reference_label=reference_label,
    )
    _write_json(output_path, comparison_report)


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


def _validate_target_coverage(sample_ids: list[str], sft_dataset_path: Path | None) -> None:
    if sft_dataset_path is None:
        return

    available_ids = set(_read_jsonl_sample_ids(sft_dataset_path))
    missing = [sample_id for sample_id in sample_ids if sample_id not in available_ids]
    if missing:
        preview = ", ".join(missing[:5])
        raise ValueError(
            f"Missing {len(missing)} sampled ids from {sft_dataset_path}: {preview}"
        )


def _parse_model_spec(spec: str) -> dict[str, Any]:
    if "=" not in spec:
        raise ValueError(
            f"Invalid --model value '{spec}'. Expected the form label=adapter_path_or_none."
        )
    label, raw_adapter = spec.split("=", 1)
    label = label.strip()
    raw_adapter = raw_adapter.strip()
    if not label:
        raise ValueError(f"Invalid --model value '{spec}': empty label.")

    if raw_adapter.lower() in BASE_MODEL_SENTINELS:
        return {"label": label, "adapter_path": None}

    adapter_path = Path(raw_adapter)
    if not adapter_path.is_absolute():
        adapter_path = (ROOT / adapter_path).resolve()
    if not adapter_path.exists():
        raise FileNotFoundError(f"Adapter path does not exist for '{label}': {adapter_path}")
    return {"label": label, "adapter_path": adapter_path}


def _model_summary(report: dict[str, Any]) -> dict[str, Any]:
    reward = report["reward"]
    matches = report["matches"]
    errors = report["errors"]
    return {
        "model_label": report["model_label"],
        "adapter_path": report["adapter_path"],
        "uses_adapter": report["uses_adapter"],
        "sample_count": report["sample_count"],
        "reward_mean": reward["mean"],
        "reward_std": reward["std"],
        "reward_p50": reward["p50"],
        "strict_parseable_rate": report["schema_rates"]["strict_forecast_parseable"],
        "strict_extra_text_rate": report["schema_rates"]["strict_forecast_with_extra_text"],
        "relaxed_parseable_rate": report["relaxed_schema_rates"]["relaxed_forecast_parseable"],
        "relaxed_clean_rate": report["relaxed_schema_rates"]["relaxed_forecast_clean"],
        "exact_match_rate": report["exact_match_rate"],
        "mean_matched_points": matches["mean_matched_points"],
        "mean_match_coverage_vs_target": matches["mean_match_coverage_vs_target"],
        "mean_time_alignment_score": matches["mean_time_alignment_score"],
        "mean_track_error_km": errors["mean_track_error_km"],
        "p50_track_error_km": errors["p50_track_error_km"],
        "mean_intensity_error_kt": errors["mean_intensity_error_kt"],
        "p50_intensity_error_kt": errors["p50_intensity_error_kt"],
    }


def _rank_models(
    summary_rows: list[dict[str, Any]],
    key: str,
    *,
    higher_is_better: bool,
) -> list[str]:
    sentinel = float("-inf") if higher_is_better else float("inf")
    return [
        row["model_label"]
        for row in sorted(
            summary_rows,
            key=lambda row: row[key] if row.get(key) is not None else sentinel,
            reverse=higher_is_better,
        )
    ]


def _build_reward_maps(model_reports: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    reward_maps: dict[str, dict[str, float]] = {}
    for report in model_reports:
        reward_maps[report["model_label"]] = {
            str(sample["sample_id"]): float(sample["reward_details"]["reward"])
            for sample in report.get("samples", [])
        }
    return reward_maps


def _pairwise_reward_matrix(
    summary_rows: list[dict[str, Any]],
    reward_maps: dict[str, dict[str, float]],
) -> dict[str, dict[str, dict[str, Any]]]:
    labels = [row["model_label"] for row in summary_rows]
    pairwise: dict[str, dict[str, dict[str, Any]]] = {}
    for left_label in labels:
        pairwise[left_label] = {}
        left_rewards = reward_maps[left_label]
        for right_label in labels:
            if left_label == right_label:
                continue
            right_rewards = reward_maps[right_label]
            wins = 0
            losses = 0
            ties = 0
            deltas: list[float] = []
            for sample_id, left_reward in left_rewards.items():
                right_reward = right_rewards[sample_id]
                delta = left_reward - right_reward
                deltas.append(delta)
                if delta > 0:
                    wins += 1
                elif delta < 0:
                    losses += 1
                else:
                    ties += 1
            decisive = wins + losses
            total = decisive + ties
            pairwise[left_label][right_label] = {
                "wins": wins,
                "losses": losses,
                "ties": ties,
                "win_rate_all_samples": (wins / total) if total else None,
                "win_rate_decisive_only": (wins / decisive) if decisive else None,
                "mean_reward_delta": (sum(deltas) / len(deltas)) if deltas else None,
            }
    return pairwise


def _reference_deltas(
    summary_rows: list[dict[str, Any]],
    pairwise_reward: dict[str, dict[str, dict[str, Any]]],
    reference_label: str | None,
) -> dict[str, dict[str, Any]] | None:
    if reference_label is None:
        return None

    row_by_label = {row["model_label"]: row for row in summary_rows}
    if reference_label not in row_by_label:
        return None

    reference = row_by_label[reference_label]
    comparisons: dict[str, dict[str, Any]] = {}
    for row in summary_rows:
        label = row["model_label"]
        if label == reference_label:
            continue
        pairwise = pairwise_reward[label][reference_label]
        comparisons[label] = {
            "delta_reward_mean": _diff_or_none(row["reward_mean"], reference["reward_mean"]),
            "delta_strict_parseable_rate": _diff_or_none(
                row["strict_parseable_rate"], reference["strict_parseable_rate"]
            ),
            "delta_exact_match_rate": (
                _diff_or_none(row["exact_match_rate"], reference["exact_match_rate"])
            ),
            "delta_match_coverage_vs_target": _diff_or_none(
                row["mean_match_coverage_vs_target"],
                reference["mean_match_coverage_vs_target"],
            ),
            "delta_track_error_km": _diff_or_none(
                row["mean_track_error_km"], reference["mean_track_error_km"]
            ),
            "delta_intensity_error_kt": _diff_or_none(
                row["mean_intensity_error_kt"], reference["mean_intensity_error_kt"]
            ),
            "pairwise_reward_vs_reference": pairwise,
        }
    return comparisons


def _diff_or_none(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return left - right


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare multiple adapters or the base model on one held-out strict-forecast subset."
    )
    parser.add_argument("--config", required=True, help="Path to the model/LoRA config.")
    parser.add_argument(
        "--reward-config",
        required=True,
        help="Path to the GRPO YAML config used to instantiate the reward function.",
    )
    parser.add_argument("--rl-dataset", required=True, help="Held-out RL JSONL with verification.")
    parser.add_argument(
        "--sft-dataset",
        help="Optional held-out SFT JSONL used to recover gold assistant targets for exact-match scoring.",
    )
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        help="One model spec in the form label=adapter_path_or_none. Use 'none' to evaluate the base model without an adapter.",
    )
    parser.add_argument(
        "--reference-label",
        help="Optional model label to use when computing deltas versus a designated reference model.",
    )
    parser.add_argument("--sample-count", type=int, default=200)
    parser.add_argument("--sample-seed", type=int, default=3407)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-prompt-tokens", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--output", required=True, help="Path to the JSON comparison report.")
    args = parser.parse_args(argv)

    config_path = Path(args.config).resolve()
    reward_config_path = Path(args.reward_config).resolve()
    rl_dataset_path = Path(args.rl_dataset).resolve()
    sft_dataset_path = Path(args.sft_dataset).resolve() if args.sft_dataset else None
    sample_ids = _select_sample_ids(rl_dataset_path, args.sample_count, args.sample_seed)
    _validate_target_coverage(sample_ids, sft_dataset_path)

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model_dir = _model_output_dir(output_path)
    model_dir.mkdir(parents=True, exist_ok=True)

    model_specs = [_parse_model_spec(spec) for spec in args.model]
    manifest = _build_manifest(
        config_path=config_path,
        reward_config_path=reward_config_path,
        rl_dataset_path=rl_dataset_path,
        sft_dataset_path=sft_dataset_path,
        sample_count=args.sample_count,
        sample_seed=args.sample_seed,
        sample_ids=sample_ids,
        max_prompt_tokens=args.max_prompt_tokens,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
        model_specs=model_specs,
        batch_size=args.batch_size,
    )
    manifest = _ensure_manifest(model_dir / "manifest.json", manifest)

    model_reports: list[dict[str, Any]] = []
    _write_aggregate_report(
        output_path=output_path,
        model_dir=model_dir,
        manifest=manifest,
        model_specs=model_specs,
        model_reports=model_reports,
        reference_label=args.reference_label,
    )

    total_models = len(model_specs)
    for index, model_spec in enumerate(model_specs, start=1):
        label = model_spec["label"]
        adapter_path = model_spec["adapter_path"]
        report_path = _model_report_path(model_dir, index, label)
        cached_report = _load_cached_model_report(
            report_path,
            model_spec=model_spec,
            manifest=manifest,
        )
        if cached_report is not None:
            print(
                f"[{index}/{total_models}] Reusing cached report for {label} from {report_path}",
                flush=True,
            )
            model_reports.append(cached_report)
            _write_aggregate_report(
                output_path=output_path,
                model_dir=model_dir,
                manifest=manifest,
                model_specs=model_specs,
                model_reports=model_reports,
                reference_label=args.reference_label,
            )
            continue

        print(
            f"[{index}/{total_models}] Evaluating {label} "
            f"(adapter={adapter_path if adapter_path is not None else 'base_model'})",
            flush=True,
        )
        report = evaluate(
            config_path=config_path,
            reward_config_path=reward_config_path,
            adapter_path=adapter_path,
            rl_dataset_path=rl_dataset_path,
            sft_dataset_path=sft_dataset_path,
            max_samples=None,
            batch_size=args.batch_size,
            max_prompt_tokens=args.max_prompt_tokens,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.do_sample,
            temperature=args.temperature,
            top_p=args.top_p,
            sample_ids=sample_ids,
            include_samples=True,
        )
        report["model_label"] = label
        report["model_kind"] = "base" if adapter_path is None else "adapter"
        report["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        report["model_report_path"] = str(report_path)
        _write_json(report_path, report)
        print(
            f"[{index}/{total_models}] Saved report for {label} to {report_path}",
            flush=True,
        )
        model_reports.append(report)
        _write_aggregate_report(
            output_path=output_path,
            model_dir=model_dir,
            manifest=manifest,
            model_specs=model_specs,
            model_reports=model_reports,
            reference_label=args.reference_label,
        )

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
