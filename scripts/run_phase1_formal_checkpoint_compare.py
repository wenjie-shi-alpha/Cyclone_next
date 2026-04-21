#!/usr/bin/env python3
"""Run a focused offline comparison for one formal Phase 1 SFT->GRPO run."""

from __future__ import annotations

import argparse
import json
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

import compare_strict_forecast_models as compare_models


PROCESS_CHECKPOINT_RE = re.compile(r"^adapter_step-(\d{6})$")
REWARD_CHECKPOINT_RE = re.compile(r"^adapter_reward-([0-9.]+)_step-(\d{6})$")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _format_optional(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def _load_pipeline_manifest(run_root: Path) -> dict[str, Any]:
    manifest_path = run_root / "pipeline_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing pipeline manifest: {manifest_path}")
    return _read_json(manifest_path)


def _resolve_eval_dataset_paths(
    dataset_root: Path,
    rl_dataset_path: Path | None,
    sft_dataset_path: Path | None,
) -> tuple[Path, Path]:
    if rl_dataset_path is None:
        rl_dataset_path = dataset_root / "views" / "forecast_only" / "rl_test.jsonl"
    if sft_dataset_path is None:
        sft_dataset_path = dataset_root / "views" / "forecast_only" / "test.jsonl"
    if not rl_dataset_path.exists():
        raise FileNotFoundError(f"Missing held-out RL dataset: {rl_dataset_path}")
    if not sft_dataset_path.exists():
        raise FileNotFoundError(f"Missing held-out SFT dataset: {sft_dataset_path}")
    return rl_dataset_path.resolve(), sft_dataset_path.resolve()


def _list_process_checkpoints(grpo_dir: Path) -> list[tuple[int, Path]]:
    checkpoints: list[tuple[int, Path]] = []
    for child in sorted(grpo_dir.iterdir()):
        if not child.is_dir():
            continue
        match = PROCESS_CHECKPOINT_RE.match(child.name)
        if match is None:
            continue
        checkpoints.append((int(match.group(1)), child.resolve()))
    return checkpoints


def _list_reward_checkpoints(grpo_dir: Path) -> list[tuple[float, int, Path]]:
    checkpoints: list[tuple[float, int, Path]] = []
    for child in sorted(grpo_dir.iterdir()):
        if not child.is_dir():
            continue
        match = REWARD_CHECKPOINT_RE.match(child.name)
        if match is None:
            continue
        checkpoints.append((float(match.group(1)), int(match.group(2)), child.resolve()))
    checkpoints.sort(key=lambda item: (-item[0], item[1]))
    return checkpoints


def _evenly_spaced_indices(length: int, count: int) -> list[int]:
    if count <= 0 or length <= 0:
        return []
    if count >= length:
        return list(range(length))
    if count == 1:
        return [length - 1]
    if count == 2:
        return [0, length - 1]
    if count == 3:
        return sorted({0, (length - 1) // 2, length - 1})

    indices = []
    for idx in range(count):
        pos = int(round(idx * (length - 1) / (count - 1)))
        indices.append(pos)
    return sorted(set(indices))


def _select_process_models(
    grpo_dir: Path,
    *,
    requested_steps: list[int] | None,
    count: int,
) -> list[dict[str, Any]]:
    available = _list_process_checkpoints(grpo_dir)
    if not available:
        raise FileNotFoundError(f"No adapter_step checkpoints found in {grpo_dir}")

    selected: list[tuple[int, Path]] = []
    if requested_steps:
        by_step = {step: path for step, path in available}
        missing = [step for step in requested_steps if step not in by_step]
        if missing:
            raise FileNotFoundError(
                f"Missing requested process checkpoints in {grpo_dir}: {missing}"
            )
        selected = [(step, by_step[step]) for step in requested_steps]
    else:
        chosen_indices = _evenly_spaced_indices(len(available), count)
        selected = [available[index] for index in chosen_indices]

    models: list[dict[str, Any]] = []
    for step, path in selected:
        models.append(
            {
                "label": f"grpo_step_{step:03d}",
                "adapter_path": path,
                "kind": "process_checkpoint",
                "step": step,
            }
        )
    return models


def _select_reward_models(grpo_dir: Path, *, count: int) -> list[dict[str, Any]]:
    available = _list_reward_checkpoints(grpo_dir)
    if not available:
        return []
    selected = available[:count]
    models: list[dict[str, Any]] = []
    for reward_value, step, path in selected:
        models.append(
            {
                "label": f"grpo_reward_{reward_value:.4f}_step_{step:03d}",
                "adapter_path": path,
                "kind": "reward_checkpoint",
                "step": step,
                "reward": reward_value,
            }
        )
    return models


def _build_model_specs(
    *,
    process_models: list[dict[str, Any]],
    reward_models: list[dict[str, Any]],
    sft_adapter_dir: Path,
) -> list[dict[str, Any]]:
    model_specs: list[dict[str, Any]] = [
        {
            "label": "base_model",
            "adapter_path": None,
            "kind": "base_model",
        },
        {
            "label": "sft_reference",
            "adapter_path": sft_adapter_dir.resolve(),
            "kind": "sft_reference",
        },
    ]
    model_specs.extend(process_models)
    model_specs.extend(reward_models)
    return model_specs


def _selection_payload(
    *,
    run_root: Path,
    sft_config_path: Path,
    reward_config_path: Path,
    rl_dataset_path: Path,
    sft_dataset_path: Path,
    process_models: list[dict[str, Any]],
    reward_models: list[dict[str, Any]],
    model_specs: list[dict[str, Any]],
    sample_count: int,
    sample_seed: int,
    max_prompt_tokens: int,
    max_new_tokens: int,
) -> dict[str, Any]:
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_root": str(run_root),
        "source_configs": {
            "sft": str(sft_config_path),
            "grpo": str(reward_config_path),
        },
        "heldout_datasets": {
            "rl_test": str(rl_dataset_path),
            "sft_test": str(sft_dataset_path),
        },
        "selection": {
            "sample_count": sample_count,
            "sample_seed": sample_seed,
            "max_prompt_tokens": max_prompt_tokens,
            "max_new_tokens": max_new_tokens,
        },
        "process_models": _json_ready_checkpoint_rows(process_models),
        "reward_models": _json_ready_checkpoint_rows(reward_models),
        "all_models": [
            {
                "label": spec["label"],
                "adapter_path": (
                    str(spec["adapter_path"]) if spec["adapter_path"] is not None else None
                ),
                "kind": spec["kind"],
            }
            for spec in model_specs
        ],
    }


def _json_ready_checkpoint_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        adapter_path = item.get("adapter_path")
        if adapter_path is not None:
            item["adapter_path"] = str(adapter_path)
        payload.append(item)
    return payload


def _base_reference_deltas(
    summary_rows: list[dict[str, Any]],
    base_label: str = "base_model",
    reference_label: str = "sft_reference",
) -> dict[str, dict[str, Any]]:
    row_by_label = {row["model_label"]: row for row in summary_rows}
    base = row_by_label.get(base_label)
    reference = row_by_label.get(reference_label)
    deltas: dict[str, dict[str, Any]] = {}
    for row in summary_rows:
        label = row["model_label"]
        if label not in {base_label, reference_label}:
            deltas[label] = {
                "vs_base": {
                    "delta_reward_mean": None if base is None else row["reward_mean"] - base["reward_mean"],
                    "delta_coverage": None
                    if base is None
                    else row["mean_match_coverage_vs_target"] - base["mean_match_coverage_vs_target"],
                    "delta_track_error_km": None
                    if base is None
                    else row["mean_track_error_km"] - base["mean_track_error_km"],
                    "delta_intensity_error_kt": None
                    if base is None
                    else row["mean_intensity_error_kt"] - base["mean_intensity_error_kt"],
                },
                "vs_sft": {
                    "delta_reward_mean": None
                    if reference is None
                    else row["reward_mean"] - reference["reward_mean"],
                    "delta_coverage": None
                    if reference is None
                    else row["mean_match_coverage_vs_target"] - reference["mean_match_coverage_vs_target"],
                    "delta_track_error_km": None
                    if reference is None
                    else row["mean_track_error_km"] - reference["mean_track_error_km"],
                    "delta_intensity_error_kt": None
                    if reference is None
                    else row["mean_intensity_error_kt"] - reference["mean_intensity_error_kt"],
                },
            }
    return deltas


def _render_summary_markdown(
    *,
    comparison_report: dict[str, Any],
    selection: dict[str, Any],
    reference_label: str,
    base_label: str,
) -> str:
    summary_rows = comparison_report.get("summary", [])
    row_by_label = {row["model_label"]: row for row in summary_rows}
    base_deltas = _base_reference_deltas(summary_rows, base_label=base_label, reference_label=reference_label)
    rankings = comparison_report.get("rankings", {})

    lines = [
        "# Phase 1 Formal Checkpoint Offline Compare",
        "",
        f"- Generated: {comparison_report.get('generated_at_utc')}",
        f"- Run root: `{selection['run_root']}`",
        f"- RL test set: `{selection['heldout_datasets']['rl_test']}`",
        f"- SFT test set: `{selection['heldout_datasets']['sft_test']}`",
        f"- Sample count: `{selection['selection']['sample_count']}`",
        f"- Sample seed: `{selection['selection']['sample_seed']}`",
        f"- Decode: greedy (`max_prompt_tokens={selection['selection']['max_prompt_tokens']}`, `max_new_tokens={selection['selection']['max_new_tokens']}`)",
        "",
        "## Selected Models",
        "",
    ]

    for spec in selection.get("all_models", []):
        adapter = spec["adapter_path"] if spec["adapter_path"] is not None else "base_model"
        lines.append(f"- `{spec['label']}`: `{adapter}` ({spec['kind']})")

    lines.extend(
        [
            "",
            "## Summary",
            "",
            "| model | reward | coverage | track error km | intensity error kt | strict parse | exact match |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in summary_rows:
        lines.append(
            "| "
            f"`{row['model_label']}` | "
            f"{_format_optional(row['reward_mean'])} | "
            f"{_format_optional(row['mean_match_coverage_vs_target'])} | "
            f"{_format_optional(row['mean_track_error_km'], 2)} | "
            f"{_format_optional(row['mean_intensity_error_kt'], 2)} | "
            f"{_format_optional(row['strict_parseable_rate'])} | "
            f"{_format_optional(row['exact_match_rate'])} |"
        )

    lines.extend(
        [
            "",
            "## Rankings",
            "",
            f"- Reward: {', '.join(f'`{label}`' for label in rankings.get('reward_mean_desc', []))}",
            f"- Coverage: {', '.join(f'`{label}`' for label in rankings.get('coverage_desc', []))}",
            f"- Track error: {', '.join(f'`{label}`' for label in rankings.get('track_error_asc', []))}",
            f"- Intensity error: {', '.join(f'`{label}`' for label in rankings.get('intensity_error_asc', []))}",
            "",
            "## Delta Vs Baselines",
            "",
        ]
    )

    for label, deltas in base_deltas.items():
        row = row_by_label[label]
        lines.append(f"### `{label}`")
        lines.append(
            "- Vs SFT: "
            f"reward `{_format_optional(deltas['vs_sft']['delta_reward_mean'])}`, "
            f"coverage `{_format_optional(deltas['vs_sft']['delta_coverage'])}`, "
            f"track error `{_format_optional(deltas['vs_sft']['delta_track_error_km'], 2)}`, "
            f"intensity error `{_format_optional(deltas['vs_sft']['delta_intensity_error_kt'], 2)}`"
        )
        lines.append(
            "- Vs base: "
            f"reward `{_format_optional(deltas['vs_base']['delta_reward_mean'])}`, "
            f"coverage `{_format_optional(deltas['vs_base']['delta_coverage'])}`, "
            f"track error `{_format_optional(deltas['vs_base']['delta_track_error_km'], 2)}`, "
            f"intensity error `{_format_optional(deltas['vs_base']['delta_intensity_error_kt'], 2)}`"
        )
        pairwise = comparison_report.get("vs_reference", {}) or {}
        if label in pairwise:
            pairwise_ref = pairwise[label]["pairwise_reward_vs_reference"]
            lines.append(
                "- Pairwise reward vs SFT: "
                f"wins `{pairwise_ref['wins']}`, "
                f"losses `{pairwise_ref['losses']}`, "
                f"ties `{pairwise_ref['ties']}`, "
                f"mean delta `{_format_optional(pairwise_ref['mean_reward_delta'])}`"
            )
        lines.append(
            "- Current metrics: "
            f"reward `{_format_optional(row['reward_mean'])}`, "
            f"coverage `{_format_optional(row['mean_match_coverage_vs_target'])}`, "
            f"track `{_format_optional(row['mean_track_error_km'], 2)}`, "
            f"intensity `{_format_optional(row['mean_intensity_error_kt'], 2)}`"
        )
        lines.append("")

    return "\n".join(lines).strip() + "\n"


def _parse_steps(raw_value: str | None) -> list[int] | None:
    if raw_value is None or not raw_value.strip():
        return None
    return [int(part.strip()) for part in raw_value.split(",") if part.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run one focused offline comparison for a formal Phase 1 SFT->GRPO run."
    )
    parser.add_argument("--run-root", required=True, help="Formal run root with pipeline_manifest.json.")
    parser.add_argument(
        "--rl-dataset",
        help="Optional held-out RL dataset override. Defaults to dataset_root/views/forecast_only/rl_test.jsonl.",
    )
    parser.add_argument(
        "--sft-dataset",
        help="Optional held-out SFT dataset override. Defaults to dataset_root/views/forecast_only/test.jsonl.",
    )
    parser.add_argument("--sample-count", type=int, default=200)
    parser.add_argument("--sample-seed", type=int, default=3407)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-prompt-tokens", type=int, default=1280)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument(
        "--process-steps",
        help="Optional comma-separated process checkpoint steps, e.g. 10,50,100. Defaults to three spread-out adapter_step checkpoints.",
    )
    parser.add_argument("--process-count", type=int, default=3)
    parser.add_argument("--reward-count", type=int, default=2)
    parser.add_argument(
        "--output-dir",
        help="Optional output directory. Defaults to run_root/evals/offline_compare_sample{N}_seed{seed}_basefixed.",
    )
    args = parser.parse_args(argv)

    run_root = Path(args.run_root).resolve()
    pipeline_manifest = _load_pipeline_manifest(run_root)
    dataset_root = Path(pipeline_manifest["datasets"]["sft"]["dataset_root"]).resolve()
    sft_config_path = Path(pipeline_manifest["source_configs"]["sft"]).resolve()
    reward_config_path = Path(pipeline_manifest["source_configs"]["grpo"]).resolve()
    sft_adapter_dir = Path(pipeline_manifest["artifacts"]["sft_adapter_dir"]).resolve()
    grpo_dir = run_root / "grpo"

    rl_dataset_path, sft_dataset_path = _resolve_eval_dataset_paths(
        dataset_root=dataset_root,
        rl_dataset_path=Path(args.rl_dataset).resolve() if args.rl_dataset else None,
        sft_dataset_path=Path(args.sft_dataset).resolve() if args.sft_dataset else None,
    )

    process_models = _select_process_models(
        grpo_dir,
        requested_steps=_parse_steps(args.process_steps),
        count=args.process_count,
    )
    reward_models = _select_reward_models(grpo_dir, count=args.reward_count)
    model_specs = _build_model_specs(
        process_models=process_models,
        reward_models=reward_models,
        sft_adapter_dir=sft_adapter_dir,
    )

    if args.output_dir:
        output_dir = Path(args.output_dir).resolve()
    else:
        output_dir = (
            run_root
            / "evals"
            / f"offline_compare_sample{args.sample_count}_seed{args.sample_seed}_basefixed"
        ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    selection_payload = _selection_payload(
        run_root=run_root,
        sft_config_path=sft_config_path,
        reward_config_path=reward_config_path,
        rl_dataset_path=rl_dataset_path,
        sft_dataset_path=sft_dataset_path,
        process_models=process_models,
        reward_models=reward_models,
        model_specs=model_specs,
        sample_count=args.sample_count,
        sample_seed=args.sample_seed,
        max_prompt_tokens=args.max_prompt_tokens,
        max_new_tokens=args.max_new_tokens,
    )
    selection_path = output_dir / "selection.json"
    _write_json(selection_path, selection_payload)

    compare_output_path = output_dir / "compare_report.json"
    compare_argv = [
        "--config",
        str(sft_config_path),
        "--reward-config",
        str(reward_config_path),
        "--rl-dataset",
        str(rl_dataset_path),
        "--sft-dataset",
        str(sft_dataset_path),
        "--reference-label",
        "sft_reference",
        "--sample-count",
        str(args.sample_count),
        "--sample-seed",
        str(args.sample_seed),
        "--batch-size",
        str(args.batch_size),
        "--max-prompt-tokens",
        str(args.max_prompt_tokens),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--output",
        str(compare_output_path),
    ]
    for spec in model_specs:
        adapter = spec["adapter_path"] if spec["adapter_path"] is not None else "none"
        compare_argv.extend(["--model", f"{spec['label']}={adapter}"])

    compare_models.main(compare_argv)

    comparison_report = _read_json(compare_output_path)
    summary_markdown = _render_summary_markdown(
        comparison_report=comparison_report,
        selection=selection_payload,
        reference_label="sft_reference",
        base_label="base_model",
    )
    summary_path = output_dir / "summary.md"
    _write_text(summary_path, summary_markdown)

    print(f"[OK] Comparison report: {compare_output_path}")
    print(f"[OK] Selection manifest: {selection_path}")
    print(f"[OK] Summary markdown: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
