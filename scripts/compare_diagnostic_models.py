#!/usr/bin/env python3
"""Compare multiple diagnostic models or synthetic baselines on one held-out subset."""

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

import eval_diagnostic_heldout as diagnostic_eval


BASE_MODEL_SENTINELS = {"", "base", "none", "no_adapter", "null"}
SYNTHETIC_BASELINES = {"majority_label", "rule_echo"}


def _slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return slug or "model"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _model_output_dir(output_path: Path) -> Path:
    return output_path.parent / f"{output_path.stem}.models"


def _model_report_path(model_dir: Path, index: int, label: str) -> Path:
    return model_dir / f"{index:02d}_{_slugify(label)}.json"


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

    rng = random.Random(sample_seed)
    selected = set(rng.sample(sample_ids, sample_count))
    return [sample_id for sample_id in sample_ids if sample_id in selected]


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
        return {
            "label": label,
            "adapter_path": None,
            "prediction_mode": "base_model",
            "kind": "base_model",
        }

    adapter_path = Path(raw_adapter)
    if not adapter_path.is_absolute():
        adapter_path = (ROOT / adapter_path).resolve()
    if not adapter_path.exists():
        raise FileNotFoundError(f"Adapter path does not exist for '{label}': {adapter_path}")
    return {
        "label": label,
        "adapter_path": adapter_path,
        "prediction_mode": "adapter",
        "kind": "adapter",
    }


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
    checks = {
        "model_label": model_spec["label"],
        "prediction_mode": model_spec["prediction_mode"],
        "adapter_path": expected_adapter_path,
        "config_path": manifest["config_path"],
        "dataset_path": manifest["dataset_path"],
        "field_names": manifest["field_names"],
        "sample_count": manifest["sample_count"],
        "sample_ids": manifest["sample_ids"],
        "runtime": manifest["runtime"],
    }
    mismatches = [
        key
        for key, expected in checks.items()
        if report.get(key) != expected
    ]
    if mismatches:
        mismatch_summary = ", ".join(mismatches)
        raise ValueError(
            f"Cached diagnostic model report at {report_path} is incompatible with the current run: "
            f"{mismatch_summary}"
        )
    report["resume_source"] = "cache"
    report["model_report_path"] = str(report_path)
    return report


def _build_manifest(
    *,
    config_path: Path | None,
    dataset_path: Path,
    train_dataset_path: Path | None,
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
    field_names: list[str],
) -> dict[str, Any]:
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path) if config_path is not None else None,
        "dataset_path": str(dataset_path),
        "train_dataset_path": str(train_dataset_path) if train_dataset_path is not None else None,
        "sample_count": len(sample_ids),
        "requested_sample_count": sample_count,
        "sample_seed": sample_seed,
        "sample_ids": sample_ids,
        "field_names": field_names,
        "runtime": {
            "requested_batch_size": batch_size,
            "max_prompt_tokens": max_prompt_tokens,
            "max_new_tokens": max_new_tokens,
            "decode": {
                "do_sample": do_sample,
                "temperature": temperature,
                "top_p": top_p,
            },
        },
        "models": [
            {
                "label": spec["label"],
                "adapter_path": (
                    str(spec["adapter_path"]) if spec["adapter_path"] is not None else None
                ),
                "prediction_mode": spec["prediction_mode"],
                "kind": spec["kind"],
            }
            for spec in model_specs
        ],
    }


def _ensure_manifest(manifest_path: Path, current_manifest: dict[str, Any]) -> dict[str, Any]:
    if not manifest_path.exists():
        _write_json(manifest_path, current_manifest)
        return current_manifest

    existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    comparable_keys = [
        "config_path",
        "dataset_path",
        "train_dataset_path",
        "sample_count",
        "sample_seed",
        "sample_ids",
        "field_names",
        "runtime",
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
    return existing_manifest


def _model_summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "model_label": report["model_label"],
        "prediction_mode": report["prediction_mode"],
        "adapter_path": report["adapter_path"],
        "sample_count": report["sample_count"],
        "json_parseable_rate": report["json_parseable_rate"],
        "json_exact_keyset_rate": report["json_exact_keyset_rate"],
        "joint_exact_match_rate": report["joint_exact_match_rate"],
        "mean_field_exact_accuracy": report["mean_field_exact_accuracy"],
        "mean_field_macro_f1": report["mean_field_macro_f1"],
        "mean_field_null_vs_non_null_f1": report["mean_field_null_vs_non_null_f1"],
    }


def _rank_models(
    summary_rows: list[dict[str, Any]],
    key: str,
) -> list[str]:
    return [
        row["model_label"]
        for row in sorted(
            summary_rows,
            key=lambda row: row[key] if row.get(key) is not None else float("-inf"),
            reverse=True,
        )
    ]


def _diff_or_none(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return left - right


def _reference_deltas(
    summary_rows: list[dict[str, Any]],
    reference_label: str | None,
) -> dict[str, dict[str, Any]] | None:
    if reference_label is None:
        return None

    row_by_label = {row["model_label"]: row for row in summary_rows}
    reference = row_by_label.get(reference_label)
    if reference is None:
        return None

    comparisons: dict[str, dict[str, Any]] = {}
    for row in summary_rows:
        label = row["model_label"]
        if label == reference_label:
            continue
        comparisons[label] = {
            "delta_joint_exact_match_rate": _diff_or_none(
                row["joint_exact_match_rate"], reference["joint_exact_match_rate"]
            ),
            "delta_json_parseable_rate": _diff_or_none(
                row["json_parseable_rate"], reference["json_parseable_rate"]
            ),
            "delta_mean_field_exact_accuracy": _diff_or_none(
                row["mean_field_exact_accuracy"], reference["mean_field_exact_accuracy"]
            ),
            "delta_mean_field_macro_f1": _diff_or_none(
                row["mean_field_macro_f1"], reference["mean_field_macro_f1"]
            ),
            "delta_mean_field_null_vs_non_null_f1": _diff_or_none(
                row["mean_field_null_vs_non_null_f1"],
                reference["mean_field_null_vs_non_null_f1"],
            ),
        }
    return comparisons


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
    completed_labels = [report["model_label"] for report in model_reports]
    completed_label_set = set(completed_labels)
    pending_labels = [
        spec["label"]
        for spec in model_specs
        if spec["label"] not in completed_label_set
    ]
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_path": str(output_path),
        "model_report_dir": str(model_dir),
        "config_path": manifest["config_path"],
        "dataset_path": manifest["dataset_path"],
        "train_dataset_path": manifest["train_dataset_path"],
        "sample_count": manifest["sample_count"],
        "sample_seed": manifest["sample_seed"],
        "sample_ids": manifest["sample_ids"],
        "field_names": manifest["field_names"],
        "runtime": manifest["runtime"],
        "completed_model_labels": completed_labels,
        "pending_model_labels": pending_labels,
        "summary": summary_rows,
        "rankings": {
            "joint_exact_match_desc": _rank_models(summary_rows, "joint_exact_match_rate"),
            "json_parseable_desc": _rank_models(summary_rows, "json_parseable_rate"),
            "field_accuracy_desc": _rank_models(summary_rows, "mean_field_exact_accuracy"),
            "field_macro_f1_desc": _rank_models(summary_rows, "mean_field_macro_f1"),
            "field_non_null_f1_desc": _rank_models(
                summary_rows, "mean_field_null_vs_non_null_f1"
            ),
        },
        "vs_reference": _reference_deltas(summary_rows, reference_label=reference_label),
        "models": model_reports,
    }


def _build_summary_markdown(comparison_report: dict[str, Any]) -> str:
    lines = [
        "# Diagnostic Compare Summary",
        "",
        f"- generated_at_utc: `{comparison_report['generated_at_utc']}`",
        f"- dataset_path: `{comparison_report['dataset_path']}`",
        f"- sample_count: `{comparison_report['sample_count']}`",
        f"- field_names: `{', '.join(comparison_report['field_names'])}`",
        "",
        "| model | mode | json parse | exact keyset | joint exact | mean field acc | mean macro-F1 | mean null/non-null F1 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in comparison_report.get("summary", []):
        lines.append(
            "| {model} | {mode} | {parse:.4f} | {keyset:.4f} | {joint:.4f} | {acc:.4f} | {macro:.4f} | {nonnull:.4f} |".format(
                model=row["model_label"],
                mode=row["prediction_mode"],
                parse=row["json_parseable_rate"] or 0.0,
                keyset=row["json_exact_keyset_rate"] or 0.0,
                joint=row["joint_exact_match_rate"] or 0.0,
                acc=row["mean_field_exact_accuracy"] or 0.0,
                macro=row["mean_field_macro_f1"] or 0.0,
                nonnull=row["mean_field_null_vs_non_null_f1"] or 0.0,
            )
        )
    return "\n".join(lines) + "\n"


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
    _write_text(output_path.with_suffix(".summary.md"), _build_summary_markdown(comparison_report))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare multiple diagnostic adapters or synthetic baselines on one held-out subset."
    )
    parser.add_argument("--config", help="Path to the diagnostic SFT YAML config.")
    parser.add_argument("--dataset", required=True, help="Path to one diagnostic SFT JSONL split.")
    parser.add_argument(
        "--train-dataset",
        help="Optional train split used to derive the majority-label synthetic baseline.",
    )
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        help="One model spec in the form label=adapter_path_or_none.",
    )
    parser.add_argument(
        "--synthetic-baseline",
        action="append",
        default=[],
        choices=sorted(SYNTHETIC_BASELINES),
        help="One synthetic baseline label to include.",
    )
    parser.add_argument("--reference-label", help="Optional model label used for delta reporting.")
    parser.add_argument("--sample-count", type=int, default=256)
    parser.add_argument("--sample-seed", type=int, default=3407)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-prompt-tokens", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--output", required=True, help="Path to the JSON comparison report.")
    args = parser.parse_args(argv)

    config_path = Path(args.config).resolve() if args.config else None
    dataset_path = Path(args.dataset).resolve()
    train_dataset_path = Path(args.train_dataset).resolve() if args.train_dataset else dataset_path.with_name("train.jsonl")
    sample_ids = _select_sample_ids(dataset_path, args.sample_count, args.sample_seed)
    field_probe = diagnostic_eval.predict_diagnostics(
        config_path=config_path,
        adapter_path=None,
        dataset_path=dataset_path,
        max_samples=None,
        batch_size=args.batch_size,
        max_prompt_tokens=args.max_prompt_tokens,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
        prediction_mode="rule_echo",
        train_dataset_path=train_dataset_path,
        sample_ids=sample_ids,
    )
    field_names = list(field_probe["field_names"])

    model_specs = [_parse_model_spec(spec) for spec in args.model]
    for synthetic_baseline in args.synthetic_baseline:
        model_specs.append(
            {
                "label": synthetic_baseline,
                "adapter_path": None,
                "prediction_mode": synthetic_baseline,
                "kind": "synthetic_baseline",
            }
        )
    if not model_specs:
        raise ValueError("At least one --model or --synthetic-baseline must be provided.")

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model_dir = _model_output_dir(output_path)
    model_dir.mkdir(parents=True, exist_ok=True)

    manifest = _build_manifest(
        config_path=config_path,
        dataset_path=dataset_path,
        train_dataset_path=train_dataset_path,
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
        field_names=field_names,
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
            f"(mode={model_spec['prediction_mode']}, adapter={model_spec['adapter_path']})",
            flush=True,
        )
        report = diagnostic_eval.evaluate(
            config_path=config_path,
            adapter_path=model_spec["adapter_path"],
            dataset_path=dataset_path,
            max_samples=None,
            batch_size=args.batch_size,
            max_prompt_tokens=args.max_prompt_tokens,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.do_sample,
            temperature=args.temperature,
            top_p=args.top_p,
            prediction_mode=model_spec["prediction_mode"],
            train_dataset_path=train_dataset_path,
            sample_ids=sample_ids,
            include_samples=True,
        )
        report["model_label"] = label
        report["prediction_mode"] = model_spec["prediction_mode"]
        report["adapter_path"] = (
            str(model_spec["adapter_path"]) if model_spec["adapter_path"] is not None else None
        )
        report["model_kind"] = model_spec["kind"]
        report["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        report["model_report_path"] = str(report_path)
        _write_json(report_path, report)
        print(f"[{index}/{total_models}] Saved report for {label} to {report_path}", flush=True)
        model_reports.append(report)
        _write_aggregate_report(
            output_path=output_path,
            model_dir=model_dir,
            manifest=manifest,
            model_specs=model_specs,
            model_reports=model_reports,
            reference_label=args.reference_label,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
