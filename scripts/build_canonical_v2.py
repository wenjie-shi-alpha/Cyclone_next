#!/usr/bin/env python3
"""Convert legacy raw training samples into canonical v2 JSONL."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

from dataset_v2 import (
    CANONICAL_VERSION,
    DIAGNOSTIC_FIELDS,
    SUPPORTED_SPLITS,
    build_canonical_json_schema,
    canonicalize_legacy_sample,
    infer_latest_legacy_raw_dir,
    iter_legacy_raw_samples,
)


def _counter_to_sorted_dict(counter: Counter[str]) -> Dict[str, int]:
    return {key: counter[key] for key in sorted(counter)}


def _safe_ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(float(numerator) / float(denominator), 4)


def _selected_missingness(record: Dict[str, Any]) -> Dict[str, bool]:
    targets = record.get("targets", {}) or {}
    diagnostics = record.get("diagnostics", {}) or {}
    return {
        "targets.official_forecast_table": bool(targets.get("official_forecast_table")),
        "targets.reasoning_text": bool(targets.get("reasoning_text")),
        "targets.risk_text": bool(targets.get("risk_text")),
        "targets.verification_target.future_best_track_points": bool(
            (targets.get("verification_target") or {}).get("future_best_track_points")
        ),
        "diagnostics.track_control_signal": bool(diagnostics.get("track_control_signal")),
        "diagnostics.turning_signal": bool(diagnostics.get("turning_signal")),
        "diagnostics.intensity_support_signal": bool(diagnostics.get("intensity_support_signal")),
        "diagnostics.shear_constraint_level": bool(diagnostics.get("shear_constraint_level")),
        "diagnostics.land_interaction_level": bool(diagnostics.get("land_interaction_level")),
        "diagnostics.model_agreement_level": bool(diagnostics.get("model_agreement_level")),
        "diagnostics.main_uncertainty_source": bool(diagnostics.get("main_uncertainty_source")),
        "diagnostics.forecast_confidence_level": bool(diagnostics.get("forecast_confidence_level")),
        "diagnostics.expert_decision_notes": bool(diagnostics.get("expert_decision_notes")),
    }


def build_canonical_dataset(
    raw_base_dir: Path,
    output_dir: Path,
) -> Dict[str, Any]:
    """Build canonical v2 JSONL plus coverage report."""
    output_dir.mkdir(parents=True, exist_ok=True)

    split_records: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    storm_ids_by_split: Dict[str, set[str]] = defaultdict(set)
    split_counts: Counter[str] = Counter()
    flag_counts: Counter[str] = Counter()
    quality_flag_counts: Counter[str] = Counter()
    missingness_present: Counter[str] = Counter()
    diagnostics_non_null: Counter[str] = Counter()

    total_records = 0
    total_lead_count = 0
    max_lead_count = 0
    time_anchor_complete_count = 0
    forecast_parseable_count = 0

    for split_name, _sample_path, sample in iter_legacy_raw_samples(raw_base_dir, splits=SUPPORTED_SPLITS):
        canonical = canonicalize_legacy_sample(sample, source_split=split_name)
        split_records[split_name].append(canonical)
        split_counts[split_name] += 1
        total_records += 1
        if canonical.get("time_anchor_complete"):
            time_anchor_complete_count += 1
        if (canonical.get("targets", {}) or {}).get("forecast_parseable"):
            forecast_parseable_count += 1

        if canonical.get("storm_id"):
            storm_ids_by_split[split_name].add(str(canonical["storm_id"]))

        for flag_name, flag_value in (canonical.get("flags", {}) or {}).items():
            if flag_value:
                flag_counts[flag_name] += 1
        for quality_flag in (canonical.get("metadata", {}) or {}).get("quality_flags", []) or []:
            quality_flag_counts[str(quality_flag)] += 1

        selected_missingness = _selected_missingness(canonical)
        for field_name, is_present in selected_missingness.items():
            if is_present:
                missingness_present[field_name] += 1

        for field_name in DIAGNOSTIC_FIELDS:
            value = (canonical.get("diagnostics", {}) or {}).get(field_name)
            if value not in (None, "", [], {}):
                diagnostics_non_null[field_name] += 1

        lead_count = len(canonical.get("lead_times", []) or [])
        total_lead_count += lead_count
        max_lead_count = max(max_lead_count, lead_count)

    for split_name, records in split_records.items():
        output_path = output_dir / f"{split_name}.jsonl"
        with output_path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    schema_path = output_dir / "schema.json"
    schema_path.write_text(
        json.dumps(build_canonical_json_schema(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    missingness_report = {}
    for field_name in sorted(_selected_missingness({}).keys()):
        present = missingness_present.get(field_name, 0)
        missingness_report[field_name] = {
            "present": present,
            "missing": total_records - present,
            "present_ratio": _safe_ratio(present, total_records),
        }

    report = {
        "canonical_version": CANONICAL_VERSION,
        "legacy_raw_dir": str(raw_base_dir),
        "output_dir": str(output_dir),
        "total_records": total_records,
        "split_counts": _counter_to_sorted_dict(split_counts),
        "split_storm_counts": {
            split_name: len(storm_ids_by_split.get(split_name, set()))
            for split_name in sorted(split_counts)
        },
        "eligibility_counts": {
            "forecast_view_eligible": flag_counts.get("forecast_view_eligible", 0),
            "reasoning_view_eligible": flag_counts.get("reasoning_view_eligible", 0),
            "diagnostic_view_eligible": flag_counts.get("diagnostic_view_eligible", 0),
        },
        "coverage": {
            "time_anchor_complete_count": time_anchor_complete_count,
            "time_anchor_complete_ratio": _safe_ratio(time_anchor_complete_count, total_records),
            "forecast_parseable_count": forecast_parseable_count,
            "forecast_parseable_ratio": _safe_ratio(forecast_parseable_count, total_records),
            "avg_lead_count": round(total_lead_count / total_records, 2) if total_records else 0.0,
            "max_lead_count": max_lead_count,
        },
        "target_presence": {
            "has_forecast": flag_counts.get("has_forecast", 0),
            "has_reasoning": flag_counts.get("has_reasoning", 0),
            "has_risk": flag_counts.get("has_risk", 0),
            "has_diagnostics": flag_counts.get("has_diagnostics", 0),
        },
        "diagnostics_non_null": {
            field_name: diagnostics_non_null.get(field_name, 0)
            for field_name in DIAGNOSTIC_FIELDS
        },
        "field_missingness": missingness_report,
        "quality_flags": _counter_to_sorted_dict(quality_flag_counts),
    }

    report_path = output_dir / "build_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Build canonical v2 dataset from legacy raw samples")
    parser.add_argument("--base-dir", type=str, default=".")
    parser.add_argument(
        "--legacy-raw-dir",
        type=str,
        default=None,
        help="Legacy raw dataset directory (expects train/val/test subdirectories)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/canonical_v2",
        help="Output directory for canonical v2 JSONL",
    )
    args = parser.parse_args()

    base_dir = Path(args.base_dir).resolve()
    raw_base_dir = (
        (base_dir / args.legacy_raw_dir).resolve()
        if args.legacy_raw_dir
        else infer_latest_legacy_raw_dir(base_dir)
    )
    if raw_base_dir is None or not raw_base_dir.exists():
        raise FileNotFoundError(
            "Could not locate a legacy raw dataset tree. Pass --legacy-raw-dir explicitly."
        )

    output_dir = (base_dir / args.output_dir).resolve()
    report = build_canonical_dataset(
        raw_base_dir=raw_base_dir,
        output_dir=output_dir,
    )
    print(
        f"canonical_v2: {report['total_records']} samples | "
        f"forecast eligible {report['eligibility_counts']['forecast_view_eligible']} | "
        f"reasoning eligible {report['eligibility_counts']['reasoning_view_eligible']} | "
        f"diagnostic eligible {report['eligibility_counts']['diagnostic_view_eligible']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
