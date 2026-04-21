#!/usr/bin/env python3
"""End-to-end rebuild for SFT/RL datasets plus readiness validation."""

from __future__ import annotations

import json
import csv
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List


def run_command(cmd: List[str], cwd: Path) -> None:
    """Run one subprocess and fail loudly if it exits non-zero."""
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def count_jsonl_lines(path: Path) -> int:
    """Count lines in a JSONL file."""
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def inspect_first_record(path: Path) -> Dict[str, Any]:
    """Read the first JSONL record for schema sanity checks."""
    with path.open("r", encoding="utf-8") as handle:
        first_line = next(handle, "")
    return json.loads(first_line) if first_line else {}


def read_json_with_retry(
    path: Path, attempts: int = 5, delay_sec: float = 0.2
) -> Dict[str, Any]:
    """Read one JSON file, tolerating short concurrent-write windows."""
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            last_error = exc
            if attempt == attempts - 1:
                raise
            time.sleep(delay_sec)
    if last_error is not None:
        raise last_error
    raise RuntimeError(f"Failed to read {path}")


def load_filtered_manifest_rows(
    manifest_csv: Path,
    year_start: int,
    year_end: int,
) -> List[Dict[str, str]]:
    """Load manifest rows using the same matched/year filter as the raw builder."""
    rows: List[Dict[str, str]] = []
    with manifest_csv.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            storm_id = (row.get("storm_id") or "").strip()
            match_status = (row.get("storm_id_match_status") or "").strip()
            year = int(storm_id[:4]) if len(storm_id) >= 4 and storm_id[:4].isdigit() else 0
            if match_status != "matched":
                continue
            if year < year_start or year > year_end:
                continue
            rows.append(row)
    return rows


def refresh_build_artifacts_from_raw(
    output_dir: Path,
    raw_base_dir: Path,
    manifest_csv: Path,
    year_start: int,
    year_end: int,
) -> Dict[str, Any]:
    """Reindex build_manifest/build_report from an already complete raw dataset tree."""
    manifest_rows = load_filtered_manifest_rows(manifest_csv, year_start, year_end)
    previous_report_path = output_dir / "build_report.json"
    previous_report: Dict[str, Any] = {}
    if previous_report_path.exists():
        previous_report = json.loads(previous_report_path.read_text(encoding="utf-8"))

    split_counts = {"train": 0, "val": 0, "test": 0, "unassigned": 0}
    build_manifest_entries: List[Dict[str, Any]] = []

    for split_name in ["train", "val", "test", "unassigned"]:
        split_dir = raw_base_dir / split_name
        if not split_dir.exists():
            continue
        for raw_path in sorted(split_dir.glob("*.json")):
            sample = read_json_with_retry(raw_path)
            prompt = sample.get("prompt", {}) or {}
            storm_meta = prompt.get("storm_meta", {}) or {}
            build_manifest_entries.append(
                {
                    "sample_id": sample.get("sample_id", raw_path.stem),
                    "storm_id": storm_meta.get("storm_id", ""),
                    "split": split_name,
                    "issue_time_utc": storm_meta.get("issue_time_utc", ""),
                    "advisory_no": storm_meta.get("advisory_no", 0),
                }
            )
            split_counts[split_name] += 1

    built = sum(split_counts.values())
    skipped = max(0, len(manifest_rows) - built)

    skip_reasons = previous_report.get("skip_reasons", {})
    if sum(skip_reasons.values()) != skipped:
        skip_reasons = previous_report.get("skip_reasons", {}) if previous_report.get("skipped") == skipped else {}

    build_manifest_path = output_dir / "build_manifest.json"
    build_manifest_path.write_text(
        json.dumps(build_manifest_entries, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    build_report = {
        "total_manifest_rows": len(manifest_rows),
        "built": built,
        "skipped": skipped,
        "skip_reasons": skip_reasons,
        "split_counts": split_counts,
        "raw_sample_audit_summary": {
            "enabled": False,
            "temporal_pass": 0,
            "verification_pass": 0,
            "identity_avg": 0.0,
            "total_audited": 0,
            "note": (
                "Refreshed from existing raw samples. Raw canonical-sample audit was not rerun. "
                "Use format_report.json for final train-view leakage checks."
            ),
        },
        "stale_raw_removed": 0,
        "reindexed_from_existing_raw": True,
    }
    build_report_path = output_dir / "build_report.json"
    build_report_path.write_text(
        json.dumps(build_report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return build_report


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Full rebuild for training datasets")
    parser.add_argument("--base-dir", type=str, default=".")
    parser.add_argument("--output-dir", type=str, default="data/training")
    parser.add_argument(
        "--raw-dir",
        type=str,
        default=None,
        help="Existing raw dataset directory to format from when --skip-build is used",
    )
    parser.add_argument("--manifest", type=str, default="data/interim/ascat/ascat_request_manifest_full.csv")
    parser.add_argument("--split-manifest", type=str, default="data/interim/leakage_prevention/split_manifest_v1.json")
    parser.add_argument("--year-start", type=int, default=2016)
    parser.add_argument("--year-end", type=int, default=2025)
    parser.add_argument("--skip-test-variants", action="store_true")
    parser.add_argument("--skip-reasoning-aux", action="store_true")
    parser.add_argument(
        "--enable-raw-audit",
        action="store_true",
        help="Run slower raw canonical-sample audit during the full raw rebuild",
    )
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Reuse the existing full raw dataset tree and refresh build_report/build_manifest",
    )

    args = parser.parse_args()
    if args.skip_build and args.enable_raw_audit:
        parser.error("--enable-raw-audit cannot be used together with --skip-build")

    base_dir = Path(args.base_dir).resolve()
    output_dir = base_dir / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir_arg = args.raw_dir or f"{args.output_dir}/raw"
    raw_base = base_dir / raw_dir_arg

    build_cmd = [
        sys.executable,
        "scripts/build_dataset_batch.py",
        "--base-dir", str(base_dir),
        "--output-dir", args.output_dir,
        "--manifest", args.manifest,
        "--split-manifest", args.split_manifest,
        "--year-start", str(args.year_start),
        "--year-end", str(args.year_end),
    ]
    if args.enable_raw_audit:
        build_cmd.append("--enable-raw-audit")
    format_cmd = [
        sys.executable,
        "scripts/dataset_formatter.py",
        "--base-dir", str(base_dir),
        "--raw-dir", raw_dir_arg,
        "--output-dir", args.output_dir,
        "--format", "both",
    ]
    if args.skip_test_variants:
        format_cmd.append("--skip-test-variants")
    if args.skip_reasoning_aux:
        format_cmd.append("--skip-reasoning-aux")

    manifest_csv = base_dir / args.manifest
    if args.skip_build:
        print("+ refresh build artifacts from existing raw dataset")
        refresh_build_artifacts_from_raw(
            output_dir=output_dir,
            raw_base_dir=raw_base,
            manifest_csv=manifest_csv,
            year_start=args.year_start,
            year_end=args.year_end,
        )
    else:
        run_command(build_cmd, cwd=base_dir)
        raw_base = output_dir / "raw"
    run_command(format_cmd, cwd=base_dir)

    build_report = json.loads((output_dir / "build_report.json").read_text(encoding="utf-8"))
    format_report = json.loads((output_dir / "format_report.json").read_text(encoding="utf-8"))

    raw_counts_actual = {
        split: sum(1 for _ in (raw_base / split).glob("*.json"))
        for split in ["train", "val", "test", "unassigned"]
    }
    sft_counts_actual = {
        split: count_jsonl_lines(output_dir / f"sft_{split}.jsonl")
        for split in ["train", "val", "test"]
    }
    sft_reasoning_counts_actual: Dict[str, int] = {}
    if not args.skip_reasoning_aux:
        sft_reasoning_counts_actual = {
            split: count_jsonl_lines(output_dir / f"sft_reasoning_{split}.jsonl")
            for split in ["train", "val", "test"]
        }
    rl_counts_actual = {
        split: count_jsonl_lines(output_dir / f"rl_{split}.jsonl")
        for split in ["train", "val", "test"]
    }
    variant_counts_actual: Dict[str, int] = {}
    if not args.skip_test_variants:
        for variant in ["anonymous", "structured_only", "perturbation"]:
            variant_counts_actual[variant] = count_jsonl_lines(
                output_dir / f"sft_test_{variant}.jsonl"
            )

    checks = {
        "build_report_refreshed": (
            "raw_sample_audit_summary" in build_report
            and "stale_raw_removed" in build_report
        ),
        "raw_counts_match_build_report": raw_counts_actual == build_report.get("split_counts", {}),
        "sft_main_accounted_against_raw": all(
            sft_counts_actual[split] + format_report.get(f"sft_{split}", {}).get("skipped", 0)
            == raw_counts_actual[split]
            for split in ["train", "val", "test"]
        ),
        "sft_main_strict_parseable": all(
            format_report.get(f"sft_{split}", {}).get("assistant_schema", {}).get("strict_forecast_parseable", -1)
            == sft_counts_actual[split]
            and format_report.get(f"sft_{split}", {}).get("assistant_schema", {}).get("strict_forecast_with_extra_text", 1)
            == 0
            and format_report.get(f"sft_{split}", {}).get("assistant_schema", {}).get("no_track_forecast", 1)
            == 0
            for split in ["train", "val", "test"]
        ),
        "sft_main_time_anchor_complete": all(
            format_report.get(f"sft_{split}", {}).get("time_anchor", {}).get("issue_time_anchor_present", -1)
            == sft_counts_actual[split]
            and format_report.get(f"sft_{split}", {}).get("time_anchor", {}).get("track_time_labels_present", -1)
            == sft_counts_actual[split]
            for split in ["train", "val", "test"]
        ),
        "rl_counts_match_format_report": all(
            rl_counts_actual[split] == format_report.get(f"rl_{split}", {}).get("count", -1)
            for split in ["train", "val", "test"]
        ),
        "prompt_leakage_clean_main_sft": all(
            format_report.get(f"sft_{split}", {}).get("train_view_leakage", {}).get(flag, 1) == 0
            for split in ["train", "val", "test"]
            for flag in [
                "prompt_contains_cjk",
                "prompt_contains_iso_datetime",
                "prompt_contains_storm_name",
                "prompt_contains_storm_id",
            ]
        ),
        "assistant_leakage_clean_main_sft": all(
            format_report.get(f"sft_{split}", {}).get("train_view_leakage", {}).get(flag, 1) == 0
            for split in ["train", "val", "test"]
            for flag in [
                "assistant_contains_cjk",
                "assistant_contains_iso_datetime",
                "assistant_contains_storm_name",
                "assistant_contains_storm_id",
            ]
        ),
        "test_variant_counts_match_main_test": (
            args.skip_test_variants
            or all(count == sft_counts_actual["test"] for count in variant_counts_actual.values())
        ),
    }
    if not args.skip_reasoning_aux:
        checks["sft_reasoning_accounted_against_raw"] = all(
            sft_reasoning_counts_actual[split]
            + format_report.get(f"sft_reasoning_{split}", {}).get("skipped", 0)
            == raw_counts_actual[split]
            for split in ["train", "val", "test"]
        )
    if args.enable_raw_audit:
        checks["raw_audit_enabled_when_requested"] = bool(
            build_report.get("raw_sample_audit_summary", {}).get("enabled")
        )

    variant_schema_ok = True
    if not args.skip_test_variants:
        for variant in ["anonymous", "structured_only", "perturbation"]:
            record = inspect_first_record(output_dir / f"sft_test_{variant}.jsonl")
            if record.get("format_version") != format_report.get("format_version"):
                variant_schema_ok = False
                break
            if record.get("evaluation_variant") != variant:
                variant_schema_ok = False
                break
    checks["test_variant_schema_synced"] = variant_schema_ok

    rl_quality_ready_counts = {
        split: format_report.get(f"rl_{split}", {}).get("count", 0)
        for split in ["train", "val", "test"]
    }

    ready_report = {
        "format_version": format_report.get("format_version"),
        "raw_dir": str(raw_base),
        "raw_counts_actual": raw_counts_actual,
        "sft_counts_actual": sft_counts_actual,
        "sft_reasoning_counts_actual": sft_reasoning_counts_actual,
        "rl_counts_actual": rl_counts_actual,
        "variant_counts_actual": variant_counts_actual,
        "rl_quality_ready_counts": rl_quality_ready_counts,
        "checks": checks,
        "all_checks_passed": all(checks.values()),
        "artifacts": {
            "build_report": str(output_dir / "build_report.json"),
            "format_report": str(output_dir / "format_report.json"),
            "sft_train": str(output_dir / "sft_train.jsonl"),
            "sft_val": str(output_dir / "sft_val.jsonl"),
            "sft_test": str(output_dir / "sft_test.jsonl"),
            "rl_train": str(output_dir / "rl_train.jsonl"),
            "rl_val": str(output_dir / "rl_val.jsonl"),
            "rl_test": str(output_dir / "rl_test.jsonl"),
        },
    }
    if not args.skip_reasoning_aux:
        ready_report["artifacts"].update({
            "sft_reasoning_train": str(output_dir / "sft_reasoning_train.jsonl"),
            "sft_reasoning_val": str(output_dir / "sft_reasoning_val.jsonl"),
            "sft_reasoning_test": str(output_dir / "sft_reasoning_test.jsonl"),
        })
    if not args.skip_test_variants:
        ready_report["artifacts"].update({
            "sft_test_anonymous": str(output_dir / "sft_test_anonymous.jsonl"),
            "sft_test_structured_only": str(output_dir / "sft_test_structured_only.jsonl"),
            "sft_test_perturbation": str(output_dir / "sft_test_perturbation.jsonl"),
        })

    ready_path = output_dir / "dataset_ready_report.json"
    ready_path.write_text(
        json.dumps(ready_report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print()
    print("=== Dataset Ready Summary ===")
    print(f"Raw counts: {raw_counts_actual}")
    print(f"SFT counts: {sft_counts_actual}")
    if sft_reasoning_counts_actual:
        print(f"SFT reasoning counts: {sft_reasoning_counts_actual}")
    print(f"RL counts: {rl_counts_actual}")
    if variant_counts_actual:
        print(f"Variant counts: {variant_counts_actual}")
    print(f"Checks passed: {ready_report['all_checks_passed']}")
    print(f"Ready report: {ready_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
