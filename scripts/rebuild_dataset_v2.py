#!/usr/bin/env python3
"""End-to-end rebuild for canonical v2 plus view exports."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from build_canonical_v2 import build_canonical_dataset
from dataset_v2 import infer_latest_legacy_raw_dir
from export_views_v2 import export_views


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Rebuild dataset into canonical v2 and exported views")
    parser.add_argument("--base-dir", type=str, default=".")
    parser.add_argument(
        "--legacy-raw-dir",
        type=str,
        default=None,
        help="Legacy raw dataset tree with train/val/test subdirectories",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output dataset root. Defaults to data/training_rebuilt_v2_<timestamp>",
    )
    parser.add_argument(
        "--skip-test-variants",
        action="store_true",
        help="Do not regenerate forecast test evaluation variants",
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

    if args.output_dir:
        output_dir = (base_dir / args.output_dir).resolve()
    else:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        output_dir = (base_dir / f"data/training_rebuilt_v2_{timestamp}").resolve()

    canonical_dir = output_dir / "canonical_v2"
    print(f"+ build canonical_v2 from {raw_base_dir}")
    canonical_report = build_canonical_dataset(
        raw_base_dir=raw_base_dir,
        output_dir=canonical_dir,
    )
    print(
        f"  canonical records: {canonical_report['total_records']} | "
        f"forecast eligible {canonical_report['eligibility_counts']['forecast_view_eligible']}"
    )

    print(f"+ export views into {output_dir}")
    export_report = export_views(
        base_dir=base_dir,
        canonical_dir=canonical_dir,
        output_dir=output_dir,
        include_test_variants=not args.skip_test_variants,
    )
    forecast_train = ((export_report.get("forecast_only", {}) or {}).get("sft_train", {}) or {}).get("count", 0)
    reasoning_train = ((export_report.get("reasoning_only", {}) or {}).get("sft_train", {}) or {}).get("count", 0)
    diagnostic_train = ((export_report.get("diagnostic_only", {}) or {}).get("sft_train", {}) or {}).get("count", 0)
    print(
        f"  view counts: forecast train {forecast_train} | "
        f"reasoning train {reasoning_train} | "
        f"diagnostic train {diagnostic_train}"
    )
    print(f"+ done: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
