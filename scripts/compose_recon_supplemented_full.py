#!/usr/bin/env python3

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compose a full Recon feature table using supplemented yearly outputs when available and baseline yearly outputs otherwise."
    )
    parser.add_argument(
        "--year-start",
        type=int,
        default=2016,
        help="First year to include.",
    )
    parser.add_argument(
        "--year-end",
        type=int,
        default=2025,
        help="Last year to include.",
    )
    parser.add_argument(
        "--base-year-feature-dir",
        type=Path,
        default=Path("data/interim/recon/full_by_year/features"),
        help="Directory containing baseline yearly Recon feature csv files.",
    )
    parser.add_argument(
        "--base-year-summary-dir",
        type=Path,
        default=Path("data/interim/recon/full_by_year/summaries"),
        help="Directory containing baseline yearly Recon summary json files.",
    )
    parser.add_argument(
        "--merged-year-feature-dir",
        type=Path,
        default=Path("data/interim/recon/supplement_secondary_fill/merged_features"),
        help="Directory containing supplemented yearly Recon feature csv files.",
    )
    parser.add_argument(
        "--merged-year-summary-dir",
        type=Path,
        default=Path("data/interim/recon/supplement_secondary_fill/merged_summaries"),
        help="Directory containing supplemented yearly Recon summary json files.",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        required=True,
        help="Output full csv.",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        required=True,
        help="Output full summary json.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)

    rows_written = 0
    available_rows = 0
    missing_rows = 0
    promoted_rows_total = 0
    years_using_merged: List[int] = []
    years_using_base: List[int] = []
    yearly_sources: Dict[str, Dict[str, Any]] = {}
    fieldnames = None  # type: Optional[List[str]]

    with args.out_csv.open("w", encoding="utf-8", newline="") as fw:
        writer = None
        for year in range(args.year_start, args.year_end + 1):
            merged_csv = args.merged_year_feature_dir / f"recon_observation_features_{year}.csv"
            merged_summary = args.merged_year_summary_dir / f"recon_observation_features_{year}_summary.json"
            base_csv = args.base_year_feature_dir / f"recon_observation_features_{year}.csv"
            base_summary = args.base_year_summary_dir / f"recon_observation_features_{year}_summary.json"

            if merged_csv.exists():
                source_csv = merged_csv
                source_summary = merged_summary if merged_summary.exists() else None
                source_kind = "supplemented"
                years_using_merged.append(year)
            elif base_csv.exists():
                source_csv = base_csv
                source_summary = base_summary if base_summary.exists() else None
                source_kind = "baseline"
                years_using_base.append(year)
            else:
                yearly_sources[str(year)] = {
                    "source_kind": "missing",
                    "source_csv": None,
                    "source_summary": None,
                }
                continue

            with source_csv.open("r", encoding="utf-8", newline="") as fr:
                reader = csv.DictReader(fr)
                if fieldnames is None:
                    fieldnames = list(reader.fieldnames or [])
                    writer = csv.DictWriter(fw, fieldnames=fieldnames)
                    writer.writeheader()
                for row in reader:
                    writer.writerow({key: row.get(key, "") for key in fieldnames})
                    rows_written += 1
                    if (row.get("recon_status") or "").strip() == "available":
                        available_rows += 1
                    else:
                        missing_rows += 1

            summary_obj: Dict[str, Any] = {}
            if source_summary is not None and source_summary.exists():
                try:
                    summary_obj = json.loads(source_summary.read_text(encoding="utf-8"))
                except Exception:
                    summary_obj = {}
            promoted_rows_total += int(summary_obj.get("promoted_rows", 0) or 0)
            yearly_sources[str(year)] = {
                "source_kind": source_kind,
                "source_csv": str(source_csv),
                "source_summary": str(source_summary) if source_summary is not None else None,
                "rows_written": int(summary_obj.get("rows_written", 0) or 0),
                "available_rows": int(summary_obj.get("available_rows", 0) or 0),
                "missing_rows": int(summary_obj.get("missing_rows", 0) or 0),
                "promoted_rows": int(summary_obj.get("promoted_rows", 0) or 0),
            }

    summary = {
        "generated_from": "compose_recon_supplemented_full",
        "year_start": args.year_start,
        "year_end": args.year_end,
        "rows_written": rows_written,
        "available_rows": available_rows,
        "missing_rows": missing_rows,
        "promoted_rows_total": promoted_rows_total,
        "years_using_supplemented_outputs": years_using_merged,
        "years_using_baseline_outputs": years_using_base,
        "yearly_sources": yearly_sources,
    }
    args.summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(args.out_csv)
    print(args.summary_json)
    print("rows_written:", rows_written)
    print("available_rows:", available_rows)
    print("missing_rows:", missing_rows)
    print("promoted_rows_total:", promoted_rows_total)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
