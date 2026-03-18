#!/usr/bin/env python3
"""Merge yearly ASCAT outputs into full and canonical tables."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Sequence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge yearly ASCAT feature outputs and summaries."
    )
    parser.add_argument(
        "--year-csv-dir",
        type=Path,
        default=Path("data/interim/ascat/full_by_year/features"),
        help="Directory containing yearly ASCAT feature csv files.",
    )
    parser.add_argument(
        "--year-summary-dir",
        type=Path,
        default=Path("data/interim/ascat/full_by_year/summaries"),
        help="Directory containing yearly ASCAT summary json files.",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path("data/interim/ascat/ascat_observation_features_full.csv"),
        help="Merged full ASCAT feature csv path.",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=Path("data/interim/ascat/ascat_observation_features_full_summary.json"),
        help="Merged ASCAT summary json path.",
    )
    parser.add_argument(
        "--canonical-out",
        type=Path,
        default=Path("data/interim/ascat/ascat_observation_features.csv"),
        help="Canonical csv sync path.",
    )
    parser.add_argument(
        "--canonical-summary",
        type=Path,
        default=Path("data/interim/ascat/ascat_observation_features_summary.json"),
        help="Canonical summary sync path.",
    )
    parser.add_argument(
        "--require-years-csv",
        type=str,
        default="",
        help="Optional comma/space separated years that must exist before merge.",
    )
    parser.add_argument(
        "--generated-from",
        type=str,
        default="yearly_controlled_runs",
        help="Tag written to merged summary json.",
    )
    return parser.parse_args()


def parse_year_tokens(value: str) -> List[str]:
    years: List[str] = []
    for token in (value or "").replace(",", " ").split():
        token = token.strip()
        if not token:
            continue
        if not token.isdigit():
            raise ValueError(f"invalid year token: {token}")
        years.append(token)
    return years


def year_from_feature_path(path: Path) -> Optional[str]:
    stem = path.stem
    prefix = "ascat_observation_features_"
    if not stem.startswith(prefix):
        return None
    year = stem[len(prefix) :]
    return year if year.isdigit() else None


def year_from_summary_path(path: Path) -> Optional[str]:
    stem = path.stem
    prefix = "ascat_observation_features_"
    if not stem.startswith(prefix):
        return None
    tail = stem[len(prefix) :]
    if not tail.endswith("_summary"):
        return None
    year = tail[: -len("_summary")]
    return year if year.isdigit() else None


def build_path_map(paths: Sequence[Path], year_fn) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for path in paths:
        year = year_fn(path)
        if year:
            out[year] = path
    return out


def ensure_required_years(
    required_years: Sequence[str],
    csv_by_year: Dict[str, Path],
    summary_by_year: Dict[str, Path],
) -> None:
    missing: List[str] = []
    for year in required_years:
        if year not in csv_by_year or year not in summary_by_year:
            missing.append(year)
    if missing:
        raise RuntimeError(
            "missing yearly ASCAT outputs for required years: " + ",".join(sorted(missing))
        )


def main() -> int:
    args = parse_args()
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(args.year_csv_dir.glob("ascat_observation_features_*.csv"))
    summary_files = sorted(args.year_summary_dir.glob("ascat_observation_features_*_summary.json"))
    if not csv_files:
        raise RuntimeError(f"no yearly ASCAT csv outputs found under {args.year_csv_dir}")

    csv_by_year = build_path_map(csv_files, year_from_feature_path)
    summary_by_year = build_path_map(summary_files, year_from_summary_path)
    required_years = parse_year_tokens(args.require_years_csv)
    if required_years:
        ensure_required_years(required_years, csv_by_year, summary_by_year)

    fieldnames: Optional[List[str]] = None
    rows_written = 0
    available_rows = 0
    missing_rows = 0
    by_year: Dict[str, Dict[str, int]] = {}

    with args.out_csv.open("w", encoding="utf-8", newline="") as fw:
        writer = None
        for year in sorted(csv_by_year.keys()):
            fp = csv_by_year[year]
            with fp.open("r", encoding="utf-8", newline="") as fr:
                reader = csv.DictReader(fr)
                if fieldnames is None:
                    fieldnames = reader.fieldnames or []
                    if not fieldnames:
                        raise RuntimeError(f"yearly csv missing header: {fp}")
                    writer = csv.DictWriter(fw, fieldnames=fieldnames)
                    writer.writeheader()
                for row in reader:
                    assert writer is not None
                    writer.writerow(row)
                    rows_written += 1
                    bucket = by_year.setdefault(year, {"total": 0, "available": 0, "missing": 0})
                    bucket["total"] += 1
                    status = (row.get("ascat_status") or "").strip()
                    if status == "available":
                        available_rows += 1
                        bucket["available"] += 1
                    else:
                        missing_rows += 1
                        bucket["missing"] += 1

    coverage_by_year: Dict[str, Dict[str, float]] = {}
    for year in sorted(by_year.keys()):
        total = by_year[year]["total"]
        available = by_year[year]["available"]
        missing = by_year[year]["missing"]
        coverage_by_year[year] = {
            "total": total,
            "available": available,
            "missing": missing,
            "coverage_rate": round((available / total) if total else 0.0, 6),
        }

    dataset_ids_used: List[str] = []
    yearly_summary_files: List[str] = []
    for year in sorted(summary_by_year.keys()):
        fp = summary_by_year[year]
        yearly_summary_files.append(str(fp))
        obj = json.loads(fp.read_text(encoding="utf-8"))
        for ds in obj.get("dataset_ids_used", []):
            if ds not in dataset_ids_used:
                dataset_ids_used.append(ds)

    merged_summary = {
        "generated_from": args.generated_from,
        "requests_total": rows_written,
        "rows_written": rows_written,
        "available_rows": available_rows,
        "missing_rows": missing_rows,
        "dataset_ids_used": dataset_ids_used,
        "coverage_by_year": coverage_by_year,
        "yearly_summary_files": yearly_summary_files,
    }
    args.summary_json.write_text(
        json.dumps(merged_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    args.canonical_out.parent.mkdir(parents=True, exist_ok=True)
    args.canonical_summary.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.out_csv, args.canonical_out)
    shutil.copyfile(args.summary_json, args.canonical_summary)

    print(args.out_csv)
    print(args.summary_json)
    print("rows_written:", rows_written)
    print("available_rows:", available_rows)
    print("missing_rows:", missing_rows)
    print("canonical_out:", args.canonical_out)
    print("canonical_summary:", args.canonical_summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
