#!/usr/bin/env python3
"""Split the ASCAT full request manifest into yearly manifest shards."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split ASCAT full manifest into yearly manifest shards."
    )
    parser.add_argument(
        "--full-manifest-csv",
        type=Path,
        default=Path("data/interim/ascat/ascat_request_manifest_full.csv"),
        help="Source ASCAT full manifest csv.",
    )
    parser.add_argument(
        "--year-manifest-dir",
        type=Path,
        default=Path("data/interim/ascat/full_by_year/manifests"),
        help="Output directory for yearly ASCAT manifest csv files.",
    )
    parser.add_argument(
        "--year-start",
        type=int,
        default=2016,
        help="Inclusive start year.",
    )
    parser.add_argument(
        "--year-end",
        type=int,
        default=2025,
        help="Inclusive end year.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.year_end < args.year_start:
        raise ValueError("year_end must be >= year_start")
    if not args.full_manifest_csv.exists():
        raise FileNotFoundError(f"full manifest not found: {args.full_manifest_csv}")

    args.year_manifest_dir.mkdir(parents=True, exist_ok=True)

    rows_by_year: Dict[str, List[Dict[str, str]]] = {}
    with args.full_manifest_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        if not fieldnames:
            raise RuntimeError(f"manifest missing header: {args.full_manifest_csv}")
        for row in reader:
            year_txt = (row.get("issue_time_utc") or "").strip()[:4]
            if not year_txt.isdigit():
                continue
            year = int(year_txt)
            if year < args.year_start or year > args.year_end:
                continue
            rows_by_year.setdefault(year_txt, []).append(row)

    total_rows = 0
    years_written = 0
    for year in range(args.year_start, args.year_end + 1):
        year_txt = str(year)
        out_path = args.year_manifest_dir / f"ascat_request_manifest_{year_txt}.csv"
        rows = rows_by_year.get(year_txt, [])
        if not rows:
            if out_path.exists():
                out_path.unlink()
            print(f"{year_txt}: 0")
            continue

        with out_path.open("w", encoding="utf-8", newline="") as fw:
            writer = csv.DictWriter(fw, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        total_rows += len(rows)
        years_written += 1
        print(f"{year_txt}: {len(rows)}")

    print("years_written:", years_written)
    print("rows_total:", total_rows)
    print("year_manifest_dir:", args.year_manifest_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
