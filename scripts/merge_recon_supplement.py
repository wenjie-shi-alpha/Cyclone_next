#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge a baseline Recon feature table with a supplement table that fills previously missing rows."
    )
    parser.add_argument("--base-features-csv", type=Path, required=True, help="Baseline yearly Recon feature csv.")
    parser.add_argument(
        "--supplement-features-csv",
        type=Path,
        required=True,
        help="Supplement Recon feature csv. Missing or empty files are treated as no supplement rows.",
    )
    parser.add_argument("--out-csv", type=Path, required=True, help="Merged output csv.")
    parser.add_argument("--summary-json", type=Path, required=True, help="Merged summary json.")
    parser.add_argument(
        "--available-value",
        type=str,
        default="available",
        help="Rows with this status are considered usable observations.",
    )
    return parser.parse_args()


def load_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def fieldnames_for(base_rows: List[Dict[str, Any]], supplement_rows: List[Dict[str, Any]]) -> List[str]:
    ordered: List[str] = []
    seen = set()
    for rows in (base_rows, supplement_rows):
        if not rows:
            continue
        for key in rows[0].keys():
            if key in seen:
                continue
            seen.add(key)
            ordered.append(key)
    return ordered


def count_status(rows: List[Dict[str, Any]], available_value: str) -> Dict[str, int]:
    available_rows = 0
    missing_rows = 0
    for row in rows:
        if (row.get("recon_status") or "").strip() == available_value:
            available_rows += 1
        else:
            missing_rows += 1
    return {
        "rows_written": len(rows),
        "available_rows": available_rows,
        "missing_rows": missing_rows,
    }


def main() -> int:
    args = parse_args()
    base_rows = load_rows(args.base_features_csv)
    supplement_rows = load_rows(args.supplement_features_csv)
    supplement_by_request_id = {
        (row.get("request_id") or "").strip(): row
        for row in supplement_rows
        if (row.get("request_id") or "").strip()
    }

    merged_rows: List[Dict[str, Any]] = []
    promoted_rows = 0
    for base_row in base_rows:
        request_id = (base_row.get("request_id") or "").strip()
        base_status = (base_row.get("recon_status") or "").strip()
        supplement_row = supplement_by_request_id.get(request_id)
        if supplement_row is not None:
            supplement_status = (supplement_row.get("recon_status") or "").strip()
            if base_status != args.available_value and supplement_status == args.available_value:
                merged_rows.append(supplement_row)
                promoted_rows += 1
                continue
        merged_rows.append(base_row)

    fieldnames = fieldnames_for(base_rows, supplement_rows)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in merged_rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    base_counts = count_status(base_rows, args.available_value)
    supplement_counts = count_status(supplement_rows, args.available_value)
    merged_counts = count_status(merged_rows, args.available_value)
    coverage_by_year: Dict[str, Dict[str, Any]] = {}
    for row in merged_rows:
        year = str(row.get("issue_time_utc") or "")[:4]
        if not year:
            continue
        bucket = coverage_by_year.setdefault(year, {"total": 0, "available": 0, "missing": 0})
        bucket["total"] += 1
        if (row.get("recon_status") or "").strip() == args.available_value:
            bucket["available"] += 1
        else:
            bucket["missing"] += 1
    for year, bucket in coverage_by_year.items():
        total = bucket["total"]
        bucket["coverage_rate"] = round((bucket["available"] / total), 6) if total else 0.0

    summary = {
        "generated_from": "baseline_plus_secondary_supplement",
        "base_features_csv": str(args.base_features_csv),
        "supplement_features_csv": str(args.supplement_features_csv),
        "out_csv": str(args.out_csv),
        "base_rows_total": base_counts["rows_written"],
        "base_available_rows": base_counts["available_rows"],
        "base_missing_rows": base_counts["missing_rows"],
        "supplement_rows_total": supplement_counts["rows_written"],
        "supplement_available_rows": supplement_counts["available_rows"],
        "supplement_missing_rows": supplement_counts["missing_rows"],
        "promoted_rows": promoted_rows,
        "rows_written": merged_counts["rows_written"],
        "available_rows": merged_counts["available_rows"],
        "missing_rows": merged_counts["missing_rows"],
        "coverage_by_year": coverage_by_year,
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(args.out_csv)
    print(args.summary_json)
    print("promoted_rows:", promoted_rows)
    print("available_rows:", merged_counts["available_rows"])
    print("missing_rows:", merged_counts["missing_rows"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
