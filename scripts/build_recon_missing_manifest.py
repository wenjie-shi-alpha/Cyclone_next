#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Set


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a Recon manifest containing only request_ids that are still missing in a base feature table."
    )
    parser.add_argument(
        "--manifest-csv",
        type=Path,
        required=True,
        help="Full request manifest csv for a year or full period.",
    )
    parser.add_argument(
        "--base-features-csv",
        type=Path,
        required=True,
        help="Existing Recon feature csv used as the baseline.",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        required=True,
        help="Output manifest csv with only missing request_ids.",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="Optional summary json path.",
    )
    parser.add_argument(
        "--status-field",
        type=str,
        default="recon_status",
        help="Status field in the base feature csv.",
    )
    parser.add_argument(
        "--available-value",
        type=str,
        default="available",
        help="Rows with this status are excluded from the missing manifest.",
    )
    return parser.parse_args()


def load_missing_request_ids(args: argparse.Namespace) -> Dict[str, str]:
    status_by_request_id: Dict[str, str] = {}
    with args.base_features_csv.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            request_id = (row.get("request_id") or "").strip()
            if not request_id:
                continue
            status_by_request_id[request_id] = (row.get(args.status_field) or "").strip()
    return status_by_request_id


def main() -> int:
    args = parse_args()
    status_by_request_id = load_missing_request_ids(args)
    missing_request_ids: Set[str] = {
        request_id
        for request_id, status in status_by_request_id.items()
        if status != args.available_value
    }

    rows_written = 0
    years: Set[str] = set()
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)

    with args.manifest_csv.open("r", encoding="utf-8", newline="") as f_in:
        reader = csv.DictReader(f_in)
        fieldnames: List[str] = list(reader.fieldnames or [])
        with args.out_csv.open("w", encoding="utf-8", newline="") as f_out:
            writer = csv.DictWriter(f_out, fieldnames=fieldnames)
            writer.writeheader()
            for row in reader:
                request_id = (row.get("request_id") or "").strip()
                if request_id not in missing_request_ids:
                    continue
                writer.writerow(row)
                rows_written += 1
                issue_time_utc = str(row.get("issue_time_utc") or "")
                if issue_time_utc:
                    years.add(issue_time_utc[:4])

    summary = {
        "manifest_csv": str(args.manifest_csv),
        "base_features_csv": str(args.base_features_csv),
        "out_csv": str(args.out_csv),
        "status_field": args.status_field,
        "available_value": args.available_value,
        "base_rows_total": len(status_by_request_id),
        "missing_request_ids_total": len(missing_request_ids),
        "rows_written": rows_written,
        "issue_years": sorted(y for y in years if y),
    }

    if args.summary_json is not None:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(args.out_csv)
    print("rows_written:", rows_written)
    print("missing_request_ids_total:", len(missing_request_ids))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
