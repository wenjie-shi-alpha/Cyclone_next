#!/usr/bin/env python3
"""P0 Step 1: Basin scope check for CDS_real vs NOAA target samples.

Outputs:
1. basin_scope_report.json
2. cds_scope_points.csv
3. cds_scope_file_summary.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple


def normalize_basin_name(raw: Optional[str]) -> Optional[str]:
    """Normalize basin names from multiple naming styles to one canonical label."""
    if raw is None:
        return None
    text = str(raw).strip().lower()
    if not text:
        return None

    token = text.replace("-", "_").replace(" ", "_")

    # Direct aliases.
    alias_map = {
        "atlantic": "Atlantic",
        "north_atlantic": "Atlantic",
        "al": "Atlantic",
        "e_pacific": "E_Pacific",
        "east_pacific": "E_Pacific",
        "eastern_pacific": "E_Pacific",
        "epac": "E_Pacific",
        "c_pacific": "C_Pacific",
        "central_pacific": "C_Pacific",
        "cpac": "C_Pacific",
        "w_pacific": "W_Pacific",
        "west_pacific": "W_Pacific",
        "western_pacific": "W_Pacific",
        "wpac": "W_Pacific",
        "n_indian": "N_Indian",
        "north_indian": "N_Indian",
        "s_indian": "S_Indian",
        "south_indian": "S_Indian",
        "s_pacific": "S_Pacific",
        "south_pacific": "S_Pacific",
    }
    if token in alias_map:
        return alias_map[token]

    # Substring heuristics for filenames like western_pacific_typhoons_superfast.csv.
    if "western_pacific" in token or "west_pacific" in token or "wpac" in token:
        return "W_Pacific"
    if "eastern_pacific" in token or "east_pacific" in token or "epac" in token:
        return "E_Pacific"
    if "central_pacific" in token or "c_pacific" in token or "cpac" in token:
        return "C_Pacific"
    if "atlantic" in token:
        return "Atlantic"
    if "north_indian" in token or "n_indian" in token:
        return "N_Indian"
    if "south_indian" in token or "s_indian" in token:
        return "S_Indian"
    if "south_pacific" in token or "s_pacific" in token:
        return "S_Pacific"

    return None


def infer_cds_basin(metadata: Dict[str, Any]) -> Optional[str]:
    """Infer CDS basin from metadata fields."""
    candidates = [
        metadata.get("basin"),
        metadata.get("tracks_file"),
        metadata.get("region"),
        metadata.get("source_tracks"),
    ]
    for candidate in candidates:
        basin = normalize_basin_name(candidate)
        if basin:
            return basin
    return None


def safe_float(value: Any) -> Optional[float]:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def collect_groundtruth_basins(csv_path: Path) -> Tuple[Counter, Counter, set[str]]:
    point_counts = Counter()
    storms_by_basin = {}

    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            basin = normalize_basin_name(row.get("noaa_basin"))
            storm_id = (row.get("storm_id") or "").strip()
            if not basin:
                continue
            point_counts[basin] += 1
            if storm_id:
                storms_by_basin.setdefault(basin, set()).add(storm_id)

    storm_counts = Counter({k: len(v) for k, v in storms_by_basin.items()})
    basins = set(point_counts)
    return point_counts, storm_counts, basins


def collect_noaa_basins(noaa_dir: Path) -> Tuple[Counter, set[str]]:
    storm_counts = Counter()
    basin_set: set[str] = set()

    for year_dir in sorted(noaa_dir.iterdir()):
        if not year_dir.is_dir():
            continue
        for basin_dir in sorted(year_dir.iterdir()):
            if not basin_dir.is_dir():
                continue
            basin = normalize_basin_name(basin_dir.name)
            if not basin:
                continue
            basin_set.add(basin)
            count = sum(1 for p in basin_dir.iterdir() if p.is_dir())
            storm_counts[basin] += count

    return storm_counts, basin_set


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run basin scope check for CDS_real data.")
    parser.add_argument(
        "--groundtruth-csv",
        default="GroundTruth_Cyclones/matched_cyclone_tracks.csv",
        help="Path to GroundTruth cyclone track CSV.",
    )
    parser.add_argument(
        "--noaa-dir",
        default="noaa",
        help="Path to NOAA local archive root directory.",
    )
    parser.add_argument(
        "--cds-dir",
        default="CDS_real",
        help="Path to CDS real environment JSON directory.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/interim/basin_scope_check",
        help="Directory for basin-scope outputs.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    groundtruth_csv = Path(args.groundtruth_csv)
    noaa_dir = Path(args.noaa_dir)
    cds_dir = Path(args.cds_dir)
    output_dir = Path(args.output_dir)

    if not groundtruth_csv.exists():
        raise FileNotFoundError(f"GroundTruth CSV not found: {groundtruth_csv}")
    if not noaa_dir.exists():
        raise FileNotFoundError(f"NOAA directory not found: {noaa_dir}")
    if not cds_dir.exists():
        raise FileNotFoundError(f"CDS directory not found: {cds_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    points_csv = output_dir / "cds_scope_points.csv"
    summary_csv = output_dir / "cds_scope_file_summary.csv"
    report_json = output_dir / "basin_scope_report.json"

    gt_point_counts, gt_storm_counts, gt_basins = collect_groundtruth_basins(groundtruth_csv)
    noaa_storm_counts, noaa_basins = collect_noaa_basins(noaa_dir)

    target_basins = sorted(gt_basins | noaa_basins)
    target_basin_set = set(target_basins)

    cds_files = sorted(cds_dir.glob("*.json"))

    tracks_file_counts = Counter()
    declared_basin_counts = Counter()
    declared_basins = set()
    total_rows = 0
    total_in_scope = 0
    total_out_scope = 0

    with points_csv.open("w", encoding="utf-8", newline="") as points_handle, summary_csv.open(
        "w", encoding="utf-8", newline=""
    ) as summary_handle:
        points_writer = csv.DictWriter(
            points_handle,
            fieldnames=[
                "source_file",
                "month_processed",
                "tracks_file",
                "time",
                "time_idx",
                "tc_lat",
                "tc_lon",
                "cds_basin_declared",
                "in_scope",
                "out_of_scope",
                "scope_reason",
            ],
        )
        summary_writer = csv.DictWriter(
            summary_handle,
            fieldnames=[
                "source_file",
                "month_processed",
                "tracks_file",
                "cds_basin_declared",
                "total_rows",
                "in_scope_rows",
                "out_of_scope_rows",
                "out_of_scope_ratio",
            ],
        )

        points_writer.writeheader()
        summary_writer.writeheader()

        for cds_file in cds_files:
            with cds_file.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)

            metadata = payload.get("metadata", {})
            rows = payload.get("environmental_analysis", [])
            tracks_file = str(metadata.get("tracks_file", ""))
            month_processed = str(metadata.get("month_processed", ""))

            tracks_file_counts[tracks_file] += 1

            cds_basin = infer_cds_basin(metadata)
            cds_basin_text = cds_basin if cds_basin else "UNKNOWN"
            declared_basin_counts[cds_basin_text] += 1
            declared_basins.add(cds_basin_text)

            file_in_scope = 0
            file_out_scope = 0

            for row in rows:
                total_rows += 1
                in_scope = cds_basin is not None and cds_basin in target_basin_set
                out_scope = not in_scope

                if in_scope:
                    scope_reason = "cds_basin_in_noaa_targets"
                    file_in_scope += 1
                    total_in_scope += 1
                elif cds_basin is None:
                    scope_reason = "cds_basin_unknown"
                    file_out_scope += 1
                    total_out_scope += 1
                else:
                    scope_reason = "cds_basin_not_in_noaa_targets"
                    file_out_scope += 1
                    total_out_scope += 1

                tc_position = row.get("tc_position") or {}
                points_writer.writerow(
                    {
                        "source_file": cds_file.name,
                        "month_processed": month_processed,
                        "tracks_file": tracks_file,
                        "time": row.get("time"),
                        "time_idx": row.get("time_idx"),
                        "tc_lat": safe_float(tc_position.get("lat")),
                        "tc_lon": safe_float(tc_position.get("lon")),
                        "cds_basin_declared": cds_basin_text,
                        "in_scope": int(in_scope),
                        "out_of_scope": int(out_scope),
                        "scope_reason": scope_reason,
                    }
                )

            out_ratio = (file_out_scope / len(rows)) if rows else 0.0
            summary_writer.writerow(
                {
                    "source_file": cds_file.name,
                    "month_processed": month_processed,
                    "tracks_file": tracks_file,
                    "cds_basin_declared": cds_basin_text,
                    "total_rows": len(rows),
                    "in_scope_rows": file_in_scope,
                    "out_of_scope_rows": file_out_scope,
                    "out_of_scope_ratio": round(out_ratio, 6),
                }
            )

    intersection = sorted(target_basin_set.intersection({b for b in declared_basins if b != "UNKNOWN"}))

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "groundtruth_csv": str(groundtruth_csv),
            "noaa_dir": str(noaa_dir),
            "cds_dir": str(cds_dir),
            "cds_file_count": len(cds_files),
        },
        "noaa_targets": {
            "basins_from_groundtruth": sorted(gt_basins),
            "basins_from_noaa_dirs": sorted(noaa_basins),
            "target_basins": target_basins,
            "groundtruth_point_counts_by_basin": dict(sorted(gt_point_counts.items())),
            "groundtruth_unique_storm_counts_by_basin": dict(sorted(gt_storm_counts.items())),
            "noaa_storm_folder_counts_by_basin": dict(sorted(noaa_storm_counts.items())),
        },
        "cds_real": {
            "declared_basins": sorted(declared_basins),
            "declared_basin_counts_by_file": dict(sorted(declared_basin_counts.items())),
            "tracks_file_counts": dict(sorted(tracks_file_counts.items())),
            "total_points_scanned": total_rows,
        },
        "scope_check": {
            "intersection_basins": intersection,
            "intersection_size": len(intersection),
            "in_scope_rows": total_in_scope,
            "out_of_scope_rows": total_out_scope,
            "out_of_scope_ratio": round(total_out_scope / total_rows, 6) if total_rows else 0.0,
            "policy": (
                "Rows are marked out_of_scope when CDS declared basin is not in NOAA target basins "
                "or CDS basin cannot be inferred from metadata."
            ),
        },
        "outputs": {
            "points_csv": str(points_csv),
            "summary_csv": str(summary_csv),
            "report_json": str(report_json),
        },
    }

    with report_json.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)

    print(f"[OK] Basin scope report: {report_json}")
    print(f"[OK] Point-level scope table: {points_csv}")
    print(f"[OK] File-level scope summary: {summary_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
