#!/usr/bin/env python3
"""Build GOES request manifest from NOAA forecast advisory files.

The manifest is a lightweight point-time table used by the GEE extractor.
Each row represents one advisory issue cycle with storm center position.
"""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple


MONTH_TO_NUM = {
    "JAN": 1,
    "FEB": 2,
    "MAR": 3,
    "APR": 4,
    "MAY": 5,
    "JUN": 6,
    "JUL": 7,
    "AUG": 8,
    "SEP": 9,
    "OCT": 10,
    "NOV": 11,
    "DEC": 12,
}

ISSUE_TIME_RE = re.compile(
    r"^\s*(\d{3,4})\s+UTC\s+[A-Z]{3}\s+([A-Z]{3})\s+(\d{1,2})\s+(\d{4})\s*$",
    re.IGNORECASE,
)
CENTER_RE = re.compile(
    r"CENTER LOCATED NEAR\s+([0-9.]+)([NS])\s+([0-9.]+)([EW])\s+AT\s+(\d{1,2})/(\d{4})Z",
    re.IGNORECASE,
)
ATCF_ID_RE = re.compile(r"\b([A-Z]{2}\d{2}\d{4})\b")
ADV_NO_RE = re.compile(r"FORECAST/ADVISORY NUMBER\s+(\d+)", re.IGNORECASE)


@dataclass
class ManifestRow:
    request_id: str
    storm_id: str
    storm_id_match_status: str
    atcf_storm_id: str
    basin: str
    storm_name: str
    advisory_no: str
    issue_time_utc: str
    center_obs_day: str
    center_obs_hhmmz: str
    lat: float
    lon: float
    source_file: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build GOES request manifest from NOAA advisory files."
    )
    parser.add_argument(
        "--noaa-root",
        type=Path,
        default=Path("noaa"),
        help="NOAA root folder.",
    )
    parser.add_argument(
        "--crosswalk-csv",
        type=Path,
        default=Path("data/interim/atcf/storm_id_crosswalk.csv"),
        help="ATCF storm crosswalk csv.",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path("data/interim/goes/goes_request_manifest.csv"),
        help="Output manifest csv path.",
    )
    parser.add_argument(
        "--year-start",
        type=int,
        default=2016,
        help="Inclusive year lower bound from NOAA folder.",
    )
    parser.add_argument(
        "--year-end",
        type=int,
        default=2025,
        help="Inclusive year upper bound from NOAA folder.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="If >0, keep only first N rows after sorting.",
    )
    return parser.parse_args()


def parse_issue_time(lines: List[str]) -> Optional[datetime]:
    for ln in lines[:50]:
        m = ISSUE_TIME_RE.match(ln.strip().upper())
        if not m:
            continue
        hhmm_raw = m.group(1).zfill(4)
        month_txt = m.group(2).upper()
        day = int(m.group(3))
        year = int(m.group(4))
        month = MONTH_TO_NUM.get(month_txt)
        if month is None:
            continue
        try:
            return datetime(
                year=year,
                month=month,
                day=day,
                hour=int(hhmm_raw[:2]),
                minute=int(hhmm_raw[2:]),
            )
        except ValueError:
            continue
    return None


def parse_center(lines: List[str]) -> Optional[Tuple[float, float, str, str]]:
    for ln in lines:
        m = CENTER_RE.search(ln)
        if not m:
            continue
        lat = float(m.group(1)) * (1.0 if m.group(2).upper() == "N" else -1.0)
        lon = float(m.group(3)) * (-1.0 if m.group(4).upper() == "W" else 1.0)
        obs_day = m.group(5).zfill(2)
        obs_hhmmz = m.group(6).zfill(4)
        return lat, lon, obs_day, obs_hhmmz
    return None


def parse_atcf_id(lines: List[str]) -> str:
    for ln in lines[:40]:
        m = ATCF_ID_RE.search(ln)
        if m:
            return m.group(1).upper()
    return ""


def parse_advisory_no(lines: List[str]) -> str:
    for ln in lines[:60]:
        m = ADV_NO_RE.search(ln)
        if m:
            try:
                return f"{int(m.group(1)):03d}"
            except Exception:
                return m.group(1).strip()
    return ""


def load_crosswalk(path: Path) -> Dict[str, Tuple[str, str]]:
    if not path.exists():
        return {}
    out: Dict[str, Tuple[str, str]] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            atcf = (row.get("atcf_storm_id") or "").strip().upper()
            if not atcf:
                continue
            matched = (row.get("matched_storm_id") or "").strip()
            status = (row.get("match_status") or "").strip() or "unknown"
            out[atcf] = (matched, status)
    return out


def iter_forecast_advisory_files(noaa_root: Path) -> List[Path]:
    return sorted(noaa_root.glob("*/*/*/forecast_advisory/*.txt"))


def year_from_path(path: Path, noaa_root: Path) -> Optional[int]:
    try:
        rel = path.relative_to(noaa_root)
    except ValueError:
        return None
    if len(rel.parts) < 1:
        return None
    try:
        return int(rel.parts[0])
    except Exception:
        return None


def build_manifest(args: argparse.Namespace) -> Tuple[List[ManifestRow], Dict[str, int]]:
    crosswalk = load_crosswalk(args.crosswalk_csv)
    rows: List[ManifestRow] = []
    stats = {
        "files_seen": 0,
        "rows_built": 0,
        "skip_issue_time": 0,
        "skip_center": 0,
        "skip_year_filter": 0,
        "with_storm_id_match": 0,
        "without_storm_id_match": 0,
    }

    for file_path in iter_forecast_advisory_files(args.noaa_root):
        stats["files_seen"] += 1
        yr = year_from_path(file_path, args.noaa_root)
        if yr is None or yr < args.year_start or yr > args.year_end:
            stats["skip_year_filter"] += 1
            continue

        lines = file_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        issue_dt = parse_issue_time(lines)
        if issue_dt is None:
            stats["skip_issue_time"] += 1
            continue

        center = parse_center(lines)
        if center is None:
            stats["skip_center"] += 1
            continue
        lat, lon, obs_day, obs_hhmmz = center

        atcf_storm_id = parse_atcf_id(lines)
        advisory_no = parse_advisory_no(lines)

        storm_id = ""
        match_status = "unresolved"
        if atcf_storm_id and atcf_storm_id in crosswalk:
            storm_id, match_status = crosswalk[atcf_storm_id]
            if storm_id:
                stats["with_storm_id_match"] += 1
            else:
                stats["without_storm_id_match"] += 1
        else:
            stats["without_storm_id_match"] += 1

        rel = file_path.relative_to(args.noaa_root)
        basin = rel.parts[1] if len(rel.parts) >= 2 else ""
        storm_name = rel.parts[2] if len(rel.parts) >= 3 else ""

        issue_time_utc = issue_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        request_id = f"{basin}_{atcf_storm_id or storm_name}_{issue_dt.strftime('%Y%m%dT%H%M')}_{advisory_no or 'UNK'}"

        rows.append(
            ManifestRow(
                request_id=request_id,
                storm_id=storm_id,
                storm_id_match_status=match_status,
                atcf_storm_id=atcf_storm_id,
                basin=basin,
                storm_name=storm_name,
                advisory_no=advisory_no,
                issue_time_utc=issue_time_utc,
                center_obs_day=obs_day,
                center_obs_hhmmz=obs_hhmmz,
                lat=lat,
                lon=lon,
                source_file=str(file_path),
            )
        )
        stats["rows_built"] += 1

    rows.sort(key=lambda x: (x.issue_time_utc, x.basin, x.storm_name, x.request_id))
    if args.limit > 0:
        rows = rows[: args.limit]
    return rows, stats


def write_manifest(path: Path, rows: List[ManifestRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "request_id",
                "storm_id",
                "storm_id_match_status",
                "atcf_storm_id",
                "basin",
                "storm_name",
                "advisory_no",
                "issue_time_utc",
                "center_obs_day",
                "center_obs_hhmmz",
                "lat",
                "lon",
                "source_file",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "request_id": row.request_id,
                    "storm_id": row.storm_id,
                    "storm_id_match_status": row.storm_id_match_status,
                    "atcf_storm_id": row.atcf_storm_id,
                    "basin": row.basin,
                    "storm_name": row.storm_name,
                    "advisory_no": row.advisory_no,
                    "issue_time_utc": row.issue_time_utc,
                    "center_obs_day": row.center_obs_day,
                    "center_obs_hhmmz": row.center_obs_hhmmz,
                    "lat": row.lat,
                    "lon": row.lon,
                    "source_file": row.source_file,
                }
            )


def main() -> int:
    args = parse_args()
    rows, stats = build_manifest(args)
    write_manifest(args.out_csv, rows)

    matched_rows = sum(1 for r in rows if r.storm_id)
    print(args.out_csv)
    print("rows:", len(rows))
    print("rows_with_storm_id:", matched_rows)
    print("rows_without_storm_id:", len(rows) - matched_rows)
    print("files_seen:", stats["files_seen"])
    print("skip_issue_time:", stats["skip_issue_time"])
    print("skip_center:", stats["skip_center"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
