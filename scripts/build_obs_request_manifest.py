#!/usr/bin/env python3
"""Build ASCAT/Recon request manifests from NOAA forecast advisory files.

The script follows the same manifest style as GOES:
- one row per advisory issue cycle
- point-time anchor from advisory center location
- optional storm_id backfill via ATCF crosswalk
"""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple


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

OUT_FIELDS = [
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
    "is_recon_candidate",
]


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
    is_recon_candidate: int

    def to_dict(self) -> Dict[str, object]:
        return {
            "request_id": self.request_id,
            "storm_id": self.storm_id,
            "storm_id_match_status": self.storm_id_match_status,
            "atcf_storm_id": self.atcf_storm_id,
            "basin": self.basin,
            "storm_name": self.storm_name,
            "advisory_no": self.advisory_no,
            "issue_time_utc": self.issue_time_utc,
            "center_obs_day": self.center_obs_day,
            "center_obs_hhmmz": self.center_obs_hhmmz,
            "lat": self.lat,
            "lon": self.lon,
            "source_file": self.source_file,
            "is_recon_candidate": self.is_recon_candidate,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build ASCAT + Recon request manifests from NOAA advisory files."
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
        "--ascat-out-csv",
        type=Path,
        default=Path("data/interim/ascat/ascat_request_manifest.csv"),
        help="ASCAT manifest output path.",
    )
    parser.add_argument(
        "--recon-out-csv",
        type=Path,
        default=Path("data/interim/recon/recon_request_manifest.csv"),
        help="Recon manifest output path.",
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
        "--recon-basins",
        type=str,
        default="Atlantic,E_Pacific",
        help="Comma separated basin folder names considered recon-priority.",
    )
    parser.add_argument(
        "--keep-all-recon-rows",
        action="store_true",
        help="If set, keep all rows in recon manifest and use is_recon_candidate as a flag only.",
    )
    parser.add_argument(
        "--only-with-storm-id",
        action="store_true",
        help="Keep only rows with non-empty storm_id in both manifests.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="If >0, keep first N rows (after sorting) for each manifest.",
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
        if not m:
            continue
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


def iter_forecast_advisory_files(noaa_root: Path) -> Iterable[Path]:
    return sorted(noaa_root.glob("*/*/*/forecast_advisory/*.txt"))


def year_from_path(path: Path, noaa_root: Path) -> Optional[int]:
    try:
        rel = path.relative_to(noaa_root)
    except ValueError:
        return None
    if not rel.parts:
        return None
    try:
        return int(rel.parts[0])
    except Exception:
        return None


def parse_recon_basins(value: str) -> Set[str]:
    out: Set[str] = set()
    for item in (value or "").split(","):
        token = item.strip()
        if token:
            out.add(token)
    return out


def build_rows(args: argparse.Namespace) -> Tuple[List[ManifestRow], Dict[str, int]]:
    crosswalk = load_crosswalk(args.crosswalk_csv)
    recon_basins = parse_recon_basins(args.recon_basins)
    rows: List[ManifestRow] = []
    stats: Dict[str, int] = {
        "files_seen": 0,
        "rows_built": 0,
        "skip_issue_time": 0,
        "skip_center": 0,
        "skip_year_filter": 0,
        "with_storm_id_match": 0,
        "without_storm_id_match": 0,
        "recon_candidate_rows": 0,
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
        rel = file_path.relative_to(args.noaa_root)
        basin = rel.parts[1] if len(rel.parts) >= 2 else ""
        storm_name = rel.parts[2] if len(rel.parts) >= 3 else ""

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

        is_recon_candidate = 1 if basin in recon_basins else 0
        if is_recon_candidate:
            stats["recon_candidate_rows"] += 1

        if args.only_with_storm_id and not storm_id:
            continue

        issue_time_utc = issue_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        request_id = (
            f"{basin}_{atcf_storm_id or storm_name}_{issue_dt.strftime('%Y%m%dT%H%M')}_{advisory_no or 'UNK'}"
        )

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
                is_recon_candidate=is_recon_candidate,
            )
        )
        stats["rows_built"] += 1

    rows.sort(key=lambda x: (x.issue_time_utc, x.basin, x.storm_name, x.request_id))
    return rows, stats


def write_csv(path: Path, rows: List[ManifestRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUT_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.to_dict())


def apply_limit(rows: List[ManifestRow], limit: int) -> List[ManifestRow]:
    if limit <= 0:
        return rows
    return rows[:limit]


def main() -> int:
    args = parse_args()
    rows, stats = build_rows(args)

    ascat_rows = apply_limit(rows, args.limit)
    recon_rows_all = rows if args.keep_all_recon_rows else [r for r in rows if r.is_recon_candidate == 1]
    recon_rows = apply_limit(recon_rows_all, args.limit)

    write_csv(args.ascat_out_csv, ascat_rows)
    write_csv(args.recon_out_csv, recon_rows)

    print("ascat_manifest:", args.ascat_out_csv)
    print("recon_manifest:", args.recon_out_csv)
    print("rows_total_built:", len(rows))
    print("ascat_rows_written:", len(ascat_rows))
    print("recon_rows_written:", len(recon_rows))
    print("recon_candidate_rows_total:", sum(1 for r in rows if r.is_recon_candidate == 1))
    print("rows_with_storm_id:", sum(1 for r in rows if r.storm_id))
    print("rows_without_storm_id:", sum(1 for r in rows if not r.storm_id))
    print("files_seen:", stats["files_seen"])
    print("skip_year_filter:", stats["skip_year_filter"])
    print("skip_issue_time:", stats["skip_issue_time"])
    print("skip_center:", stats["skip_center"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
