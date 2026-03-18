#!/usr/bin/env python3
"""Systematic data review for Cyclone_next.

Produces an audit JSON with:
- data inventory by source
- NOAA forecast advisory parse/match coverage
- constructable sample funnel for dataset_v0
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple


MONTHS = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}


def normalize_lon(lon: float) -> float:
    while lon >= 180:
        lon -= 360
    while lon < -180:
        lon += 360
    return lon


def lon_diff(a: float, b: float) -> float:
    d = abs(a - b)
    return min(d, 360.0 - d)


def parse_float(value: str) -> Optional[float]:
    try:
        return float(value)
    except Exception:
        return None


@dataclass
class GtPoint:
    dt: datetime
    lat: float
    lon: float
    basin: str
    storm_id: str


@dataclass
class Advisory:
    path: str
    basin: str
    storm_name: str
    advisory_no: Optional[int]
    issue_dt: Optional[datetime]
    lat: Optional[float]
    lon: Optional[float]
    atcf_storm_token: Optional[str]


@dataclass
class MatchResult:
    matched: bool
    storm_id: Optional[str] = None
    basin: Optional[str] = None
    gt_dt: Optional[datetime] = None
    time_diff_h: Optional[float] = None
    lat_diff: Optional[float] = None
    lon_diff: Optional[float] = None


def parse_gt(gt_csv: Path) -> Tuple[List[GtPoint], Dict[datetime, List[GtPoint]], Set[str], Counter, Counter]:
    points: List[GtPoint] = []
    by_dt: Dict[datetime, List[GtPoint]] = defaultdict(list)
    storms: Set[str] = set()
    basin_point_counts: Counter = Counter()
    basin_storms: Dict[str, Set[str]] = defaultdict(set)

    with gt_csv.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            dt_text = (row.get("datetime") or "")[:19]
            if not dt_text:
                continue
            try:
                dt = datetime.strptime(dt_text, "%Y-%m-%d %H:%M:%S")
            except ValueError:
                continue
            lat = parse_float(row.get("latitude") or "")
            lon = parse_float(row.get("longitude") or "")
            storm_id = (row.get("storm_id") or "").strip()
            basin = (row.get("noaa_basin") or "").strip()
            if lat is None or lon is None or not storm_id or not basin:
                continue

            lon = normalize_lon(lon)
            p = GtPoint(dt=dt, lat=lat, lon=lon, basin=basin, storm_id=storm_id)
            points.append(p)
            by_dt[dt].append(p)
            storms.add(storm_id)
            basin_point_counts[basin] += 1
            basin_storms[basin].add(storm_id)

    basin_storm_counts = Counter({k: len(v) for k, v in basin_storms.items()})
    return points, by_dt, storms, basin_point_counts, basin_storm_counts


def extract_month_year_from_lines(lines: List[str]) -> Optional[Tuple[int, int]]:
    # Search near top for tokens like "AUG 12 2004" or "Aug 28 2023"
    p = re.compile(r"\b([A-Za-z]{3})\s+(\d{1,2})\s+(\d{4})\b")
    for line in lines[:80]:
        m = p.search(line)
        if not m:
            continue
        month_token = m.group(1).lower()
        year = int(m.group(3))
        month = MONTHS.get(month_token)
        if month:
            return year, month
    return None


def parse_advisory_file(path: Path) -> Advisory:
    parts = path.parts
    # .../noaa/<year>/<basin>/<storm>/forecast_advisory/<file>
    basin = parts[-4] if len(parts) >= 5 else "UNKNOWN"
    storm_name = parts[-3] if len(parts) >= 4 else "UNKNOWN"
    fname = path.name

    advisory_no = None
    m_no = re.search(r"\.fstadv\.(\d+)\.txt$", fname, flags=re.IGNORECASE)
    if m_no:
        try:
            advisory_no = int(m_no.group(1))
        except ValueError:
            advisory_no = None

    atcf_token = None
    m_token = re.search(r"([a-z]{2}\d{2}\d{4})\.fstadv", fname, flags=re.IGNORECASE)
    if m_token:
        atcf_token = m_token.group(1).lower()

    text = path.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()

    month_year = extract_month_year_from_lines(lines)
    year_from_header = month_year[0] if month_year else None
    month_from_header = month_year[1] if month_year else None

    day = None
    hhmm = None
    lat = None
    lon = None

    # INIT line (modern+old styles)
    init_re = re.compile(
        r"^\s*INIT\s+(\d{1,2})/(\d{4})Z\s+([0-9.]+)([NS])\s+([0-9.]+)([EW])",
        flags=re.IGNORECASE,
    )
    # CENTER LOCATED line
    center_re = re.compile(
        r"([0-9.]+)([NS])\s+([0-9.]+)([EW]).*?AT\s+(\d{1,2})/(\d{4})Z",
        flags=re.IGNORECASE,
    )

    for line in lines:
        m = init_re.search(line)
        if m:
            day = int(m.group(1))
            hhmm = m.group(2)
            lat_v = float(m.group(3))
            lat_h = m.group(4).upper()
            lon_v = float(m.group(5))
            lon_h = m.group(6).upper()
            lat = lat_v if lat_h == "N" else -lat_v
            lon = lon_v if lon_h == "E" else -lon_v
            lon = normalize_lon(lon)
            break

    if day is None or hhmm is None or lat is None or lon is None:
        for line in lines:
            if "CENTER LOCATED NEAR" not in line.upper() and "REPEAT...CENTER LOCATED NEAR" not in line.upper():
                continue
            m = center_re.search(line)
            if m:
                lat_v = float(m.group(1))
                lat_h = m.group(2).upper()
                lon_v = float(m.group(3))
                lon_h = m.group(4).upper()
                day = int(m.group(5))
                hhmm = m.group(6)
                lat = lat_v if lat_h == "N" else -lat_v
                lon = lon_v if lon_h == "E" else -lon_v
                lon = normalize_lon(lon)
                break

    issue_dt = None
    if year_from_header and month_from_header and day is not None and hhmm and len(hhmm) == 4:
        hour = int(hhmm[:2])
        minute = int(hhmm[2:])
        # Handle potential month rollover edge-cases by trying neighboring months.
        candidate_months = [(year_from_header, month_from_header)]
        if month_from_header == 1:
            candidate_months.append((year_from_header - 1, 12))
        else:
            candidate_months.append((year_from_header, month_from_header - 1))
        if month_from_header == 12:
            candidate_months.append((year_from_header + 1, 1))
        else:
            candidate_months.append((year_from_header, month_from_header + 1))

        for yy, mm in candidate_months:
            try:
                issue_dt = datetime(yy, mm, day, hour, minute)
                break
            except ValueError:
                continue

    return Advisory(
        path=str(path),
        basin=basin,
        storm_name=storm_name,
        advisory_no=advisory_no,
        issue_dt=issue_dt,
        lat=lat,
        lon=lon,
        atcf_storm_token=atcf_token,
    )


def match_advisory_to_gt(advisory: Advisory, gt_by_dt: Dict[datetime, List[GtPoint]], time_window_h: int = 3) -> MatchResult:
    if advisory.issue_dt is None or advisory.lat is None or advisory.lon is None:
        return MatchResult(matched=False)

    best: Optional[Tuple[GtPoint, float, float, float]] = None
    # tuple: point, time_diff_h, lat_diff, lon_diff

    for delta_h in range(-time_window_h, time_window_h + 1):
        dt = advisory.issue_dt + timedelta(hours=delta_h)
        for p in gt_by_dt.get(dt, []):
            td = abs((p.dt - advisory.issue_dt).total_seconds()) / 3600.0
            dlat = abs(advisory.lat - p.lat)
            dlon = lon_diff(advisory.lon, p.lon)
            score = (td, dlat + dlon)
            if best is None:
                best = (p, td, dlat, dlon)
            else:
                prev = (best[1], best[2] + best[3])
                if score < prev:
                    best = (p, td, dlat, dlon)

    if best is None:
        return MatchResult(matched=False)

    p, td, dlat, dlon = best
    # conservative spatial threshold, advisory centers are usually 0.1-deg precision
    if dlat > 1.5 or dlon > 1.5:
        return MatchResult(matched=False)

    return MatchResult(
        matched=True,
        storm_id=p.storm_id,
        basin=p.basin,
        gt_dt=p.dt,
        time_diff_h=td,
        lat_diff=dlat,
        lon_diff=dlon,
    )


def count_noaa_products(noaa_root: Path) -> Counter:
    c = Counter()
    for p in noaa_root.rglob("*.txt"):
        if not p.is_file():
            continue
        parent = p.parent.name
        c[parent] += 1
    return c


def parse_model_cycles_hres(track_dir: Path) -> Tuple[Set[str], Dict[str, Set[datetime]], int]:
    sid_set: Set[str] = set()
    cycles: Dict[str, Set[datetime]] = defaultdict(set)
    file_count = 0
    pat = re.compile(r"track_(\d{4}\d{3}[NS]\d{5})_(\d{8})_(\d{4})\.csv$", flags=re.IGNORECASE)

    for p in track_dir.glob("*.csv"):
        file_count += 1
        m = pat.match(p.name)
        if not m:
            continue
        sid = m.group(1)
        date = m.group(2)
        hhmm = m.group(3)
        try:
            dt = datetime.strptime(f"{date}{hhmm}", "%Y%m%d%H%M")
        except ValueError:
            continue
        sid_set.add(sid)
        cycles[sid].add(dt)

    return sid_set, cycles, file_count


def parse_model_cycles_gfs(track_dir: Path) -> Tuple[Set[str], Dict[str, Set[datetime]], int, int, int]:
    sid_set: Set[str] = set()
    cycles: Dict[str, Set[datetime]] = defaultdict(set)
    total_files = 0
    usable_csv_files = 0
    zone_identifier_files = 0

    dt_pat = re.compile(r"gfs_(\d{4}-\d{2}-\d{2})(\d{2})_f000", flags=re.IGNORECASE)
    sid_in_name_pat = re.compile(r"^track_(\d{4}\d{3}[NS]\d{5})_gfs_", flags=re.IGNORECASE)

    for p in track_dir.iterdir():
        if not p.is_file():
            continue
        total_files += 1
        if p.name.endswith(":Zone.Identifier"):
            zone_identifier_files += 1
            continue
        if p.suffix.lower() != ".csv":
            continue
        usable_csv_files += 1

        m_dt = dt_pat.search(p.name)
        if not m_dt:
            continue
        try:
            dt = datetime.strptime(m_dt.group(1) + m_dt.group(2), "%Y-%m-%d%H")
        except ValueError:
            continue

        sid = None
        m_sid = sid_in_name_pat.match(p.name)
        if m_sid:
            sid = m_sid.group(1)
        else:
            # tracks_auto_* files: infer storm_id from first data row particle field.
            try:
                with p.open("r", encoding="utf-8", newline="") as f:
                    reader = csv.DictReader(f)
                    first = next(reader, None)
                if first:
                    sid = (first.get("particle") or first.get("storm_id") or "").strip()
            except Exception:
                sid = None

        if not sid:
            continue

        sid_set.add(sid)
        cycles[sid].add(dt)

    return sid_set, cycles, total_files, usable_csv_files, zone_identifier_files


def count_nonempty_system_json(system_dir: Path, include_zone_identifier: bool = False) -> Tuple[int, int, int]:
    total = 0
    nonempty = 0
    zone_identifier = 0

    for p in system_dir.iterdir():
        if not p.is_file():
            continue
        if p.name.endswith(":Zone.Identifier"):
            zone_identifier += 1
            if not include_zone_identifier:
                continue
        if p.suffix.lower() != ".json":
            continue
        total += 1
        try:
            payload = json.loads(p.read_text(encoding="utf-8", errors="ignore"))
        except Exception:
            continue

        is_nonempty = False
        if isinstance(payload, dict):
            ts = payload.get("time_series")
            if isinstance(ts, list):
                for row in ts:
                    if not isinstance(row, dict):
                        continue
                    env = row.get("environmental_systems")
                    if isinstance(env, list) and len(env) > 0:
                        is_nonempty = True
                        break
        if is_nonempty:
            nonempty += 1

    return total, nonempty, zone_identifier


def map_cds_to_gt(cds_dir: Path, gt_by_dt: Dict[datetime, List[GtPoint]]) -> Tuple[int, int, Set[str], Set[Tuple[str, datetime]], Counter]:
    total_rows = 0
    matched_rows = 0
    sid_set: Set[str] = set()
    sid_time_set: Set[Tuple[str, datetime]] = set()
    tracks_file_counter: Counter = Counter()

    for p in sorted(cds_dir.glob("*.json")):
        try:
            payload = json.loads(p.read_text(encoding="utf-8", errors="ignore"))
        except Exception:
            continue
        metadata = payload.get("metadata", {})
        tracks_file_counter[str(metadata.get("tracks_file", ""))] += 1
        rows = payload.get("environmental_analysis", [])
        for row in rows:
            total_rows += 1
            time_text = str(row.get("time", ""))[:19].replace("T", " ")
            try:
                dt = datetime.strptime(time_text, "%Y-%m-%d %H:%M:%S")
            except ValueError:
                continue
            tc = row.get("tc_position") or {}
            lat = parse_float(str(tc.get("lat", "")))
            lon = parse_float(str(tc.get("lon", "")))
            if lat is None or lon is None:
                continue
            lon = normalize_lon(lon)

            cands = gt_by_dt.get(dt, [])
            if not cands:
                continue
            best = None
            best_dist = None
            for p_gt in cands:
                dlat = abs(lat - p_gt.lat)
                dlon = lon_diff(lon, p_gt.lon)
                dist = dlat * dlat + dlon * dlon
                if best is None or dist < best_dist:
                    best = p_gt
                    best_dist = dist
            if best is None:
                continue
            # strict threshold used in previous alignment check
            if abs(lat - best.lat) <= 0.3 and lon_diff(lon, best.lon) <= 0.3:
                matched_rows += 1
                sid_set.add(best.storm_id)
                sid_time_set.add((best.storm_id, best.dt))

    return total_rows, matched_rows, sid_set, sid_time_set, tracks_file_counter


def has_cycle_within(cycles: Dict[str, Set[datetime]], sid: str, target_dt: datetime, window_h: int) -> bool:
    if sid not in cycles:
        return False
    for dt in cycles[sid]:
        if abs((dt - target_dt).total_seconds()) <= window_h * 3600:
            return True
    return False


def basin_counter_to_dict(counter: Counter) -> Dict[str, int]:
    return {k: counter[k] for k in sorted(counter.keys())}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--noaa-dir", default="noaa")
    parser.add_argument("--groundtruth-csv", default="GroundTruth_Cyclones/matched_cyclone_tracks.csv")
    parser.add_argument("--cds-dir", default="CDS_real")
    parser.add_argument("--hres-track-dir", default="HRES_forecast/HRES_track")
    parser.add_argument("--hres-system-dir", default="HRES_forecast/HRES_system")
    parser.add_argument("--gfs-track-dir", default="GFS_forecast/GFS_track")
    parser.add_argument("--gfs-system-dir", default="GFS_forecast/GFS_system")
    parser.add_argument("--output-json", default="data/interim/review/data_system_review_2026-03-03.json")
    args = parser.parse_args()

    noaa_dir = Path(args.noaa_dir)
    gt_csv = Path(args.groundtruth_csv)
    cds_dir = Path(args.cds_dir)
    hres_track_dir = Path(args.hres_track_dir)
    hres_system_dir = Path(args.hres_system_dir)
    gfs_track_dir = Path(args.gfs_track_dir)
    gfs_system_dir = Path(args.gfs_system_dir)
    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    # Inventory and GT
    product_counts = count_noaa_products(noaa_dir)
    gt_points, gt_by_dt, gt_storms, gt_basin_point_counts, gt_basin_storm_counts = parse_gt(gt_csv)

    # Parse NOAA forecast advisories
    fstadv_files = sorted(noaa_dir.rglob("forecast_advisory/*.txt"))
    advisories: List[Advisory] = [parse_advisory_file(p) for p in fstadv_files]

    parsed_issue_dt = sum(1 for a in advisories if a.issue_dt is not None)
    parsed_position = sum(1 for a in advisories if a.lat is not None and a.lon is not None)
    parsed_full = sum(1 for a in advisories if a.issue_dt is not None and a.lat is not None and a.lon is not None)

    # Match NOAA advisories to GroundTruth
    matched_rows = []
    unmatched_rows = 0
    basin_counts_matched = Counter()
    time_diff_counter = Counter()

    for a in advisories:
        m = match_advisory_to_gt(a, gt_by_dt, time_window_h=3)
        if not m.matched:
            unmatched_rows += 1
            continue
        matched_rows.append((a, m))
        basin_counts_matched[m.basin or "UNKNOWN"] += 1
        td = int(round(m.time_diff_h or 0))
        time_diff_counter[td] += 1

    # CDS alignment
    cds_total_rows, cds_matched_rows, cds_storm_ids, cds_sid_time_set, cds_tracks_counter = map_cds_to_gt(cds_dir, gt_by_dt)

    # HRES/GFS cycles and system coverage
    hres_sid_set, hres_cycles, hres_track_files = parse_model_cycles_hres(hres_track_dir)
    hres_sys_total, hres_sys_nonempty, hres_sys_zone = count_nonempty_system_json(hres_system_dir)

    gfs_sid_set, gfs_cycles, gfs_track_total, gfs_track_usable, gfs_track_zone = parse_model_cycles_gfs(gfs_track_dir)
    gfs_sys_total, gfs_sys_nonempty, gfs_sys_zone = count_nonempty_system_json(gfs_system_dir)

    # Funnel counts
    funnel = {
        "official_outputs_forecast_advisory_total": len(advisories),
        "official_outputs_parseable_issue_time_and_center": parsed_full,
        "matched_to_groundtruth_within_3h": len(matched_rows),
    }

    # Coverage checks on matched NOAA advisories
    with_cds = 0
    with_hres_exact = 0
    with_gfs_exact = 0
    with_hres_relaxed = 0
    with_gfs_relaxed = 0
    with_any_guidance_relaxed = 0
    with_both_guidance_relaxed = 0
    dataset_v0_relaxed = 0

    basin_funnel = Counter()

    for a, m in matched_rows:
        sid = m.storm_id
        if sid is None or m.gt_dt is None:
            continue

        has_cds = (sid, m.gt_dt) in cds_sid_time_set
        has_hres_0h = has_cycle_within(hres_cycles, sid, a.issue_dt, window_h=0) if a.issue_dt else False
        has_gfs_0h = has_cycle_within(gfs_cycles, sid, a.issue_dt, window_h=0) if a.issue_dt else False
        has_hres_3h = has_cycle_within(hres_cycles, sid, a.issue_dt, window_h=3) if a.issue_dt else False
        has_gfs_3h = has_cycle_within(gfs_cycles, sid, a.issue_dt, window_h=3) if a.issue_dt else False

        if has_cds:
            with_cds += 1
        if has_hres_0h:
            with_hres_exact += 1
        if has_gfs_0h:
            with_gfs_exact += 1
        if has_hres_3h:
            with_hres_relaxed += 1
        if has_gfs_3h:
            with_gfs_relaxed += 1

        any_guidance = has_hres_3h or has_gfs_3h
        both_guidance = has_hres_3h and has_gfs_3h
        if any_guidance:
            with_any_guidance_relaxed += 1
        if both_guidance:
            with_both_guidance_relaxed += 1

        # dataset_v0 minimal requirement from current P0 definition
        if has_cds and any_guidance:
            dataset_v0_relaxed += 1
            basin_funnel[m.basin or "UNKNOWN"] += 1

    funnel.update(
        {
            "matched_noaa_with_cds_now_env": with_cds,
            "matched_noaa_with_hres_guidance_exact_0h": with_hres_exact,
            "matched_noaa_with_gfs_guidance_exact_0h": with_gfs_exact,
            "matched_noaa_with_hres_guidance_relaxed_pm3h": with_hres_relaxed,
            "matched_noaa_with_gfs_guidance_relaxed_pm3h": with_gfs_relaxed,
            "matched_noaa_with_any_guidance_relaxed_pm3h": with_any_guidance_relaxed,
            "matched_noaa_with_both_guidance_relaxed_pm3h": with_both_guidance_relaxed,
            "dataset_v0_constructable_samples_relaxed_pm3h": dataset_v0_relaxed,
        }
    )

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "noaa_dir": str(noaa_dir),
            "groundtruth_csv": str(gt_csv),
            "cds_dir": str(cds_dir),
            "hres_track_dir": str(hres_track_dir),
            "hres_system_dir": str(hres_system_dir),
            "gfs_track_dir": str(gfs_track_dir),
            "gfs_system_dir": str(gfs_system_dir),
        },
        "inventory": {
            "noaa_product_file_counts": basin_counter_to_dict(product_counts),
            "groundtruth": {
                "rows": len(gt_points),
                "unique_storm_ids": len(gt_storms),
                "point_counts_by_basin": basin_counter_to_dict(gt_basin_point_counts),
                "storm_counts_by_basin": basin_counter_to_dict(gt_basin_storm_counts),
            },
            "cds_real": {
                "monthly_json_files": len(list(cds_dir.glob("*.json"))),
                "total_environment_points": cds_total_rows,
                "matched_to_groundtruth_points": cds_matched_rows,
                "matched_ratio": round(cds_matched_rows / cds_total_rows, 6) if cds_total_rows else None,
                "unique_storm_ids": len(cds_storm_ids),
                "tracks_file_counts": basin_counter_to_dict(cds_tracks_counter),
            },
            "hres": {
                "track_csv_files": hres_track_files,
                "track_unique_storm_ids": len(hres_sid_set),
                "track_unique_cycles": sum(len(v) for v in hres_cycles.values()),
                "system_json_files": hres_sys_total,
                "system_nonempty_json_files": hres_sys_nonempty,
                "system_zone_identifier_files": hres_sys_zone,
            },
            "gfs": {
                "track_total_files": gfs_track_total,
                "track_usable_csv_files": gfs_track_usable,
                "track_zone_identifier_files": gfs_track_zone,
                "track_unique_storm_ids": len(gfs_sid_set),
                "track_unique_cycles": sum(len(v) for v in gfs_cycles.values()),
                "system_json_files": gfs_sys_total,
                "system_nonempty_json_files": gfs_sys_nonempty,
                "system_zone_identifier_files": gfs_sys_zone,
            },
        },
        "noaa_forecast_advisory_quality": {
            "total_files": len(advisories),
            "parsed_issue_datetime": parsed_issue_dt,
            "parsed_center_position": parsed_position,
            "parsed_issue_datetime_and_center": parsed_full,
            "matched_to_groundtruth_within_pm3h": len(matched_rows),
            "unmatched_after_parse": unmatched_rows,
            "matched_basin_counts": basin_counter_to_dict(basin_counts_matched),
            "matched_time_diff_hours_distribution": basin_counter_to_dict(time_diff_counter),
        },
        "sample_funnel": funnel,
        "dataset_v0_constructable_samples_by_basin_relaxed_pm3h": basin_counter_to_dict(basin_funnel),
        "gaps": [
            "ATCF A-deck missing (full multi-model guidance not available).",
            "ATCF B-deck/IBTrACS missing in current workspace (standard verification target not complete).",
            "NOAA advisories around 11.6% cannot map to GroundTruth via current parser+matching and need fallback parser/rules.",
            "Guidance cycle coverage is limited by HRES/GFS storm/time overlap (especially exact issue-time alignment).",
            "GFS Zone.Identifier sidecar files still present and should be cleaned in P0 step 3.",
        ],
    }

    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[OK] Wrote review report: {out_json}")
    print(json.dumps(report["sample_funnel"], ensure_ascii=False, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
