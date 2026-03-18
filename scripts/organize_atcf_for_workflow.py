#!/usr/bin/env python3
"""Organize ATCF A/B-deck data into workflow-ready artifacts by storm_id.

Outputs:
1. storm_id crosswalk:
   data/interim/atcf/storm_id_crosswalk.csv
2. by-storm guidance/verification tables:
   data/interim/atcf/by_storm/<storm_id>/a_deck_guidance.csv
   data/interim/atcf/by_storm/<storm_id>/a_deck_spread.csv
   data/interim/atcf/by_storm/<storm_id>/b_deck_best_track.csv
   data/interim/atcf/by_storm/<storm_id>/ibtracs_best_track.csv
   data/interim/atcf/by_storm/<storm_id>/verification_groundtruth_preferred.csv
3. summary report:
   data/interim/atcf/summary.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


ATCF_FILE_RE = re.compile(r"^[ab]([a-z]{2}\d{2}\d{4})\.dat$", re.IGNORECASE)
ATCF_TO_GT_BASIN = {
    "AL": "Atlantic",
    "EP": "E_Pacific",
    "CP": "C_Pacific",
}
EARTH_RADIUS_KM = 6371.0088


@dataclass
class GtPoint:
    dt: datetime
    lat: float
    lon: float
    storm_id: str
    basin: str


@dataclass
class GtTrackRow:
    storm_id: str
    dt: datetime
    lat: float
    lon: float
    vmax_kt_wmo: Optional[float]
    min_pressure_mb_wmo: Optional[float]
    vmax_kt_usa: Optional[float]
    min_pressure_mb_usa: Optional[float]
    noaa_name: str
    noaa_basin: str


@dataclass
class AtcfRecord:
    atcf_storm_id: str
    basin_code: str
    storm_num: int
    season_year: int
    tech_num: str
    tech: str
    init_dt: datetime
    tau_h: int
    valid_dt: datetime
    lat: float
    lon: float
    vmax_kt: Optional[int]
    mslp_hpa: Optional[float]
    storm_name: Optional[str]
    source_file: str
    line_no: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate ATCF into workflow artifacts grouped by storm_id."
    )
    parser.add_argument(
        "--a-root",
        type=Path,
        default=Path("data/processed/atcf/plaintext/a_deck"),
        help="Path to plaintext A-deck root.",
    )
    parser.add_argument(
        "--b-root",
        type=Path,
        default=Path("data/processed/atcf/plaintext/b_deck"),
        help="Path to plaintext B-deck root.",
    )
    parser.add_argument(
        "--groundtruth-csv",
        type=Path,
        default=Path("GroundTruth_Cyclones/matched_cyclone_tracks.csv"),
        help="GroundTruth CSV path for storm_id mapping.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/interim/atcf"),
        help="Output directory.",
    )
    parser.add_argument(
        "--max-time-diff-hours",
        type=int,
        default=3,
        help="Maximum time difference (hours) for B-deck -> GroundTruth matching.",
    )
    parser.add_argument(
        "--max-geo-distance-deg",
        type=float,
        default=3.0,
        help="Maximum lat/lon distance (degree) for B-deck -> GroundTruth matching.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove prior outputs under out-dir before writing new results.",
    )
    return parser.parse_args()


def parse_float(value: str) -> Optional[float]:
    try:
        if value is None:
            return None
        txt = str(value).strip()
        if not txt:
            return None
        return float(txt)
    except Exception:
        return None


def parse_int(value: str) -> Optional[int]:
    try:
        if value is None:
            return None
        txt = str(value).strip()
        if not txt:
            return None
        return int(txt)
    except Exception:
        return None


def parse_latlon_token(token: str) -> Optional[float]:
    txt = (token or "").strip().upper()
    if len(txt) < 2:
        return None
    hemi = txt[-1]
    digits = txt[:-1].strip()
    if hemi not in {"N", "S", "E", "W"}:
        return None
    if not re.fullmatch(r"-?\d+", digits):
        return None
    value = int(digits) / 10.0
    if hemi in {"S", "W"}:
        value = -value
    return value


def normalize_lon(lon: float) -> float:
    x = lon
    while x >= 180.0:
        x -= 360.0
    while x < -180.0:
        x += 360.0
    return x


def lon_diff(a: float, b: float) -> float:
    d = abs(a - b)
    return min(d, 360.0 - d)


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    rlat1 = math.radians(lat1)
    rlat2 = math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    sin_dlat = math.sin(dlat / 2.0)
    sin_dlon = math.sin(dlon / 2.0)
    a = sin_dlat * sin_dlat + math.cos(rlat1) * math.cos(rlat2) * sin_dlon * sin_dlon
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1.0 - a)))
    return EARTH_RADIUS_KM * c


def circular_mean_lon(lons: List[float]) -> Optional[float]:
    if not lons:
        return None
    x = sum(math.cos(math.radians(v)) for v in lons) / len(lons)
    y = sum(math.sin(math.radians(v)) for v in lons) / len(lons)
    if x == 0.0 and y == 0.0:
        return normalize_lon(sum(lons) / len(lons))
    return normalize_lon(math.degrees(math.atan2(y, x)))


def parse_atcf_file_token(file_path: Path) -> Optional[str]:
    m = ATCF_FILE_RE.match(file_path.name)
    if not m:
        return None
    return m.group(1).upper()


def parse_atcf_line(
    line: str,
    atcf_storm_id: str,
    source_file: str,
    line_no: int,
) -> Optional[AtcfRecord]:
    parts = [p.strip() for p in line.rstrip("\n").split(",")]
    if len(parts) < 10:
        return None

    basin_code = ((parts[0] or "").strip() or atcf_storm_id[:2]).upper()
    storm_num = parse_int((parts[1] or "").strip())
    if storm_num is None:
        storm_num = parse_int(atcf_storm_id[2:4]) or 0
    season_year = parse_int(atcf_storm_id[4:8]) or 0
    init_raw = (parts[2] or "").strip()
    tech_num = (parts[3] or "").strip()
    tech = ((parts[4] or "").strip() or "UNKNOWN").upper()
    tau_h = parse_int((parts[5] or "").strip())
    lat = parse_latlon_token((parts[6] or "").strip())
    lon = parse_latlon_token((parts[7] or "").strip())
    vmax_kt = parse_int((parts[8] or "").strip())
    mslp = parse_float((parts[9] or "").strip())
    storm_name = (parts[27] or "").strip().upper() if len(parts) > 27 else ""

    if tau_h is None:
        return None
    if lat is None or lon is None:
        return None
    try:
        init_dt = datetime.strptime(init_raw, "%Y%m%d%H")
    except ValueError:
        return None

    lon = normalize_lon(lon)
    valid_dt = init_dt + timedelta(hours=tau_h)
    if vmax_kt is not None and vmax_kt <= 0:
        vmax_kt = None
    if mslp is not None and mslp <= 0:
        mslp = None
    mslp_hpa = float(mslp) if mslp is not None else None

    return AtcfRecord(
        atcf_storm_id=atcf_storm_id,
        basin_code=basin_code,
        storm_num=storm_num,
        season_year=season_year,
        tech_num=tech_num,
        tech=tech,
        init_dt=init_dt,
        tau_h=tau_h,
        valid_dt=valid_dt,
        lat=float(lat),
        lon=float(lon),
        vmax_kt=vmax_kt,
        mslp_hpa=mslp_hpa,
        storm_name=storm_name or None,
        source_file=source_file,
        line_no=line_no,
    )


def load_groundtruth(
    gt_csv: Path,
) -> Tuple[Dict[datetime, List[GtPoint]], Counter, Counter, Dict[str, List[GtTrackRow]]]:
    by_dt: Dict[datetime, List[GtPoint]] = defaultdict(list)
    basin_counter: Counter = Counter()
    storm_counter: Counter = Counter()
    by_storm_rows: Dict[str, List[GtTrackRow]] = defaultdict(list)

    with gt_csv.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            dt_text = (row.get("datetime") or "")[:19]
            storm_id = (row.get("storm_id") or "").strip()
            basin = (row.get("noaa_basin") or "").strip()
            lat = parse_float(row.get("latitude") or "")
            lon = parse_float(row.get("longitude") or "")
            if not dt_text or not storm_id or not basin or lat is None or lon is None:
                continue
            try:
                dt = datetime.strptime(dt_text, "%Y-%m-%d %H:%M:%S")
            except ValueError:
                continue
            point = GtPoint(
                dt=dt,
                lat=float(lat),
                lon=normalize_lon(float(lon)),
                storm_id=storm_id,
                basin=basin,
            )
            by_dt[dt].append(point)
            basin_counter[basin] += 1
            storm_counter[storm_id] += 1

            by_storm_rows[storm_id].append(
                GtTrackRow(
                    storm_id=storm_id,
                    dt=dt,
                    lat=float(lat),
                    lon=normalize_lon(float(lon)),
                    vmax_kt_wmo=parse_float(row.get("max_wind_wmo") or ""),
                    min_pressure_mb_wmo=parse_float(row.get("min_pressure_wmo") or ""),
                    vmax_kt_usa=parse_float(row.get("max_wind_usa") or ""),
                    min_pressure_mb_usa=parse_float(row.get("min_pressure_usa") or ""),
                    noaa_name=(row.get("noaa_name") or "").strip(),
                    noaa_basin=(row.get("noaa_basin") or "").strip(),
                )
            )

    for sid in list(by_storm_rows.keys()):
        by_storm_rows[sid].sort(key=lambda r: r.dt)

    return by_dt, basin_counter, storm_counter, by_storm_rows


def list_deck_files(root: Path) -> List[Path]:
    if not root.exists():
        return []
    return sorted(p for p in root.rglob("*.dat") if p.is_file())


def parse_and_dedup_file(file_path: Path, atcf_storm_id: str) -> Tuple[int, int, List[AtcfRecord]]:
    raw_rows = 0
    parsed_rows = 0
    dedup: Dict[Tuple[str, datetime, int], AtcfRecord] = {}
    source_file = str(file_path)

    with file_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line_no, line in enumerate(f, start=1):
            raw_rows += 1
            record = parse_atcf_line(line, atcf_storm_id=atcf_storm_id, source_file=source_file, line_no=line_no)
            if record is None:
                continue
            parsed_rows += 1
            dedup_key = (record.tech, record.init_dt, record.tau_h)
            # Keep last row under the same key (ATCF append updates pattern).
            dedup[dedup_key] = record

    rows = sorted(dedup.values(), key=lambda r: (r.valid_dt, r.init_dt, r.tech, r.tau_h, r.line_no))
    return raw_rows, parsed_rows, rows


def match_bdeck_token_to_storm(
    records: List[AtcfRecord],
    gt_by_dt: Dict[datetime, List[GtPoint]],
    max_time_diff_hours: int,
    max_geo_distance_deg: float,
) -> Dict[str, Any]:
    total_points = len(records)
    if total_points == 0:
        return {
            "matched_storm_id": None,
            "matched_points": 0,
            "match_ratio": 0.0,
            "mean_geo_distance_deg": None,
            "mean_time_diff_h": None,
            "status": "unmatched_no_records",
        }

    basin_code = records[0].basin_code
    target_basin = ATCF_TO_GT_BASIN.get(basin_code)
    votes: Dict[str, Dict[str, float]] = {}

    for record in records:
        best: Optional[Tuple[GtPoint, float, int]] = None
        for delta_h in range(-max_time_diff_hours, max_time_diff_hours + 1):
            cand_dt = record.valid_dt + timedelta(hours=delta_h)
            for point in gt_by_dt.get(cand_dt, []):
                if target_basin is not None and point.basin != target_basin:
                    continue
                d_geo = math.hypot(record.lat - point.lat, lon_diff(record.lon, point.lon))
                if best is None:
                    best = (point, d_geo, abs(delta_h))
                    continue
                best_geo = best[1]
                best_dt_h = best[2]
                if (d_geo, abs(delta_h)) < (best_geo, best_dt_h):
                    best = (point, d_geo, abs(delta_h))

        if best is None:
            continue
        point, d_geo, d_t = best
        if d_geo > max_geo_distance_deg:
            continue
        stat = votes.setdefault(point.storm_id, {"count": 0.0, "geo_sum": 0.0, "time_sum": 0.0})
        stat["count"] += 1.0
        stat["geo_sum"] += d_geo
        stat["time_sum"] += float(d_t)

    if not votes:
        status = "unmatched_no_candidate"
        if records[0].storm_num >= 90:
            status = "unmatched_invest"
        return {
            "matched_storm_id": None,
            "matched_points": 0,
            "match_ratio": 0.0,
            "mean_geo_distance_deg": None,
            "mean_time_diff_h": None,
            "status": status,
        }

    def rank_key(item: Tuple[str, Dict[str, float]]) -> Tuple[float, float, float]:
        _, stat = item
        count = stat["count"]
        mean_geo = stat["geo_sum"] / count if count else 9999.0
        mean_dt = stat["time_sum"] / count if count else 9999.0
        return (count, -mean_geo, -mean_dt)

    best_sid, best_stat = max(votes.items(), key=rank_key)
    matched_points = int(best_stat["count"])
    match_ratio = best_stat["count"] / float(total_points)
    mean_geo = best_stat["geo_sum"] / best_stat["count"]
    mean_dt = best_stat["time_sum"] / best_stat["count"]
    status = "matched"

    if records[0].storm_num >= 90:
        status = "matched_invest"

    return {
        "matched_storm_id": best_sid,
        "matched_points": matched_points,
        "match_ratio": round(match_ratio, 6),
        "mean_geo_distance_deg": round(mean_geo, 6),
        "mean_time_diff_h": round(mean_dt, 6),
        "status": status,
    }


def append_csv_rows(path: Path, fieldnames: List[str], rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def write_csv_rows(path: Path, fieldnames: List[str], rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_a_deck_spread_rows(
    storm_id: str,
    atcf_storm_id: str,
    records: List[AtcfRecord],
) -> List[Dict[str, Any]]:
    by_group: Dict[Tuple[datetime, int, datetime], List[AtcfRecord]] = defaultdict(list)
    for rec in records:
        if rec.tau_h < 0:
            continue
        by_group[(rec.init_dt, rec.tau_h, rec.valid_dt)].append(rec)

    rows: List[Dict[str, Any]] = []
    for key in sorted(by_group.keys()):
        init_dt, tau_h, valid_dt = key
        pts = by_group[key]
        lats = [r.lat for r in pts]
        lons = [r.lon for r in pts]
        vmaxs = [float(r.vmax_kt) for r in pts if r.vmax_kt is not None]
        mslps = [float(r.mslp_hpa) for r in pts if r.mslp_hpa is not None]

        c_lat = sum(lats) / len(lats) if lats else None
        c_lon = circular_mean_lon(lons)
        c_vmax = sum(vmaxs) / len(vmaxs) if vmaxs else None
        c_mslp = sum(mslps) / len(mslps) if mslps else None

        if c_lat is None or c_lon is None:
            continue

        dists_km = [haversine_km(r.lat, r.lon, c_lat, c_lon) for r in pts]
        track_spread = math.sqrt(sum(d * d for d in dists_km) / len(dists_km)) if dists_km else 0.0
        wind_spread = (
            math.sqrt(sum((v - c_vmax) ** 2 for v in vmaxs) / len(vmaxs))
            if vmaxs and c_vmax is not None
            else 0.0
        )

        rows.append(
            {
                "storm_id": storm_id,
                "atcf_storm_id": atcf_storm_id,
                "init_time_utc": init_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "tau_h": tau_h,
                "valid_time_utc": valid_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "model_count": len(pts),
                "consensus_lat": round(c_lat, 3),
                "consensus_lon": round(c_lon, 3),
                "consensus_vmax_kt": round(c_vmax, 2) if c_vmax is not None else "",
                "consensus_mslp_hpa": round(c_mslp, 2) if c_mslp is not None else "",
                "track_spread_km": round(track_spread, 3),
                "wind_spread_kt": round(wind_spread, 3),
                "models": "|".join(sorted({r.tech for r in pts if r.tech})),
            }
        )

    return rows


def to_iso_utc(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def pick_ibtracs_vmax_kt(row: GtTrackRow) -> Optional[float]:
    if row.vmax_kt_usa is not None:
        return row.vmax_kt_usa
    return row.vmax_kt_wmo


def pick_ibtracs_min_pressure_mb(row: GtTrackRow) -> Optional[float]:
    if row.min_pressure_mb_usa is not None:
        return row.min_pressure_mb_usa
    return row.min_pressure_mb_wmo


def main() -> int:
    args = parse_args()

    if args.clean and args.out_dir.exists():
        shutil.rmtree(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    by_storm_root = args.out_dir / "by_storm"
    by_storm_root.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading GroundTruth: {args.groundtruth_csv}")
    gt_by_dt, gt_basin_counter, gt_storm_counter, gt_rows_by_storm = load_groundtruth(args.groundtruth_csv)
    print(
        f"[INFO] GroundTruth loaded: points={sum(gt_basin_counter.values())} "
        f"storms={len(gt_storm_counter)}"
    )

    a_files = list_deck_files(args.a_root)
    b_files = list_deck_files(args.b_root)
    a_tokens = {parse_atcf_file_token(p) for p in a_files}
    b_tokens = {parse_atcf_file_token(p) for p in b_files}
    a_tokens.discard(None)
    b_tokens.discard(None)

    print(f"[INFO] B-deck files: {len(b_files)}")
    bdeck_by_token: Dict[str, List[AtcfRecord]] = {}
    bdeck_file_stats: Dict[str, Dict[str, int]] = {}

    for idx, bfile in enumerate(b_files, start=1):
        token = parse_atcf_file_token(bfile)
        if token is None:
            continue
        raw_rows, parsed_rows, records = parse_and_dedup_file(bfile, token)
        bdeck_by_token[token] = records
        bdeck_file_stats[token] = {
            "raw_rows": raw_rows,
            "parsed_rows": parsed_rows,
            "dedup_rows": len(records),
        }
        if idx % 25 == 0:
            print(f"[PROGRESS] B-deck parsed {idx}/{len(b_files)} files")

    crosswalk_rows: List[Dict[str, Any]] = []
    crosswalk_map: Dict[str, Dict[str, Any]] = {}
    for token in sorted(bdeck_by_token.keys()):
        records = bdeck_by_token[token]
        stats = bdeck_file_stats.get(token, {})
        match = match_bdeck_token_to_storm(
            records=records,
            gt_by_dt=gt_by_dt,
            max_time_diff_hours=args.max_time_diff_hours,
            max_geo_distance_deg=args.max_geo_distance_deg,
        )
        storm_num = parse_int(token[2:4]) or 0
        first_valid = to_iso_utc(records[0].valid_dt) if records else ""
        last_valid = to_iso_utc(records[-1].valid_dt) if records else ""
        names = [r.storm_name for r in records if r.storm_name]
        common_name = Counter(names).most_common(1)[0][0] if names else ""

        row = {
            "atcf_storm_id": token,
            "basin_code": token[:2],
            "storm_num": storm_num,
            "season_year": parse_int(token[4:8]) or "",
            "is_invest": 1 if storm_num >= 90 else 0,
            "has_a_deck_file": 1 if token in a_tokens else 0,
            "has_b_deck_file": 1 if token in b_tokens else 0,
            "matched_storm_id": match["matched_storm_id"] or "",
            "matched_points": match["matched_points"],
            "total_points": len(records),
            "match_ratio": match["match_ratio"],
            "mean_geo_distance_deg": ""
            if match["mean_geo_distance_deg"] is None
            else match["mean_geo_distance_deg"],
            "mean_time_diff_h": ""
            if match["mean_time_diff_h"] is None
            else match["mean_time_diff_h"],
            "match_status": match["status"],
            "sample_storm_name_from_bdeck": common_name,
            "first_valid_time_utc": first_valid,
            "last_valid_time_utc": last_valid,
            "raw_rows": stats.get("raw_rows", 0),
            "parsed_rows": stats.get("parsed_rows", 0),
            "dedup_rows": stats.get("dedup_rows", 0),
        }
        crosswalk_rows.append(row)
        crosswalk_map[token] = row

    crosswalk_path = args.out_dir / "storm_id_crosswalk.csv"
    crosswalk_fields = [
        "atcf_storm_id",
        "basin_code",
        "storm_num",
        "season_year",
        "is_invest",
        "has_a_deck_file",
        "has_b_deck_file",
        "matched_storm_id",
        "matched_points",
        "total_points",
        "match_ratio",
        "mean_geo_distance_deg",
        "mean_time_diff_h",
        "match_status",
        "sample_storm_name_from_bdeck",
        "first_valid_time_utc",
        "last_valid_time_utc",
        "raw_rows",
        "parsed_rows",
        "dedup_rows",
    ]
    with crosswalk_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=crosswalk_fields)
        writer.writeheader()
        writer.writerows(crosswalk_rows)
    print(f"[OK] Crosswalk written: {crosswalk_path}")

    b_best_fields = [
        "storm_id",
        "atcf_storm_id",
        "basin_code",
        "storm_num",
        "season_year",
        "tech",
        "init_time_utc",
        "tau_h",
        "valid_time_utc",
        "lat",
        "lon",
        "vmax_kt",
        "mslp_hpa",
        "storm_name",
        "source_file",
    ]
    bdeck_rows_written = 0
    matched_tokens = 0
    bdeck_rows_by_sid: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for token, records in sorted(bdeck_by_token.items()):
        mapped_sid = (crosswalk_map.get(token, {}).get("matched_storm_id") or "").strip()
        if not mapped_sid:
            continue
        matched_tokens += 1
        out_path = by_storm_root / mapped_sid / "b_deck_best_track.csv"
        rows: List[Dict[str, Any]] = []
        for rec in records:
            rows.append(
                {
                    "storm_id": mapped_sid,
                    "atcf_storm_id": rec.atcf_storm_id,
                    "basin_code": rec.basin_code,
                    "storm_num": rec.storm_num,
                    "season_year": rec.season_year,
                    "tech": rec.tech,
                    "init_time_utc": to_iso_utc(rec.init_dt),
                    "tau_h": rec.tau_h,
                    "valid_time_utc": to_iso_utc(rec.valid_dt),
                    "lat": round(rec.lat, 3),
                    "lon": round(rec.lon, 3),
                    "vmax_kt": rec.vmax_kt if rec.vmax_kt is not None else "",
                    "mslp_hpa": rec.mslp_hpa if rec.mslp_hpa is not None else "",
                    "storm_name": rec.storm_name or "",
                    "source_file": rec.source_file,
                }
            )
        append_csv_rows(out_path, b_best_fields, rows)
        bdeck_rows_written += len(rows)
        bdeck_rows_by_sid[mapped_sid].extend(rows)

    print(
        f"[OK] B-deck by-storm outputs written: tokens={matched_tokens} rows={bdeck_rows_written}"
    )

    # Build IBTrACS-by-storm tables and preferred verification tables.
    ibtracs_fields = [
        "storm_id",
        "valid_time_utc",
        "lat",
        "lon",
        "vmax_kt_wmo",
        "min_pressure_mb_wmo",
        "vmax_kt_usa",
        "min_pressure_mb_usa",
        "noaa_name",
        "noaa_basin",
        "source_file",
    ]
    preferred_fields = [
        "storm_id",
        "valid_time_utc",
        "lat",
        "lon",
        "vmax_kt",
        "min_pressure_mb",
        "storm_phase",
        "source_used",
        "source_priority",
        "atcf_storm_id",
        "source_file",
    ]

    preferred_rows_total = 0
    ibtracs_rows_total = 0
    preferred_source_counter: Counter = Counter()
    preferred_storm_count = 0
    preferred_bdeck_storms = 0
    preferred_ibtracs_storms = 0

    all_storm_ids_for_verify = sorted(set(gt_rows_by_storm.keys()) | set(bdeck_rows_by_sid.keys()))
    for sid in all_storm_ids_for_verify:
        storm_dir = by_storm_root / sid

        ib_rows: List[Dict[str, Any]] = []
        for row in gt_rows_by_storm.get(sid, []):
            ib_rows.append(
                {
                    "storm_id": sid,
                    "valid_time_utc": to_iso_utc(row.dt),
                    "lat": round(row.lat, 3),
                    "lon": round(row.lon, 3),
                    "vmax_kt_wmo": row.vmax_kt_wmo if row.vmax_kt_wmo is not None else "",
                    "min_pressure_mb_wmo": row.min_pressure_mb_wmo if row.min_pressure_mb_wmo is not None else "",
                    "vmax_kt_usa": row.vmax_kt_usa if row.vmax_kt_usa is not None else "",
                    "min_pressure_mb_usa": row.min_pressure_mb_usa if row.min_pressure_mb_usa is not None else "",
                    "noaa_name": row.noaa_name,
                    "noaa_basin": row.noaa_basin,
                    "source_file": str(args.groundtruth_csv),
                }
            )

        ibtracs_fp = storm_dir / "ibtracs_best_track.csv"
        write_csv_rows(ibtracs_fp, ibtracs_fields, ib_rows)
        ibtracs_rows_total += len(ib_rows)

        b_rows = bdeck_rows_by_sid.get(sid, [])
        preferred_rows: List[Dict[str, Any]] = []
        if b_rows:
            preferred_bdeck_storms += 1
            for r in sorted(b_rows, key=lambda x: x["valid_time_utc"]):
                preferred_rows.append(
                    {
                        "storm_id": sid,
                        "valid_time_utc": r["valid_time_utc"],
                        "lat": r["lat"],
                        "lon": r["lon"],
                        "vmax_kt": r["vmax_kt"],
                        "min_pressure_mb": r["mslp_hpa"],
                        "storm_phase": r["tech"],
                        "source_used": "atcf_b_deck",
                        "source_priority": 1,
                        "atcf_storm_id": r["atcf_storm_id"],
                        "source_file": r["source_file"],
                    }
                )
        else:
            preferred_ibtracs_storms += 1
            for row in gt_rows_by_storm.get(sid, []):
                preferred_rows.append(
                    {
                        "storm_id": sid,
                        "valid_time_utc": to_iso_utc(row.dt),
                        "lat": round(row.lat, 3),
                        "lon": round(row.lon, 3),
                        "vmax_kt": pick_ibtracs_vmax_kt(row) if pick_ibtracs_vmax_kt(row) is not None else "",
                        "min_pressure_mb": pick_ibtracs_min_pressure_mb(row)
                        if pick_ibtracs_min_pressure_mb(row) is not None
                        else "",
                        "storm_phase": "",
                        "source_used": "ibtracs_matched_groundtruth",
                        "source_priority": 2,
                        "atcf_storm_id": "",
                        "source_file": str(args.groundtruth_csv),
                    }
                )

        preferred_fp = storm_dir / "verification_groundtruth_preferred.csv"
        write_csv_rows(preferred_fp, preferred_fields, preferred_rows)
        preferred_rows_total += len(preferred_rows)
        preferred_storm_count += 1
        for r in preferred_rows:
            preferred_source_counter[r["source_used"]] += 1

    print(f"[INFO] A-deck files: {len(a_files)}")
    a_guidance_fields = [
        "storm_id",
        "atcf_storm_id",
        "basin_code",
        "storm_num",
        "season_year",
        "model",
        "tech_num",
        "init_time_utc",
        "tau_h",
        "valid_time_utc",
        "lat",
        "lon",
        "vmax_kt",
        "mslp_hpa",
        "storm_name",
        "source_file",
    ]
    a_spread_fields = [
        "storm_id",
        "atcf_storm_id",
        "init_time_utc",
        "tau_h",
        "valid_time_utc",
        "model_count",
        "consensus_lat",
        "consensus_lon",
        "consensus_vmax_kt",
        "consensus_mslp_hpa",
        "track_spread_km",
        "wind_spread_kt",
        "models",
    ]
    a_summary_rows: List[Dict[str, Any]] = []

    a_raw_rows_total = 0
    a_parsed_rows_total = 0
    a_dedup_rows_total = 0
    a_rows_written = 0
    a_spread_rows_written = 0
    mapped_a_files = 0

    for idx, afile in enumerate(a_files, start=1):
        token = parse_atcf_file_token(afile)
        if token is None:
            continue
        mapped_sid = (crosswalk_map.get(token, {}).get("matched_storm_id") or "").strip()
        raw_rows, parsed_rows, records = parse_and_dedup_file(afile, token)
        a_raw_rows_total += raw_rows
        a_parsed_rows_total += parsed_rows
        a_dedup_rows_total += len(records)

        summary_row = {
            "atcf_storm_id": token,
            "raw_rows": raw_rows,
            "parsed_rows": parsed_rows,
            "dedup_rows": len(records),
            "mapped_storm_id": mapped_sid,
            "is_mapped": 1 if mapped_sid else 0,
            "unique_models": len({r.tech for r in records if r.tech}),
            "unique_inits": len({r.init_dt for r in records}),
            "first_valid_time_utc": to_iso_utc(records[0].valid_dt) if records else "",
            "last_valid_time_utc": to_iso_utc(records[-1].valid_dt) if records else "",
        }
        a_summary_rows.append(summary_row)

        if not mapped_sid:
            if idx % 20 == 0:
                print(f"[PROGRESS] A-deck parsed {idx}/{len(a_files)} files")
            continue

        mapped_a_files += 1
        out_guidance = by_storm_root / mapped_sid / "a_deck_guidance.csv"
        guidance_rows: List[Dict[str, Any]] = []
        for rec in records:
            guidance_rows.append(
                {
                    "storm_id": mapped_sid,
                    "atcf_storm_id": rec.atcf_storm_id,
                    "basin_code": rec.basin_code,
                    "storm_num": rec.storm_num,
                    "season_year": rec.season_year,
                    "model": rec.tech,
                    "tech_num": rec.tech_num,
                    "init_time_utc": to_iso_utc(rec.init_dt),
                    "tau_h": rec.tau_h,
                    "valid_time_utc": to_iso_utc(rec.valid_dt),
                    "lat": round(rec.lat, 3),
                    "lon": round(rec.lon, 3),
                    "vmax_kt": rec.vmax_kt if rec.vmax_kt is not None else "",
                    "mslp_hpa": rec.mslp_hpa if rec.mslp_hpa is not None else "",
                    "storm_name": rec.storm_name or "",
                    "source_file": rec.source_file,
                }
            )
        append_csv_rows(out_guidance, a_guidance_fields, guidance_rows)
        a_rows_written += len(guidance_rows)

        spread_rows = build_a_deck_spread_rows(mapped_sid, token, records)
        out_spread = by_storm_root / mapped_sid / "a_deck_spread.csv"
        append_csv_rows(out_spread, a_spread_fields, spread_rows)
        a_spread_rows_written += len(spread_rows)

        if idx % 20 == 0:
            print(f"[PROGRESS] A-deck parsed {idx}/{len(a_files)} files")

    a_summary_path = args.out_dir / "a_deck_file_summary.csv"
    with a_summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "atcf_storm_id",
                "raw_rows",
                "parsed_rows",
                "dedup_rows",
                "mapped_storm_id",
                "is_mapped",
                "unique_models",
                "unique_inits",
                "first_valid_time_utc",
                "last_valid_time_utc",
            ],
        )
        writer.writeheader()
        writer.writerows(a_summary_rows)
    print(f"[OK] A-deck summary written: {a_summary_path}")

    mapped_storm_ids = sorted(
        {
            row["matched_storm_id"]
            for row in crosswalk_rows
            if row.get("matched_storm_id")
        }
    )
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "inputs": {
            "a_root": str(args.a_root),
            "b_root": str(args.b_root),
            "groundtruth_csv": str(args.groundtruth_csv),
        },
        "matching_policy": {
            "method": "bdeck_time_space_nearest",
            "max_time_diff_hours": args.max_time_diff_hours,
            "max_geo_distance_deg": args.max_geo_distance_deg,
            "basin_constraint": ATCF_TO_GT_BASIN,
            "verification_source_priority": ["atcf_b_deck", "ibtracs_matched_groundtruth"],
        },
        "groundtruth": {
            "total_points": sum(gt_basin_counter.values()),
            "total_unique_storm_ids": len(gt_storm_counter),
            "points_by_basin": dict(gt_basin_counter),
        },
        "atcf_inventory": {
            "a_deck_files": len(a_files),
            "b_deck_files": len(b_files),
            "a_deck_tokens": len(a_tokens),
            "b_deck_tokens": len(b_tokens),
        },
        "crosswalk": {
            "rows": len(crosswalk_rows),
            "matched_rows": sum(1 for r in crosswalk_rows if r.get("matched_storm_id")),
            "matched_invest_rows": sum(1 for r in crosswalk_rows if r.get("match_status") == "matched_invest"),
            "unmatched_rows": sum(1 for r in crosswalk_rows if not r.get("matched_storm_id")),
            "unique_mapped_storm_ids": len(set(mapped_storm_ids)),
        },
        "outputs": {
            "crosswalk_csv": str(crosswalk_path),
            "a_deck_file_summary_csv": str(a_summary_path),
            "by_storm_root": str(by_storm_root),
            "b_deck_rows_written": bdeck_rows_written,
            "a_deck_rows_written": a_rows_written,
            "a_deck_spread_rows_written": a_spread_rows_written,
            "a_deck_raw_rows_total": a_raw_rows_total,
            "a_deck_parsed_rows_total": a_parsed_rows_total,
            "a_deck_dedup_rows_total": a_dedup_rows_total,
            "a_deck_mapped_files": mapped_a_files,
            "ibtracs_rows_written": ibtracs_rows_total,
            "verification_preferred_rows_written": preferred_rows_total,
            "verification_preferred_storm_count": preferred_storm_count,
            "verification_preferred_bdeck_storm_count": preferred_bdeck_storms,
            "verification_preferred_ibtracs_storm_count": preferred_ibtracs_storms,
            "verification_preferred_source_row_counts": dict(preferred_source_counter),
        },
    }
    summary_path = args.out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] Summary written: {summary_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
