#!/usr/bin/env python3
"""Guidance data quality check for 2016-2025 and usable-cycle filtering.

Outputs:
- data/interim/qc/guidance_cycle_qc_2016_2025.csv
- data/interim/qc/guidance_usable_cycles_2016_2025.csv
- data/interim/qc/guidance_cycle_qc_summary_2016_2025.json
- data/interim/qc/dataset_sample_coverage_after_guidance_qc_2016_2025.json
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

# Reuse robust NOAA/GT/CDS utilities from previous review script.
from data_system_review import (  # type: ignore
    has_cycle_within,
    map_cds_to_gt,
    match_advisory_to_gt,
    parse_advisory_file,
    parse_gt,
)


def parse_time_ymdhms(text: str) -> Optional[datetime]:
    t = (text or "").strip()
    if not t:
        return None
    t = t[:19].replace("T", " ")
    try:
        return datetime.strptime(t, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None


def parse_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


@dataclass
class TrackStats:
    model: str
    storm_id: str
    init_time_utc: datetime
    source_file: str
    source_kind: str
    rows_total: int
    rows_valid: int
    unique_valid_times: int
    horizon_h: float
    duplicate_ratio: float
    invalid_time_ratio: float
    invalid_geo_ratio: float
    invalid_phys_ratio: float
    tau_off6_ratio: float
    track_pass_hard: bool
    track_fail_reasons: List[str]


@dataclass
class SystemStats:
    model: str
    storm_id: str
    init_time_utc: datetime
    source_file: str
    exists: bool
    parse_ok: bool
    time_steps: int
    nonempty_steps: int
    nonempty_ratio: float
    unique_names: Set[str]
    has_required_any: bool
    has_core_any: bool
    system_pass_hard: bool
    system_fail_reasons: List[str]


REQUIRED_SYSTEMS = {
    "VerticalWindShear",
    "UpperLevelDivergence",
    "OceanHeatContent",
    "SubtropicalHigh",
    "WesterlyTrough",
    "MonsoonTrough",
    "LowLevelFlow",
}
CORE_SYSTEMS = {
    "VerticalWindShear",
    "UpperLevelDivergence",
    "OceanHeatContent",
}


def evaluate_track_group(
    *,
    model: str,
    storm_id: str,
    init_dt: datetime,
    source_file: str,
    source_kind: str,
    rows: List[Dict[str, str]],
) -> TrackStats:
    rows_total = len(rows)
    valid_points: List[Tuple[datetime, float, float, float, float]] = []

    invalid_time = 0
    invalid_geo = 0
    invalid_phys = 0

    for row in rows:
        dt = parse_time_ymdhms(row.get("time", ""))
        if dt is None:
            invalid_time += 1
            continue

        lat = parse_float(row.get("lat"))
        lon = parse_float(row.get("lon"))
        msl = parse_float(row.get("msl"))
        wind = parse_float(row.get("wind"))

        geo_bad = False
        phys_bad = False

        if lat is None or lon is None:
            geo_bad = True
        else:
            if not (-90.0 <= lat <= 90.0):
                geo_bad = True
            if not (-180.0 <= lon <= 360.0):
                geo_bad = True

        if msl is None or wind is None:
            phys_bad = True
        else:
            if not (85000.0 <= msl <= 106000.0):
                phys_bad = True
            if not (0.0 <= wind <= 90.0):
                phys_bad = True

        if geo_bad:
            invalid_geo += 1
        if phys_bad:
            invalid_phys += 1

        if not geo_bad and not phys_bad:
            valid_points.append((dt, lat, lon, msl, wind))

    invalid_time_ratio = (invalid_time / rows_total) if rows_total else 1.0
    invalid_geo_ratio = (invalid_geo / rows_total) if rows_total else 1.0
    invalid_phys_ratio = (invalid_phys / rows_total) if rows_total else 1.0

    unique_times = sorted({p[0] for p in valid_points})
    unique_valid_times = len(unique_times)
    rows_valid = len(valid_points)

    if unique_times:
        horizon_h = max((t - init_dt).total_seconds() / 3600.0 for t in unique_times)
    else:
        horizon_h = -999.0

    duplicate_ratio = 1.0
    if rows_valid > 0:
        duplicate_ratio = 1.0 - (unique_valid_times / rows_valid)

    tau_values = []
    for t in unique_times:
        tau_values.append((t - init_dt).total_seconds() / 3600.0)
    tau_off6 = sum(1 for tau in tau_values if tau >= 0 and abs((tau / 6.0) - round(tau / 6.0)) > 1e-6)
    tau_off6_ratio = (tau_off6 / len(tau_values)) if tau_values else 1.0

    fail_reasons: List[str] = []
    if rows_total == 0:
        fail_reasons.append("track_no_rows")
    if rows_valid < 2:
        fail_reasons.append("track_valid_rows_lt2")
    if unique_valid_times < 2:
        fail_reasons.append("track_unique_time_lt2")
    if horizon_h < 6:
        fail_reasons.append("track_horizon_lt6h")
    if invalid_geo_ratio > 0.20:
        fail_reasons.append("track_invalid_geo_ratio_gt20pct")
    if invalid_phys_ratio > 0.20:
        fail_reasons.append("track_invalid_phys_ratio_gt20pct")
    if invalid_time_ratio > 0.20:
        fail_reasons.append("track_invalid_time_ratio_gt20pct")

    track_pass_hard = len(fail_reasons) == 0

    return TrackStats(
        model=model,
        storm_id=storm_id,
        init_time_utc=init_dt,
        source_file=source_file,
        source_kind=source_kind,
        rows_total=rows_total,
        rows_valid=rows_valid,
        unique_valid_times=unique_valid_times,
        horizon_h=horizon_h,
        duplicate_ratio=duplicate_ratio,
        invalid_time_ratio=invalid_time_ratio,
        invalid_geo_ratio=invalid_geo_ratio,
        invalid_phys_ratio=invalid_phys_ratio,
        tau_off6_ratio=tau_off6_ratio,
        track_pass_hard=track_pass_hard,
        track_fail_reasons=fail_reasons,
    )


def parse_hres_track_groups(track_dir: Path, year_start: int, year_end: int) -> Dict[Tuple[str, str, datetime], List[TrackStats]]:
    groups: Dict[Tuple[str, str, datetime], List[TrackStats]] = defaultdict(list)
    pat = re.compile(r"^track_(\d{4}\d{3}[NS]\d{5})_(\d{8})_(\d{4})\.csv$", re.IGNORECASE)

    for fp in sorted(track_dir.glob("*.csv")):
        m = pat.match(fp.name)
        if not m:
            continue
        sid = m.group(1)
        init_dt = datetime.strptime(m.group(2) + m.group(3), "%Y%m%d%H%M")
        if not (year_start <= init_dt.year <= year_end):
            continue

        with fp.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        stats = evaluate_track_group(
            model="HRES",
            storm_id=sid,
            init_dt=init_dt,
            source_file=str(fp),
            source_kind="track",
            rows=rows,
        )
        groups[("HRES", sid, init_dt)].append(stats)

    return groups


def parse_gfs_track_groups(track_dir: Path, year_start: int, year_end: int) -> Dict[Tuple[str, str, datetime], List[TrackStats]]:
    groups: Dict[Tuple[str, str, datetime], List[TrackStats]] = defaultdict(list)
    dt_pat = re.compile(r"gfs_(\d{4}-\d{2}-\d{2})(\d{2})_f000", re.IGNORECASE)

    for fp in sorted(track_dir.iterdir()):
        if not fp.is_file():
            continue
        if fp.name.endswith(":Zone.Identifier"):
            continue
        if fp.suffix.lower() != ".csv":
            continue

        m = dt_pat.search(fp.name)
        if not m:
            continue
        init_dt = datetime.strptime(m.group(1) + m.group(2), "%Y-%m-%d%H")
        if not (year_start <= init_dt.year <= year_end):
            continue

        with fp.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        # Split tracks_auto files by particle (storm id).
        by_sid: Dict[str, List[Dict[str, str]]] = defaultdict(list)
        for row in rows:
            sid = (row.get("particle") or row.get("storm_id") or "").strip()
            if sid:
                by_sid[sid].append(row)

        source_kind = "tracks_auto" if fp.name.startswith("tracks_auto_") else "track"

        for sid, sid_rows in by_sid.items():
            stats = evaluate_track_group(
                model="GFS",
                storm_id=sid,
                init_dt=init_dt,
                source_file=str(fp),
                source_kind=source_kind,
                rows=sid_rows,
            )
            groups[("GFS", sid, init_dt)].append(stats)

    return groups


def pick_best_track(candidates: List[TrackStats]) -> TrackStats:
    def score(s: TrackStats) -> Tuple[int, int, int, float, float, float, int]:
        source_priority = 2 if s.source_kind == "track" else 1
        return (
            1 if s.track_pass_hard else 0,
            source_priority,
            s.unique_valid_times,
            -s.invalid_geo_ratio,
            -s.invalid_phys_ratio,
            -s.duplicate_ratio,
            s.rows_valid,
        )

    return max(candidates, key=score)


def parse_hres_system(system_dir: Path, year_start: int, year_end: int) -> Dict[Tuple[str, str, datetime], SystemStats]:
    out: Dict[Tuple[str, str, datetime], SystemStats] = {}
    pat = re.compile(
        r"^(\d{4}\d{3}[NS]\d{5})_(\d{8})_(\d{4})_TC_Analysis_(\d{4}\d{3}[NS]\d{5})\.json$",
        re.IGNORECASE,
    )

    for fp in sorted(system_dir.glob("*.json")):
        m = pat.match(fp.name)
        if not m:
            continue
        sid = m.group(1)
        init_dt = datetime.strptime(m.group(2) + m.group(3), "%Y%m%d%H%M")
        if not (year_start <= init_dt.year <= year_end):
            continue

        out[("HRES", sid, init_dt)] = evaluate_system_file("HRES", sid, init_dt, fp)

    return out


def parse_gfs_system(system_dir: Path, year_start: int, year_end: int) -> Dict[Tuple[str, str, datetime], SystemStats]:
    out: Dict[Tuple[str, str, datetime], SystemStats] = {}
    pat = re.compile(
        r"^gfs_(\d{4}-\d{2}-\d{2})(\d{2})_f000_f240_6h_TC_Analysis_(\d{4}\d{3}[NS]\d{5})\.json$",
        re.IGNORECASE,
    )

    for fp in sorted(system_dir.iterdir()):
        if not fp.is_file():
            continue
        if fp.name.endswith(":Zone.Identifier"):
            continue
        if fp.name == "_analysis_manifest.json":
            continue
        if fp.suffix.lower() != ".json":
            continue
        m = pat.match(fp.name)
        if not m:
            continue

        init_dt = datetime.strptime(m.group(1) + m.group(2), "%Y-%m-%d%H")
        if not (year_start <= init_dt.year <= year_end):
            continue
        sid = m.group(3)

        out[("GFS", sid, init_dt)] = evaluate_system_file("GFS", sid, init_dt, fp)

    return out


def evaluate_system_file(model: str, sid: str, init_dt: datetime, fp: Path) -> SystemStats:
    parse_ok = True
    time_steps = 0
    nonempty_steps = 0
    unique_names: Set[str] = set()
    fail_reasons: List[str] = []

    try:
        payload = json.loads(fp.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        payload = None
        parse_ok = False

    if not isinstance(payload, dict):
        parse_ok = False

    ts = payload.get("time_series") if isinstance(payload, dict) else None
    if not isinstance(ts, list):
        ts = []

    time_steps = len(ts)
    for row in ts:
        if not isinstance(row, dict):
            continue
        env = row.get("environmental_systems")
        if isinstance(env, list) and len(env) > 0:
            nonempty_steps += 1
            for item in env:
                if isinstance(item, dict):
                    nm = item.get("system_name")
                    if isinstance(nm, str) and nm:
                        unique_names.add(nm)

    nonempty_ratio = (nonempty_steps / time_steps) if time_steps else 0.0
    has_required_any = len(unique_names.intersection(REQUIRED_SYSTEMS)) > 0
    has_core_any = len(unique_names.intersection(CORE_SYSTEMS)) > 0

    if not parse_ok:
        fail_reasons.append("system_parse_failed")
    if time_steps == 0:
        fail_reasons.append("system_time_steps_eq0")
    if nonempty_steps == 0:
        fail_reasons.append("system_nonempty_steps_eq0")
    if len(unique_names) == 0:
        fail_reasons.append("system_unique_names_eq0")

    system_pass_hard = len(fail_reasons) == 0

    return SystemStats(
        model=model,
        storm_id=sid,
        init_time_utc=init_dt,
        source_file=str(fp),
        exists=True,
        parse_ok=parse_ok,
        time_steps=time_steps,
        nonempty_steps=nonempty_steps,
        nonempty_ratio=nonempty_ratio,
        unique_names=unique_names,
        has_required_any=has_required_any,
        has_core_any=has_core_any,
        system_pass_hard=system_pass_hard,
        system_fail_reasons=fail_reasons,
    )


def combine_qc(
    track_candidates: Dict[Tuple[str, str, datetime], List[TrackStats]],
    system_map: Dict[Tuple[str, str, datetime], SystemStats],
    gt_storm_ids: Set[str],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    for key, cands in track_candidates.items():
        model, sid, init_dt = key
        best_track = pick_best_track(cands)
        system_stats = system_map.get(key)

        system_exists = system_stats is not None
        if system_exists:
            system_pass_hard = system_stats.system_pass_hard
            system_time_steps = system_stats.time_steps
            system_nonempty_steps = system_stats.nonempty_steps
            system_nonempty_ratio = system_stats.nonempty_ratio
            system_unique_names = sorted(system_stats.unique_names)
            system_has_required_any = system_stats.has_required_any
            system_has_core_any = system_stats.has_core_any
            system_file = system_stats.source_file
            system_fail_reasons = list(system_stats.system_fail_reasons)
        else:
            system_pass_hard = False
            system_time_steps = 0
            system_nonempty_steps = 0
            system_nonempty_ratio = 0.0
            system_unique_names = []
            system_has_required_any = False
            system_has_core_any = False
            system_file = ""
            system_fail_reasons = ["system_file_missing"]

        in_groundtruth = sid in gt_storm_ids

        fail_reasons = list(best_track.track_fail_reasons) + system_fail_reasons
        if not in_groundtruth:
            fail_reasons.append("storm_id_not_in_groundtruth")

        usable_guidance_cycle = best_track.track_pass_hard and system_pass_hard and in_groundtruth

        rows.append(
            {
                "model": model,
                "storm_id": sid,
                "init_time_utc": init_dt.strftime("%Y-%m-%d %H:%M:%S"),
                "init_year": init_dt.year,
                "track_source_file": best_track.source_file,
                "track_source_kind": best_track.source_kind,
                "track_rows_total": best_track.rows_total,
                "track_rows_valid": best_track.rows_valid,
                "track_unique_valid_times": best_track.unique_valid_times,
                "track_horizon_h": round(best_track.horizon_h, 3),
                "track_duplicate_ratio": round(best_track.duplicate_ratio, 6),
                "track_invalid_time_ratio": round(best_track.invalid_time_ratio, 6),
                "track_invalid_geo_ratio": round(best_track.invalid_geo_ratio, 6),
                "track_invalid_phys_ratio": round(best_track.invalid_phys_ratio, 6),
                "track_tau_off6_ratio": round(best_track.tau_off6_ratio, 6),
                "track_pass_hard": int(best_track.track_pass_hard),
                "system_source_file": system_file,
                "system_exists": int(system_exists),
                "system_time_steps": system_time_steps,
                "system_nonempty_steps": system_nonempty_steps,
                "system_nonempty_ratio": round(system_nonempty_ratio, 6),
                "system_unique_names": "|".join(system_unique_names),
                "system_has_required_any": int(system_has_required_any),
                "system_has_core_any": int(system_has_core_any),
                "system_pass_hard": int(system_pass_hard),
                "in_groundtruth": int(in_groundtruth),
                "usable_guidance_cycle": int(usable_guidance_cycle),
                "fail_reasons": "|".join(sorted(set(fail_reasons))),
            }
        )

    rows.sort(key=lambda r: (r["model"], r["init_time_utc"], r["storm_id"]))
    return rows


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize_qc(rows: List[Dict[str, Any]], year_start: int, year_end: int) -> Dict[str, Any]:
    total = len(rows)
    usable = [r for r in rows if r["usable_guidance_cycle"] == 1]
    by_model_total = Counter(r["model"] for r in rows)
    by_model_usable = Counter(r["model"] for r in usable)

    fail_counter = Counter()
    for r in rows:
        reasons = [x for x in (r.get("fail_reasons") or "").split("|") if x]
        for reason in reasons:
            fail_counter[reason] += 1

    missing_required_by_model = Counter()
    missing_core_by_model = Counter()
    for r in rows:
        if r["system_pass_hard"] == 1 and r["system_has_required_any"] == 0:
            missing_required_by_model[r["model"]] += 1
        if r["system_pass_hard"] == 1 and r["system_has_core_any"] == 0:
            missing_core_by_model[r["model"]] += 1

    unique_storm_total = len(set((r["storm_id"] for r in rows)))
    unique_storm_usable = len(set((r["storm_id"] for r in usable)))

    year_model_total = Counter((r["init_year"], r["model"]) for r in rows)
    year_model_usable = Counter((r["init_year"], r["model"]) for r in usable)

    year_coverage = []
    for year in range(year_start, year_end + 1):
        for model in ["HRES", "GFS"]:
            t = year_model_total[(year, model)]
            u = year_model_usable[(year, model)]
            if t == 0 and u == 0:
                continue
            year_coverage.append(
                {
                    "year": year,
                    "model": model,
                    "total_cycles": t,
                    "usable_cycles": u,
                    "usable_ratio": round(u / t, 6) if t else 0.0,
                }
            )

    return {
        "year_window": [year_start, year_end],
        "total_cycles": total,
        "usable_cycles": len(usable),
        "usable_ratio": round(len(usable) / total, 6) if total else 0.0,
        "total_unique_storm_ids": unique_storm_total,
        "usable_unique_storm_ids": unique_storm_usable,
        "cycles_by_model_total": dict(sorted(by_model_total.items())),
        "cycles_by_model_usable": dict(sorted(by_model_usable.items())),
        "top_fail_reasons": fail_counter.most_common(20),
        "system_missing_required_any_by_model": dict(sorted(missing_required_by_model.items())),
        "system_missing_core_any_by_model": dict(sorted(missing_core_by_model.items())),
        "year_model_coverage": year_coverage,
    }


def build_usable_cycle_index(rows: List[Dict[str, Any]]) -> Dict[Tuple[str, str], List[datetime]]:
    out: Dict[Tuple[str, str], List[datetime]] = defaultdict(list)
    for r in rows:
        if r["usable_guidance_cycle"] != 1:
            continue
        model = r["model"]
        sid = r["storm_id"]
        init_dt = datetime.strptime(r["init_time_utc"], "%Y-%m-%d %H:%M:%S")
        out[(model, sid)].append(init_dt)

    for k in list(out.keys()):
        out[k] = sorted(set(out[k]))
    return out


def sample_coverage_after_qc(
    *,
    noaa_dir: Path,
    gt_by_dt: Dict[datetime, List[Any]],
    cds_sid_time_set: Set[Tuple[str, datetime]],
    usable_cycle_index: Dict[Tuple[str, str], List[datetime]],
    year_start: int,
    year_end: int,
) -> Dict[str, Any]:
    adv_files = sorted(noaa_dir.rglob("forecast_advisory/*.txt"))
    advisories = [parse_advisory_file(p) for p in adv_files]

    matched = []
    for a in advisories:
        if a.issue_dt is None:
            continue
        if not (year_start <= a.issue_dt.year <= year_end):
            continue
        m = match_advisory_to_gt(a, gt_by_dt, time_window_h=3)
        if m.matched and m.storm_id and m.gt_dt:
            matched.append((a, m))

    with_cds = 0
    with_hres = 0
    with_gfs = 0
    with_any_guidance = 0
    with_both_guidance = 0
    dataset_v0_ready = 0
    by_basin = Counter()

    def has_usable(model: str, sid: str, issue_dt: datetime) -> bool:
        times = usable_cycle_index.get((model, sid), [])
        for t in times:
            if abs((t - issue_dt).total_seconds()) <= 3 * 3600:
                return True
        return False

    for a, m in matched:
        sid = m.storm_id
        gt_dt = m.gt_dt
        assert sid is not None and gt_dt is not None and a.issue_dt is not None

        has_cds = (sid, gt_dt) in cds_sid_time_set
        has_h = has_usable("HRES", sid, a.issue_dt)
        has_g = has_usable("GFS", sid, a.issue_dt)

        if has_cds:
            with_cds += 1
        if has_h:
            with_hres += 1
        if has_g:
            with_gfs += 1
        if has_h or has_g:
            with_any_guidance += 1
        if has_h and has_g:
            with_both_guidance += 1

        if has_cds and (has_h or has_g):
            dataset_v0_ready += 1
            by_basin[m.basin or "UNKNOWN"] += 1

    return {
        "year_window": [year_start, year_end],
        "matched_noaa_advisories_with_groundtruth_pm3h": len(matched),
        "matched_noaa_with_cds_now_env": with_cds,
        "matched_noaa_with_hres_guidance_after_qc_pm3h": with_hres,
        "matched_noaa_with_gfs_guidance_after_qc_pm3h": with_gfs,
        "matched_noaa_with_any_guidance_after_qc_pm3h": with_any_guidance,
        "matched_noaa_with_both_guidance_after_qc_pm3h": with_both_guidance,
        "dataset_v0_constructable_after_guidance_qc_pm3h": dataset_v0_ready,
        "dataset_v0_constructable_by_basin_after_guidance_qc_pm3h": dict(sorted(by_basin.items())),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--year-start", type=int, default=2016)
    parser.add_argument("--year-end", type=int, default=2025)
    parser.add_argument("--hres-track-dir", default="HRES_forecast/HRES_track")
    parser.add_argument("--gfs-track-dir", default="GFS_forecast/GFS_track")
    parser.add_argument("--hres-system-dir", default="HRES_forecast/HRES_system")
    parser.add_argument("--gfs-system-dir", default="GFS_forecast/GFS_system")
    parser.add_argument("--noaa-dir", default="noaa")
    parser.add_argument("--groundtruth-csv", default="GroundTruth_Cyclones/matched_cyclone_tracks.csv")
    parser.add_argument("--cds-dir", default="CDS_real")
    parser.add_argument("--output-dir", default="data/interim/qc")
    args = parser.parse_args()

    year_start = args.year_start
    year_end = args.year_end

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    qc_csv = output_dir / f"guidance_cycle_qc_{year_start}_{year_end}.csv"
    usable_csv = output_dir / f"guidance_usable_cycles_{year_start}_{year_end}.csv"
    summary_json = output_dir / f"guidance_cycle_qc_summary_{year_start}_{year_end}.json"
    coverage_json = output_dir / f"dataset_sample_coverage_after_guidance_qc_{year_start}_{year_end}.json"

    # GroundTruth + CDS mapping for downstream sample-level coverage.
    _, gt_by_dt, gt_storm_ids, _, _ = parse_gt(Path(args.groundtruth_csv))
    _, _, _, cds_sid_time_set, _ = map_cds_to_gt(Path(args.cds_dir), gt_by_dt)

    # Parse track/system and run QC.
    hres_track_candidates = parse_hres_track_groups(Path(args.hres_track_dir), year_start, year_end)
    gfs_track_candidates = parse_gfs_track_groups(Path(args.gfs_track_dir), year_start, year_end)
    all_track_candidates = {**hres_track_candidates}
    for k, v in gfs_track_candidates.items():
        all_track_candidates.setdefault(k, []).extend(v)

    hres_system = parse_hres_system(Path(args.hres_system_dir), year_start, year_end)
    gfs_system = parse_gfs_system(Path(args.gfs_system_dir), year_start, year_end)
    all_system = {**hres_system, **gfs_system}

    qc_rows = combine_qc(all_track_candidates, all_system, gt_storm_ids)
    usable_rows = [r for r in qc_rows if r["usable_guidance_cycle"] == 1]

    write_csv(qc_csv, qc_rows)
    write_csv(usable_csv, usable_rows)

    summary = summarize_qc(qc_rows, year_start, year_end)
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    usable_cycle_index = build_usable_cycle_index(qc_rows)
    coverage = sample_coverage_after_qc(
        noaa_dir=Path(args.noaa_dir),
        gt_by_dt=gt_by_dt,
        cds_sid_time_set=cds_sid_time_set,
        usable_cycle_index=usable_cycle_index,
        year_start=year_start,
        year_end=year_end,
    )
    coverage_json.write_text(json.dumps(coverage, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[OK] QC table: {qc_csv}")
    print(f"[OK] Usable cycles: {usable_csv}")
    print(f"[OK] QC summary: {summary_json}")
    print(f"[OK] Sample coverage: {coverage_json}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(coverage, ensure_ascii=False, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
