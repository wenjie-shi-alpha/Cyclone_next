#!/usr/bin/env python3
"""Build a leak-safe v0.1 sample preview with final-shape placeholders.

This script keeps the sample close to the target training schema:
{
  "sample_id": ...,
  "prompt": {...},
  "target": {...},
  "verification_targets": {...}
}

It also keeps explicit placeholders for missing datasets (A-deck, B-deck,
satellite/ASCAT/recon), while constructing proxy blocks from available data.
"""

from __future__ import annotations

import csv
import json
import math
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


BASE = Path(".")
ATCF_BLOCKED_GUIDANCE_MODELS = {"CARQ", "WRNG", "OFCL", "OFCI"}


def parse_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def circular_mean_lon(lons: List[float]) -> Optional[float]:
    if not lons:
        return None
    radians = [math.radians(v) for v in lons]
    s = sum(math.sin(v) for v in radians)
    c = sum(math.cos(v) for v in radians)
    if abs(s) < 1e-12 and abs(c) < 1e-12:
        return norm_lon_to_minus180_180(lons[0])
    return norm_lon_to_minus180_180(math.degrees(math.atan2(s, c)))


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2.0) ** 2 + math.cos(p1) * math.cos(p2) * (math.sin(dl / 2.0) ** 2)
    return 2.0 * r * math.asin(math.sqrt(a))


def sanitize_goes_metric(signal_name: str, value: Optional[float]) -> Optional[float]:
    if value is None or not math.isfinite(value):
        return None
    if value <= -9000:
        return None

    if signal_name in {
        "cloud_top_temp_min_k",
        "cloud_top_temp_p10_k",
        "cloud_top_temp_mean_k",
    }:
        if 400 < abs(value) < 10000:
            value /= 10.0
        if value < 120 or value > 380:
            return None
        return value

    if signal_name == "cloud_top_temp_std_k":
        if 400 < abs(value) < 10000:
            value /= 10.0
        if value < 0 or value > 150:
            return None
        return value

    if signal_name == "eye_ring_temp_contrast_k":
        if 400 < abs(value) < 10000:
            value /= 10.0
        if value < -200 or value > 200:
            return None
        return value

    if signal_name in {"cold_cloud_fraction_inner", "cold_cloud_fraction_ring"}:
        if value < 0 or value > 1:
            return None
        return value

    if signal_name in {"cold_cloud_area_inner_km2", "cold_cloud_area_ring_km2"}:
        if value < 0:
            return None
        return value

    return value


def norm_lon_to_minus180_180(lon: float) -> float:
    while lon >= 180:
        lon -= 360
    while lon < -180:
        lon += 360
    return lon


def read_text(path: Path) -> List[str]:
    return path.read_text(encoding="utf-8", errors="ignore").splitlines()


def compact_ws(text: str) -> str:
    return " ".join(text.split())


def parse_iso_utc(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ")


def parse_iso_utc_flexible(s: str) -> Optional[datetime]:
    text = (s or "").strip()
    if not text:
        return None
    fmts = [
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S.%f",
    ]
    for fmt in fmts:
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def load_goes_coverage_summary() -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    candidates = [
        BASE / "data" / "interim" / "goes" / "goes_observation_features_full_summary.json",
        BASE / "data" / "interim" / "goes" / "goes_observation_features_summary.json",
    ]
    trace: Dict[str, Any] = {"goes_summary_file": None}
    for fp in candidates:
        if not fp.exists():
            continue
        try:
            payload = json.loads(fp.read_text(encoding="utf-8"))
        except Exception:
            continue
        trace["goes_summary_file"] = str(fp)
        return payload, trace
    return None, trace


def build_goes_coverage_snapshot(summary: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not summary:
        return None

    rows_written = int(parse_float(summary.get("rows_written")) or 0)
    available_rows = int(parse_float(summary.get("available_rows")) or 0)
    missing_rows = int(parse_float(summary.get("missing_rows")) or 0)
    coverage_rate = round((available_rows / rows_written), 6) if rows_written > 0 else None

    by_year_raw = summary.get("coverage_by_year")
    by_year: Dict[str, Any] = {}
    if isinstance(by_year_raw, dict):
        for year in sorted(by_year_raw.keys()):
            yr_obj = by_year_raw.get(year) or {}
            by_year[year] = {
                "total": int(parse_float(yr_obj.get("total")) or 0),
                "available": int(parse_float(yr_obj.get("available")) or 0),
                "missing": int(parse_float(yr_obj.get("missing")) or 0),
                "coverage_rate": parse_float(yr_obj.get("coverage_rate")),
            }

    return {
        "rows_written": rows_written,
        "available_rows": available_rows,
        "missing_rows": missing_rows,
        "coverage_rate": coverage_rate,
        "dataset_ids_used": summary.get("dataset_ids_used") or [],
        "generated_from": summary.get("generated_from") or "single_run_or_unknown",
    }


def extract_advisory_summary(lines: List[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "center": {},
        "motion": {},
        "intensity": {},
        "forecast_table": [],
        "watch_warning_text": None,
    }

    for ln in lines:
        m = re.search(
            r"CENTER LOCATED NEAR\s+([0-9.]+)([NS])\s+([0-9.]+)([EW])\s+AT\s+(\d{2})/(\d{4})Z",
            ln,
            re.I,
        )
        if m:
            lat = float(m.group(1)) * (1 if m.group(2).upper() == "N" else -1)
            lon = float(m.group(3)) * (-1 if m.group(4).upper() == "W" else 1)
            out["center"] = {
                "lat": lat,
                "lon": lon,
                "obs_day": int(m.group(5)),
                "obs_hhmmz": m.group(6),
            }
            break

    for ln in lines:
        m = re.search(
            r"PRESENT MOVEMENT TOWARD THE\s+(.+?)\s+OR\s+(\d+)\s+DEGREES AT\s+(\d+)\s+KT",
            ln,
            re.I,
        )
        if m:
            out["motion"] = {
                "motion_text": m.group(1).strip(),
                "direction_deg": int(m.group(2)),
                "speed_kt": int(m.group(3)),
            }
            break

    min_p = None
    max_w = None
    for ln in lines:
        m = re.search(r"ESTIMATED MINIMUM CENTRAL PRESSURE\s+(\d+)\s+MB", ln, re.I)
        if m:
            min_p = int(m.group(1))
        m2 = re.search(r"MAX SUSTAINED WINDS\s+(\d+)\s+KT", ln, re.I)
        if m2:
            max_w = int(m2.group(1))
        if min_p is not None and max_w is not None:
            break
    out["intensity"] = {"min_pressure_mb": min_p, "max_wind_kt": max_w}

    for ln in lines:
        if "WATCHES OR WARNINGS" in ln.upper():
            out["watch_warning_text"] = ln.strip()
            break

    fcst: List[Dict[str, Any]] = []
    current: Optional[Dict[str, Any]] = None
    for ln in lines:
        m = re.search(
            r"^\s*FORECAST VALID\s+(\d{2})/(\d{4})Z\s+([0-9.]+)([NS])\s+([0-9.]+)([EW])",
            ln,
            re.I,
        )
        if m:
            lat = float(m.group(3)) * (1 if m.group(4).upper() == "N" else -1)
            lon = float(m.group(5)) * (-1 if m.group(6).upper() == "W" else 1)
            current = {
                "valid_day": int(m.group(1)),
                "valid_hhmmz": m.group(2),
                "lat": lat,
                "lon": lon,
                "vmax_kt": None,
            }
            fcst.append(current)
            continue
        if current is not None:
            m2 = re.search(r"^\s*MAX WIND\s+(\d+)\s+KT", ln, re.I)
            if m2:
                current["vmax_kt"] = int(m2.group(1))
    out["forecast_table"] = fcst

    return out


def extract_discussion_sections(lines: List[str]) -> Dict[str, Any]:
    body: List[str] = []
    started = False
    for ln in lines:
        if "FORECAST POSITIONS AND MAX WINDS" in ln.upper():
            break
        if not started:
            if ln.strip().startswith("Satellite"):
                started = True
                body.append(ln)
            continue
        body.append(ln)

    paragraphs: List[str] = []
    chunk: List[str] = []
    for ln in body:
        if not ln.strip():
            if chunk:
                paragraphs.append(compact_ws(" ".join(chunk)))
                chunk = []
            continue
        chunk.append(ln.strip())
    if chunk:
        paragraphs.append(compact_ws(" ".join(chunk)))

    current_analysis_text = paragraphs[0] if len(paragraphs) >= 1 else ""
    forecast_reasoning_text = paragraphs[1] if len(paragraphs) >= 2 else ""
    additional_context_text = "\n\n".join(paragraphs[2:]) if len(paragraphs) >= 3 else ""
    full_reasoning_text = "\n\n".join(paragraphs)

    track: List[Dict[str, Any]] = []
    in_table = False
    for ln in lines:
        if "FORECAST POSITIONS AND MAX WINDS" in ln.upper():
            in_table = True
            continue
        if not in_table:
            continue
        if ln.strip().startswith("$$"):
            break
        m = re.search(
            r"^\s*(INIT|\d+H)\s+(\d{2})/(\d{4})Z\s+([0-9.]+)([NS])\s+([0-9.]+)([EW])\s+(\d+)\s+KT",
            ln,
            re.I,
        )
        if not m:
            continue
        tau_label = m.group(1).upper()
        tau_h = 0 if tau_label == "INIT" else int(tau_label[:-1])
        lat = float(m.group(4)) * (1 if m.group(5).upper() == "N" else -1)
        lon = float(m.group(6)) * (-1 if m.group(7).upper() == "W" else 1)
        track.append(
            {
                "tau_h": tau_h,
                "valid_day": int(m.group(2)),
                "valid_hhmmz": m.group(3),
                "lat": lat,
                "lon": lon,
                "vmax_kt": int(m.group(8)),
            }
        )

    return {
        "current_analysis_text": current_analysis_text,
        "forecast_reasoning_text": forecast_reasoning_text,
        "additional_context_text": additional_context_text,
        "full_reasoning_text": full_reasoning_text,
        "forecast_positions": track,
    }


def extract_public_summary(lines: List[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for i, ln in enumerate(lines):
        if "SUMMARY OF" in ln.upper():
            out["summary_block_header"] = ln.strip()
            out["summary_lines"] = [x.strip() for x in lines[i + 2 : i + 9] if x.strip()]
            break
    return out


def extract_wind_prob_summary(lines: List[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "has_location_rows": False,
        "location_rows": 0,
    }
    for ln in lines:
        m = re.search(r"WIND SPEED PROBABILITIES NUMBER\s+(\d+)", ln, re.I)
        if m:
            out["product_number"] = int(m.group(1))
            break

    row_count = 0
    in_location_table = False
    for ln in lines:
        if ln.strip().startswith("LOCATION"):
            in_location_table = True
            out["has_location_rows"] = True
            continue
        if not in_location_table:
            continue
        if ln.strip().startswith("$$"):
            break
        if not ln.strip():
            continue
        if "KT" in ln and "LOCATION" in ln:
            continue
        if re.match(r"^[A-Z0-9].*[A-Z0-9]$", ln.strip()):
            row_count += 1
    out["location_rows"] = row_count
    return out


def load_groundtruth_state(storm_id: str, issue_dt: datetime) -> Dict[str, Any]:
    best = None
    best_dt: Optional[datetime] = None
    gt_path = BASE / "GroundTruth_Cyclones" / "matched_cyclone_tracks.csv"
    with gt_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("storm_id") != storm_id:
                continue
            dt = datetime.strptime((row.get("datetime") or "")[:19], "%Y-%m-%d %H:%M:%S")
            if best is None or abs((dt - issue_dt).total_seconds()) < abs((best_dt - issue_dt).total_seconds()):
                best = row
                best_dt = dt

    if best is None or best_dt is None:
        return {}

    return {
        "matched_datetime_utc": best_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "lat": parse_float(best.get("latitude")),
        "lon": parse_float(best.get("longitude")),
        "max_wind_wmo": parse_float(best.get("max_wind_wmo")),
        "min_pressure_wmo": parse_float(best.get("min_pressure_wmo")),
        "max_wind_usa": parse_float(best.get("max_wind_usa")),
        "min_pressure_usa": parse_float(best.get("min_pressure_usa")),
        "storm_speed": parse_float(best.get("storm_speed")),
        "storm_direction": parse_float(best.get("storm_direction")),
    }


def pick_cds_row(issue_dt: datetime, target_lat: float, target_lon_w: float) -> Tuple[Dict[str, Any], Path]:
    month_file = BASE / "CDS_real" / f"cds_environment_analysis_{issue_dt.strftime('%Y-%m')}.json"
    payload = json.loads(month_file.read_text(encoding="utf-8"))
    best = None
    best_d = None

    for row in payload.get("environmental_analysis", []):
        t = str(row.get("time", ""))[:19]
        try:
            dt = datetime.strptime(t.replace("T", " "), "%Y-%m-%d %H:%M:%S")
        except ValueError:
            continue
        if dt != issue_dt:
            continue

        tc = row.get("tc_position") or {}
        lat = parse_float(tc.get("lat"))
        lon = parse_float(tc.get("lon"))
        if lat is None or lon is None:
            continue
        lon_w = norm_lon_to_minus180_180(lon)
        d = (lat - target_lat) ** 2 + (lon_w - target_lon_w) ** 2
        if best is None or d < best_d:
            best = row
            best_d = d

    if best is None:
        raise RuntimeError("CDS row not found for sample time")
    return best, month_file


def extract_cds_features(row: Dict[str, Any]) -> Dict[str, Any]:
    features: Dict[str, Any] = {}
    systems = row.get("environmental_systems") or []
    by_name = {s.get("system_name"): s for s in systems if isinstance(s, dict) and s.get("system_name")}

    def num_from_intensity(sys_obj: Dict[str, Any]) -> Optional[float]:
        intensity = sys_obj.get("intensity") if isinstance(sys_obj, dict) else None
        if not isinstance(intensity, dict):
            return None
        for k in ["value", "average_value", "max_value", "speed", "surface_temp"]:
            if k in intensity:
                v = parse_float(intensity.get(k))
                if v is not None:
                    return v
        return None

    mapping = {
        "VerticalWindShear": "vertical_wind_shear",
        "UpperLevelDivergence": "upper_level_divergence",
        "OceanHeatContent": "ocean_heat_content_or_sst",
        "SubtropicalHigh": "subtropical_high",
        "WesterlyTrough": "westerly_trough",
        "MonsoonTrough": "monsoon_trough",
        "LowLevelFlow": "low_level_flow",
    }

    for src_name, dst_name in mapping.items():
        s = by_name.get(src_name)
        if s is None:
            features[dst_name] = None
            continue
        intensity = s.get("intensity") if isinstance(s.get("intensity"), dict) else {}
        features[dst_name] = {
            "value": num_from_intensity(s),
            "unit": intensity.get("unit"),
            "level": intensity.get("level"),
            "description": s.get("description"),
        }
    return features


def extract_hres_track(storm_id: str, init_dt: datetime) -> Tuple[List[Dict[str, Any]], Path]:
    fp = (
        BASE
        / "HRES_forecast"
        / "HRES_track"
        / f"track_{storm_id}_{init_dt.strftime('%Y%m%d')}_{init_dt.strftime('%H%M')}.csv"
    )
    with fp.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    by_time: Dict[datetime, Dict[str, str]] = {}
    for row in rows:
        dt = datetime.strptime((row["time"] or "")[:19], "%Y-%m-%d %H:%M:%S")
        cur = by_time.get(dt)
        if cur is None:
            by_time[dt] = row
            continue
        cur_msl = parse_float(cur.get("msl")) or 1e18
        new_msl = parse_float(row.get("msl")) or 1e18
        if new_msl < cur_msl:
            by_time[dt] = row

    out: List[Dict[str, Any]] = []
    for dt in sorted(by_time.keys()):
        row = by_time[dt]
        tau_h = int(round((dt - init_dt).total_seconds() / 3600))
        out.append(
            {
                "tau_h": tau_h,
                "valid_time_utc": dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "lat": parse_float(row.get("lat")),
                "lon": norm_lon_to_minus180_180(parse_float(row.get("lon")) or 0.0),
                "mslp_hpa": round((parse_float(row.get("msl")) or 0.0) / 100.0, 1),
                "wind_kt": round((parse_float(row.get("wind")) or 0.0) * 1.94384, 1),
            }
        )
    return out, fp


def extract_hres_system(storm_id: str, init_dt: datetime) -> Tuple[List[Dict[str, Any]], Path]:
    fp = (
        BASE
        / "HRES_forecast"
        / "HRES_system"
        / f"{storm_id}_{init_dt.strftime('%Y%m%d')}_{init_dt.strftime('%H%M')}_TC_Analysis_{storm_id}.json"
    )
    payload = json.loads(fp.read_text(encoding="utf-8"))

    target_names = {
        "VerticalWindShear",
        "UpperLevelDivergence",
        "OceanHeatContent",
        "SubtropicalHigh",
        "WesterlyTrough",
        "MonsoonTrough",
        "LowLevelFlow",
    }

    def pick_num(system: Dict[str, Any]) -> Optional[float]:
        intensity = system.get("intensity")
        if not isinstance(intensity, dict):
            return None
        for key in ["value", "average_value", "max_value", "speed", "surface_temp"]:
            if key in intensity:
                v = parse_float(intensity.get(key))
                if v is not None:
                    return v
        return None

    by_time: Dict[datetime, Dict[str, Any]] = {}
    for row in payload.get("time_series", []):
        t = row.get("time")
        if not t:
            continue
        dt = datetime.strptime(str(t)[:19].replace("T", " "), "%Y-%m-%d %H:%M:%S")
        sys_map: Dict[str, Any] = {}
        for s in row.get("environmental_systems") or []:
            if not isinstance(s, dict):
                continue
            name = s.get("system_name")
            if name not in target_names:
                continue
            intensity = s.get("intensity") if isinstance(s.get("intensity"), dict) else {}
            sys_map[name] = {
                "value": pick_num(s),
                "unit": intensity.get("unit"),
                "level": intensity.get("level"),
            }
        if dt not in by_time:
            by_time[dt] = {}
        by_time[dt].update(sys_map)

    out: List[Dict[str, Any]] = []
    for dt in sorted(by_time.keys()):
        tau_h = int(round((dt - init_dt).total_seconds() / 3600))
        if tau_h < 0:
            continue
        out.append(
            {
                "tau_h": tau_h,
                "valid_time_utc": dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "systems": by_time[dt],
            }
        )
    return out, fp


def add_issue_lead_fields(
    points: List[Dict[str, Any]], issue_dt: datetime, tau_key_name: str
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for p in points:
        dt = parse_iso_utc(p["valid_time_utc"])
        q = dict(p)
        q["lead_from_issue_h"] = int(round((dt - issue_dt).total_seconds() / 3600))
        q[tau_key_name] = q.pop("tau_h")
        out.append(q)
    return out


def latest_pre_issue_point(points: List[Dict[str, Any]], issue_dt: datetime) -> Optional[Dict[str, Any]]:
    cands = [p for p in points if parse_iso_utc(p["valid_time_utc"]) <= issue_dt]
    if not cands:
        return None
    cands.sort(key=lambda x: parse_iso_utc(x["valid_time_utc"]))
    return cands[-1]


def future_only(points: List[Dict[str, Any]], issue_dt: datetime) -> List[Dict[str, Any]]:
    return [p for p in points if parse_iso_utc(p["valid_time_utc"]) > issue_dt]


def build_observation_placeholder(
    missing_reason: Optional[str] = None,
    source_file: Optional[str] = None,
) -> Dict[str, Any]:
    reason = (
        missing_reason
        or "GOES/ASCAT/Recon structured observation table not ingested in workspace yet"
    )
    return {
        "status": "missing_real_data",
        "value": None,
        "expected_real_fields": ["obs_time_utc", "obs_type", "signal", "value", "qc_flag", "source_platform"],
        "missing_reason": reason,
        "source_file": source_file,
        "policy_note": "no forecast_discussion text or derived flags are allowed in prompt under strict anti-leak mode",
    }


def load_goes_observation_structured(
    storm_id: str,
    issue_dt: datetime,
    max_time_diff_hours: int = 3,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    goes_fp = BASE / "data" / "interim" / "goes" / "goes_observation_features.csv"
    trace: Dict[str, Any] = {"goes_feature_file": str(goes_fp)}

    if not goes_fp.exists():
        return (
            build_observation_placeholder(
                missing_reason="GOES structured feature file not found in workspace",
                source_file=str(goes_fp),
            ),
            trace,
        )

    best_row: Optional[Dict[str, str]] = None
    best_abs_hours: Optional[float] = None
    with goes_fp.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row_storm_id = (row.get("storm_id") or "").strip()
            if row_storm_id != storm_id:
                continue
            row_issue_dt = parse_iso_utc_flexible(row.get("issue_time_utc") or "")
            if row_issue_dt is None:
                continue
            dt_abs_h = abs((row_issue_dt - issue_dt).total_seconds()) / 3600.0
            if best_abs_hours is None or dt_abs_h < best_abs_hours:
                best_abs_hours = dt_abs_h
                best_row = row

    if best_row is None or best_abs_hours is None:
        return (
            build_observation_placeholder(
                missing_reason="No GOES rows matched storm_id and issue_time window",
                source_file=str(goes_fp),
            ),
            trace,
        )

    if best_abs_hours > max_time_diff_hours:
        return (
            build_observation_placeholder(
                missing_reason=(
                    f"Nearest GOES row exceeds time tolerance: "
                    f"{best_abs_hours:.2f}h > {max_time_diff_hours}h"
                ),
                source_file=str(goes_fp),
            ),
            trace,
        )

    goes_status = (best_row.get("goes_status") or "").strip()
    if goes_status != "available":
        missing_reason = (best_row.get("missing_reason") or "").strip() or "GOES row marked missing_real_data"
        return (
            build_observation_placeholder(
                missing_reason=missing_reason,
                source_file=str(goes_fp),
            ),
            trace,
        )

    obs_time_utc = (best_row.get("obs_time_utc") or "").strip() or (best_row.get("issue_time_utc") or "").strip()
    source_platform = (
        (best_row.get("goes_source_collection") or "").strip()
        or (best_row.get("goes_platform") or "").strip()
        or "GOES_MCMIPC"
    )
    qc_has_image = str(best_row.get("qc_has_image") or "").strip() in {"1", "true", "True", "TRUE"}
    qc_flag = "ok" if qc_has_image else "warn"

    metric_specs = [
        ("cloud_top_temp_min_k", "c13_min_k", "K"),
        ("cloud_top_temp_p10_k", "c13_p10_k", "K"),
        ("cloud_top_temp_mean_k", "c13_mean_k", "K"),
        ("cloud_top_temp_std_k", "c13_std_k", "K"),
        ("cold_cloud_area_inner_km2", "cold_area_inner_km2", "km2"),
        ("cold_cloud_fraction_inner", "cold_fraction_inner", "ratio"),
        ("cold_cloud_area_ring_km2", "cold_area_ring_km2", "km2"),
        ("cold_cloud_fraction_ring", "cold_fraction_ring", "ratio"),
        ("eye_ring_temp_contrast_k", "eye_ring_temp_contrast_k", "K"),
    ]

    value_rows: List[Dict[str, Any]] = []
    for signal_name, col_name, unit in metric_specs:
        val = sanitize_goes_metric(signal_name, parse_float(best_row.get(col_name)))
        if val is None:
            continue
        value_rows.append(
            {
                "obs_time_utc": obs_time_utc,
                "obs_type": "goes_ir_structured",
                "signal": signal_name,
                "value": round(val, 4),
                "unit": unit,
                "qc_flag": qc_flag,
                "source_platform": source_platform,
            }
        )

    if not value_rows:
        return (
            build_observation_placeholder(
                missing_reason="GOES row available but no numeric feature columns parsed",
                source_file=str(goes_fp),
            ),
            trace,
        )

    trace.update(
        {
            "goes_request_id": (best_row.get("request_id") or "").strip() or None,
            "goes_issue_time_selected_utc": (best_row.get("issue_time_utc") or "").strip() or None,
            "goes_obs_time_selected_utc": obs_time_utc or None,
        }
    )

    block = {
        "status": "available",
        "source_file": str(goes_fp),
        "match_rule": "storm_id + nearest_issue_time_within_3h",
        "time_match_delta_hours": round(best_abs_hours, 3),
        "obs_time_utc": obs_time_utc,
        "source_platform": source_platform,
        "value": value_rows,
        "qc": {
            "has_image": 1 if qc_has_image else 0,
            "time_within_window": str(best_row.get("qc_time_within_window") or ""),
            "obs_offset_minutes": parse_float(best_row.get("obs_offset_minutes")),
            "obs_offset_abs_minutes": parse_float(best_row.get("obs_offset_abs_minutes")),
        },
        "expected_real_fields": ["obs_time_utc", "obs_type", "signal", "value", "qc_flag", "source_platform"],
        "policy_note": "GOES metrics are extracted from satellite radiances only, no forecast text leakage",
    }
    return block, trace


def load_best_row_by_storm_issue(
    feature_fp: Path,
    storm_id: str,
    issue_dt: datetime,
    max_time_diff_hours: int = 3,
) -> Tuple[Optional[Dict[str, str]], Optional[float], str]:
    if not feature_fp.exists():
        return None, None, f"feature file not found: {feature_fp}"

    best_row: Optional[Dict[str, str]] = None
    best_abs_hours: Optional[float] = None
    with feature_fp.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row_storm_id = (row.get("storm_id") or "").strip()
            if row_storm_id != storm_id:
                continue
            row_issue_dt = parse_iso_utc_flexible(row.get("issue_time_utc") or "")
            if row_issue_dt is None:
                continue
            dt_abs_h = abs((row_issue_dt - issue_dt).total_seconds()) / 3600.0
            if best_abs_hours is None or dt_abs_h < best_abs_hours:
                best_abs_hours = dt_abs_h
                best_row = row

    if best_row is None or best_abs_hours is None:
        return None, None, "No rows matched storm_id and issue_time window"
    if best_abs_hours > max_time_diff_hours:
        return None, best_abs_hours, (
            f"Nearest row exceeds time tolerance: {best_abs_hours:.2f}h > {max_time_diff_hours}h"
        )
    return best_row, best_abs_hours, ""


def build_obs_block_from_feature_row(
    row: Dict[str, str],
    source_file: str,
    status_col: str,
    expected_status: str,
    obs_time_col: str,
    source_platform: str,
    obs_type: str,
    metric_specs: List[Tuple[str, str, str]],
    qc_has_col: str,
    qc_time_col: str,
    offset_col: str,
    offset_abs_col: str,
    policy_note: str,
    time_match_delta_hours: Optional[float],
) -> Dict[str, Any]:
    status = (row.get(status_col) or "").strip()
    if status != expected_status:
        missing_reason = (row.get("missing_reason") or "").strip() or f"{status_col} marked missing_real_data"
        return build_observation_placeholder(missing_reason=missing_reason, source_file=source_file)

    obs_time_utc = (row.get(obs_time_col) or "").strip() or (row.get("issue_time_utc") or "").strip()
    qc_flag = "ok" if str(row.get(qc_has_col) or "").strip() in {"1", "true", "True", "TRUE"} else "warn"

    value_rows: List[Dict[str, Any]] = []
    for signal_name, col_name, unit in metric_specs:
        val = parse_float(row.get(col_name))
        if val is None:
            continue
        value_rows.append(
            {
                "obs_time_utc": obs_time_utc,
                "obs_type": obs_type,
                "signal": signal_name,
                "value": round(val, 4),
                "unit": unit,
                "qc_flag": qc_flag,
                "source_platform": source_platform,
            }
        )

    if not value_rows:
        return build_observation_placeholder(
            missing_reason="Row available but no numeric feature columns parsed",
            source_file=source_file,
        )

    return {
        "status": "available",
        "source_file": source_file,
        "match_rule": "storm_id + nearest_issue_time_within_3h",
        "time_match_delta_hours": round(time_match_delta_hours or 0.0, 3),
        "obs_time_utc": obs_time_utc,
        "source_platform": source_platform,
        "value": value_rows,
        "qc": {
            "has_data": 1 if qc_flag == "ok" else 0,
            "time_within_window": str(row.get(qc_time_col) or ""),
            "obs_offset_minutes": parse_float(row.get(offset_col)),
            "obs_offset_abs_minutes": parse_float(row.get(offset_abs_col)),
        },
        "expected_real_fields": ["obs_time_utc", "obs_type", "signal", "value", "qc_flag", "source_platform"],
        "policy_note": policy_note,
    }


def load_ascat_observation_structured(
    storm_id: str,
    issue_dt: datetime,
    max_time_diff_hours: int = 3,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    fp = BASE / "data" / "interim" / "ascat" / "ascat_observation_features.csv"
    trace: Dict[str, Any] = {"ascat_feature_file": str(fp)}
    row, dt_abs_h, err = load_best_row_by_storm_issue(fp, storm_id, issue_dt, max_time_diff_hours=max_time_diff_hours)
    if row is None:
        return (
            build_observation_placeholder(missing_reason=f"ASCAT {err}", source_file=str(fp)),
            trace,
        )

    source_platform = (
        (row.get("ascat_platform") or "").strip()
        or (row.get("ascat_dataset_id") or "").strip()
        or "ASCAT_L3"
    )
    block = build_obs_block_from_feature_row(
        row=row,
        source_file=str(fp),
        status_col="ascat_status",
        expected_status="available",
        obs_time_col="obs_time_utc",
        source_platform=source_platform,
        obs_type="ascat_surface_wind_structured",
        metric_specs=[
            ("wind_mean_inner_kt", "wind_mean_inner_kt", "kt"),
            ("wind_p90_inner_kt", "wind_p90_inner_kt", "kt"),
            ("wind_max_inner_kt", "wind_max_inner_kt", "kt"),
            ("wind_mean_ring_kt", "wind_mean_ring_kt", "kt"),
            ("wind_p90_ring_kt", "wind_p90_ring_kt", "kt"),
            ("wind_max_ring_kt", "wind_max_ring_kt", "kt"),
            ("wind_area_ge34kt_inner_km2", "wind_area_ge34kt_inner_km2", "km2"),
            ("wind_area_ge50kt_inner_km2", "wind_area_ge50kt_inner_km2", "km2"),
            ("valid_cell_count", "valid_cell_count", "count"),
        ],
        qc_has_col="qc_has_data",
        qc_time_col="qc_time_within_window",
        offset_col="obs_offset_minutes",
        offset_abs_col="obs_offset_abs_minutes",
        policy_note="ASCAT metrics are extracted from scatterometer wind products only, no forecast text leakage",
        time_match_delta_hours=dt_abs_h,
    )

    trace.update(
        {
            "ascat_request_id": (row.get("request_id") or "").strip() or None,
            "ascat_issue_time_selected_utc": (row.get("issue_time_utc") or "").strip() or None,
            "ascat_obs_time_selected_utc": (row.get("obs_time_utc") or "").strip() or None,
            "ascat_dataset_id": (row.get("ascat_dataset_id") or "").strip() or None,
        }
    )
    return block, trace


def load_recon_observation_structured(
    storm_id: str,
    issue_dt: datetime,
    max_time_diff_hours: int = 6,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    candidates = [
        BASE / "data" / "interim" / "recon" / "recon_observation_features.csv",
        BASE / "data" / "interim" / "recon" / "recon_observation_features_full.csv",
    ]
    fp = candidates[0]
    for cand in candidates:
        if cand.exists():
            fp = cand
            break
    trace: Dict[str, Any] = {"recon_feature_file": str(fp)}
    row, dt_abs_h, err = load_best_row_by_storm_issue(fp, storm_id, issue_dt, max_time_diff_hours=max_time_diff_hours)
    if row is None:
        return (
            build_observation_placeholder(missing_reason=f"Recon {err}", source_file=str(fp)),
            trace,
        )

    source_platform = (row.get("recon_message_type") or "").strip() or "NHC_Recon"
    block = build_obs_block_from_feature_row(
        row=row,
        source_file=str(fp),
        status_col="recon_status",
        expected_status="available",
        obs_time_col="recon_obs_time_utc",
        source_platform=source_platform,
        obs_type="recon_aircraft_structured",
        metric_specs=[
            ("vdm_min_slp_mb", "vdm_min_slp_mb", "mb"),
            ("vdm_max_flight_level_wind_kt", "vdm_max_flight_level_wind_kt", "kt"),
            ("vdm_center_lat", "vdm_center_lat", "deg"),
            ("vdm_center_lon", "vdm_center_lon", "deg"),
            ("hdob_max_sfmr_wind_kt", "hdob_max_sfmr_wind_kt", "kt"),
            ("hdob_max_flight_level_wind_kt", "hdob_max_flight_level_wind_kt", "kt"),
            ("dropsonde_min_slp_mb", "dropsonde_min_slp_mb", "mb"),
            ("message_count", "message_count", "count"),
        ],
        qc_has_col="qc_has_data",
        qc_time_col="qc_time_within_window",
        offset_col="obs_offset_minutes",
        offset_abs_col="obs_offset_abs_minutes",
        policy_note="Recon metrics are extracted from NHC aircraft text products only, no forecast text leakage",
        time_match_delta_hours=dt_abs_h,
    )

    trace.update(
        {
            "recon_request_id": (row.get("request_id") or "").strip() or None,
            "recon_issue_time_selected_utc": (row.get("issue_time_utc") or "").strip() or None,
            "recon_obs_time_selected_utc": (row.get("recon_obs_time_utc") or "").strip() or None,
            "recon_source_file": (row.get("recon_source_file") or "").strip() or None,
        }
    )
    return block, trace


def merge_observation_blocks(
    goes_block: Dict[str, Any],
    ascat_block: Dict[str, Any],
    recon_block: Dict[str, Any],
) -> Dict[str, Any]:
    components = {
        "goes_structured_obs": goes_block,
        "ascat_structured_obs": ascat_block,
        "recon_structured_obs": recon_block,
    }
    statuses = {name: (blk.get("status") or "missing_real_data") for name, blk in components.items()}
    available = [name for name, st in statuses.items() if st == "available"]

    if not available:
        out = build_observation_placeholder(
            missing_reason="GOES/ASCAT/Recon structured observation tables are all missing for this sample",
            source_file=None,
        )
        out["component_status"] = statuses
        out["component_missing_reason"] = {
            name: (blk.get("missing_reason") or "")
            for name, blk in components.items()
            if blk.get("status") != "available"
        }
        return out

    merged_value: List[Dict[str, Any]] = []
    source_files: List[str] = []
    component_missing_reason: Dict[str, str] = {}
    for name, blk in components.items():
        if blk.get("status") == "available":
            vals = blk.get("value")
            if isinstance(vals, list):
                merged_value.extend(vals)
            source_file = blk.get("source_file")
            if isinstance(source_file, str) and source_file:
                source_files.append(source_file)
        else:
            component_missing_reason[name] = str(blk.get("missing_reason") or "")

    overall_status = "available" if len(available) == len(components) else "partial_available"
    obs_time = None
    for name in ["goes_structured_obs", "ascat_structured_obs", "recon_structured_obs"]:
        blk = components[name]
        if blk.get("status") == "available":
            obs_time = blk.get("obs_time_utc")
            break

    return {
        "status": overall_status,
        "obs_time_utc": obs_time,
        "value": merged_value,
        "source_files": source_files,
        "component_status": statuses,
        "component_missing_reason": component_missing_reason,
        "expected_real_fields": ["obs_time_utc", "obs_type", "signal", "value", "qc_flag", "source_platform"],
        "policy_note": "Merged observation evidence from GOES + ASCAT + Recon under strict anti-leak mode",
    }


def build_multimodel_proxy(future_track_points: List[Dict[str, Any]]) -> Dict[str, Any]:
    spread_proxy = []
    for p in future_track_points:
        spread_proxy.append(
            {
                "lead_from_issue_h": p["lead_from_issue_h"],
                "model_count": 1,
                "consensus_lat": p["lat"],
                "consensus_lon": p["lon"],
                "consensus_vmax_kt": p["wind_kt"],
                "consensus_mslp_hpa": p["mslp_hpa"],
                "track_spread_km": 0.0,
                "wind_spread_kt": 0.0,
                "method": "single_model_proxy_not_real_multimodel_spread",
            }
        )
    return {
        "status": "missing_real_data",
        "missing_reason": "ATCF A-deck multimodel guidance not downloaded/cleaned in workspace yet",
        "expected_real_fields": [
            "model_count",
            "consensus_lat",
            "consensus_lon",
            "consensus_vmax_kt",
            "track_spread_km",
            "wind_spread_kt",
        ],
        "proxy_constructed_from_available_data": {
            "consensus_spread_proxy": spread_proxy,
        },
    }


def strip_atcf_model_source_tokens(value: Any) -> Any:
    """Remove high-cardinality model-id payload from ATCF multimodel block."""
    if isinstance(value, dict):
        out: Dict[str, Any] = {}
        for k, v in value.items():
            if k in {"model_ids", "models"}:
                continue
            if k == "expected_real_fields" and isinstance(v, list):
                out[k] = [x for x in v if x not in {"model_ids", "models"}]
                continue
            out[k] = strip_atcf_model_source_tokens(v)
        return out
    if isinstance(value, list):
        return [strip_atcf_model_source_tokens(x) for x in value]
    return value


def load_atcf_multimodel_guidance(
    storm_id: str,
    issue_dt: datetime,
    max_horizon_h: int = 120,
) -> Tuple[Optional[Dict[str, Any]], Dict[str, str]]:
    storm_dir = BASE / "data" / "interim" / "atcf" / "by_storm" / storm_id
    guidance_fp = storm_dir / "a_deck_guidance.csv"
    spread_fp = storm_dir / "a_deck_spread.csv"
    trace = {
        "guidance_file": str(guidance_fp),
        "spread_file": str(spread_fp),
    }

    if not guidance_fp.exists():
        return None, trace

    cutoff_dt = issue_dt + timedelta(hours=max_horizon_h)
    raw_rows: List[Dict[str, Any]] = []
    with guidance_fp.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            init_time = row.get("init_time_utc") or ""
            valid_time = row.get("valid_time_utc") or ""
            model = (row.get("model") or "").strip().upper()
            if not init_time or not valid_time or not model:
                continue
            if model in ATCF_BLOCKED_GUIDANCE_MODELS:
                continue
            try:
                init_dt = parse_iso_utc(init_time)
                valid_dt = parse_iso_utc(valid_time)
            except ValueError:
                continue
            if init_dt > issue_dt:
                continue
            if valid_dt <= issue_dt or valid_dt > cutoff_dt:
                continue
            tau_h = int(parse_float(row.get("tau_h")) or 0)
            raw_rows.append(
                {
                    "atcf_storm_id": (row.get("atcf_storm_id") or "").strip().upper(),
                    "model": model,
                    "init_dt": init_dt,
                    "tau_h": tau_h,
                    "valid_dt": valid_dt,
                    "lat": parse_float(row.get("lat")),
                    "lon": parse_float(row.get("lon")),
                    "vmax_kt": parse_float(row.get("vmax_kt")),
                    "mslp_hpa": parse_float(row.get("mslp_hpa")),
                }
            )

    if not raw_rows:
        return None, trace

    selected_init = max(r["init_dt"] for r in raw_rows)
    selected = [r for r in raw_rows if r["init_dt"] == selected_init]
    if not selected:
        return None, trace
    selected.sort(key=lambda x: (x["valid_dt"], x["model"], x["tau_h"]))

    model_track_points: List[Dict[str, Any]] = []
    for r in selected:
        lead_h = int(round((r["valid_dt"] - issue_dt).total_seconds() / 3600))
        model_track_points.append(
            {
                "model_id": r["model"],
                "tau_from_model_init_h": r["tau_h"],
                "lead_from_issue_h": lead_h,
                "valid_time_utc": r["valid_dt"].strftime("%Y-%m-%dT%H:%M:%SZ"),
                "lat": r["lat"],
                "lon": r["lon"],
                "vmax_kt": r["vmax_kt"],
                "mslp_hpa": r["mslp_hpa"],
            }
        )

    spread_groups: Dict[Tuple[int, int, datetime], List[Dict[str, Any]]] = {}
    for r in selected:
        lead_h = int(round((r["valid_dt"] - issue_dt).total_seconds() / 3600))
        spread_groups.setdefault((lead_h, r["tau_h"], r["valid_dt"]), []).append(r)

    spread_points: List[Dict[str, Any]] = []
    for lead_h, tau_h, valid_dt in sorted(spread_groups.keys(), key=lambda x: (x[0], x[1])):
        pts = spread_groups[(lead_h, tau_h, valid_dt)]
        lats = [p["lat"] for p in pts if p.get("lat") is not None]
        lons = [p["lon"] for p in pts if p.get("lon") is not None]
        if not lats or not lons:
            continue

        consensus_lat = sum(lats) / len(lats)
        consensus_lon = circular_mean_lon(lons)
        if consensus_lon is None:
            continue

        vmaxs = [p["vmax_kt"] for p in pts if p.get("vmax_kt") is not None]
        mslps = [p["mslp_hpa"] for p in pts if p.get("mslp_hpa") is not None]
        consensus_vmax = (sum(vmaxs) / len(vmaxs)) if vmaxs else None
        consensus_mslp = (sum(mslps) / len(mslps)) if mslps else None

        dists_km = [
            haversine_km(float(p["lat"]), float(p["lon"]), consensus_lat, consensus_lon)
            for p in pts
            if p.get("lat") is not None and p.get("lon") is not None
        ]
        track_spread = math.sqrt(sum(d * d for d in dists_km) / len(dists_km)) if dists_km else 0.0
        wind_spread = (
            math.sqrt(sum((v - consensus_vmax) ** 2 for v in vmaxs) / len(vmaxs))
            if vmaxs and consensus_vmax is not None
            else 0.0
        )
        models = sorted({str(p.get("model") or "").upper() for p in pts if p.get("model")})
        spread_points.append(
            {
                "lead_from_issue_h": lead_h,
                "tau_from_model_init_h": tau_h,
                "valid_time_utc": valid_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "model_count": len(models),
                "consensus_lat": round(consensus_lat, 3),
                "consensus_lon": round(consensus_lon, 3),
                "consensus_vmax_kt": round(consensus_vmax, 2) if consensus_vmax is not None else None,
                "consensus_mslp_hpa": round(consensus_mslp, 2) if consensus_mslp is not None else None,
                "track_spread_km": round(track_spread, 3),
                "wind_spread_kt": round(wind_spread, 3),
                "models": "|".join(models),
            }
        )

    # Omit raw per-model points: 2000+ rows per sample would dominate the context window.
    # The consensus_spread_points_future aggregates the same information at each lead time.
    block = {
        "status": "available",
        "source_track_file": str(guidance_fp),
        "source_spread_file": str(spread_fp) if spread_fp.exists() else None,
        "selected_model_init_time_utc": selected_init.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "atcf_storm_ids": sorted({r["atcf_storm_id"] for r in selected if r["atcf_storm_id"]}),
        "model_ids": sorted({r["model"] for r in selected}),
        "model_count": len(set(r["model"] for r in selected)),
        "consensus_spread_points_future": spread_points,
    }
    return block, trace


def load_atcf_bdeck_future_series(
    storm_id: str,
    issue_dt: datetime,
    max_horizon_h: int = 120,
) -> Tuple[Optional[Dict[str, Any]], Dict[str, str]]:
    storm_dir = BASE / "data" / "interim" / "atcf" / "by_storm" / storm_id
    bdeck_fp = storm_dir / "b_deck_best_track.csv"
    trace = {"bdeck_file": str(bdeck_fp)}

    if not bdeck_fp.exists():
        return None, trace

    cutoff_dt = issue_dt + timedelta(hours=max_horizon_h)
    rows: List[Dict[str, Any]] = []
    with bdeck_fp.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            valid_time = row.get("valid_time_utc") or ""
            if not valid_time:
                continue
            try:
                valid_dt = parse_iso_utc(valid_time)
            except ValueError:
                continue
            if valid_dt <= issue_dt or valid_dt > cutoff_dt:
                continue
            rows.append(
                {
                    "lead_from_issue_h": int(round((valid_dt - issue_dt).total_seconds() / 3600)),
                    "valid_time_utc": valid_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "lat": parse_float(row.get("lat")),
                    "lon": parse_float(row.get("lon")),
                    "vmax_kt": parse_float(row.get("vmax_kt")),
                    "min_pressure_mb": parse_float(row.get("mslp_hpa")),
                    "storm_phase": row.get("tech"),
                }
            )
    rows.sort(key=lambda x: x["lead_from_issue_h"])
    if not rows:
        return None, trace

    block = {
        "status": "available",
        "source_file": str(bdeck_fp),
        "note": "ATCF B-deck best track, verification only",
        "points_future": rows,
    }
    return block, trace


def load_preferred_verification_future_series(
    storm_id: str,
    issue_dt: datetime,
    max_horizon_h: int = 120,
) -> Tuple[Optional[Dict[str, Any]], Dict[str, str]]:
    storm_dir = BASE / "data" / "interim" / "atcf" / "by_storm" / storm_id
    preferred_fp = storm_dir / "verification_groundtruth_preferred.csv"
    trace = {"preferred_verification_file": str(preferred_fp)}

    if not preferred_fp.exists():
        return None, trace

    cutoff_dt = issue_dt + timedelta(hours=max_horizon_h)
    rows: List[Dict[str, Any]] = []
    source_used_set = set()
    with preferred_fp.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            valid_time = row.get("valid_time_utc") or ""
            if not valid_time:
                continue
            try:
                valid_dt = parse_iso_utc(valid_time)
            except ValueError:
                continue
            if valid_dt <= issue_dt or valid_dt > cutoff_dt:
                continue
            source_used = (row.get("source_used") or "").strip()
            if source_used:
                source_used_set.add(source_used)
            rows.append(
                {
                    "lead_from_issue_h": int(round((valid_dt - issue_dt).total_seconds() / 3600)),
                    "valid_time_utc": valid_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "lat": parse_float(row.get("lat")),
                    "lon": parse_float(row.get("lon")),
                    "vmax_kt": parse_float(row.get("vmax_kt")),
                    "min_pressure_mb": parse_float(row.get("min_pressure_mb")),
                    "storm_phase": row.get("storm_phase"),
                    "source_used": source_used,
                }
            )
    rows.sort(key=lambda x: x["lead_from_issue_h"])
    if not rows:
        return None, trace

    block = {
        "status": "available",
        "policy": "prefer_atcf_b_deck_then_ibtracs_matched_groundtruth",
        "source_file": str(preferred_fp),
        "sources_used_in_points": sorted(source_used_set),
        "points_future": rows,
    }
    return block, trace


def build_sample() -> Dict[str, Any]:
    storm_id = "2020186N30289"
    basin = "Atlantic"
    storm_name = "EDOUARD"
    issue_dt = datetime(2020, 7, 6, 3, 0)
    advisory_no = 7
    hres_init = datetime(2020, 7, 6, 0, 0)

    adv_path = BASE / "noaa" / "2020" / "Atlantic" / "EDOUARD" / "forecast_advisory" / "al052020.fstadv.007.txt"
    dis_path = BASE / "noaa" / "2020" / "Atlantic" / "EDOUARD" / "forecast_discussion" / "al052020.discus.007.txt"
    pub_path = BASE / "noaa" / "2020" / "Atlantic" / "EDOUARD" / "public_advisory" / "al052020.public.007.txt"
    wnd_path = BASE / "noaa" / "2020" / "Atlantic" / "EDOUARD" / "wind_speed_probabilities" / "al052020.wndprb.007.txt"

    adv_lines = read_text(adv_path)
    dis_lines = read_text(dis_path)
    pub_lines = read_text(pub_path)
    wnd_lines = read_text(wnd_path)

    adv_summary = extract_advisory_summary(adv_lines)
    dis_summary = extract_discussion_sections(dis_lines)
    public_summary = extract_public_summary(pub_lines)
    wind_prob_summary = extract_wind_prob_summary(wnd_lines)

    advisory_center = adv_summary.get("center") or {}
    target_lat = parse_float(advisory_center.get("lat"))
    target_lon = parse_float(advisory_center.get("lon"))
    if target_lat is None or target_lon is None:
        raise RuntimeError("Unable to parse advisory center for CDS matching.")

    cds_row, cds_file = pick_cds_row(issue_dt, target_lat=target_lat, target_lon_w=target_lon)
    cds_features = extract_cds_features(cds_row)

    hres_track, hres_track_file = extract_hres_track(storm_id, hres_init)
    hres_env, hres_system_file = extract_hres_system(storm_id, hres_init)

    keep_taus = {0, 12, 24, 36, 48}
    hres_track = [x for x in hres_track if x["tau_h"] in keep_taus]
    hres_env = [x for x in hres_env if x["tau_h"] in keep_taus]

    hres_track_issue = add_issue_lead_fields(hres_track, issue_dt, "tau_from_model_init_h")
    hres_env_issue = add_issue_lead_fields(hres_env, issue_dt, "tau_from_model_init_h")

    pre_issue_track = latest_pre_issue_point(hres_track_issue, issue_dt)
    pre_issue_env = latest_pre_issue_point(hres_env_issue, issue_dt)
    future_track = future_only(hres_track_issue, issue_dt)
    future_env = future_only(hres_env_issue, issue_dt)

    gt_state = load_groundtruth_state(storm_id, issue_dt)
    goes_obs_block, goes_trace = load_goes_observation_structured(storm_id, issue_dt)
    ascat_obs_block, ascat_trace = load_ascat_observation_structured(storm_id, issue_dt)
    recon_obs_block, recon_trace = load_recon_observation_structured(storm_id, issue_dt)
    observation_structured_block = merge_observation_blocks(goes_obs_block, ascat_obs_block, recon_obs_block)
    goes_available = goes_obs_block.get("status") == "available"
    ascat_available = ascat_obs_block.get("status") == "available"
    recon_available = recon_obs_block.get("status") == "available"
    obs_all_available = goes_available and ascat_available and recon_available
    obs_any_available = goes_available or ascat_available or recon_available
    goes_summary_raw, goes_summary_trace = load_goes_coverage_summary()
    goes_coverage_snapshot = build_goes_coverage_snapshot(goes_summary_raw)
    goes_coverage_rate = None
    if goes_coverage_snapshot is not None:
        goes_coverage_rate = goes_coverage_snapshot.get("coverage_rate")
    atcf_guidance_block, atcf_guidance_trace = load_atcf_multimodel_guidance(storm_id, issue_dt)
    atcf_guidance_available = atcf_guidance_block is not None
    if atcf_guidance_block is None:
        atcf_guidance_block = build_multimodel_proxy(future_track)
    guidance_model_ids = set((atcf_guidance_block or {}).get("model_ids") or [])
    atcf_guidance_block = strip_atcf_model_source_tokens(atcf_guidance_block)

    preferred_verify_block, preferred_verify_trace = load_preferred_verification_future_series(storm_id, issue_dt)
    preferred_verify_available = preferred_verify_block is not None

    prompt = {
        "storm_meta": {
            "storm_id": storm_id,
            "storm_name": storm_name,
            "basin": basin,
            "issue_time_utc": issue_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "advisory_no": advisory_no,
            "time_match_rule": "nearest_within_3h",
        },
        "now_inputs": {
            "current_state_from_noaa_forecast_advisory": {
                "center": adv_summary.get("center"),
                "motion": adv_summary.get("motion"),
                "intensity": adv_summary.get("intensity"),
            },
            "environment_now_ec_reanalysis": {
                "source_file": str(cds_file),
                "source_time": str(cds_row.get("time")),
                "tc_position": cds_row.get("tc_position"),
                "features": cds_features,
            },
            "observation_evidence_structured": observation_structured_block,
            "pre_issue_guidance_context": {
                "ec_hres_latest_point_at_or_before_issue_track": pre_issue_track,
                "ec_hres_latest_point_at_or_before_issue_environment": pre_issue_env,
            },
        },
        "guidance_inputs": {
            "guidance_time_reference": {
                "issue_time_utc": issue_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "model_init_time_utc": hres_init.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "tau_reference": "tau_from_model_init_h",
                "lead_reference": "lead_from_issue_h",
                "rule": "future_guidance_blocks_include_only_valid_time_after_issue",
            },
            "ec_single_model_guidance_hres": {
                "model": "HRES",
                "init_time_utc": hres_init.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "source_track_file": str(hres_track_file),
                "source_environment_file": str(hres_system_file),
                "track_intensity_points_future": future_track,
                "environment_points_future": future_env,
            },
            "multimodel_guidance_a_deck": atcf_guidance_block,
        },
    }

    target = {
        "official_outputs": {
            "track_intensity_table": {
                "from_forecast_advisory": adv_summary.get("forecast_table"),
            },
            "risk_messages": {
                "watch_warning_text": adv_summary.get("watch_warning_text"),
                "public_advisory_summary": public_summary,
                "wind_speed_probabilities": wind_prob_summary,
            },
            "reasoning_text": {
                "sections": {
                    "current_analysis_text": dis_summary.get("current_analysis_text"),
                    "forecast_reasoning_text": dis_summary.get("forecast_reasoning_text"),
                    "additional_context_text": dis_summary.get("additional_context_text"),
                },
            },
        }
    }

    if preferred_verify_available and preferred_verify_block is not None:
        future_best_track_series = preferred_verify_block
    else:
        future_best_track_series = {
            "status": "missing_real_data",
            "missing_reason": "Preferred verification file not generated (B-deck primary, IBTrACS fallback).",
            "expected_real_fields": [
                "valid_time_utc",
                "lat",
                "lon",
                "vmax_kt",
                "min_pressure_mb",
                "storm_phase",
            ],
            "proxy_for_schema_preview_only": {
                "official_forecast_reference": adv_summary.get("forecast_table"),
                "note": "official forecast is not truth, proxy only for showing final data shape",
            },
        }

    verification_targets = {
        "policy": "offline_evaluation_or_rft_reward_only_never_in_prompt",
        "groundtruth_source_policy": {
            "preferred_order": ["atcf_b_deck", "ibtracs_matched_groundtruth"],
            "selection_rule": "use storm-level preferred verification table generated by organizer script",
        },
        "best_track_point_near_issue": {
            "status": "available",
            "source_file": "GroundTruth_Cyclones/matched_cyclone_tracks.csv",
            "value": gt_state or None,
            "note": "kept outside prompt to avoid leakage",
        },
        "future_best_track_series": future_best_track_series,
    }

    prompt_blob = json.dumps(prompt, ensure_ascii=False)
    current_text = dis_summary.get("current_analysis_text") or ""
    forecast_text = dis_summary.get("forecast_reasoning_text") or ""
    full_text = dis_summary.get("full_reasoning_text") or ""
    public_lines = public_summary.get("summary_lines") or []
    gt_marker_keys = ["max_wind_wmo", "min_pressure_wmo", "max_wind_usa", "min_pressure_usa", "storm_speed", "storm_direction"]
    guidance_future_ok = all(p.get("lead_from_issue_h", 0) > 0 for p in future_track + future_env)

    leakage_audit = {
        "checks": {
            "full_discussion_text_in_prompt": bool(full_text) and full_text in prompt_blob,
            "forecast_reasoning_text_in_prompt": bool(forecast_text) and forecast_text in prompt_blob,
            "public_advisory_summary_in_prompt": bool(public_lines)
            and all(line in prompt_blob for line in public_lines),
            "groundtruth_verification_fields_in_prompt": any(f'"{k}"' in prompt_blob for k in gt_marker_keys),
            "future_guidance_points_only": guidance_future_ok,
            "current_analysis_text_in_prompt": bool(current_text) and current_text in prompt_blob,
            "ofcl_not_leaked_into_guidance": not bool(guidance_model_ids & ATCF_BLOCKED_GUIDANCE_MODELS),
        },
        "excluded_from_prompt": [
            "current_analysis_text_from_discussion",
            "forecast_reasoning_text_from_discussion",
            "public_advisory_summary_lines",
            "groundtruth_verification_fields",
            "official_track_from_discussion",
        ],
    }

    missing_blocks: List[str] = []
    if not obs_all_available:
        missing_blocks.append("goes_ascat_recon_structured_obs")
    if not atcf_guidance_available:
        missing_blocks.insert(0, "a_deck_multimodel")
    if not preferred_verify_available:
        missing_blocks.insert(1 if not atcf_guidance_available else 0, "b_deck_or_ibtracs")

    sample = {
        "sample_id": f"{basin}_{storm_id}_{issue_dt.strftime('%Y-%m-%dT%H:%M:%SZ')}_{advisory_no:03d}",
        "schema_version": "tc_sft_dataset_v0.1.3_strict",
        "task_type": "tc_forecast_sft",
        "time_window": "2016-2025",
        "build_info": {
            "assembled_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "design_policy": "strict_anti_leak_with_bdeck_primary_ibtracs_fallback_verify",
        },
        "prompt": prompt,
        "target": target,
        "verification_targets": verification_targets,
        "data_gap_plan": {
            "a_deck_multimodel": {
                "status": "available" if atcf_guidance_available else "missing_real_data",
                "priority": "p0",
                "why_it_matters": "required for real consensus and spread features",
                "current_proxy": None if atcf_guidance_available else "ec_single_model_proxy_with_model_count_1",
            },
            "b_deck_or_ibtracs": {
                "status": "available" if preferred_verify_available else "missing_real_data",
                "priority": "p0",
                "why_it_matters": "required for outcome-based evaluation and rft reward",
                "current_proxy": None if preferred_verify_available else "official_forecast_reference_only_not_truth",
                "policy": "prefer_atcf_b_deck_then_ibtracs_matched_groundtruth",
            },
            "goes_ascat_recon_structured_obs": {
                "status": "available" if obs_all_available else ("partial_available" if obs_any_available else "missing_real_data"),
                "priority": "p1",
                "why_it_matters": "improves now-state evidence fidelity beyond text proxy",
                "current_proxy": None if obs_any_available else "strict_placeholder_no_text_proxy",
                "pipeline_coverage_snapshot": goes_coverage_snapshot,
                "next_gap_focus": "prioritize 2016/2017 GOES coverage gaps, then add ASCAT/Recon",
                "component_status": {
                    "goes": goes_obs_block.get("status"),
                    "ascat": ascat_obs_block.get("status"),
                    "recon": recon_obs_block.get("status"),
                },
            },
        },
        "leakage_audit": leakage_audit,
        "quality_flags": {
            "guidance_qc_pass": 1,
            "coverage_flag": "atcf_integrated_when_available",
            "goes_overall_coverage_rate": goes_coverage_rate,
            "goes_status_for_sample": goes_obs_block.get("status"),
            "ascat_status_for_sample": ascat_obs_block.get("status"),
            "recon_status_for_sample": recon_obs_block.get("status"),
            "observation_status_for_sample": observation_structured_block.get("status"),
            "missing_blocks": missing_blocks,
        },
        "source_trace": {
            "forecast_advisory": str(adv_path),
            "forecast_discussion": str(dis_path),
            "public_advisory": str(pub_path),
            "wind_probabilities": str(wnd_path),
            "groundtruth": "GroundTruth_Cyclones/matched_cyclone_tracks.csv",
            "cds_real": str(cds_file),
            "hres_track": str(hres_track_file),
            "hres_system": str(hres_system_file),
            "goes_observation_features": goes_trace.get("goes_feature_file"),
            "goes_observation_summary": goes_summary_trace.get("goes_summary_file"),
            "goes_request_id": goes_trace.get("goes_request_id"),
            "goes_issue_time_selected_utc": goes_trace.get("goes_issue_time_selected_utc"),
            "goes_obs_time_selected_utc": goes_trace.get("goes_obs_time_selected_utc"),
            "ascat_observation_features": ascat_trace.get("ascat_feature_file"),
            "ascat_request_id": ascat_trace.get("ascat_request_id"),
            "ascat_issue_time_selected_utc": ascat_trace.get("ascat_issue_time_selected_utc"),
            "ascat_obs_time_selected_utc": ascat_trace.get("ascat_obs_time_selected_utc"),
            "ascat_dataset_id": ascat_trace.get("ascat_dataset_id"),
            "recon_observation_features": recon_trace.get("recon_feature_file"),
            "recon_request_id": recon_trace.get("recon_request_id"),
            "recon_issue_time_selected_utc": recon_trace.get("recon_issue_time_selected_utc"),
            "recon_obs_time_selected_utc": recon_trace.get("recon_obs_time_selected_utc"),
            "recon_source_file": recon_trace.get("recon_source_file"),
            "atcf_a_deck_guidance": atcf_guidance_trace.get("guidance_file"),
            "atcf_a_deck_spread": atcf_guidance_trace.get("spread_file"),
            "verification_preferred_file": preferred_verify_trace.get("preferred_verification_file"),
        },
    }

    # Extended leakage audit via data_leakage_prevention module (optional)
    try:
        from data_leakage_prevention import InputLeakageAuditor as _InputLeakageAuditor
        _auditor = _InputLeakageAuditor(groundtruth_csv=BASE / "GroundTruth_Cyclones" / "matched_cyclone_tracks.csv")
        _extended_audit = _auditor.audit_sample(sample)
        sample["leakage_audit"]["extended_checks"] = _extended_audit
    except ImportError:
        pass

    return sample


def main() -> int:
    out_dir = BASE / "data" / "interim" / "schema"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "dataset_v0_1_sample_preview_ec_single_source.json"

    sample = build_sample()
    out_file.write_text(json.dumps(sample, ensure_ascii=False, indent=2), encoding="utf-8")

    print(out_file)
    print("sample_id:", sample["sample_id"])
    print("schema_version:", sample["schema_version"])
    print(
        "future_track_points:",
        len(
            sample["prompt"]["guidance_inputs"]["ec_single_model_guidance_hres"][
                "track_intensity_points_future"
            ]
        ),
    )
    print(
        "future_env_points:",
        len(
            sample["prompt"]["guidance_inputs"]["ec_single_model_guidance_hres"][
                "environment_points_future"
            ]
        ),
    )
    print(
        "atcf_multimodel_status:",
        sample["prompt"]["guidance_inputs"]["multimodel_guidance_a_deck"].get("status"),
    )
    print(
        "atcf_future_best_track_status:",
        sample["verification_targets"]["future_best_track_series"].get("status"),
    )
    print("leakage_checks:", json.dumps(sample["leakage_audit"]["checks"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
