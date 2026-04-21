#!/usr/bin/env python3
"""Convert raw dataset samples into compact LLM training formats.

Training-view rules:
  - Prompt remains English only
  - Main SFT view is strict forecast-only, reward-parser aligned
  - Reasoning text is exported as a separate auxiliary SFT view
  - Observations and guidance are compact summaries with explicit DayDD HHMMZ anchors
  - Verification data never enters the prompt
  - Current-analysis reasoning is kept only when structured observations exist
  - Public advisory boilerplate and narrative background stay out of main SFT targets
  - RL export keeps only quality-ready rows with >=4 future best-track points
  - Test-set evaluation variants are regenerated in the same schema as main SFT
"""

from __future__ import annotations

import json
import math
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


FORMAT_VERSION = "train_v0_2_compact"
KEY_LEAD_HOURS = [12, 24, 48, 72, 96, 120]
TRACK_CORRECTION_TARGET_LEADS = (48, 72)
TRACK_CORRECTION_LEAD_TOLERANCE_H = 6.0
SLOT_LOCKED_CORRECTION_MAX_SLOTS = 6
FORECAST_VALID_TIME_LOOKAHEAD_H = 10 * 24
RL_MIN_FUTURE_BT_POINTS = 4
SFT_VIEW_STRICT_FORECAST = "strict_forecast"
SFT_VIEW_REASONING = "reasoning_only"
PRIMARY_ENV_FEATURES = [
    "vertical_wind_shear",
    "upper_level_divergence",
    "ocean_heat_content_or_sst",
    "subtropical_high",
    "westerly_trough",
]

OBS_COMPONENT_ORDER = [
    "goes_structured_obs",
    "ascat_structured_obs",
    "recon_structured_obs",
]
OBS_COMPONENT_LABELS = {
    "goes": "GOES",
    "ascat": "ASCAT",
    "recon": "Recon",
}
OBS_TYPE_LABELS = {
    "goes_ir_structured": "GOES",
    "ascat_surface_wind_structured": "ASCAT",
    "recon_aircraft_structured": "Recon",
}
OBS_SIGNAL_PRIORITY = {
    "goes_ir_structured": [
        "cloud_top_temp_min_k",
        "cloud_top_temp_p10_k",
        "cold_cloud_fraction_inner",
        "cold_cloud_area_inner_km2",
        "eye_ring_temp_contrast_k",
    ],
    "ascat_surface_wind_structured": [
        "max_wind_kt",
        "mean_wind_kt",
        "wind_radius_34kt_km",
    ],
    "recon_aircraft_structured": [
        "message_count",
        "vdm_min_slp_mb",
        "vdm_max_flight_level_wind_kt",
        "hdob_max_flight_level_wind_kt",
    ],
}
OBS_SIGNAL_LABELS = {
    "cloud_top_temp_min_k": "min Tb",
    "cloud_top_temp_p10_k": "p10 Tb",
    "cold_cloud_fraction_inner": "inner cold frac",
    "cold_cloud_area_inner_km2": "inner cold area",
    "eye_ring_temp_contrast_k": "eye-ring contrast",
    "max_wind_kt": "max wind",
    "mean_wind_kt": "mean wind",
    "wind_radius_34kt_km": "34-kt radius",
    "message_count": "messages",
    "vdm_min_slp_mb": "VDM min SLP",
    "vdm_max_flight_level_wind_kt": "VDM FL wind",
    "hdob_max_flight_level_wind_kt": "HDOB FL wind",
}
RISK_BOILERPLATE_PHRASES = (
    "NO COASTAL WATCHES OR WARNINGS IN EFFECT",
)
STORM_IDENTITY_PREFIXES = (
    "Tropical Depression",
    "Tropical Storm",
    "Subtropical Depression",
    "Subtropical Storm",
    "Post-Tropical Cyclone",
    "Potential Tropical Cyclone",
    "Remnants of",
    "Hurricane",
    "Typhoon",
    "Cyclone",
    "Storm",
    "Depression",
)
KNOWN_LEAKAGE_FLAGS = [
    "prompt_contains_cjk",
    "assistant_contains_cjk",
    "prompt_contains_iso_datetime",
    "assistant_contains_iso_datetime",
    "prompt_contains_storm_name",
    "assistant_contains_storm_name",
    "prompt_contains_storm_id",
    "assistant_contains_storm_id",
]
ISO_DATETIME_RE = re.compile(r"\b20\d{2}-\d{2}-\d{2}T\d{2}:\d{2}(?::\d{2})?Z\b")
CJK_RE = re.compile(r"[\u4e00-\u9fff]")
STRICT_FORECAST_LINE_RE = re.compile(
    r"^-\s*Day(?P<day>\d{1,2})\s+"
    r"(?P<hh>\d{2})(?P<mm>\d{2})Z\s*\|\s*"
    r"(?P<lat>\d+(?:\.\d+)?)°(?P<lat_hemi>[NS])\s+"
    r"(?P<lon>\d+(?:\.\d+)?)°(?P<lon_hemi>[EW])\s*\|\s*"
    r"(?P<vmax>\d+(?:\.\d+)?)\s*kt$"
)


# ---------------------------------------------------------------------------
# CDS description compression
# ---------------------------------------------------------------------------

CDS_LEVEL_MAP = {
    "强": "strong",
    "中等": "moderate",
    "弱": "weak",
    "低": "low",
    "高": "high",
    "极强": "very strong",
    "极高": "very high",
    "负值": "negative",
}

CDS_FEATURE_LABELS = {
    "vertical_wind_shear": "Vertical wind shear",
    "upper_level_divergence": "Upper-level divergence",
    "ocean_heat_content_or_sst": "Sea surface temp / OHC",
    "subtropical_high": "Subtropical high",
    "westerly_trough": "Westerly trough",
    "monsoon_trough": "Monsoon trough",
    "low_level_flow": "Low-level flow",
}

CDS_SUMMARY_RULES = {
    "vertical_wind_shear": {
        "极强": "very unfavorable for organization",
        "强": "generally unfavorable for organization",
        "中等": "mixed support for organization",
        "弱": "favorable for organization",
    },
    "upper_level_divergence": {
        "强": "supports upper-level outflow",
        "中等": "provides some outflow support",
        "弱": "limited outflow support",
        "负值": "upper-level convergence, unfavorable for deep convection",
    },
    "ocean_heat_content_or_sst": {
        "低": "limited ocean support",
        "中等": "marginal ocean support",
        "高": "favorable ocean support",
        "极高": "strong ocean support",
    },
    "subtropical_high": {
        "强": "strong steering influence",
        "中等": "moderate steering influence",
        "弱": "weak steering influence",
    },
}

STEERING_LEVEL_ORDER = {
    "weak": 1,
    "moderate": 2,
    "strong": 3,
    "very strong": 4,
    "very_strong": 4,
}


def strip_trailing_zeros(text: str) -> str:
    """Normalize decimal strings like 12.0 -> 12."""
    if "." not in text:
        return text
    return text.rstrip("0").rstrip(".")


def format_number(value: Any, unit: str = "") -> str:
    """Format numeric values compactly for prompts/targets."""
    if value is None:
        return "N/A"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    if not isinstance(value, float):
        return str(value)

    abs_value = abs(value)
    if unit == "ratio":
        digits = 2
    elif abs_value >= 1000:
        digits = 0
    elif abs_value >= 10:
        digits = 1
    else:
        digits = 2
    return strip_trailing_zeros(f"{value:.{digits}f}")


def format_value_with_unit(value: Any, unit: str = "") -> str:
    """Render value with unit while avoiding redundant spaces."""
    value_text = format_number(value, unit)
    return f"{value_text} {unit}".strip()


def normalize_text(text: Any) -> str:
    """Collapse multiline or repeated whitespace text."""
    return " ".join(str(text or "").split())


def parse_utc_datetime(value: Any) -> Optional[datetime]:
    """Parse an ISO-like UTC timestamp into one timezone-aware datetime."""
    text = normalize_text(value)
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def format_day_hhmm_label(day: Any, hhmmz: Any) -> str:
    """Format day-of-month and HHMM into the strict forecast time label."""
    day_text = str(day or "").strip()
    hhmm_text = str(hhmmz or "").strip().upper().removesuffix("Z")
    if not day_text.isdigit() or len(hhmm_text) != 4 or not hhmm_text.isdigit():
        return ""
    return f"Day{int(day_text):02d} {hhmm_text}Z"


def format_datetime_label(value: datetime) -> str:
    """Render one datetime using the strict DayDD HHMMZ label."""
    return f"Day{value.day:02d} {value.strftime('%H%M')}Z"


def shift_month_anchor(value: datetime, month_offset: int) -> datetime:
    """Shift one first-of-month UTC anchor without day overflow issues."""
    month_index = value.year * 12 + (value.month - 1) + month_offset
    year, month_zero_based = divmod(month_index, 12)
    return value.replace(year=year, month=month_zero_based + 1, day=1)


def resolve_valid_datetime_from_day_hhmm(
    issue_time_utc: Any,
    valid_day: Any,
    valid_hhmmz: Any,
) -> Optional[datetime]:
    """Infer one future valid time from DayDD HHMMZ plus the issue timestamp."""
    issue_dt = parse_utc_datetime(issue_time_utc)
    day_text = str(valid_day or "").strip()
    hhmm_text = str(valid_hhmmz or "").strip().upper().removesuffix("Z")
    if issue_dt is None or not day_text.isdigit() or len(hhmm_text) != 4 or not hhmm_text.isdigit():
        return None

    day = int(day_text)
    hour = int(hhmm_text[:2])
    minute = int(hhmm_text[2:])
    issue_dt = issue_dt.astimezone(timezone.utc)
    month_anchor = issue_dt.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

    candidates: List[Tuple[float, datetime]] = []
    for month_offset in (-1, 0, 1):
        candidate_anchor = shift_month_anchor(month_anchor, month_offset)
        try:
            candidate = candidate_anchor.replace(day=day, hour=hour, minute=minute)
        except ValueError:
            continue
        lead_hours = (candidate - issue_dt).total_seconds() / 3600.0
        if 0 <= lead_hours <= FORECAST_VALID_TIME_LOOKAHEAD_H:
            candidates.append((lead_hours, candidate))

    if not candidates:
        return None
    return min(candidates, key=lambda item: item[0])[1]


def resolve_point_valid_datetime(
    point: Dict[str, Any],
    issue_time_utc: Any,
) -> Optional[datetime]:
    """Resolve one guidance or forecast point to a UTC valid time when possible."""
    valid_dt = parse_utc_datetime(point.get("valid_time_utc"))
    if valid_dt is not None:
        return valid_dt

    valid_dt = resolve_valid_datetime_from_day_hhmm(
        issue_time_utc=issue_time_utc,
        valid_day=point.get("valid_day"),
        valid_hhmmz=point.get("valid_hhmmz"),
    )
    if valid_dt is not None:
        return valid_dt

    issue_dt = parse_utc_datetime(issue_time_utc)
    lead = point.get("lead_from_issue_h")
    if issue_dt is not None and isinstance(lead, (int, float)):
        return issue_dt + timedelta(hours=float(lead))
    return None


def resolve_time_label(
    *,
    valid_day: Any = None,
    valid_hhmmz: Any = None,
    valid_time_utc: Any = None,
    issue_time_utc: Any = None,
    lead_from_issue_h: Any = None,
) -> str:
    """Resolve the best available DayDD HHMMZ label for prompt-visible timing."""
    explicit_label = format_day_hhmm_label(valid_day, valid_hhmmz)
    if explicit_label:
        return explicit_label

    valid_dt = parse_utc_datetime(valid_time_utc)
    if valid_dt is not None:
        return format_datetime_label(valid_dt)

    issue_dt = parse_utc_datetime(issue_time_utc)
    if issue_dt is not None and isinstance(lead_from_issue_h, (int, float)):
        resolved_dt = issue_dt + timedelta(hours=float(lead_from_issue_h))
        return format_datetime_label(resolved_dt)

    return ""


def format_coord(lat: Any, lon: Any) -> str:
    """Format coordinates as compass notation."""
    if lat is None or lon is None:
        return "N/A"
    lat_value = float(lat)
    lon_value = float(lon)
    lat_hemi = "N" if lat_value >= 0 else "S"
    lon_hemi = "W" if lon_value <= 0 else "E"
    return f"{abs(lat_value):.1f}°{lat_hemi} {abs(lon_value):.1f}°{lon_hemi}"


def contains_term(text: str, term: str) -> bool:
    """Match a whole-word-ish term in text."""
    if not term:
        return False
    pattern = re.compile(rf"\b{re.escape(term)}\b", re.IGNORECASE)
    return bool(pattern.search(text))


def sanitize_identity_references(text: str, storm_meta: Dict[str, Any]) -> str:
    """Remove storm identity terms from train-view text while preserving structure."""
    cleaned = str(text or "")
    storm_name = normalize_text(storm_meta.get("storm_name"))
    storm_id = normalize_text(storm_meta.get("storm_id"))

    if storm_name:
        for prefix in sorted(STORM_IDENTITY_PREFIXES, key=len, reverse=True):
            cleaned = re.sub(
                rf"\b{re.escape(prefix)}\s+{re.escape(storm_name)}\b",
                "the cyclone",
                cleaned,
                flags=re.IGNORECASE,
            )
        cleaned = re.sub(
            rf"\b{re.escape(storm_name)}'s\b",
            "the cyclone's",
            cleaned,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(
            rf"\b{re.escape(storm_name)}\b",
            "the cyclone",
            cleaned,
            flags=re.IGNORECASE,
        )

    if storm_id:
        cleaned = re.sub(
            rf"\b{re.escape(storm_id)}\b",
            "the cyclone",
            cleaned,
            flags=re.IGNORECASE,
        )

    return cleaned


def read_json_with_retry(
    path: Path, attempts: int = 5, delay_sec: float = 0.2
) -> Dict[str, Any]:
    """Read one JSON file, tolerating short concurrent-write windows."""
    last_error: Optional[Exception] = None
    for attempt in range(attempts):
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            last_error = exc
            if attempt == attempts - 1:
                raise
            time.sleep(delay_sec)
    if last_error is not None:
        raise last_error
    raise RuntimeError(f"Failed to read {path}")


def component_short_name(component_key: str) -> str:
    """Convert goes_structured_obs -> goes."""
    return component_key.replace("_structured_obs", "")


def get_available_observation_components(obs: Dict[str, Any]) -> List[str]:
    """Return short observation component names that are available."""
    component_status = obs.get("component_status") or {}
    available: List[str] = []
    for component_key in OBS_COMPONENT_ORDER:
        if component_status.get(component_key) == "available":
            available.append(component_short_name(component_key))
    return available


def classify_observation_support(obs: Dict[str, Any]) -> str:
    """Bucket structured observation support strength."""
    available = get_available_observation_components(obs)
    if len(available) >= 2:
        return "multi_source"
    if len(available) == 1:
        return "single_source"
    return "none"


def compress_cds_description(feature_key: str, feature: Dict[str, Any]) -> str:
    """Compress a CDS feature into a compact English line."""
    if not feature:
        return f"{feature_key}: no data"

    label = CDS_FEATURE_LABELS.get(feature_key, feature_key.replace("_", " ").title())
    value = feature.get("value")
    unit = feature.get("unit", "")
    level = feature.get("level", "")
    level_en = CDS_LEVEL_MAP.get(level, level)
    summary = CDS_SUMMARY_RULES.get(feature_key, {}).get(level, "")

    text = f"{label}: {format_value_with_unit(value, unit)}"
    if level_en:
        text += f" ({level_en})"
    if summary:
        text += f" — {summary}"
    return text


def _feature_level_slug(feature: Dict[str, Any]) -> str:
    level = normalize_text((feature or {}).get("level")).lower()
    return CDS_LEVEL_MAP.get(level, level).replace(" ", "_")


def _feature_level_text(feature: Dict[str, Any]) -> str:
    slug = _feature_level_slug(feature)
    if not slug:
        return "not available"
    return slug.replace("_", " ")


def _steering_level_rank(feature: Dict[str, Any]) -> int:
    return STEERING_LEVEL_ORDER.get(_feature_level_text(feature), 0)


# ---------------------------------------------------------------------------
# Observation formatting
# ---------------------------------------------------------------------------

def select_observation_items(
    obs_type: str, items: List[Dict[str, Any]], max_items: int = 3
) -> List[Dict[str, Any]]:
    """Pick the highest-value observation signals for each source."""
    item_map: Dict[str, Dict[str, Any]] = {}
    for item in items:
        signal = item.get("signal")
        if signal and item.get("value") is not None and signal not in item_map:
            item_map[signal] = item

    selected: List[Dict[str, Any]] = []
    for signal in OBS_SIGNAL_PRIORITY.get(obs_type, []):
        if signal in item_map:
            selected.append(item_map[signal])
        if len(selected) >= max_items:
            return selected

    for signal in sorted(item_map):
        if item_map[signal] not in selected:
            selected.append(item_map[signal])
        if len(selected) >= max_items:
            break
    return selected


def format_observation_item(item: Dict[str, Any]) -> str:
    """Format one compact observation signal summary."""
    signal = item.get("signal", "unknown")
    label = OBS_SIGNAL_LABELS.get(signal, signal.replace("_", " "))
    value_text = format_value_with_unit(item.get("value"), item.get("unit", ""))
    return f"{label} {value_text}"


def format_observations(obs: Dict[str, Any]) -> str:
    """Format structured observation evidence into compact text."""
    component_status = obs.get("component_status") or {}
    available_components = get_available_observation_components(obs)
    missing_components = [
        OBS_COMPONENT_LABELS[component_short_name(component_key)]
        for component_key in OBS_COMPONENT_ORDER
        if component_status.get(component_key) != "available"
    ]

    if not available_components:
        missing_text = ", ".join(missing_components or OBS_COMPONENT_LABELS.values())
        return f"- Coverage: none available; missing {missing_text}"

    coverage_text = " + ".join(OBS_COMPONENT_LABELS[name] for name in available_components)
    lines = [f"- Coverage: {coverage_text} available"]
    if missing_components:
        lines[0] += f"; missing {', '.join(missing_components)}"

    by_type: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for item in obs.get("value", []) or []:
        by_type[item.get("obs_type", "unknown")].append(item)

    for obs_type in [
        "goes_ir_structured",
        "ascat_surface_wind_structured",
        "recon_aircraft_structured",
    ]:
        selected = select_observation_items(obs_type, by_type.get(obs_type, []))
        if not selected:
            continue
        item_text = "; ".join(format_observation_item(item) for item in selected)
        lines.append(f"- {OBS_TYPE_LABELS.get(obs_type, obs_type)}: {item_text}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Track-turn analysis
# ---------------------------------------------------------------------------

def _safe_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_longitude_delta(delta_lon: float) -> float:
    while delta_lon > 180.0:
        delta_lon -= 360.0
    while delta_lon < -180.0:
        delta_lon += 360.0
    return delta_lon


def _movement_label(dlat: float, dlon: float) -> str:
    lat_dir = "north" if dlat > 0.2 else ("south" if dlat < -0.2 else "")
    lon_dir = "east" if dlon > 0.2 else ("west" if dlon < -0.2 else "")
    if lat_dir and lon_dir:
        return f"{lat_dir}-{lon_dir}"
    return lat_dir or lon_dir or "steady"


def _heading_degrees(dlat: float, dlon: float) -> float:
    heading = math.degrees(math.atan2(dlat, dlon))
    return heading if heading >= 0.0 else heading + 360.0


def _heading_change_degrees(left: float, right: float) -> float:
    delta = abs(left - right) % 360.0
    return min(delta, 360.0 - delta)


def analyze_track_turn(
    points: List[Dict[str, Any]],
    *,
    lat_key: str,
    lon_key: str,
) -> Optional[Dict[str, Any]]:
    valid_points: List[tuple[float, float]] = []
    for point in points or []:
        lat = _safe_float(point.get(lat_key))
        lon = _safe_float(point.get(lon_key))
        if lat is None or lon is None:
            continue
        valid_points.append((lat, lon))

    if len(valid_points) < 3:
        return None

    segments: List[Dict[str, float | str]] = []
    for (lat_a, lon_a), (lat_b, lon_b) in zip(valid_points, valid_points[1:]):
        dlat = lat_b - lat_a
        dlon = _normalize_longitude_delta(lon_b - lon_a)
        if abs(dlat) < 0.05 and abs(dlon) < 0.05:
            continue
        segments.append(
            {
                "dlat": dlat,
                "dlon": dlon,
                "heading_deg": _heading_degrees(dlat, dlon),
                "label": _movement_label(dlat, dlon),
            }
        )

    if len(segments) < 2:
        return None

    window = min(2, len(segments))

    def averaged_motion(segment_slice: List[Dict[str, float | str]]) -> tuple[float, float]:
        mean_dlat = sum(float(segment["dlat"]) for segment in segment_slice) / len(segment_slice)
        mean_dlon = sum(float(segment["dlon"]) for segment in segment_slice) / len(segment_slice)
        return mean_dlat, mean_dlon

    early_dlat, early_dlon = averaged_motion(segments[:window])
    late_dlat, late_dlon = averaged_motion(segments[-window:])
    early_heading = _heading_degrees(early_dlat, early_dlon)
    late_heading = _heading_degrees(late_dlat, late_dlon)
    heading_change_deg = _heading_change_degrees(early_heading, late_heading)
    max_segment_heading_change_deg = max(
        _heading_change_degrees(
            float(left["heading_deg"]),
            float(right["heading_deg"]),
        )
        for left, right in zip(segments, segments[1:])
    )

    mean_latitude = sum(lat for lat, _lon in valid_points) / len(valid_points)
    poleward_sign = 1.0 if mean_latitude >= 0.0 else -1.0
    poleward_late = late_dlat * poleward_sign >= 0.2
    zonal_reversal = early_dlon <= -0.25 and late_dlon >= 0.15
    meridional_reorientation = abs(late_dlat - early_dlat) >= 0.45

    signal = "steady"
    if zonal_reversal and poleward_late and (
        heading_change_deg >= 45.0 or max_segment_heading_change_deg >= 50.0
    ):
        signal = "recurvature"
    elif (
        heading_change_deg >= 35.0
        or max_segment_heading_change_deg >= 45.0
        or meridional_reorientation
    ):
        signal = "notable_turn"

    return {
        "signal": signal,
        "early_label": _movement_label(early_dlat, early_dlon),
        "late_label": _movement_label(late_dlat, late_dlon),
        "heading_change_deg": heading_change_deg,
        "max_segment_heading_change_deg": max_segment_heading_change_deg,
        "segment_count": len(segments),
    }


def classify_turning_signal_from_points(
    points: List[Dict[str, Any]],
    *,
    lat_key: str,
    lon_key: str,
) -> Optional[str]:
    analysis = analyze_track_turn(points, lat_key=lat_key, lon_key=lon_key)
    if analysis is None:
        return None
    return str(analysis["signal"])


def _turn_magnitude_bucket_from_analysis(analysis: Dict[str, Any]) -> str:
    magnitude = max(
        float(analysis["heading_change_deg"]),
        float(analysis["max_segment_heading_change_deg"]),
    )
    if magnitude < 20.0:
        return "none"
    if magnitude < 45.0:
        return "modest"
    if magnitude < 75.0:
        return "strong"
    return "sharp"


def _turn_direction_family_from_analysis(analysis: Dict[str, Any]) -> str:
    early_label = str(analysis["early_label"])
    late_label = str(analysis["late_label"])
    magnitude = max(
        float(analysis["heading_change_deg"]),
        float(analysis["max_segment_heading_change_deg"]),
    )
    if str(analysis["signal"]) == "steady" or magnitude < 20.0:
        return "hold"
    if "east" in late_label and "east" not in early_label:
        return "eastward_escape"
    if "west" in late_label and "west" not in early_label:
        return "westward_or_equatorward_bend"
    if "south" in late_label and "south" not in early_label:
        return "westward_or_equatorward_bend"
    return "poleward_turn"


def _steering_regime_phase_from_analysis(analysis: Dict[str, Any]) -> str:
    if str(analysis["signal"]) == "recurvature":
        return "recurving"
    if _turn_magnitude_bucket_from_analysis(analysis) != "none":
        return "transition"
    return "locked"


def _turn_timing_bucket_from_points(
    points: List[Dict[str, Any]],
    *,
    lat_key: str,
    lon_key: str,
    lead_key: str,
) -> Optional[str]:
    valid_points: List[Tuple[float, float, float]] = []
    for point in points or []:
        lead_h = _safe_float(point.get(lead_key))
        lat = _safe_float(point.get(lat_key))
        lon = _safe_float(point.get(lon_key))
        if lead_h is None or lat is None or lon is None:
            continue
        valid_points.append((lead_h, lat, lon))
    valid_points.sort(key=lambda item: item[0])
    if len(valid_points) < 3:
        return None

    segments: List[Tuple[float, float]] = []
    for (lead_a, lat_a, lon_a), (lead_b, lat_b, lon_b) in zip(valid_points, valid_points[1:]):
        del lead_a
        dlat = lat_b - lat_a
        dlon = _normalize_longitude_delta(lon_b - lon_a)
        if abs(dlat) < 0.05 and abs(dlon) < 0.05:
            continue
        segments.append((lead_b, _heading_degrees(dlat, dlon)))
    if len(segments) < 2:
        return None

    baseline_heading = segments[0][1]
    first_turn_lead_h: float | None = None
    for lead_h, heading in segments[1:]:
        if _heading_change_degrees(baseline_heading, heading) >= 25.0:
            first_turn_lead_h = lead_h
            break
    if first_turn_lead_h is None:
        return "none_or_locked"
    if first_turn_lead_h <= 48.0:
        return "24_to_48h"
    return "48_to_72h"


def analyze_track_inflection(
    points: List[Dict[str, Any]],
    *,
    lat_key: str,
    lon_key: str,
    lead_key: str | None = None,
) -> Optional[Dict[str, Any]]:
    analysis = analyze_track_turn(points, lat_key=lat_key, lon_key=lon_key)
    if analysis is None:
        return None

    turn_timing_bucket = (
        _turn_timing_bucket_from_points(
            points,
            lat_key=lat_key,
            lon_key=lon_key,
            lead_key=lead_key,
        )
        if lead_key
        else None
    )
    return {
        **analysis,
        "steering_regime_phase": _steering_regime_phase_from_analysis(analysis),
        "turn_timing_bucket": turn_timing_bucket,
        "turn_direction_family": _turn_direction_family_from_analysis(analysis),
        "turn_magnitude_bucket": _turn_magnitude_bucket_from_analysis(analysis),
    }


def _format_turning_cue_line(
    source_label: str,
    analysis: Dict[str, Any],
    *,
    spread_start_km: float | None = None,
    spread_end_km: float | None = None,
) -> str:
    signal_text = str(analysis["signal"]).replace("_", " ")
    parts = [
        f"- {source_label}: early {analysis['early_label']}, late {analysis['late_label']}",
        f"heading change {int(round(float(analysis['heading_change_deg'])))} deg",
        f"{signal_text} signal",
    ]
    if spread_start_km is not None and spread_end_km is not None:
        spread_delta = spread_end_km - spread_start_km
        if abs(spread_delta) >= 100.0:
            trend = "widens" if spread_delta > 0.0 else "tightens"
            parts.append(
                f"spread {trend} {format_number(spread_start_km)}->{format_number(spread_end_km)} km"
            )
    return "; ".join(parts)


def _guidance_regime_text(
    source_label: str,
    analysis: Dict[str, Any] | None,
) -> str | None:
    if analysis is None:
        return None
    signal_text = str(analysis["signal"]).replace("_", " ")
    return (
        f"{source_label} shows {analysis['early_label']} early and {analysis['late_label']} late "
        f"({signal_text})"
    )


def format_steering_competition_cues(
    env_features: Dict[str, Any],
    atcf: Dict[str, Any],
    hres: Dict[str, Any],
    *,
    issue_time_utc: Any,
    target_leads: Optional[List[int]] = None,
) -> str:
    del issue_time_utc
    subtropical_high = env_features.get("subtropical_high") or {}
    westerly_trough = env_features.get("westerly_trough") or {}
    monsoon_trough = env_features.get("monsoon_trough") or {}

    ridge_rank = _steering_level_rank(subtropical_high)
    trough_rank = max(_steering_level_rank(westerly_trough), _steering_level_rank(monsoon_trough))
    ridge_text = _feature_level_text(subtropical_high)
    trough_text = _feature_level_text(westerly_trough)
    monsoon_text = _feature_level_text(monsoon_trough)

    lines = [
        (
            f"- Steering environment: subtropical high {ridge_text}; "
            f"westerly trough {trough_text}; monsoon trough {monsoon_text}"
        )
    ]
    if ridge_rank >= 2 and trough_rank >= 2:
        lines.append(
            "- Steering balance: ridge and trough are both material, so a later poleward or eastward bend would favor a competition regime rather than pure ridge control"
        )
    elif ridge_rank > trough_rank:
        lines.append(
            "- Steering balance: the ridge signal is stronger than the trough signal, so a mainly westward or northwestward track would fit ridge-dominant control"
        )
    elif trough_rank > ridge_rank:
        lines.append(
            "- Steering balance: the trough signal is stronger than the ridge signal, so poleward pickup or eastward escape would fit trough-dominant control"
        )
    else:
        lines.append(
            "- Steering balance: neither ridge nor trough is clearly dominant from the environment alone, so rely on guidance bends to judge whether control stays ridge-led or becomes mixed"
        )

    atcf_points = select_representative_guidance_points(
        atcf.get("consensus_spread_points_future", []) or [],
        target_leads=target_leads,
    )
    hres_points = select_representative_guidance_points(
        hres.get("track_intensity_points_future", []) or [],
        target_leads=target_leads,
    )
    regime_bits = [
        text
        for text in [
            _guidance_regime_text(
                "ATCF",
                analyze_track_turn(atcf_points, lat_key="consensus_lat", lon_key="consensus_lon"),
            ),
            _guidance_regime_text(
                "HRES",
                analyze_track_turn(hres_points, lat_key="lat", lon_key="lon"),
            ),
        ]
        if text
    ]
    if regime_bits:
        lines.append("- Guidance regime cue: " + " ; ".join(regime_bits))

    return "\n".join(lines)


def _format_track_inflection_cue_line(
    source_label: str,
    analysis: Dict[str, Any],
    *,
    spread_start_km: float | None = None,
    spread_end_km: float | None = None,
) -> str:
    parts = [
        f"- {source_label}: regime {str(analysis['steering_regime_phase']).replace('_', ' ')}",
        f"turn window {str(analysis.get('turn_timing_bucket') or 'unknown').replace('_', ' ')}",
        f"direction {str(analysis['turn_direction_family']).replace('_', ' ')}",
        f"magnitude {str(analysis['turn_magnitude_bucket']).replace('_', ' ')}",
        f"early {analysis['early_label']}",
        f"late {analysis['late_label']}",
    ]
    if spread_start_km is not None and spread_end_km is not None:
        spread_delta = spread_end_km - spread_start_km
        if abs(spread_delta) >= 100.0:
            trend = "widens" if spread_delta > 0.0 else "tightens"
            parts.append(
                f"spread {trend} {format_number(spread_start_km)}->{format_number(spread_end_km)} km"
            )
    return "; ".join(parts)


def format_track_inflection_guidance_cues(
    atcf: Dict[str, Any],
    hres: Dict[str, Any],
    *,
    issue_time_utc: Any,
    target_leads: Optional[List[int]] = None,
) -> str:
    del issue_time_utc
    lines: List[str] = []

    atcf_points = select_representative_guidance_points(
        atcf.get("consensus_spread_points_future", []) or [],
        target_leads=target_leads,
    )
    atcf_analysis = analyze_track_inflection(
        atcf_points,
        lat_key="consensus_lat",
        lon_key="consensus_lon",
        lead_key="lead_from_issue_h",
    )
    if atcf_analysis is not None:
        track_spreads = [
            _safe_float(point.get("track_spread_km"))
            for point in atcf_points
            if _safe_float(point.get("track_spread_km")) is not None
        ]
        spread_start = track_spreads[0] if track_spreads else None
        spread_end = track_spreads[-1] if track_spreads else None
        lines.append(
            _format_track_inflection_cue_line(
                "ATCF",
                atcf_analysis,
                spread_start_km=spread_start,
                spread_end_km=spread_end,
            )
        )

    hres_points = select_representative_guidance_points(
        hres.get("track_intensity_points_future", []) or [],
        target_leads=target_leads,
    )
    hres_analysis = analyze_track_inflection(
        hres_points,
        lat_key="lat",
        lon_key="lon",
        lead_key="lead_from_issue_h",
    )
    if hres_analysis is not None:
        lines.append(_format_track_inflection_cue_line("HRES", hres_analysis))

    if atcf_analysis is not None and hres_analysis is not None:
        if (
            atcf_analysis["turn_direction_family"] == hres_analysis["turn_direction_family"]
            and atcf_analysis.get("turn_timing_bucket") == hres_analysis.get("turn_timing_bucket")
        ):
            lines.append(
                "- Cross-model cue: ATCF and HRES align on both turn direction family and turn window"
            )
        elif atcf_analysis["turn_direction_family"] == hres_analysis["turn_direction_family"]:
            lines.append(
                "- Cross-model cue: ATCF and HRES agree on direction family but differ on turn timing"
            )
        else:
            lines.append(
                "- Cross-model cue: ATCF and HRES disagree on turn direction family, so treat inflection timing as lower-confidence"
            )

    return "\n".join(lines)


def format_turning_guidance_cues(
    atcf: Dict[str, Any],
    hres: Dict[str, Any],
    *,
    issue_time_utc: Any,
    target_leads: Optional[List[int]] = None,
) -> str:
    lines: List[str] = []

    atcf_points = select_representative_guidance_points(
        atcf.get("consensus_spread_points_future", []) or [],
        target_leads=target_leads,
    )
    atcf_analysis = analyze_track_turn(atcf_points, lat_key="consensus_lat", lon_key="consensus_lon")
    if atcf_analysis is not None:
        track_spreads = [
            _safe_float(point.get("track_spread_km"))
            for point in atcf_points
            if _safe_float(point.get("track_spread_km")) is not None
        ]
        spread_start = track_spreads[0] if track_spreads else None
        spread_end = track_spreads[-1] if track_spreads else None
        lines.append(
            _format_turning_cue_line(
                "ATCF",
                atcf_analysis,
                spread_start_km=spread_start,
                spread_end_km=spread_end,
            )
        )

    hres_points = select_representative_guidance_points(
        hres.get("track_intensity_points_future", []) or [],
        target_leads=target_leads,
    )
    hres_analysis = analyze_track_turn(hres_points, lat_key="lat", lon_key="lon")
    if hres_analysis is not None:
        lines.append(_format_turning_cue_line("HRES", hres_analysis))

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Guidance formatting
# ---------------------------------------------------------------------------

def select_representative_guidance_points(
    points: List[Dict[str, Any]], target_leads: Optional[List[int]] = None
) -> List[Dict[str, Any]]:
    """Select guidance points aligned to target forecast leads when available."""
    valid: List[Tuple[int, float, Dict[str, Any]]] = []
    for idx, point in enumerate(points or []):
        lead = point.get("lead_from_issue_h")
        if isinstance(lead, (int, float)):
            valid.append((idx, float(lead), point))
    valid.sort(key=lambda item: item[1])

    if not valid:
        return []

    selected: List[Dict[str, Any]] = []
    used_indices = set()
    last_selected_lead = float("-inf")
    for target_lead in target_leads or KEY_LEAD_HOURS:
        candidates = [
            (
                abs(lead - target_lead),
                0 if lead >= target_lead else 1,
                -float(point.get("model_count") or 0),
                lead,
                idx,
                point,
            )
            for idx, lead, point in valid
            if idx not in used_indices and lead > last_selected_lead
        ]
        if not candidates:
            break
        _, _, _, lead, idx, point = min(candidates)
        used_indices.add(idx)
        selected.append(point)
        last_selected_lead = lead

    if not selected:
        return [point for _, _, point in valid[: len(KEY_LEAD_HOURS)]]

    selected.sort(key=lambda point: point.get("lead_from_issue_h") or 0)
    return selected


def select_nearest_lead_point(
    points: List[Dict[str, Any]],
    *,
    target_lead_h: float,
    lead_key: str = "lead_from_issue_h",
    tolerance_h: float = TRACK_CORRECTION_LEAD_TOLERANCE_H,
) -> Optional[Dict[str, Any]]:
    """Return the closest point to one target lead within a fixed tolerance."""
    candidates: List[Tuple[float, int, float, int, Dict[str, Any]]] = []
    for idx, point in enumerate(points or []):
        lead_h = _safe_float(point.get(lead_key))
        if lead_h is None:
            continue
        delta_h = abs(lead_h - float(target_lead_h))
        if delta_h > float(tolerance_h):
            continue
        candidates.append(
            (
                delta_h,
                0 if lead_h <= float(target_lead_h) else 1,
                lead_h,
                idx,
                point,
            )
        )
    if not candidates:
        return None
    return min(candidates)[-1]


def derive_guidance_target_leads(
    forecast_table: List[Dict[str, Any]],
    issue_time_utc: Any,
) -> List[int]:
    """Convert official forecast slots into guidance-selection target leads."""
    issue_dt = parse_utc_datetime(issue_time_utc)
    leads: List[int] = []
    seen = set()
    for point in forecast_table or []:
        resolved_valid_dt = resolve_point_valid_datetime(point, issue_time_utc=issue_time_utc)
        if resolved_valid_dt is not None and issue_dt is not None:
            lead_hours = int(round((resolved_valid_dt - issue_dt).total_seconds() / 3600.0))
        else:
            lead = point.get("lead_from_issue_h")
            if not isinstance(lead, (int, float)):
                continue
            lead_hours = int(round(float(lead)))

        if lead_hours <= 0 or lead_hours in seen:
            continue
        seen.add(lead_hours)
        leads.append(lead_hours)

    return leads or list(KEY_LEAD_HOURS)


def format_hres_guidance(
    hres: Dict[str, Any],
    issue_time_utc: Any,
    target_leads: Optional[List[int]] = None,
) -> str:
    """Format deterministic HRES guidance aligned to official forecast slots."""
    points = select_representative_guidance_points(
        hres.get("track_intensity_points_future", []) or [],
        target_leads=target_leads,
    )
    if not points:
        return ""

    lines = ["Deterministic HRES:"]
    for point in points:
        time_label = resolve_time_label(
            valid_day=point.get("valid_day"),
            valid_hhmmz=point.get("valid_hhmmz"),
            valid_time_utc=point.get("valid_time_utc"),
            issue_time_utc=issue_time_utc,
            lead_from_issue_h=point.get("lead_from_issue_h"),
        )
        if not time_label:
            continue
        coord = format_coord(point.get("lat"), point.get("lon"))
        wind = format_value_with_unit(point.get("wind_kt"), "kt")
        mslp = format_value_with_unit(point.get("mslp_hpa"), "hPa")
        lines.append(f"- {time_label} {coord} | {wind}, {mslp}")
    return "\n".join(lines) if len(lines) > 1 else ""


def format_atcf_guidance(
    atcf: Dict[str, Any],
    issue_time_utc: Any,
    target_leads: Optional[List[int]] = None,
) -> str:
    """Format ATCF guidance using official-slot-aligned representative lead times."""
    if atcf.get("status") != "available":
        return ""

    points = select_representative_guidance_points(
        atcf.get("consensus_spread_points_future", []) or [],
        target_leads=target_leads,
    )
    if not points:
        return ""

    lines = ["ATCF guidance:"]
    for point in points:
        time_label = resolve_time_label(
            valid_day=point.get("valid_day"),
            valid_hhmmz=point.get("valid_hhmmz"),
            valid_time_utc=point.get("valid_time_utc"),
            issue_time_utc=issue_time_utc,
            lead_from_issue_h=point.get("lead_from_issue_h"),
        )
        if not time_label:
            continue
        coord = format_coord(point.get("consensus_lat"), point.get("consensus_lon"))
        vmax = format_value_with_unit(point.get("consensus_vmax_kt"), "kt")
        mslp = format_value_with_unit(point.get("consensus_mslp_hpa"), "hPa")
        track_spread = format_value_with_unit(point.get("track_spread_km"), "km")
        wind_spread = format_value_with_unit(point.get("wind_spread_kt"), "kt")
        model_count = point.get("model_count")
        model_count_text = ""
        if isinstance(model_count, (int, float)):
            count_value = int(round(float(model_count)))
            model_label = "model" if count_value == 1 else "models"
            model_count_text = f" | {count_value} {model_label}"
        lines.append(
            f"- {time_label} {coord} | {vmax}, {mslp} | spread {track_spread}/{wind_spread}{model_count_text}"
        )
    return "\n".join(lines) if len(lines) > 1 else ""


def _format_track_correction_anchor_line(
    target_lead_h: int,
    point: Dict[str, Any],
    *,
    issue_time_utc: Any,
) -> str:
    time_label = resolve_time_label(
        valid_day=point.get("valid_day"),
        valid_hhmmz=point.get("valid_hhmmz"),
        valid_time_utc=point.get("valid_time_utc"),
        issue_time_utc=issue_time_utc,
        lead_from_issue_h=point.get("lead_from_issue_h"),
    )
    lead_h = _safe_float(point.get("lead_from_issue_h"))
    lead_text = f"lead {int(round(lead_h))}h" if lead_h is not None else "lead unknown"
    coord = format_coord(point.get("consensus_lat"), point.get("consensus_lon"))

    parts = [f"- {target_lead_h}h anchor: nearest ATCF point"]
    if time_label:
        parts.append(time_label)
    parts.append(f"({lead_text})")
    parts.append(coord)

    spread_km = _safe_float(point.get("track_spread_km"))
    if spread_km is not None:
        parts.append(f"track spread {format_number(spread_km)} km")

    model_count = point.get("model_count")
    if isinstance(model_count, (int, float)):
        count_value = int(round(float(model_count)))
        model_label = "model" if count_value == 1 else "models"
        parts.append(f"{count_value} {model_label}")
    return ", ".join(parts)


def format_track_correction_guidance_cues(
    atcf: Dict[str, Any],
    *,
    issue_time_utc: Any,
) -> str:
    """Surface the fixed-lead ATCF anchors used by the track-correction schema."""
    if atcf.get("status") != "available":
        return ""

    points = atcf.get("consensus_spread_points_future", []) or []
    if not points:
        return ""

    lines: List[str] = [
        (
            "- Reference frame: interpret the 48h/72h correction labels relative to the ATCF "
            "consensus track already shown above, using the nearest visible anchors below."
        )
    ]
    anchor_count = 0
    for target_lead_h in TRACK_CORRECTION_TARGET_LEADS:
        point = select_nearest_lead_point(points, target_lead_h=target_lead_h)
        if point is None:
            lines.append(
                f"- {target_lead_h}h anchor: no ATCF consensus point is available within "
                f"+/-{int(TRACK_CORRECTION_LEAD_TOLERANCE_H)}h."
            )
            continue
        lines.append(
            _format_track_correction_anchor_line(
                target_lead_h,
                point,
                issue_time_utc=issue_time_utc,
            )
        )
        anchor_count += 1

    if anchor_count == 0:
        return ""

    lines.append(
        "- These anchors are track-only reference points. They do not imply any forecast timing shift or intensity change."
    )
    return "\n".join(lines)


def format_slot_locked_correction_guidance_cues(
    atcf: Dict[str, Any],
    *,
    issue_time_utc: Any,
    official_track_table: List[Dict[str, Any]],
) -> str:
    """Map visible ATCF representative points onto fixed output slots for correction tasks."""
    if atcf.get("status") != "available":
        return ""

    target_leads = derive_guidance_target_leads(official_track_table, issue_time_utc)
    points = select_representative_guidance_points(
        atcf.get("consensus_spread_points_future", []) or [],
        target_leads=target_leads,
    )
    if not points:
        return ""

    lines: List[str] = [
        (
            "- Use the ATCF guidance block above as the base trajectory. For each slot field, "
            "predict only the lat/lon correction bucket relative to the matching ATCF point."
        ),
        "- Keep slot count and slot order fixed. Do not invent new slots, drop slots, or shift Day/HHMMZ labels.",
        "- These correction fields modify track position only; they do not modify intensity.",
        "- Within each slot, judge lat and lon separately.",
        "- For *_lat_* fields, focus on north/south displacement relative to the ATCF point; east/west agreement alone is not evidence for near_consensus lat.",
        "- For *_lon_* fields, focus on east/west displacement relative to the ATCF point.",
    ]
    for slot_index, point in enumerate(points[:SLOT_LOCKED_CORRECTION_MAX_SLOTS], start=1):
        time_label = resolve_time_label(
            valid_day=point.get("valid_day"),
            valid_hhmmz=point.get("valid_hhmmz"),
            valid_time_utc=point.get("valid_time_utc"),
            issue_time_utc=issue_time_utc,
            lead_from_issue_h=point.get("lead_from_issue_h"),
        )
        lead_h = _safe_float(point.get("lead_from_issue_h"))
        lead_text = f"lead {int(round(lead_h))}h" if lead_h is not None else "lead unknown"
        coord = format_coord(point.get("consensus_lat"), point.get("consensus_lon"))
        spread_km = _safe_float(point.get("track_spread_km"))
        spread_text = (
            f"track spread {format_number(spread_km)} km"
            if spread_km is not None
            else "track spread unknown"
        )
        lines.append(
            f"- slot_{slot_index}: {time_label} | {coord} | {lead_text} | {spread_text}"
        )
    lines.append(
        "- If a later slot field has no matching ATCF point above, leave that field null."
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Target formatting
# ---------------------------------------------------------------------------

def format_track_table(forecast_table: List[Dict[str, Any]]) -> str:
    """Format the official track/intensity table."""
    if not forecast_table:
        return "Official forecast:\n- No official track forecast"

    lines = ["Official forecast:"]
    for point in forecast_table:
        coord = format_coord(point.get("lat"), point.get("lon"))
        day = point.get("valid_day", 0)
        hhmm = point.get("valid_hhmmz", "")
        vmax = format_value_with_unit(point.get("vmax_kt"), "kt")
        lines.append(f"- Day{int(day):02d} {hhmm}Z | {coord} | {vmax}")
    return "\n".join(lines)


def build_reward_forecast_slots(forecast_table: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Keep only the strict forecast slot times needed by the RL reward."""
    slots: List[Dict[str, Any]] = []
    for point in forecast_table:
        day = point.get("valid_day")
        hhmm = str(point.get("valid_hhmmz") or "").strip().upper().removesuffix("Z")
        if day in (None, "") or len(hhmm) != 4 or not hhmm.isdigit():
            continue
        slots.append(
            {
                "valid_day": int(day),
                "valid_hhmmz": hhmm,
            }
        )
    return slots


def format_risk_messages(risk: Dict[str, Any]) -> str:
    """Keep only material risk text and drop boilerplate public summaries."""
    watch_warning = normalize_text(risk.get("watch_warning_text"))
    if not watch_warning:
        return ""

    watch_warning_upper = watch_warning.upper()
    if any(phrase in watch_warning_upper for phrase in RISK_BOILERPLATE_PHRASES):
        return ""
    return watch_warning


def should_keep_current_analysis(
    current_analysis: str, observation_support: str
) -> Tuple[bool, Optional[str]]:
    """Gate current-analysis reasoning on available structured observations."""
    if not current_analysis:
        return False, "not_present_in_source"
    if observation_support == "none":
        return False, "missing_structured_observation_evidence"
    return True, None


def get_future_best_track_points(sample: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return verification future best-track points."""
    verification = sample.get("verification_targets", {}) or {}
    return verification.get("future_best_track_series", {}).get("points_future") or []


def classify_reward_signal_tier(point_count: int) -> str:
    """Tier RL readiness by how much future truth is available."""
    if point_count <= 0:
        return "pending_truth_backfill"
    if point_count < RL_MIN_FUTURE_BT_POINTS:
        return "weak_reward"
    if point_count < 8:
        return "usable_reward"
    return "strong_reward"


# ---------------------------------------------------------------------------
# SFT formatter
# ---------------------------------------------------------------------------

FORECAST_SYSTEM_PROMPT = (
    "You are a tropical cyclone forecaster. Use only the evidence and guidance "
    "provided in the prompt. Return only the official forecast table. The first "
    "line must be exactly 'Official forecast:'. Each remaining non-empty line "
    "must be exactly '- DayDD HHMMZ | LAT LON | NN kt'. Do not output reasoning, "
    "risk text, markdown headings, prose, or any additional text."
)

REASONING_SYSTEM_PROMPT = (
    "You are a tropical cyclone forecaster. The official forecast table is "
    "already fixed in the prompt. Explain the current analysis and forecast "
    "reasoning using only the provided evidence and the fixed official forecast. "
    "Do not repeat the forecast table and do not add risk text."
)


def build_train_metadata(
    sft_view: str,
    observation_support: str,
    available_components: List[str],
    atcf_guidance_included: bool,
    hres_guidance_included: bool,
    pre_issue_hres_included: bool,
    track_point_count: int,
    issue_time_anchor_present: bool,
    track_time_labels_present: bool,
    risk_source_present: bool,
    risk_included: bool,
    current_analysis_source_present: bool,
    current_analysis_included: bool,
    current_analysis_drop_reason: Optional[str],
    forecast_reasoning_source_present: bool,
    forecast_reasoning_included: bool,
    reward_ready: bool,
    future_best_track_point_count: int,
    reward_signal_tier: str,
) -> Dict[str, Any]:
    """Store side metadata for filtering and reporting."""
    return {
        "format_version": FORMAT_VERSION,
        "sft_view": sft_view,
        "observation_support": observation_support,
        "available_observation_components": available_components,
        "atcf_guidance_included": atcf_guidance_included,
        "hres_guidance_included": hres_guidance_included,
        "pre_issue_hres_included": pre_issue_hres_included,
        "track_point_count": track_point_count,
        "issue_time_anchor_present": issue_time_anchor_present,
        "track_time_labels_present": track_time_labels_present,
        "risk_source_present": risk_source_present,
        "risk_included": risk_included,
        "current_analysis_source_present": current_analysis_source_present,
        "current_analysis_included": current_analysis_included,
        "current_analysis_drop_reason": current_analysis_drop_reason,
        "forecast_reasoning_source_present": forecast_reasoning_source_present,
        "forecast_reasoning_included": forecast_reasoning_included,
        "reward_ready": reward_ready,
        "future_best_track_point_count": future_best_track_point_count,
        "reward_signal_tier": reward_signal_tier,
    }


def format_sft_sample(
    sample: Dict[str, Any],
    view: str = SFT_VIEW_STRICT_FORECAST,
) -> Optional[Dict[str, Any]]:
    """Convert a raw sample into one compact SFT view."""
    prompt_data = sample.get("prompt", {}) or {}
    target_data = sample.get("target", {}) or {}
    storm_meta = prompt_data.get("storm_meta", {}) or {}
    now = prompt_data.get("now_inputs", {}) or {}
    guidance = prompt_data.get("guidance_inputs", {}) or {}
    guidance_time_reference = guidance.get("guidance_time_reference", {}) or {}
    issue_time_utc = (
        storm_meta.get("issue_time_utc")
        or guidance_time_reference.get("issue_time_utc")
        or None
    )

    # Current state
    current_state = now.get("current_state_from_noaa_forecast_advisory", {}) or {}
    center = current_state.get("center", {}) or {}
    motion = current_state.get("motion", {}) or {}
    intensity = current_state.get("intensity", {}) or {}

    state_parts = [f"Position {format_coord(center.get('lat'), center.get('lon'))}"]
    motion_text = normalize_text(motion.get("motion_text"))
    if motion_text:
        speed = format_value_with_unit(motion.get("speed_kt"), "kt")
        state_parts.append(f"Motion {motion_text} at {speed}")
    if intensity:
        max_wind = format_value_with_unit(intensity.get("max_wind_kt"), "kt")
        min_pressure = format_value_with_unit(intensity.get("min_pressure_mb"), "mb")
        state_parts.append(f"Intensity {max_wind} / {min_pressure}")

    issue_time_label = resolve_time_label(
        valid_day=center.get("obs_day"),
        valid_hhmmz=center.get("obs_hhmmz"),
        valid_time_utc=issue_time_utc,
    )

    user_parts: List[str] = []
    if issue_time_label:
        user_parts.append(f"## Time Anchor\n- Advisory issue {issue_time_label}")
    user_parts.append(f"## Current State\n- {' | '.join(state_parts)}")

    # Environmental diagnostics
    env_features = (
        now.get("environment_now_ec_reanalysis", {}).get("features", {}) or {}
    )
    env_lines = []
    for feature_key in PRIMARY_ENV_FEATURES:
        feature = env_features.get(feature_key)
        if feature:
            env_lines.append(f"- {compress_cds_description(feature_key, feature)}")
    if env_lines:
        user_parts.append("## Environmental Diagnostics\n" + "\n".join(env_lines))

    # Latest pre-issue deterministic analysis
    pre_track = (
        now.get("pre_issue_guidance_context", {})
        .get("ec_hres_latest_point_at_or_before_issue_track")
    )
    pre_issue_hres_included = bool(pre_track)
    if pre_track:
        coord = format_coord(pre_track.get("lat"), pre_track.get("lon"))
        wind = format_value_with_unit(pre_track.get("wind_kt"), "kt")
        mslp = format_value_with_unit(pre_track.get("mslp_hpa"), "hPa")
        user_parts.append(
            "## Pre-Issue HRES Analysis\n"
            f"- {coord} | {wind}, {mslp}"
        )

    # Observations
    observation_block = now.get("observation_evidence_structured", {}) or {}
    observation_support = classify_observation_support(observation_block)
    available_components = get_available_observation_components(observation_block)
    user_parts.append(
        "## Observation Evidence\n" + format_observations(observation_block)
    )

    official_outputs = target_data.get("official_outputs", {}) or {}
    track_table = (
        official_outputs.get("track_intensity_table", {}).get("from_forecast_advisory", [])
        or []
    )
    if not track_table:
        return None
    guidance_target_leads = derive_guidance_target_leads(track_table, issue_time_utc)

    # Model guidance
    atcf_text = format_atcf_guidance(
        guidance.get("multimodel_guidance_a_deck", {}) or {},
        issue_time_utc=issue_time_utc,
        target_leads=guidance_target_leads,
    )
    hres_text = format_hres_guidance(
        guidance.get("ec_single_model_guidance_hres", {}) or {},
        issue_time_utc=issue_time_utc,
        target_leads=guidance_target_leads,
    )
    guidance_parts = [text for text in [atcf_text, hres_text] if text]
    if guidance_parts:
        user_parts.append("## Model Guidance\n" + "\n\n".join(guidance_parts))

    strict_forecast_table = format_track_table(track_table)
    track_time_labels_present = all(
        bool(format_day_hhmm_label(point.get("valid_day"), point.get("valid_hhmmz")))
        for point in track_table
    )

    risk_text = sanitize_identity_references(
        format_risk_messages(official_outputs.get("risk_messages", {}) or {}),
        storm_meta,
    )

    reasoning = official_outputs.get("reasoning_text", {}).get("sections", {}) or {}
    current_analysis = sanitize_identity_references(
        normalize_text(reasoning.get("current_analysis_text")),
        storm_meta,
    )
    forecast_reasoning = sanitize_identity_references(
        normalize_text(reasoning.get("forecast_reasoning_text")),
        storm_meta,
    )

    keep_current_analysis, drop_reason = should_keep_current_analysis(
        current_analysis,
        observation_support,
    )
    reasoning_lines = []
    if keep_current_analysis:
        reasoning_lines.append(f"- Current analysis: {current_analysis}")
    if forecast_reasoning:
        reasoning_lines.append(f"- Forecast reasoning: {forecast_reasoning}")

    if view == SFT_VIEW_STRICT_FORECAST:
        system_prompt = FORECAST_SYSTEM_PROMPT
        user_content = "\n\n".join(user_parts)
        assistant_content = strict_forecast_table
        risk_included = False
        current_analysis_included = False
        forecast_reasoning_included = False
    elif view == SFT_VIEW_REASONING:
        if not reasoning_lines:
            return None
        system_prompt = REASONING_SYSTEM_PROMPT
        user_content = "\n\n".join(
            [
                *user_parts,
                "## Fixed Official Forecast\n" + strict_forecast_table,
            ]
        )
        assistant_content = "## Forecast Reasoning\n" + "\n".join(reasoning_lines)
        risk_included = False
        current_analysis_included = keep_current_analysis
        forecast_reasoning_included = bool(forecast_reasoning)
    else:
        raise ValueError(f"Unsupported SFT view: {view}")

    future_best_track = get_future_best_track_points(sample)
    future_best_track_point_count = len(future_best_track)
    reward_ready = future_best_track_point_count > 0
    reward_signal_tier = classify_reward_signal_tier(future_best_track_point_count)
    train_metadata = build_train_metadata(
        sft_view=view,
        observation_support=observation_support,
        available_components=available_components,
        atcf_guidance_included=bool(atcf_text),
        hres_guidance_included=bool(hres_text),
        pre_issue_hres_included=pre_issue_hres_included,
        track_point_count=len(track_table),
        issue_time_anchor_present=bool(issue_time_label),
        track_time_labels_present=track_time_labels_present,
        risk_source_present=bool(risk_text),
        risk_included=risk_included,
        current_analysis_source_present=bool(current_analysis),
        current_analysis_included=current_analysis_included,
        current_analysis_drop_reason=drop_reason,
        forecast_reasoning_source_present=bool(forecast_reasoning),
        forecast_reasoning_included=forecast_reasoning_included,
        reward_ready=reward_ready,
        future_best_track_point_count=future_best_track_point_count,
        reward_signal_tier=reward_signal_tier,
    )

    return {
        "sample_id": sample.get("sample_id", ""),
        "format_version": FORMAT_VERSION,
        "train_metadata": train_metadata,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ],
    }


def analyze_assistant_schema(assistant_content: str) -> Dict[str, int]:
    """Audit whether one assistant target obeys the strict forecast schema."""
    nonempty_lines = [line.strip() for line in str(assistant_content or "").splitlines() if line.strip()]
    if not nonempty_lines:
        return {
            "strict_forecast_parseable": 0,
            "strict_forecast_with_extra_text": 0,
            "no_track_forecast": 0,
            "reasoning_only": 0,
        }

    first_line = nonempty_lines[0]
    is_reasoning_only = int(first_line == "## Forecast Reasoning")
    no_track_forecast = int(any("No official track forecast" in line for line in nonempty_lines))

    if first_line != "Official forecast:":
        return {
            "strict_forecast_parseable": 0,
            "strict_forecast_with_extra_text": 0,
            "no_track_forecast": no_track_forecast,
            "reasoning_only": is_reasoning_only,
        }

    extra_text = 0
    forecast_lines = 0
    for line in nonempty_lines[1:]:
        if STRICT_FORECAST_LINE_RE.match(line):
            forecast_lines += 1
        else:
            extra_text += 1

    strict_parseable = int(forecast_lines > 0 and extra_text == 0 and not no_track_forecast)
    return {
        "strict_forecast_parseable": strict_parseable,
        "strict_forecast_with_extra_text": int(extra_text > 0),
        "no_track_forecast": no_track_forecast,
        "reasoning_only": is_reasoning_only,
    }


# ---------------------------------------------------------------------------
# RL formatter
# ---------------------------------------------------------------------------

def format_rl_sample(sample: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Convert a raw sample into RL format, keeping only quality-ready rows."""
    verification = sample.get("verification_targets", {}) or {}
    future_best_track = get_future_best_track_points(sample)
    if len(future_best_track) < RL_MIN_FUTURE_BT_POINTS:
        return None

    sft_sample = format_sft_sample(sample)
    if sft_sample is None:
        return None
    track_table = (
        (sample.get("target", {}) or {})
        .get("official_outputs", {})
        .get("track_intensity_table", {})
        .get("from_forecast_advisory", [])
        or []
    )
    forecast_slots = build_reward_forecast_slots(track_table)
    if not forecast_slots:
        return None
    reward_verification = {
        "future_best_track": future_best_track,
        "forecast_slots": forecast_slots,
        "best_track_at_issue": verification.get("best_track_point_near_issue", {}).get("value"),
    }
    prompt_messages = [
        message for message in sft_sample["messages"] if message["role"] != "assistant"
    ]

    return {
        "sample_id": sample.get("sample_id", ""),
        "format_version": FORMAT_VERSION,
        "train_metadata": sft_sample.get("train_metadata", {}),
        "messages": prompt_messages,
        "verification": reward_verification,
    }


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

def estimate_tokens(text: str) -> int:
    """Coarse token estimate used for relative comparisons."""
    if not text:
        return 0
    return max(1, len(text) // 4)


def summarize_numeric_series(values: Iterable[int]) -> Dict[str, int]:
    """Return compact median/p90 summary for a numeric series."""
    items = sorted(int(value) for value in values)
    if not items:
        return {}
    return {
        "median": items[len(items) // 2],
        "p90": items[min(len(items) - 1, int(len(items) * 0.9))],
    }


def scan_train_view_leakage(
    user_content: str, assistant_content: str, sample: Dict[str, Any]
) -> Dict[str, bool]:
    """Scan the final train view instead of the raw canonical sample."""
    storm_meta = sample.get("prompt", {}).get("storm_meta", {}) or {}
    storm_name = normalize_text(storm_meta.get("storm_name"))
    storm_id = normalize_text(storm_meta.get("storm_id"))

    return {
        "prompt_contains_cjk": bool(CJK_RE.search(user_content)),
        "assistant_contains_cjk": bool(CJK_RE.search(assistant_content)),
        "prompt_contains_iso_datetime": bool(ISO_DATETIME_RE.search(user_content)),
        "assistant_contains_iso_datetime": bool(ISO_DATETIME_RE.search(assistant_content)),
        "prompt_contains_storm_name": contains_term(user_content, storm_name),
        "assistant_contains_storm_name": contains_term(assistant_content, storm_name),
        "prompt_contains_storm_id": bool(storm_id and storm_id in user_content),
        "assistant_contains_storm_id": bool(storm_id and storm_id in assistant_content),
    }


def attach_evaluation_variant(
    formatted: Dict[str, Any], evaluation_variant: Optional[str]
) -> Dict[str, Any]:
    """Attach evaluation variant metadata without changing message content."""
    if not evaluation_variant:
        return formatted

    updated = dict(formatted)
    train_metadata = dict(updated.get("train_metadata", {}) or {})
    train_metadata["evaluation_variant"] = evaluation_variant
    updated["train_metadata"] = train_metadata
    updated["evaluation_variant"] = evaluation_variant
    return updated


def convert_split(
    raw_dir: Optional[Path],
    output_dir: Path,
    split_name: str,
    fmt: str = "sft",
    sft_view: str = SFT_VIEW_STRICT_FORECAST,
    samples: Optional[List[Dict[str, Any]]] = None,
    output_name: Optional[str] = None,
    evaluation_variant: Optional[str] = None,
) -> Dict[str, Any]:
    """Convert one split directory into compact training JSONL."""
    raw_files: List[Path] = []
    if samples is None:
        if raw_dir is None or not raw_dir.exists():
            return {"count": 0, "errors": 0}
        raw_files = sorted(raw_dir.glob("*.json"))
        if not raw_files:
            return {"count": 0, "errors": 0}
    elif not samples:
        return {"count": 0, "errors": 0}

    output_file = output_dir / (output_name or f"{fmt}_{split_name}.jsonl")
    count = 0
    errors = 0
    skipped = 0
    skip_reasons = Counter()
    reward_tier_counts = Counter()
    total_token_stats: List[int] = []
    user_token_stats: List[int] = []
    assistant_token_stats: List[int] = []
    observation_support_counts = Counter()
    observation_source_combo_counts = Counter()
    target_section_counts = Counter()
    current_analysis_drop_counts = Counter()
    guidance_counts = Counter()
    leakage_counts = Counter()
    schema_counts = Counter()
    view_counts = Counter()
    anchor_counts = Counter()

    with output_file.open("w", encoding="utf-8") as handle:
        if samples is None:
            sample_stream = []
            for raw_path in raw_files:
                try:
                    sample = read_json_with_retry(raw_path)
                except Exception:
                    errors += 1
                    continue
                sample_stream.append(sample)
        else:
            sample_stream = samples

        for sample in sample_stream:
            try:
                future_best_track_point_count = len(get_future_best_track_points(sample))
                raw_track_table = (
                    ((sample.get("target", {}) or {}).get("official_outputs", {}) or {})
                    .get("track_intensity_table", {})
                    .get("from_forecast_advisory", [])
                    or []
                )
                reasoning_sections = (
                    ((sample.get("target", {}) or {}).get("official_outputs", {}) or {})
                    .get("reasoning_text", {})
                    .get("sections", {})
                    or {}
                )
                has_reasoning_source = bool(
                    normalize_text(reasoning_sections.get("current_analysis_text"))
                    or normalize_text(reasoning_sections.get("forecast_reasoning_text"))
                )
                reward_signal_tier = classify_reward_signal_tier(
                    future_best_track_point_count
                )
                if fmt == "rl":
                    reward_tier_counts[reward_signal_tier] += 1

                if fmt == "sft":
                    formatted = format_sft_sample(sample, view=sft_view)
                    if formatted is None:
                        skipped += 1
                        if not raw_track_table:
                            skip_reasons["missing_track_table"] += 1
                        elif sft_view == SFT_VIEW_REASONING and not has_reasoning_source:
                            skip_reasons["missing_reasoning_text"] += 1
                        else:
                            skip_reasons["unsupported_sample"] += 1
                        continue
                elif fmt == "rl":
                    formatted = format_rl_sample(sample)
                    if formatted is None:
                        skipped += 1
                        if not raw_track_table:
                            skip_reasons["missing_track_table"] += 1
                        elif reward_signal_tier == "pending_truth_backfill":
                            skip_reasons["missing_future_best_track"] += 1
                        else:
                            skip_reasons["insufficient_future_best_track_points"] += 1
                        continue
                else:
                    raise ValueError(f"Unsupported format: {fmt}")

                formatted = attach_evaluation_variant(formatted, evaluation_variant)
                messages = formatted.get("messages", [])
                user_content = next(
                    (message.get("content", "") for message in messages if message.get("role") == "user"),
                    "",
                )
                assistant_content = next(
                    (
                        message.get("content", "")
                        for message in messages
                        if message.get("role") == "assistant"
                    ),
                    "",
                )

                total_text = " ".join(message.get("content", "") for message in messages)
                total_token_stats.append(estimate_tokens(total_text))
                user_token_stats.append(estimate_tokens(user_content))
                if assistant_content:
                    assistant_token_stats.append(estimate_tokens(assistant_content))

                train_metadata = formatted.get("train_metadata", {}) or {}
                if train_metadata.get("sft_view"):
                    view_counts[str(train_metadata.get("sft_view"))] += 1
                observation_support = train_metadata.get("observation_support", "unknown")
                observation_support_counts[observation_support] += 1
                combo = "+".join(train_metadata.get("available_observation_components") or ["none"])
                observation_source_combo_counts[combo] += 1

                if train_metadata.get("issue_time_anchor_present"):
                    anchor_counts["issue_time_anchor_present"] += 1
                if train_metadata.get("track_time_labels_present"):
                    anchor_counts["track_time_labels_present"] += 1
                if train_metadata.get("risk_source_present"):
                    target_section_counts["risk_source_present"] += 1
                if train_metadata.get("risk_included"):
                    target_section_counts["risk_included"] += 1
                if train_metadata.get("current_analysis_source_present"):
                    target_section_counts["current_analysis_source_present"] += 1
                if train_metadata.get("current_analysis_included"):
                    target_section_counts["current_analysis_included"] += 1
                if train_metadata.get("forecast_reasoning_source_present"):
                    target_section_counts["forecast_reasoning_source_present"] += 1
                if train_metadata.get("forecast_reasoning_included"):
                    target_section_counts["forecast_reasoning_included"] += 1
                if train_metadata.get("atcf_guidance_included"):
                    guidance_counts["atcf_consensus"] += 1
                if train_metadata.get("hres_guidance_included"):
                    guidance_counts["hres_deterministic"] += 1
                if train_metadata.get("pre_issue_hres_included"):
                    guidance_counts["pre_issue_hres"] += 1

                drop_reason = train_metadata.get("current_analysis_drop_reason")
                if drop_reason:
                    current_analysis_drop_counts[drop_reason] += 1

                leakage_flags = scan_train_view_leakage(
                    user_content=user_content,
                    assistant_content=assistant_content,
                    sample=sample,
                )
                for flag_name, is_present in leakage_flags.items():
                    if is_present:
                        leakage_counts[flag_name] += 1

                for schema_name, schema_value in analyze_assistant_schema(assistant_content).items():
                    if schema_value:
                        schema_counts[schema_name] += schema_value

                handle.write(json.dumps(formatted, ensure_ascii=False) + "\n")
                count += 1
            except Exception:
                errors += 1

    stats: Dict[str, Any] = {
        "count": count,
        "errors": errors,
        "skipped": skipped,
    }
    if fmt == "sft":
        stats["sft_view"] = sft_view
    total_summary = summarize_numeric_series(total_token_stats)
    user_summary = summarize_numeric_series(user_token_stats)
    assistant_summary = summarize_numeric_series(assistant_token_stats)
    if total_summary:
        stats["token_est_median"] = total_summary["median"]
        stats["token_est_p90"] = total_summary["p90"]
    if user_summary:
        stats["user_token_est_median"] = user_summary["median"]
        stats["user_token_est_p90"] = user_summary["p90"]
    if assistant_summary:
        stats["assistant_token_est_median"] = assistant_summary["median"]
        stats["assistant_token_est_p90"] = assistant_summary["p90"]

    stats["observation_support"] = {
        "none": observation_support_counts.get("none", 0),
        "single_source": observation_support_counts.get("single_source", 0),
        "multi_source": observation_support_counts.get("multi_source", 0),
    }
    stats["observation_source_combos"] = dict(sorted(observation_source_combo_counts.items()))
    stats["target_sections"] = {
        "risk_source_present": target_section_counts.get("risk_source_present", 0),
        "risk_included": target_section_counts.get("risk_included", 0),
        "current_analysis_source_present": target_section_counts.get("current_analysis_source_present", 0),
        "current_analysis_included": target_section_counts.get("current_analysis_included", 0),
        "forecast_reasoning_source_present": target_section_counts.get("forecast_reasoning_source_present", 0),
        "forecast_reasoning_included": target_section_counts.get("forecast_reasoning_included", 0),
    }
    stats["current_analysis_drop_reasons"] = dict(sorted(current_analysis_drop_counts.items()))
    stats["guidance_included"] = {
        "atcf_consensus": guidance_counts.get("atcf_consensus", 0),
        "hres_deterministic": guidance_counts.get("hres_deterministic", 0),
        "pre_issue_hres": guidance_counts.get("pre_issue_hres", 0),
    }
    if anchor_counts or fmt == "sft":
        stats["time_anchor"] = {
            "issue_time_anchor_present": anchor_counts.get("issue_time_anchor_present", 0),
            "track_time_labels_present": anchor_counts.get("track_time_labels_present", 0),
        }
    if schema_counts or fmt == "sft":
        stats["assistant_schema"] = {
            "strict_forecast_parseable": schema_counts.get("strict_forecast_parseable", 0),
            "strict_forecast_with_extra_text": schema_counts.get("strict_forecast_with_extra_text", 0),
            "no_track_forecast": schema_counts.get("no_track_forecast", 0),
            "reasoning_only": schema_counts.get("reasoning_only", 0),
        }
    if view_counts:
        stats["view_counts"] = dict(sorted(view_counts.items()))
    if fmt == "rl":
        stats["reward_signal_tiers"] = {
            "pending_truth_backfill": reward_tier_counts.get("pending_truth_backfill", 0),
            "weak_reward": reward_tier_counts.get("weak_reward", 0),
            "usable_reward": reward_tier_counts.get("usable_reward", 0),
            "strong_reward": reward_tier_counts.get("strong_reward", 0),
        }
    stats["train_view_leakage"] = {
        flag_name: leakage_counts.get(flag_name, 0)
        for flag_name in KNOWN_LEAKAGE_FLAGS
    }
    if skip_reasons:
        stats["skipped_reasons"] = dict(sorted(skip_reasons.items()))
    return stats


def generate_test_variants(
    base_dir: Path,
    raw_base: Path,
    output_dir: Path,
) -> Dict[str, Any]:
    """Regenerate SFT test-set variants in the current compact schema."""
    test_dir = raw_base / "test"
    raw_files = sorted(test_dir.glob("*.json"))
    if not raw_files:
        return {}

    test_samples = [
        read_json_with_retry(raw_path)
        for raw_path in raw_files
    ]

    try:
        scripts_dir = base_dir / "scripts"
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))
        from data_leakage_prevention import Anonymizer
    except ImportError:
        return {"error": "failed_to_import_anonymizer"}

    mapping_file = base_dir / "data" / "interim" / "leakage_prevention" / "anonymization_mapping_v1.json"
    groundtruth_csv = base_dir / "GroundTruth_Cyclones" / "matched_cyclone_tracks.csv"
    anonymizer = Anonymizer(
        groundtruth_csv=groundtruth_csv,
        mapping_file=mapping_file if mapping_file.exists() else None,
    )
    if mapping_file.exists():
        mapping = anonymizer.load_mapping(mapping_file)
    else:
        storm_ids = sorted(
            {
                (sample.get("prompt", {}).get("storm_meta", {}) or {}).get("storm_id", "")
                for sample in test_samples
            }
        )
        mapping = anonymizer.generate_mapping([storm_id for storm_id in storm_ids if storm_id])

    variant_sources = {
        "anonymous": anonymizer.generate_anonymous_test_set(test_samples, mapping),
        "structured_only": anonymizer.generate_structured_only_test_set(test_samples),
        "perturbation": anonymizer.generate_perturbation_test_set(test_samples, mapping),
    }

    results: Dict[str, Any] = {}
    for variant_name, variant_samples in variant_sources.items():
        results[variant_name] = convert_split(
            raw_dir=None,
            output_dir=output_dir,
            split_name="test",
            fmt="sft",
            sft_view=SFT_VIEW_STRICT_FORECAST,
            samples=variant_samples,
            output_name=f"sft_test_{variant_name}.jsonl",
            evaluation_variant=variant_name,
        )
    return results


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Dataset formatter: raw JSON -> compact SFT/RL JSONL")
    parser.add_argument("--base-dir", type=str, default=".")
    parser.add_argument("--raw-dir", type=str, default="data/training/raw")
    parser.add_argument("--output-dir", type=str, default="data/training")
    parser.add_argument("--format", choices=["sft", "rl", "both"], default="both")
    parser.add_argument(
        "--skip-test-variants",
        action="store_true",
        help="Do not regenerate sft_test_{anonymous,structured_only,perturbation}.jsonl",
    )
    parser.add_argument(
        "--skip-reasoning-aux",
        action="store_true",
        help="Do not export sft_reasoning_{train,val,test}.jsonl auxiliary views",
    )

    args = parser.parse_args()
    base_dir = Path(args.base_dir)
    raw_base = base_dir / args.raw_dir
    output_dir = base_dir / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    results: Dict[str, Any] = {"format_version": FORMAT_VERSION}

    for split_name in ["train", "val", "test", "unassigned"]:
        raw_dir = raw_base / split_name
        if not raw_dir.exists():
            continue

        if args.format in ("sft", "both"):
            stats = convert_split(
                raw_dir,
                output_dir,
                split_name,
                fmt="sft",
                sft_view=SFT_VIEW_STRICT_FORECAST,
            )
            results[f"sft_{split_name}"] = stats
            print(
                f"SFT {split_name}: {stats['count']} samples | "
                f"median tokens {stats.get('token_est_median', 'N/A')}"
            )
            if split_name in {"train", "val", "test"} and not args.skip_reasoning_aux:
                reasoning_stats = convert_split(
                    raw_dir,
                    output_dir,
                    split_name,
                    fmt="sft",
                    sft_view=SFT_VIEW_REASONING,
                    output_name=f"sft_reasoning_{split_name}.jsonl",
                )
                results[f"sft_reasoning_{split_name}"] = reasoning_stats
                print(
                    f"SFT reasoning {split_name}: {reasoning_stats['count']} samples | "
                    f"median tokens {reasoning_stats.get('token_est_median', 'N/A')}"
                )

        if args.format in ("rl", "both"):
            stats = convert_split(raw_dir, output_dir, split_name, fmt="rl")
            results[f"rl_{split_name}"] = stats
            print(
                f"RL  {split_name}: {stats['count']} samples | "
                f"skipped {stats.get('skipped', 0)}"
            )

    if args.format in ("sft", "both") and not args.skip_test_variants:
        variant_stats = generate_test_variants(
            base_dir=base_dir,
            raw_base=raw_base,
            output_dir=output_dir,
        )
        if variant_stats:
            results["sft_test_variants"] = variant_stats
            for variant_name, stats in variant_stats.items():
                if "error" in stats:
                    print(f"SFT test variant {variant_name}: ERROR {stats['error']}")
                else:
                    print(
                        f"SFT test variant {variant_name}: {stats['count']} samples | "
                        f"median tokens {stats.get('token_est_median', 'N/A')}"
                    )

    report_path = output_dir / "format_report.json"
    report_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
