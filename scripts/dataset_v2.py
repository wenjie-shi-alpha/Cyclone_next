#!/usr/bin/env python3
"""Shared helpers for canonical v2 dataset rebuild and view export."""

from __future__ import annotations

import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from dataset_formatter import (
    TRACK_CORRECTION_LEAD_TOLERANCE_H,
    analyze_assistant_schema,
    analyze_track_inflection,
    classify_turning_signal_from_points,
    format_risk_messages,
    format_track_table,
    normalize_text,
    select_nearest_lead_point,
)
from slot_locked_forecast_correction import (
    SLOT_CORRECTION_FIELD_CANONICAL_VALUES,
    SLOT_CORRECTION_FIELD_NAMES,
    SLOT_CORRECTION_FIELD_VALUE_ALIASES,
    derive_slot_correction_payload,
)


CANONICAL_VERSION = "tc_canonical_v2.0.0"
DIAGNOSTIC_DERIVATION_VERSION = "heuristic_v2"
SUPPORTED_SPLITS = ("train", "val", "test", "unassigned")
DIAGNOSTIC_FIELDS = (
    "track_control_signal",
    "turning_signal",
    "intensity_support_signal",
    "shear_constraint_level",
    "land_interaction_level",
    "model_agreement_level",
    "main_uncertainty_source",
    "forecast_confidence_level",
    "expert_decision_notes",
)
DIAGNOSTIC_TRACK_CORE_FIELDS = (
    "track_control_signal",
    "turning_signal",
    "model_agreement_level",
    "forecast_confidence_level",
)
DIAGNOSTIC_TRACK_TURN_FIELDS = (
    "track_control_signal",
    "turning_signal",
)
DIAGNOSTIC_TRACK_INFLECTION_FIELDS = (
    "steering_regime_phase",
    "turn_timing_bucket",
    "turn_direction_family",
    "turn_magnitude_bucket",
)
DIAGNOSTIC_TRACK_CORRECTION_FIELDS = (
    "lat_bias_vs_consensus_48h_bucket",
    "lon_bias_vs_consensus_48h_bucket",
    "lat_bias_vs_consensus_72h_bucket",
    "lon_bias_vs_consensus_72h_bucket",
)
DIAGNOSTIC_SLOT_CORRECTION_FIELDS = tuple(SLOT_CORRECTION_FIELD_NAMES)
DIAGNOSTIC_SLOT_TURN_CORRECTION_FIELDS = tuple(SLOT_CORRECTION_FIELD_NAMES) + (
    "turning_signal",
)
DIAGNOSTIC_CORE_FIELDS = (
    "track_control_signal",
    "turning_signal",
    "intensity_support_signal",
    "shear_constraint_level",
    "model_agreement_level",
    "forecast_confidence_level",
)

_TRACK_CORRECTION_LAT_BUCKET_VALUES = (
    "south_large",
    "south_small",
    "near_consensus",
    "north_small",
    "north_large",
)
_TRACK_CORRECTION_LON_BUCKET_VALUES = (
    "east_large",
    "east_small",
    "near_consensus",
    "west_small",
    "west_large",
)
_TRACK_CORRECTION_BUCKET_THRESHOLDS_KM = {
    48: {"near": 50.0, "large": 150.0},
    72: {"near": 75.0, "large": 175.0},
}

DIAGNOSTIC_FIELD_CANONICAL_VALUES: Dict[str, Optional[Tuple[str, ...]]] = {
    "track_control_signal": (
        "competing_ridge_and_trough",
        "subtropical_high",
        "midlatitude_trough",
    ),
    "turning_signal": (
        "steady",
        "notable_turn",
        "recurvature",
    ),
    "steering_regime_phase": (
        "locked",
        "transition",
        "recurving",
    ),
    "turn_timing_bucket": (
        "none_or_locked",
        "24_to_48h",
        "48_to_72h",
    ),
    "turn_direction_family": (
        "hold",
        "poleward_turn",
        "eastward_escape",
        "westward_or_equatorward_bend",
    ),
    "turn_magnitude_bucket": (
        "none",
        "modest",
        "strong",
        "sharp",
    ),
    "lat_bias_vs_consensus_48h_bucket": _TRACK_CORRECTION_LAT_BUCKET_VALUES,
    "lon_bias_vs_consensus_48h_bucket": _TRACK_CORRECTION_LON_BUCKET_VALUES,
    "lat_bias_vs_consensus_72h_bucket": _TRACK_CORRECTION_LAT_BUCKET_VALUES,
    "lon_bias_vs_consensus_72h_bucket": _TRACK_CORRECTION_LON_BUCKET_VALUES,
    "intensity_support_signal": (
        "supportive",
        "mixed",
        "constraining",
    ),
    "shear_constraint_level": (
        "weak",
        "moderate",
        "strong",
        "very_strong",
    ),
    "land_interaction_level": (
        "moderate",
        "high",
    ),
    "model_agreement_level": (
        "low",
        "medium",
        "high",
    ),
    "main_uncertainty_source": (
        "model_spread",
        "midlatitude_trough_interaction",
        "ridge_evolution",
        "vertical_wind_shear",
        "land_interaction",
        "intensity_change",
    ),
    "forecast_confidence_level": (
        "low",
        "medium",
        "high",
    ),
    "expert_decision_notes": None,
}
DIAGNOSTIC_FIELD_CANONICAL_VALUES.update(SLOT_CORRECTION_FIELD_CANONICAL_VALUES)

_DIAGNOSTIC_NULL_STRINGS = {
    "",
    "null",
    "none",
    "na",
    "n_a",
    "n/a",
    "<null>",
    "unknown",
    "not_available",
    "not_applicable",
    "missing",
}

_TRACK_CORRECTION_LAT_VALUE_ALIASES = {
    "south_large": "south_large",
    "south_big": "south_large",
    "south_small": "south_small",
    "near_consensus": "near_consensus",
    "near": "near_consensus",
    "near_guidance": "near_consensus",
    "aligned_with_consensus": "near_consensus",
    "north_small": "north_small",
    "north_large": "north_large",
    "north_big": "north_large",
}
_TRACK_CORRECTION_LON_VALUE_ALIASES = {
    "east_large": "east_large",
    "east_big": "east_large",
    "east_small": "east_small",
    "near_consensus": "near_consensus",
    "near": "near_consensus",
    "near_guidance": "near_consensus",
    "aligned_with_consensus": "near_consensus",
    "west_small": "west_small",
    "west_large": "west_large",
    "west_big": "west_large",
}

_DIAGNOSTIC_VALUE_ALIASES: Dict[str, Dict[str, Optional[str]]] = {
    "track_control_signal": {
        "subtropical_high": "subtropical_high",
        "subtropicalhigh": "subtropical_high",
        "subtropical_ridge": "subtropical_high",
        "ridge": "subtropical_high",
        "subtropical": "subtropical_high",
        "midlatitude_trough": "midlatitude_trough",
        "mid_latitude_trough": "midlatitude_trough",
        "trough": "midlatitude_trough",
        "westerly_trough": "midlatitude_trough",
        "competing_ridge_and_trough": "competing_ridge_and_trough",
        "ridge_and_trough": "competing_ridge_and_trough",
        "ridge_trough_interaction": "competing_ridge_and_trough",
        "competing_ridge_trough": "competing_ridge_and_trough",
    },
    "turning_signal": {
        "steady": "steady",
        "straight": "steady",
        "no_turn": "steady",
        "notable_turn": "notable_turn",
        "turn": "notable_turn",
        "turning": "notable_turn",
        "recurvature": "recurvature",
        "recurving": "recurvature",
    },
    "steering_regime_phase": {
        "locked": "locked",
        "transition": "transition",
        "recurving": "recurving",
        "recurvature": "recurving",
    },
    "turn_timing_bucket": {
        "none_or_locked": "none_or_locked",
        "locked": "none_or_locked",
        "24_to_48h": "24_to_48h",
        "24_48h": "24_to_48h",
        "48_to_72h": "48_to_72h",
        "48_72h": "48_to_72h",
    },
    "turn_direction_family": {
        "hold": "hold",
        "poleward_turn": "poleward_turn",
        "poleward": "poleward_turn",
        "eastward_escape": "eastward_escape",
        "eastward": "eastward_escape",
        "westward_or_equatorward_bend": "westward_or_equatorward_bend",
        "westward_bend": "westward_or_equatorward_bend",
        "equatorward_turn": "westward_or_equatorward_bend",
    },
    "turn_magnitude_bucket": {
        "none": "none",
        "modest": "modest",
        "moderate": "modest",
        "strong": "strong",
        "sharp": "sharp",
        "large": "sharp",
    },
    "lat_bias_vs_consensus_48h_bucket": _TRACK_CORRECTION_LAT_VALUE_ALIASES,
    "lon_bias_vs_consensus_48h_bucket": _TRACK_CORRECTION_LON_VALUE_ALIASES,
    "lat_bias_vs_consensus_72h_bucket": _TRACK_CORRECTION_LAT_VALUE_ALIASES,
    "lon_bias_vs_consensus_72h_bucket": _TRACK_CORRECTION_LON_VALUE_ALIASES,
    "model_agreement_level": {
        "low": "low",
        "medium": "medium",
        "moderate": "medium",
        "med": "medium",
        "high": "high",
    },
    "forecast_confidence_level": {
        "low": "low",
        "medium": "medium",
        "moderate": "medium",
        "med": "medium",
        "high": "high",
    },
    "intensity_support_signal": {
        "supportive": "supportive",
        "support": "supportive",
        "favorable": "supportive",
        "mixed": "mixed",
        "neutral": "mixed",
        "constraining": "constraining",
        "constraint": "constraining",
        "unfavorable": "constraining",
    },
    "shear_constraint_level": {
        "weak": "weak",
        "moderate": "moderate",
        "medium": "moderate",
        "strong": "strong",
        "very_strong": "very_strong",
        "verystrong": "very_strong",
    },
    "land_interaction_level": {
        "moderate": "moderate",
        "medium": "moderate",
        "high": "high",
    },
    "main_uncertainty_source": {
        "model_spread": "model_spread",
        "spread": "model_spread",
        "midlatitude_trough_interaction": "midlatitude_trough_interaction",
        "trough_interaction": "midlatitude_trough_interaction",
        "ridge_evolution": "ridge_evolution",
        "ridge": "ridge_evolution",
        "vertical_wind_shear": "vertical_wind_shear",
        "shear": "vertical_wind_shear",
        "land_interaction": "land_interaction",
        "land": "land_interaction",
        "intensity_change": "intensity_change",
    },
}
_DIAGNOSTIC_VALUE_ALIASES.update(SLOT_CORRECTION_FIELD_VALUE_ALIASES)

_LEVEL_MAP = {
    "极强": "very_strong",
    "强": "strong",
    "中等": "moderate",
    "弱": "weak",
    "低": "low",
    "高": "high",
    "极高": "very_high",
    "负值": "negative",
}

_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")


def read_json_with_retry(
    path: Path,
    attempts: int = 5,
    delay_sec: float = 0.2,
) -> Dict[str, Any]:
    """Read one JSON file while tolerating short concurrent-write windows."""
    import time

    last_error: Exception | None = None
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


def write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> int:
    """Write JSONL records and return the written row count."""
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def parse_utc_datetime(value: Any) -> Optional[datetime]:
    """Parse one ISO-like UTC datetime into a timezone-aware object."""
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


def format_utc_datetime(value: Optional[datetime]) -> Optional[str]:
    """Render one UTC datetime as ISO text with Z suffix."""
    if value is None:
        return None
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def infer_latest_legacy_raw_dir(base_dir: Path) -> Optional[Path]:
    """Find the newest legacy raw dataset tree produced by the old pipeline."""
    candidates = [
        path / "raw"
        for path in sorted(base_dir.glob("data/training_canonical_*"))
        if (path / "raw").exists()
    ]
    return candidates[-1] if candidates else None


def iter_legacy_raw_samples(
    raw_base_dir: Path,
    splits: Sequence[str] = SUPPORTED_SPLITS,
) -> Iterable[Tuple[str, Path, Dict[str, Any]]]:
    """Yield all legacy raw samples split-by-split."""
    for split_name in splits:
        split_dir = raw_base_dir / split_name
        if not split_dir.exists():
            continue
        for sample_path in sorted(split_dir.glob("*.json")):
            yield split_name, sample_path, read_json_with_retry(sample_path)


def iter_canonical_split_records(
    canonical_dir: Path,
    splits: Sequence[str] = SUPPORTED_SPLITS,
) -> Iterable[Tuple[str, Path, Dict[str, Any]]]:
    """Yield all canonical records split-by-split."""
    for split_name in splits:
        split_path = canonical_dir / f"{split_name}.jsonl"
        if not split_path.exists():
            continue
        with split_path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield split_name, split_path, json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSONL record in {split_path}:{line_no}"
                    ) from exc


def _candidate_valid_datetimes(
    issue_dt: datetime,
    valid_day: Any,
    valid_hhmmz: Any,
) -> List[datetime]:
    """Build month-aware valid-time candidates from DayDD HHMMZ labels."""
    day_text = str(valid_day or "").strip()
    hhmm_text = str(valid_hhmmz or "").strip().upper().removesuffix("Z")
    if not day_text.isdigit() or len(hhmm_text) != 4 or not hhmm_text.isdigit():
        return []

    day = int(day_text)
    hour = int(hhmm_text[:2])
    minute = int(hhmm_text[2:])
    if hour > 23 or minute > 59:
        return []

    candidates: List[datetime] = []
    base_month = issue_dt.month - 1
    for month_offset in (-1, 0, 1, 2):
        absolute_month = base_month + month_offset
        year = issue_dt.year + absolute_month // 12
        month = absolute_month % 12 + 1
        try:
            candidates.append(datetime(year, month, day, hour, minute, tzinfo=timezone.utc))
        except ValueError:
            continue
    return candidates


def resolve_valid_datetime_from_day_label(
    issue_dt: Optional[datetime],
    valid_day: Any,
    valid_hhmmz: Any,
) -> Optional[datetime]:
    """Resolve one forecast valid time relative to the sample issue time."""
    if issue_dt is None:
        return None

    candidates = _candidate_valid_datetimes(issue_dt, valid_day, valid_hhmmz)
    if not candidates:
        return None

    future = []
    for candidate in candidates:
        delta_h = (candidate - issue_dt).total_seconds() / 3600.0
        if -1.0 <= delta_h <= 240.0:
            future.append((max(delta_h, 0.0), candidate))
    if future:
        return min(future, key=lambda item: item[0])[1]

    nearest = min(
        candidates,
        key=lambda candidate: abs((candidate - issue_dt).total_seconds()),
    )
    return nearest


def compute_lead_times(
    issue_dt: Optional[datetime],
    track_points: Sequence[Dict[str, Any]],
) -> List[int]:
    """Compute issue-relative lead hours from DayDD HHMMZ forecast labels."""
    leads: List[int] = []
    for point in track_points or []:
        valid_dt = resolve_valid_datetime_from_day_label(
            issue_dt,
            point.get("valid_day"),
            point.get("valid_hhmmz"),
        )
        if valid_dt is None or issue_dt is None:
            continue
        lead_h = int(round((valid_dt - issue_dt).total_seconds() / 3600.0))
        if lead_h not in leads:
            leads.append(lead_h)
    return leads


def _collect_datetimes(values: Iterable[Any]) -> List[datetime]:
    collected: List[datetime] = []
    for value in values:
        parsed = parse_utc_datetime(value)
        if parsed is not None:
            collected.append(parsed)
    return collected


def build_input_window_spec(
    issue_time: Any,
    observation_block: Dict[str, Any],
    environment_block: Dict[str, Any],
    guidance_time_reference: Dict[str, Any],
    pre_issue_track: Dict[str, Any],
    pre_issue_environment: Dict[str, Any],
) -> Dict[str, Optional[str]]:
    """Summarize the aligned time window for one issuance unit."""
    issue_dt = parse_utc_datetime(issue_time)

    obs_times = _collect_datetimes(
        [issue_time]
        + [item.get("obs_time_utc") for item in observation_block.get("value", []) or []]
    )
    env_times = _collect_datetimes(
        [
            environment_block.get("source_time"),
            pre_issue_environment.get("valid_time_utc"),
            pre_issue_track.get("valid_time_utc"),
        ]
    )

    if issue_dt is not None and not obs_times:
        obs_times = [issue_dt]
    if issue_dt is not None and not env_times:
        env_times = [issue_dt]

    return {
        "obs_start": format_utc_datetime(min(obs_times) if obs_times else None),
        "obs_end": format_utc_datetime(max(obs_times) if obs_times else None),
        "env_start": format_utc_datetime(min(env_times) if env_times else None),
        "env_end": format_utc_datetime(max(env_times) if env_times else None),
        "guidance_cycle": normalize_text(
            guidance_time_reference.get("model_init_time_utc")
            or guidance_time_reference.get("issue_time_utc")
        )
        or None,
    }


def _normalize_level(value: Any) -> Optional[str]:
    text = normalize_text(value)
    if not text:
        return None
    return _LEVEL_MAP.get(text, text.lower().replace(" ", "_"))


def _slug_diagnostic_value(value: Any) -> Optional[str]:
    text = normalize_text(value)
    if not text:
        return None
    lowered = text.casefold().replace("-", "_")
    lowered = _NON_ALNUM_RE.sub("_", lowered)
    lowered = lowered.strip("_")
    return lowered or None


def diagnostic_allowed_values(field_name: str) -> Optional[Tuple[str, ...]]:
    """Return the frozen canonical label space for one diagnostic field."""
    return DIAGNOSTIC_FIELD_CANONICAL_VALUES.get(field_name)


def describe_diagnostic_field(field_name: str) -> str:
    """Render one short schema line for prompt-side constrained diagnostic output."""
    allowed = diagnostic_allowed_values(field_name)
    if allowed is None:
        return f'"{field_name}": one short string or null'
    allowed_text = ", ".join(f'"{label}"' for label in allowed)
    return f'"{field_name}": one of {allowed_text}, or null'


def normalize_diagnostic_field_value(
    field_name: str,
    value: Any,
) -> Any:
    """Canonicalize one diagnostic value into the frozen label space when possible."""
    if value in (None, [], {}):
        return None
    if isinstance(value, str):
        text = normalize_text(value)
        slug = _slug_diagnostic_value(text)
        if slug in _DIAGNOSTIC_NULL_STRINGS:
            return None

        allowed = diagnostic_allowed_values(field_name)
        aliases = _DIAGNOSTIC_VALUE_ALIASES.get(field_name, {})
        if slug is not None:
            mapped = aliases.get(slug)
            if mapped is not None:
                return mapped
            if allowed is not None and slug in allowed:
                return slug

        if field_name == "expert_decision_notes":
            return text or None
        if allowed is None:
            return text or None
        return slug or text or None

    if isinstance(value, (bool, int, float)):
        return str(value)
    if field_name == "expert_decision_notes":
        return normalize_text(value) or None
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def normalize_diagnostic_payload(
    payload: Dict[str, Any] | None,
    *,
    field_names: Sequence[str],
) -> Dict[str, Any]:
    """Canonicalize one diagnostic payload field-by-field."""
    source = payload or {}
    return {
        str(field_name): normalize_diagnostic_field_value(str(field_name), source.get(field_name))
        for field_name in field_names
    }


def _classify_model_agreement(atcf_guidance: Dict[str, Any]) -> Optional[str]:
    points = atcf_guidance.get("consensus_spread_points_future", []) or []
    track_spreads = [
        float(point.get("track_spread_km"))
        for point in points
        if point.get("track_spread_km") not in (None, "")
    ]
    wind_spreads = [
        float(point.get("wind_spread_kt"))
        for point in points
        if point.get("wind_spread_kt") not in (None, "")
    ]
    if not track_spreads:
        return None

    track_spreads = sorted(track_spreads)
    wind_spreads = sorted(wind_spreads) if wind_spreads else [999.0]
    median_track = track_spreads[len(track_spreads) // 2]
    median_wind = wind_spreads[len(wind_spreads) // 2]

    if median_track <= 350.0 and median_wind <= 8.0:
        return "high"
    if median_track <= 800.0 and median_wind <= 15.0:
        return "medium"
    return "low"


def _infer_track_control_signal(
    reasoning_text: str,
    env_features: Dict[str, Any],
) -> Optional[str]:
    reasoning_lower = reasoning_text.lower()
    if "trough" in reasoning_lower and ("ridge" in reasoning_lower or "subtropical" in reasoning_lower):
        return "competing_ridge_and_trough"
    if "trough" in reasoning_lower:
        return "midlatitude_trough"
    if "ridge" in reasoning_lower or "subtropical" in reasoning_lower:
        return "subtropical_high"

    subtropical_high = _normalize_level(
        (env_features.get("subtropical_high") or {}).get("level")
    )
    westerly_trough = _normalize_level(
        (env_features.get("westerly_trough") or {}).get("level")
    )
    monsoon_trough = _normalize_level(
        (env_features.get("monsoon_trough") or {}).get("level")
    )
    if subtropical_high in {"strong", "moderate"} and westerly_trough in {"strong", "moderate"}:
        return "competing_ridge_and_trough"
    if subtropical_high:
        return "subtropical_high"
    if westerly_trough:
        return "westerly_trough"
    if monsoon_trough:
        return "monsoon_trough"
    return None


def _direction_bucket(point_a: Dict[str, Any], point_b: Dict[str, Any]) -> Optional[str]:
    lat_a = point_a.get("lat")
    lon_a = point_a.get("lon")
    lat_b = point_b.get("lat")
    lon_b = point_b.get("lon")
    if None in {lat_a, lon_a, lat_b, lon_b}:
        return None

    dlat = float(lat_b) - float(lat_a)
    dlon = float(lon_b) - float(lon_a)
    lat_dir = "north" if dlat > 0.4 else ("south" if dlat < -0.4 else "")
    lon_dir = "east" if dlon > 0.4 else ("west" if dlon < -0.4 else "")
    if lat_dir and lon_dir:
        return f"{lat_dir}_{lon_dir}"
    return lat_dir or lon_dir or "steady"


def _infer_turning_signal(
    track_points: Sequence[Dict[str, Any]],
    reasoning_text: str,
) -> Optional[str]:
    reasoning_lower = reasoning_text.lower()
    if any(
        token in reasoning_lower
        for token in (
            "recurv",
            "turning around",
            "turn northeast",
            "turn southeast",
            "east-northeastward",
            "south-eastward",
            "north-eastward",
        )
    ):
        return "recurvature"

    geometric_signal = classify_turning_signal_from_points(
        list(track_points or []),
        lat_key="lat",
        lon_key="lon",
    )
    if geometric_signal is not None:
        return geometric_signal
    if "turn" in reasoning_lower:
        return "notable_turn"
    return None


def _official_track_points_with_leads(
    issue_time: Any,
    track_points: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    issue_dt = parse_utc_datetime(issue_time)
    if issue_dt is None:
        return []

    points_with_leads: List[Dict[str, Any]] = []
    for point in track_points or []:
        valid_dt = resolve_valid_datetime_from_day_label(
            issue_dt,
            point.get("valid_day"),
            point.get("valid_hhmmz"),
        )
        lat = point.get("lat")
        lon = point.get("lon")
        if valid_dt is None or lat in (None, "") or lon in (None, ""):
            continue
        lead_h = (valid_dt - issue_dt).total_seconds() / 3600.0
        if lead_h < 0.0:
            continue
        points_with_leads.append(
            {
                "lead_from_issue_h": lead_h,
                "lat": lat,
                "lon": lon,
            }
        )
    return points_with_leads


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


def _track_correction_bucket(
    delta_km: float,
    *,
    near_threshold_km: float,
    large_threshold_km: float,
    negative_prefix: str,
    positive_prefix: str,
) -> str:
    abs_delta_km = abs(delta_km)
    if abs_delta_km < near_threshold_km:
        return "near_consensus"
    magnitude = "large" if abs_delta_km >= large_threshold_km else "small"
    direction = positive_prefix if delta_km >= 0.0 else negative_prefix
    return f"{direction}_{magnitude}"


def _track_correction_bias_km(
    official_point: Dict[str, Any],
    consensus_point: Dict[str, Any],
) -> Optional[Tuple[float, float]]:
    official_lat = _safe_float(official_point.get("lat"))
    official_lon = _safe_float(official_point.get("lon"))
    consensus_lat = _safe_float(consensus_point.get("consensus_lat"))
    consensus_lon = _safe_float(consensus_point.get("consensus_lon"))
    if None in {official_lat, official_lon, consensus_lat, consensus_lon}:
        return None

    lat_bias_km = (official_lat - consensus_lat) * 111.32
    lon_delta_deg = _normalize_longitude_delta(official_lon - consensus_lon)
    reference_lat = (official_lat + consensus_lat) / 2.0
    lon_bias_east_km = lon_delta_deg * 111.32 * math.cos(math.radians(reference_lat))
    return lat_bias_km, lon_bias_east_km


def _derive_track_correction_candidate(
    issue_time: Any,
    consensus_points: Sequence[Dict[str, Any]],
    track_points: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    defaults = {
        field_name: None
        for field_name in DIAGNOSTIC_TRACK_CORRECTION_FIELDS
    }
    official_points_with_leads = _official_track_points_with_leads(issue_time, track_points)
    if not official_points_with_leads or not consensus_points:
        return defaults

    candidate = dict(defaults)
    for target_lead_h, thresholds in _TRACK_CORRECTION_BUCKET_THRESHOLDS_KM.items():
        official_point = select_nearest_lead_point(
            official_points_with_leads,
            target_lead_h=target_lead_h,
            tolerance_h=TRACK_CORRECTION_LEAD_TOLERANCE_H,
        )
        consensus_point = select_nearest_lead_point(
            list(consensus_points),
            target_lead_h=target_lead_h,
            tolerance_h=TRACK_CORRECTION_LEAD_TOLERANCE_H,
        )
        if official_point is None or consensus_point is None:
            continue

        bias_km = _track_correction_bias_km(official_point, consensus_point)
        if bias_km is None:
            continue
        lat_bias_km, lon_bias_east_km = bias_km
        candidate[f"lat_bias_vs_consensus_{target_lead_h}h_bucket"] = _track_correction_bucket(
            lat_bias_km,
            near_threshold_km=thresholds["near"],
            large_threshold_km=thresholds["large"],
            negative_prefix="south",
            positive_prefix="north",
        )
        candidate[f"lon_bias_vs_consensus_{target_lead_h}h_bucket"] = _track_correction_bucket(
            lon_bias_east_km,
            near_threshold_km=thresholds["near"],
            large_threshold_km=thresholds["large"],
            negative_prefix="west",
            positive_prefix="east",
        )
    return candidate


def _derive_track_inflection_candidate(
    issue_time: Any,
    track_points: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    analyzed = analyze_track_inflection(
        _official_track_points_with_leads(issue_time, track_points),
        lat_key="lat",
        lon_key="lon",
        lead_key="lead_from_issue_h",
    )
    if analyzed is None:
        return {
            "steering_regime_phase": None,
            "turn_timing_bucket": None,
            "turn_direction_family": None,
            "turn_magnitude_bucket": None,
        }
    return {
        "steering_regime_phase": analyzed.get("steering_regime_phase"),
        "turn_timing_bucket": analyzed.get("turn_timing_bucket"),
        "turn_direction_family": analyzed.get("turn_direction_family"),
        "turn_magnitude_bucket": analyzed.get("turn_magnitude_bucket"),
    }


def _infer_intensity_support_signal(env_features: Dict[str, Any]) -> Optional[str]:
    shear_level = _normalize_level((env_features.get("vertical_wind_shear") or {}).get("level"))
    divergence_level = _normalize_level((env_features.get("upper_level_divergence") or {}).get("level"))
    ocean_level = _normalize_level((env_features.get("ocean_heat_content_or_sst") or {}).get("level"))

    good = 0.0
    bad = 0.0

    if shear_level in {"weak"}:
        good += 2.0
    elif shear_level in {"moderate"}:
        bad += 0.5
    elif shear_level in {"strong", "very_strong"}:
        bad += 2.0

    if divergence_level in {"strong", "moderate"}:
        good += 1.0
    elif divergence_level in {"weak", "negative"}:
        bad += 1.0

    if ocean_level in {"high", "very_high"}:
        good += 2.0
    elif ocean_level in {"moderate"}:
        good += 0.5
    elif ocean_level in {"low"}:
        bad += 2.0

    if good == 0.0 and bad == 0.0:
        return None
    if good - bad >= 1.5:
        return "supportive"
    if bad - good >= 1.5:
        return "constraining"
    return "mixed"


def _infer_land_interaction_level(
    risk_text: Optional[str],
    reasoning_text: str,
) -> Optional[str]:
    source_text = " ".join(filter(None, [normalize_text(risk_text), reasoning_text])).lower()
    if not source_text:
        return None
    if any(token in source_text for token in ("landfall", "inland", "over land")):
        return "high"
    if any(token in source_text for token in ("coast", "coastal", "near land", "land interaction")):
        return "moderate"
    return None


def _infer_main_uncertainty_source(
    reasoning_text: str,
    model_agreement_level: Optional[str],
) -> Optional[str]:
    reasoning_lower = reasoning_text.lower()
    if any(token in reasoning_lower for token in ("ensemble", "spread", "deterministic runs", "uncertain")):
        return "model_spread"
    if "trough" in reasoning_lower:
        return "midlatitude_trough_interaction"
    if "ridge" in reasoning_lower or "subtropical" in reasoning_lower:
        return "ridge_evolution"
    if "shear" in reasoning_lower:
        return "vertical_wind_shear"
    if any(token in reasoning_lower for token in ("land", "coast")):
        return "land_interaction"
    if any(token in reasoning_lower for token in ("intensif", "weaken")):
        return "intensity_change"
    if model_agreement_level == "low":
        return "model_spread"
    return None


def _infer_forecast_confidence(
    reasoning_text: str,
    model_agreement_level: Optional[str],
) -> Optional[str]:
    reasoning_lower = reasoning_text.lower()
    if any(token in reasoning_lower for token in ("very low confidence", "low confidence", "highly uncertain")):
        return "low"
    if any(token in reasoning_lower for token in ("high confidence", "good confidence")):
        return "high"
    if model_agreement_level == "high":
        return "high"
    if model_agreement_level == "medium":
        return "medium"
    if model_agreement_level == "low":
        return "low"
    return None


def derive_diagnostics(
    env_features: Dict[str, Any],
    atcf_guidance: Dict[str, Any],
    track_points: Sequence[Dict[str, Any]],
    reasoning_sections: Dict[str, Any],
    risk_text: Optional[str],
) -> Dict[str, Any]:
    """Build structured diagnostic signals from current inputs and source text."""
    forecast_reasoning = normalize_text(reasoning_sections.get("forecast_reasoning_text"))
    additional_context = normalize_text(reasoning_sections.get("additional_context_text"))
    reasoning_blob = " ".join(text for text in [forecast_reasoning, additional_context] if text)

    model_agreement_level = _classify_model_agreement(atcf_guidance)
    expert_notes = forecast_reasoning or additional_context or None

    return {
        "track_control_signal": _infer_track_control_signal(reasoning_blob, env_features),
        "turning_signal": _infer_turning_signal(track_points, reasoning_blob),
        "intensity_support_signal": _infer_intensity_support_signal(env_features),
        "shear_constraint_level": _normalize_level(
            (env_features.get("vertical_wind_shear") or {}).get("level")
        ),
        "land_interaction_level": _infer_land_interaction_level(risk_text, reasoning_blob),
        "model_agreement_level": model_agreement_level,
        "main_uncertainty_source": _infer_main_uncertainty_source(
            reasoning_blob,
            model_agreement_level,
        ),
        "forecast_confidence_level": _infer_forecast_confidence(
            reasoning_blob,
            model_agreement_level,
        ),
        "expert_decision_notes": expert_notes,
    }


def rederive_diagnostics_from_canonical(record: Dict[str, Any]) -> Dict[str, Any]:
    """Recompute diagnostics from canonical evidence without requiring legacy raw files."""
    inputs = record.get("inputs", {}) or {}
    targets = record.get("targets", {}) or {}
    environment_context = inputs.get("environment_context", {}) or {}
    model_guidance = inputs.get("model_guidance", {}) or {}
    official_track_points = targets.get("official_forecast_points", []) or []

    diagnostics = derive_diagnostics(
        (environment_context.get("current_environment", {}) or {}).get("features", {}) or {},
        model_guidance.get("atcf_consensus", {}) or {},
        official_track_points,
        targets.get("reasoning_sections", {}) or {},
        format_risk_messages(targets.get("risk_messages", {}) or {}),
    )
    diagnostics.update(
        _derive_track_inflection_candidate(
            record.get("issue_time"),
            official_track_points,
        )
    )
    diagnostics.update(
        _derive_track_correction_candidate(
            record.get("issue_time"),
            (model_guidance.get("atcf_consensus", {}) or {}).get("consensus_spread_points_future", []) or [],
            official_track_points,
        )
    )
    diagnostics.update(
        derive_slot_correction_payload(
            record.get("issue_time"),
            official_track_points,
            (model_guidance.get("atcf_consensus", {}) or {}).get("consensus_spread_points_future", []) or [],
        )
    )
    return diagnostics


def _verification_target_block(
    verification_targets: Dict[str, Any],
) -> Dict[str, Any]:
    near_issue = (verification_targets.get("best_track_point_near_issue") or {}).get("value")
    future_points = (
        (verification_targets.get("future_best_track_series") or {}).get("points_future")
        or []
    )
    return {
        "best_track_point_near_issue": near_issue,
        "future_best_track_points": future_points,
        "track": [
            {
                "lead_from_issue_h": point.get("lead_from_issue_h"),
                "valid_time_utc": point.get("valid_time_utc"),
                "lat": point.get("lat"),
                "lon": point.get("lon"),
                "storm_phase": point.get("storm_phase"),
                "source_used": point.get("source_used"),
            }
            for point in future_points
        ],
        "intensity": [
            {
                "lead_from_issue_h": point.get("lead_from_issue_h"),
                "valid_time_utc": point.get("valid_time_utc"),
                "vmax_kt": point.get("vmax_kt"),
                "min_pressure_mb": point.get("min_pressure_mb"),
                "storm_phase": point.get("storm_phase"),
                "source_used": point.get("source_used"),
            }
            for point in future_points
        ],
    }


def _build_quality_flags(
    raw_quality_flags: Dict[str, Any],
    leakage_checks: Dict[str, Any],
    has_diagnostics: bool,
) -> List[str]:
    flags: List[str] = []

    observation_status = normalize_text(raw_quality_flags.get("observation_status")) or None
    if observation_status == "missing_real_data":
        flags.append("missing_observation_evidence")
    elif observation_status == "partial_available":
        flags.append("partial_observation_evidence")

    if raw_quality_flags.get("guidance_qc_pass") in {0, False}:
        flags.append("guidance_qc_failed")
    if not raw_quality_flags.get("atcf_guidance_available", False):
        flags.append("missing_atcf_guidance")
    if not raw_quality_flags.get("verification_available", False):
        flags.append("missing_verification")

    for check_name, check_value in (leakage_checks or {}).items():
        if check_name == "future_guidance_points_only" and not check_value:
            flags.append("future_guidance_filter_failed")
        elif check_name == "ofcl_not_leaked_into_guidance" and not check_value:
            flags.append("official_forecast_leaked_into_guidance")
        elif check_name.endswith("_in_prompt") and bool(check_value):
            flags.append(f"prompt_leakage:{check_name}")

    if has_diagnostics:
        flags.append("diagnostics_available")
        flags.append(f"diagnostics_source:{DIAGNOSTIC_DERIVATION_VERSION}")

    return sorted(set(flags))


def canonicalize_legacy_sample(
    sample: Dict[str, Any],
    source_split: str,
) -> Dict[str, Any]:
    """Convert one legacy raw sample into canonical v2."""
    prompt = sample.get("prompt", {}) or {}
    target = sample.get("target", {}) or {}
    verification_targets = sample.get("verification_targets", {}) or {}
    official_outputs = target.get("official_outputs", {}) or {}
    storm_meta = prompt.get("storm_meta", {}) or {}
    now_inputs = prompt.get("now_inputs", {}) or {}
    guidance_inputs = prompt.get("guidance_inputs", {}) or {}
    observation_block = now_inputs.get("observation_evidence_structured", {}) or {}
    environment_block = now_inputs.get("environment_now_ec_reanalysis", {}) or {}
    pre_issue_context = now_inputs.get("pre_issue_guidance_context", {}) or {}
    pre_issue_track = (
        pre_issue_context.get("ec_hres_latest_point_at_or_before_issue_track") or {}
    )
    pre_issue_environment = (
        pre_issue_context.get("ec_hres_latest_point_at_or_before_issue_environment") or {}
    )
    hres_guidance = guidance_inputs.get("ec_single_model_guidance_hres", {}) or {}
    atcf_guidance = guidance_inputs.get("multimodel_guidance_a_deck", {}) or {}
    guidance_time_reference = guidance_inputs.get("guidance_time_reference", {}) or {}
    track_points = (
        (official_outputs.get("track_intensity_table") or {}).get("from_forecast_advisory")
        or []
    )
    reasoning_sections = (
        (official_outputs.get("reasoning_text") or {}).get("sections") or {}
    )
    issue_time = normalize_text(
        storm_meta.get("issue_time_utc") or guidance_time_reference.get("issue_time_utc")
    ) or None
    issue_dt = parse_utc_datetime(issue_time)
    lead_times = compute_lead_times(issue_dt, track_points)
    official_forecast_table = format_track_table(track_points) if track_points else None
    forecast_parseable = bool(
        official_forecast_table
        and analyze_assistant_schema(official_forecast_table).get("strict_forecast_parseable")
    )
    risk_messages = official_outputs.get("risk_messages", {}) or {}
    risk_text = normalize_text(format_risk_messages(risk_messages)) or None

    reasoning_parts = []
    for label, key in [
        ("Current analysis", "current_analysis_text"),
        ("Forecast reasoning", "forecast_reasoning_text"),
        ("Additional context", "additional_context_text"),
    ]:
        text = normalize_text(reasoning_sections.get(key))
        if text:
            reasoning_parts.append(f"{label}: {text}")
    reasoning_text = "\n".join(reasoning_parts) or None

    diagnostics = derive_diagnostics(
        env_features=environment_block.get("features", {}) or {},
        atcf_guidance=atcf_guidance,
        track_points=track_points,
        reasoning_sections=reasoning_sections,
        risk_text=risk_text,
    )
    has_diagnostics = any(value not in (None, "", [], {}) for value in diagnostics.values())

    issue_time_anchor_present = issue_dt is not None
    track_time_labels_present = len(lead_times) == len(track_points) if track_points else False
    guidance_cycle = normalize_text(
        guidance_time_reference.get("model_init_time_utc")
        or hres_guidance.get("init_time_utc")
    )
    time_anchor_complete = bool(issue_time_anchor_present and track_time_labels_present and guidance_cycle)

    leakage_checks = (sample.get("leakage_audit", {}) or {}).get("checks", {}) or {}
    raw_quality_flags = sample.get("quality_flags", {}) or {}
    quality_flags = _build_quality_flags(
        raw_quality_flags=raw_quality_flags,
        leakage_checks=leakage_checks,
        has_diagnostics=has_diagnostics,
    )

    canonical = {
        "sample_id": sample.get("sample_id", ""),
        "storm_id": normalize_text(storm_meta.get("storm_id")) or None,
        "basin": normalize_text(storm_meta.get("basin")) or None,
        "issue_time": issue_time,
        "lead_times": lead_times,
        "source_split": source_split,
        "time_anchor_complete": time_anchor_complete,
        "input_window_spec": build_input_window_spec(
            issue_time=issue_time,
            observation_block=observation_block,
            environment_block=environment_block,
            guidance_time_reference=guidance_time_reference,
            pre_issue_track=pre_issue_track,
            pre_issue_environment=pre_issue_environment,
        ),
        "inputs": {
            "observation_context": {
                "current_state": now_inputs.get("current_state_from_noaa_forecast_advisory", {}) or {},
                "structured_observations": observation_block,
            },
            "environment_context": {
                "current_environment": environment_block,
                "pre_issue_environment_guidance": pre_issue_environment,
            },
            "model_guidance": {
                "guidance_time_reference": guidance_time_reference,
                "hres_deterministic": hres_guidance,
                "atcf_consensus": atcf_guidance,
            },
            "historical_track_context": {
                "status": "limited_pre_issue_guidance_only",
                "historical_best_track_points": [],
                "pre_issue_track_guidance": pre_issue_track,
            },
        },
        "targets": {
            "official_forecast_table": official_forecast_table,
            "official_forecast_points": track_points,
            "forecast_parseable": forecast_parseable,
            "verification_target": _verification_target_block(verification_targets),
            "reasoning_text": reasoning_text,
            "reasoning_sections": {
                "current_analysis_text": normalize_text(reasoning_sections.get("current_analysis_text")) or None,
                "forecast_reasoning_text": normalize_text(reasoning_sections.get("forecast_reasoning_text")) or None,
                "additional_context_text": normalize_text(reasoning_sections.get("additional_context_text")) or None,
            },
            "risk_text": risk_text,
            "risk_messages": risk_messages,
        },
        "diagnostics": diagnostics,
        "flags": {
            "has_forecast": bool(official_forecast_table),
            "has_reasoning": bool(reasoning_text),
            "has_risk": bool(risk_text),
            "has_diagnostics": has_diagnostics,
            "forecast_view_eligible": bool(official_forecast_table and forecast_parseable and time_anchor_complete),
            "diagnostic_view_eligible": bool(has_diagnostics and time_anchor_complete),
            "reasoning_view_eligible": bool(reasoning_text and official_forecast_table and time_anchor_complete),
        },
        "metadata": {
            "raw_source_ids": [
                value
                for value in (sample.get("source_trace", {}) or {}).values()
                if normalize_text(value)
            ],
            "parser_version": sample.get("schema_version") or None,
            "canonical_version": CANONICAL_VERSION,
            "diagnostic_derivation_version": DIAGNOSTIC_DERIVATION_VERSION,
            "quality_flags": quality_flags,
            "raw_quality_flags": raw_quality_flags,
            "source_trace": sample.get("source_trace", {}) or {},
            "leakage_checks": leakage_checks,
            "legacy_leakage_audit": sample.get("leakage_audit", {}) or {},
            "storm_name": normalize_text(storm_meta.get("storm_name")) or None,
            "advisory_no": storm_meta.get("advisory_no"),
            "design_policy": (sample.get("build_info", {}) or {}).get("design_policy"),
        },
    }
    return canonical


def legacy_sample_from_canonical(record: Dict[str, Any]) -> Dict[str, Any]:
    """Reconstruct a legacy raw-sample view from canonical v2."""
    inputs = record.get("inputs", {}) or {}
    targets = record.get("targets", {}) or {}
    metadata = record.get("metadata", {}) or {}
    guidance = inputs.get("model_guidance", {}) or {}
    observation_context = inputs.get("observation_context", {}) or {}
    environment_context = inputs.get("environment_context", {}) or {}
    historical_track = inputs.get("historical_track_context", {}) or {}
    verification_target = targets.get("verification_target", {}) or {}

    return {
        "sample_id": record.get("sample_id", ""),
        "schema_version": metadata.get("parser_version"),
        "prompt": {
            "storm_meta": {
                "storm_id": record.get("storm_id"),
                "storm_name": metadata.get("storm_name"),
                "basin": record.get("basin"),
                "issue_time_utc": record.get("issue_time"),
                "advisory_no": metadata.get("advisory_no"),
                "time_match_rule": "legacy_roundtrip_from_canonical_v2",
            },
            "now_inputs": {
                "current_state_from_noaa_forecast_advisory": observation_context.get("current_state", {}) or {},
                "environment_now_ec_reanalysis": environment_context.get("current_environment", {}) or {},
                "observation_evidence_structured": observation_context.get("structured_observations", {}) or {},
                "pre_issue_guidance_context": {
                    "ec_hres_latest_point_at_or_before_issue_track": historical_track.get("pre_issue_track_guidance", {}) or {},
                    "ec_hres_latest_point_at_or_before_issue_environment": environment_context.get("pre_issue_environment_guidance", {}) or {},
                },
            },
            "guidance_inputs": {
                "guidance_time_reference": guidance.get("guidance_time_reference", {}) or {},
                "ec_single_model_guidance_hres": guidance.get("hres_deterministic", {}) or {},
                "multimodel_guidance_a_deck": guidance.get("atcf_consensus", {}) or {},
            },
        },
        "target": {
            "official_outputs": {
                "track_intensity_table": {
                    "from_forecast_advisory": targets.get("official_forecast_points", []) or [],
                },
                "risk_messages": targets.get("risk_messages", {}) or {},
                "reasoning_text": {
                    "sections": targets.get("reasoning_sections", {}) or {},
                },
            }
        },
        "verification_targets": {
            "best_track_point_near_issue": {
                "value": verification_target.get("best_track_point_near_issue"),
            },
            "future_best_track_series": {
                "points_future": verification_target.get("future_best_track_points", []) or [],
            },
        },
        "quality_flags": metadata.get("raw_quality_flags", {}) or {},
        "leakage_audit": metadata.get("legacy_leakage_audit", {}) or {},
        "source_trace": metadata.get("source_trace", {}) or {},
    }


def build_canonical_json_schema() -> Dict[str, Any]:
    """Return a compact JSON Schema for canonical v2 records."""
    nullable_string = {"type": ["string", "null"]}
    nullable_boolean = {"type": ["boolean", "null"]}
    nullable_object = {"type": ["object", "null"]}
    nullable_array = {"type": ["array", "null"]}

    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "Tropical Cyclone Canonical Dataset v2",
        "type": "object",
        "required": [
            "sample_id",
            "storm_id",
            "basin",
            "issue_time",
            "lead_times",
            "source_split",
            "time_anchor_complete",
            "input_window_spec",
            "inputs",
            "targets",
            "diagnostics",
            "flags",
            "metadata",
        ],
        "properties": {
            "sample_id": {"type": "string"},
            "storm_id": nullable_string,
            "basin": nullable_string,
            "issue_time": nullable_string,
            "lead_times": {"type": "array", "items": {"type": "integer"}},
            "source_split": {"type": "string"},
            "time_anchor_complete": {"type": "boolean"},
            "input_window_spec": {
                "type": "object",
                "properties": {
                    "obs_start": nullable_string,
                    "obs_end": nullable_string,
                    "env_start": nullable_string,
                    "env_end": nullable_string,
                    "guidance_cycle": nullable_string,
                },
            },
            "inputs": {
                "type": "object",
                "properties": {
                    "observation_context": nullable_object,
                    "environment_context": nullable_object,
                    "model_guidance": nullable_object,
                    "historical_track_context": nullable_object,
                },
            },
            "targets": {
                "type": "object",
                "properties": {
                    "official_forecast_table": nullable_string,
                    "official_forecast_points": nullable_array,
                    "forecast_parseable": nullable_boolean,
                    "verification_target": nullable_object,
                    "reasoning_text": nullable_string,
                    "reasoning_sections": nullable_object,
                    "risk_text": nullable_string,
                    "risk_messages": nullable_object,
                },
            },
            "diagnostics": {
                "type": "object",
                "properties": {
                    key: nullable_string
                    for key in DIAGNOSTIC_FIELDS
                },
            },
            "flags": {
                "type": "object",
                "properties": {
                    "has_forecast": {"type": "boolean"},
                    "has_reasoning": {"type": "boolean"},
                    "has_risk": {"type": "boolean"},
                    "has_diagnostics": {"type": "boolean"},
                    "forecast_view_eligible": {"type": "boolean"},
                    "diagnostic_view_eligible": {"type": "boolean"},
                    "reasoning_view_eligible": {"type": "boolean"},
                },
            },
            "metadata": {
                "type": "object",
                "properties": {
                    "raw_source_ids": {"type": "array", "items": {"type": "string"}},
                    "parser_version": nullable_string,
                    "canonical_version": {"type": "string"},
                    "diagnostic_derivation_version": {"type": "string"},
                    "quality_flags": {"type": "array", "items": {"type": "string"}},
                    "raw_quality_flags": nullable_object,
                    "source_trace": nullable_object,
                    "leakage_checks": nullable_object,
                    "legacy_leakage_audit": nullable_object,
                    "storm_name": nullable_string,
                    "advisory_no": {"type": ["integer", "null"]},
                    "design_policy": nullable_string,
                },
            },
        },
    }
