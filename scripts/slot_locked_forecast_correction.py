#!/usr/bin/env python3
"""Helpers for slot-locked forecast correction relative to visible ATCF guidance."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from statistics import median
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import dataset_formatter as df
from cyclone_training.rewards import parse_forecast_points


MAX_VISIBLE_ATCF_SLOTS = 6
LAT_CORRECTION_BUCKET_VALUES = (
    "south_large",
    "south_small",
    "near_consensus",
    "north_small",
    "north_large",
)
LON_CORRECTION_BUCKET_VALUES = (
    "east_large",
    "east_small",
    "near_consensus",
    "west_small",
    "west_large",
)
TURNING_SIGNAL_VALUES = (
    "steady",
    "notable_turn",
    "recurvature",
)
SLOT_CORRECTION_FIELD_NAMES = tuple(
    field_name
    for slot_index in range(1, MAX_VISIBLE_ATCF_SLOTS + 1)
    for field_name in (
        f"slot_{slot_index}_lat_bias_vs_consensus_bucket",
        f"slot_{slot_index}_lon_bias_vs_consensus_bucket",
    )
)
SLOT_CORRECTION_FIELD_CANONICAL_VALUES: Dict[str, Tuple[str, ...]] = {
    f"slot_{slot_index}_lat_bias_vs_consensus_bucket": LAT_CORRECTION_BUCKET_VALUES
    for slot_index in range(1, MAX_VISIBLE_ATCF_SLOTS + 1)
}
SLOT_CORRECTION_FIELD_CANONICAL_VALUES.update(
    {
        f"slot_{slot_index}_lon_bias_vs_consensus_bucket": LON_CORRECTION_BUCKET_VALUES
        for slot_index in range(1, MAX_VISIBLE_ATCF_SLOTS + 1)
    }
)
SLOT_CORRECTION_FIELD_VALUE_ALIASES: Dict[str, Dict[str, str]] = {
    f"slot_{slot_index}_lat_bias_vs_consensus_bucket": {
        "south_large": "south_large",
        "south_big": "south_large",
        "south_small": "south_small",
        "near_consensus": "near_consensus",
        "near": "near_consensus",
        "aligned_with_consensus": "near_consensus",
        "north_small": "north_small",
        "north_large": "north_large",
        "north_big": "north_large",
    }
    for slot_index in range(1, MAX_VISIBLE_ATCF_SLOTS + 1)
}
SLOT_CORRECTION_FIELD_VALUE_ALIASES.update(
    {
        f"slot_{slot_index}_lon_bias_vs_consensus_bucket": {
            "east_large": "east_large",
            "east_big": "east_large",
            "east_small": "east_small",
            "near_consensus": "near_consensus",
            "near": "near_consensus",
            "aligned_with_consensus": "near_consensus",
            "west_small": "west_small",
            "west_large": "west_large",
            "west_big": "west_large",
        }
        for slot_index in range(1, MAX_VISIBLE_ATCF_SLOTS + 1)
    }
)

# Frozen from train-split q25/q75 audit against visible ATCF representative slots.
SLOT_CORRECTION_BUCKET_THRESHOLDS_KM: Dict[int, Dict[str, Dict[str, float]]] = {
    1: {
        "lat": {"near": 68.2, "large": 133.7},
        "lon": {"near": 134.6, "large": 264.1},
    },
    2: {
        "lat": {"near": 68.8, "large": 142.9},
        "lon": {"near": 130.0, "large": 263.3},
    },
    3: {
        "lat": {"near": 65.8, "large": 147.9},
        "lon": {"near": 123.8, "large": 252.4},
    },
    4: {
        "lat": {"near": 65.5, "large": 156.0},
        "lon": {"near": 116.5, "large": 252.2},
    },
    5: {
        "lat": {"near": 62.6, "large": 170.2},
        "lon": {"near": 107.6, "large": 263.6},
    },
    6: {
        "lat": {"near": 47.4, "large": 154.8},
        "lon": {"near": 72.2, "large": 217.3},
    },
}


@dataclass(frozen=True)
class SlotCorrectionPair:
    slot_index: int
    official_point: Dict[str, Any]
    consensus_point: Dict[str, Any]


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


def _slot_field_name(slot_index: int, axis: str) -> str:
    return f"slot_{slot_index}_{axis}_bias_vs_consensus_bucket"


def _bucket_allowed_values(axis: str) -> Tuple[str, ...]:
    return LAT_CORRECTION_BUCKET_VALUES if axis == "lat" else LON_CORRECTION_BUCKET_VALUES


def _normalize_bucket_label(axis: str, value: Any) -> str:
    label = str(value or "near_consensus")
    return label if label in _bucket_allowed_values(axis) else "near_consensus"


def _normalize_turning_signal(value: Any) -> Optional[str]:
    label = str(value or "").strip().lower()
    if label in TURNING_SIGNAL_VALUES:
        return label
    return None


def _record_turning_signal(record: Dict[str, Any]) -> Optional[str]:
    diagnostics = (record.get("diagnostics") or {}) if isinstance(record, dict) else {}
    return _normalize_turning_signal(diagnostics.get("turning_signal"))


def visible_consensus_slot_points(
    issue_time: Any,
    official_points: Sequence[Dict[str, Any]],
    atcf_consensus_points: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Match visible ATCF points to official forecast slots using the prompt export logic."""
    if not official_points or not atcf_consensus_points:
        return []
    target_leads = df.derive_guidance_target_leads(list(official_points), issue_time)
    selected = df.select_representative_guidance_points(
        list(atcf_consensus_points),
        target_leads=target_leads,
    )
    return list(selected[:MAX_VISIBLE_ATCF_SLOTS])


def iter_slot_correction_pairs(
    issue_time: Any,
    official_points: Sequence[Dict[str, Any]],
    atcf_consensus_points: Sequence[Dict[str, Any]],
) -> List[SlotCorrectionPair]:
    selected_consensus_points = visible_consensus_slot_points(
        issue_time,
        official_points,
        atcf_consensus_points,
    )
    pair_count = min(len(official_points), len(selected_consensus_points), MAX_VISIBLE_ATCF_SLOTS)
    return [
        SlotCorrectionPair(
            slot_index=slot_index,
            official_point=dict(official_points[slot_index - 1]),
            consensus_point=dict(selected_consensus_points[slot_index - 1]),
        )
        for slot_index in range(1, pair_count + 1)
    ]


def slot_bias_km(
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


def slot_correction_bucket(
    delta_km: float,
    *,
    slot_index: int,
    axis: str,
) -> str:
    thresholds = SLOT_CORRECTION_BUCKET_THRESHOLDS_KM[slot_index][axis]
    abs_delta_km = abs(delta_km)
    if abs_delta_km < thresholds["near"]:
        return "near_consensus"
    magnitude = "large" if abs_delta_km >= thresholds["large"] else "small"
    if axis == "lat":
        direction = "north" if delta_km >= 0.0 else "south"
    else:
        direction = "east" if delta_km >= 0.0 else "west"
    return f"{direction}_{magnitude}"


def derive_slot_correction_payload(
    issue_time: Any,
    official_points: Sequence[Dict[str, Any]],
    atcf_consensus_points: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    payload = {field_name: None for field_name in SLOT_CORRECTION_FIELD_NAMES}
    for pair in iter_slot_correction_pairs(issue_time, official_points, atcf_consensus_points):
        bias = slot_bias_km(pair.official_point, pair.consensus_point)
        if bias is None:
            continue
        lat_bias_km, lon_bias_east_km = bias
        payload[_slot_field_name(pair.slot_index, "lat")] = slot_correction_bucket(
            lat_bias_km,
            slot_index=pair.slot_index,
            axis="lat",
        )
        payload[_slot_field_name(pair.slot_index, "lon")] = slot_correction_bucket(
            lon_bias_east_km,
            slot_index=pair.slot_index,
            axis="lon",
        )
    return payload


def _fallback_bucket_offset_km(
    *,
    slot_index: int,
    axis: str,
    label: str,
) -> float:
    thresholds = SLOT_CORRECTION_BUCKET_THRESHOLDS_KM[slot_index][axis]
    near = thresholds["near"]
    large = thresholds["large"]
    small_mag = (near + large) / 2.0
    large_mag = large + max(20.0, (large - near) / 2.0)
    if label == "near_consensus":
        return 0.0
    if label.endswith("_small"):
        magnitude = small_mag
    else:
        magnitude = large_mag

    if axis == "lat":
        return magnitude if label.startswith("north") else -magnitude
    return magnitude if label.startswith("east") else -magnitude


def build_slot_correction_calibration(
    records: Iterable[Dict[str, Any]],
) -> Dict[str, Any]:
    """Build per-slot bucket medians used by the deterministic renderer."""
    bucket_offsets: Dict[Tuple[int, str, str], List[float]] = defaultdict(list)
    turning_bucket_offsets: Dict[Tuple[str, int, str, str], List[float]] = defaultdict(list)
    lat_all: List[float] = []
    lon_all: List[float] = []
    for record in records:
        official_points = (record.get("targets", {}) or {}).get("official_forecast_points", []) or []
        consensus_points = (
            ((record.get("inputs", {}) or {}).get("model_guidance", {}) or {})
            .get("atcf_consensus", {})
            .get("consensus_spread_points_future", [])
            or []
        )
        for pair in iter_slot_correction_pairs(record.get("issue_time"), official_points, consensus_points):
            bias = slot_bias_km(pair.official_point, pair.consensus_point)
            if bias is None:
                continue
            lat_bias_km, lon_bias_east_km = bias
            lat_all.append(lat_bias_km)
            lon_all.append(lon_bias_east_km)
            lat_label = slot_correction_bucket(lat_bias_km, slot_index=pair.slot_index, axis="lat")
            lon_label = slot_correction_bucket(lon_bias_east_km, slot_index=pair.slot_index, axis="lon")
            bucket_offsets[(pair.slot_index, "lat", lat_label)].append(lat_bias_km)
            bucket_offsets[(pair.slot_index, "lon", lon_label)].append(lon_bias_east_km)
            turning_label = _record_turning_signal(record)
            if turning_label is not None:
                turning_bucket_offsets[(turning_label, pair.slot_index, "lat", lat_label)].append(
                    lat_bias_km
                )
                turning_bucket_offsets[(turning_label, pair.slot_index, "lon", lon_label)].append(
                    lon_bias_east_km
                )

    slot_bucket_offsets_km: Dict[str, Dict[str, Dict[str, float]]] = {}
    for slot_index in range(1, MAX_VISIBLE_ATCF_SLOTS + 1):
        slot_bucket_offsets_km[str(slot_index)] = {}
        for axis in ("lat", "lon"):
            allowed = _bucket_allowed_values(axis)
            slot_bucket_offsets_km[str(slot_index)][axis] = {
                label: (
                    median(bucket_offsets[(slot_index, axis, label)])
                    if bucket_offsets[(slot_index, axis, label)]
                    else _fallback_bucket_offset_km(
                        slot_index=slot_index,
                        axis=axis,
                        label=label,
                    )
                )
                for label in allowed
            }

    turning_slot_bucket_offsets_km: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {}
    for turning_label in TURNING_SIGNAL_VALUES:
        turning_slot_bucket_offsets_km[turning_label] = {}
        for slot_index in range(1, MAX_VISIBLE_ATCF_SLOTS + 1):
            turning_slot_bucket_offsets_km[turning_label][str(slot_index)] = {}
            for axis in ("lat", "lon"):
                turning_slot_bucket_offsets_km[turning_label][str(slot_index)][axis] = {}
                for label in _bucket_allowed_values(axis):
                    values = turning_bucket_offsets[(turning_label, slot_index, axis, label)]
                    turning_slot_bucket_offsets_km[turning_label][str(slot_index)][axis][label] = (
                        median(values)
                        if values
                        else slot_bucket_offsets_km[str(slot_index)][axis][label]
                    )

    calibration: Dict[str, Any] = {
        "global_median_offset_km": {
            "lat": median(lat_all) if lat_all else 0.0,
            "lon": median(lon_all) if lon_all else 0.0,
        },
        "slot_bucket_offsets_km": slot_bucket_offsets_km,
        "turning_slot_bucket_offsets_km": turning_slot_bucket_offsets_km,
    }
    return calibration


def _apply_lat_offset_km(base_lat: float, offset_km: float) -> float:
    return base_lat + (offset_km / 111.32)


def _apply_lon_offset_km(base_lon: float, *, reference_lat: float, east_km: float) -> float:
    cos_lat = max(0.2, math.cos(math.radians(reference_lat)))
    return base_lon + east_km / (111.32 * cos_lat)


def render_visible_consensus_forecast_text(record: Dict[str, Any]) -> str:
    official_points = (record.get("targets", {}) or {}).get("official_forecast_points", []) or []
    consensus_points = (
        ((record.get("inputs", {}) or {}).get("model_guidance", {}) or {})
        .get("atcf_consensus", {})
        .get("consensus_spread_points_future", [])
        or []
    )
    rendered_points: List[Dict[str, Any]] = []
    for pair in iter_slot_correction_pairs(record.get("issue_time"), official_points, consensus_points):
        base_lat = _safe_float(pair.consensus_point.get("consensus_lat"))
        base_lon = _safe_float(pair.consensus_point.get("consensus_lon"))
        if base_lat is None or base_lon is None:
            continue
        rendered_points.append(
            {
                "valid_day": pair.official_point.get("valid_day"),
                "valid_hhmmz": pair.official_point.get("valid_hhmmz"),
                "lat": base_lat,
                "lon": base_lon,
                "vmax_kt": round(_safe_float(pair.consensus_point.get("consensus_vmax_kt")) or 0.0),
            }
        )
    return df.format_track_table(rendered_points)


def render_slot_correction_forecast_text(
    record: Dict[str, Any],
    correction_payload: Dict[str, Any],
    *,
    calibration: Dict[str, Any],
    intensity_source: str = "consensus",
    intensity_reference_text: str | None = None,
    offset_scale: float = 1.0,
) -> str:
    """Render one strict forecast table by correcting visible ATCF points slot-by-slot."""
    official_points = (record.get("targets", {}) or {}).get("official_forecast_points", []) or []
    consensus_points = (
        ((record.get("inputs", {}) or {}).get("model_guidance", {}) or {})
        .get("atcf_consensus", {})
        .get("consensus_spread_points_future", [])
        or []
    )
    rendered_points: List[Dict[str, Any]] = []
    slot_offsets = calibration.get("slot_bucket_offsets_km", {}) or {}
    turning_slot_offsets = calibration.get("turning_slot_bucket_offsets_km", {}) or {}
    turning_label = _normalize_turning_signal(correction_payload.get("turning_signal"))
    issue_time = record.get("issue_time")
    baseline_intensity_by_time: Dict[str, float] = {}
    if intensity_source == "baseline_forecast" and intensity_reference_text:
        issue_dt = df.parse_utc_datetime(issue_time)
        if issue_dt is not None:
            for point in parse_forecast_points(intensity_reference_text, issue_dt):
                baseline_intensity_by_time[
                    point.valid_time.isoformat().replace("+00:00", "Z")
                ] = float(point.vmax_kt)
    for pair in iter_slot_correction_pairs(record.get("issue_time"), official_points, consensus_points):
        base_lat = _safe_float(pair.consensus_point.get("consensus_lat"))
        base_lon = _safe_float(pair.consensus_point.get("consensus_lon"))
        if base_lat is None or base_lon is None:
            continue

        lat_field = _slot_field_name(pair.slot_index, "lat")
        lon_field = _slot_field_name(pair.slot_index, "lon")
        lat_label = _normalize_bucket_label("lat", correction_payload.get(lat_field))
        lon_label = _normalize_bucket_label("lon", correction_payload.get(lon_field))

        conditional_slot_offsets = (
            ((turning_slot_offsets.get(turning_label, {}) or {}).get(str(pair.slot_index), {}) or {})
            if turning_label is not None
            else {}
        )
        lat_offset_km = ((conditional_slot_offsets.get("lat", {}) or {}).get(lat_label))
        if lat_offset_km is None:
            lat_offset_km = (
                ((slot_offsets.get(str(pair.slot_index), {}) or {}).get("lat", {}) or {}).get(lat_label)
            )
        if lat_offset_km is None:
            lat_offset_km = _fallback_bucket_offset_km(
                slot_index=pair.slot_index,
                axis="lat",
                label=lat_label,
            )
        lon_offset_km = ((conditional_slot_offsets.get("lon", {}) or {}).get(lon_label))
        if lon_offset_km is None:
            lon_offset_km = (
                ((slot_offsets.get(str(pair.slot_index), {}) or {}).get("lon", {}) or {}).get(lon_label)
            )
        if lon_offset_km is None:
            lon_offset_km = _fallback_bucket_offset_km(
                slot_index=pair.slot_index,
                axis="lon",
                label=lon_label,
            )

        corrected_lat = _apply_lat_offset_km(base_lat, float(lat_offset_km) * float(offset_scale))
        corrected_lon = _apply_lon_offset_km(
            base_lon,
            reference_lat=corrected_lat,
            east_km=float(lon_offset_km) * float(offset_scale),
        )
        vmax_kt = round(_safe_float(pair.consensus_point.get("consensus_vmax_kt")) or 0.0)
        if intensity_source == "baseline_forecast":
            valid_dt = df.resolve_point_valid_datetime(pair.official_point, issue_time)
            if valid_dt is not None:
                vmax_kt = round(
                    baseline_intensity_by_time.get(
                        valid_dt.isoformat().replace("+00:00", "Z"),
                        float(vmax_kt),
                    )
                )
        rendered_points.append(
            {
                "valid_day": pair.official_point.get("valid_day"),
                "valid_hhmmz": pair.official_point.get("valid_hhmmz"),
                "lat": corrected_lat,
                "lon": corrected_lon,
                "vmax_kt": vmax_kt,
            }
        )
    return df.format_track_table(rendered_points)
