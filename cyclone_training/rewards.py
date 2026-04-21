"""Structured GRPO rewards for cyclone forecast generation."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping, Sequence

from .config import RewardRuntimeConfig


FORECAST_LINE_RE = re.compile(
    r"^-\s*Day(?P<day>\d{1,2})\s+"
    r"(?P<hh>\d{2})(?P<mm>\d{2})Z\s*\|\s*"
    r"(?P<lat>\d+(?:\.\d+)?)°(?P<lat_hemi>[NS])\s+"
    r"(?P<lon>\d+(?:\.\d+)?)°(?P<lon_hemi>[EW])\s*\|\s*"
    r"(?P<vmax>\d+(?:\.\d+)?)\s*kt\s*$",
    re.MULTILINE,
)
HEADER_LINE_RE = re.compile(r"^official forecast:?\s*$", re.IGNORECASE)


@dataclass(slots=True)
class ParsedForecastPoint:
    valid_time: datetime
    lead_from_issue_h: float
    lat: float
    lon: float
    vmax_kt: float


@dataclass(slots=True)
class ForecastSchemaStats:
    has_header: bool
    parsed_line_count: int
    invalid_line_count: int


@dataclass(slots=True)
class TargetForecastSlot:
    valid_time: datetime


@dataclass(slots=True)
class MatchedForecastSlot:
    target_slot: TargetForecastSlot
    forecast_point: ParsedForecastPoint
    truth_point: Mapping[str, Any]
    forecast_time_delta_h: float
    time_alignment_score: float


def _to_datetime(value: str) -> datetime:
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _hemi_signed(value: float, hemisphere: str) -> float:
    if hemisphere in {"S", "W"}:
        return -abs(value)
    return abs(value)


def _candidate_months(issue_time: datetime) -> list[tuple[int, int]]:
    current = issue_time.replace(day=1)
    next_month_seed = (current + timedelta(days=35)).replace(day=1)
    return [
        (current.year, current.month),
        (next_month_seed.year, next_month_seed.month),
    ]


def _resolve_valid_time(issue_time: datetime, day: int, hour: int, minute: int) -> datetime | None:
    best: datetime | None = None
    best_delta: timedelta | None = None
    for year, month in _candidate_months(issue_time):
        try:
            candidate = datetime(
                year,
                month,
                day,
                hour,
                minute,
                tzinfo=timezone.utc,
            )
        except ValueError:
            continue
        delta = candidate - issue_time
        if delta <= timedelta(0) or delta > timedelta(days=7):
            continue
        if best is None or delta < best_delta:
            best = candidate
            best_delta = delta
    return best


def completion_to_text(completion: Any) -> str:
    """Normalize TRL completions into one plain string."""
    if completion is None:
        return ""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, dict):
        if "content" in completion:
            return completion_to_text(completion["content"])
        if "text" in completion:
            return str(completion["text"])
        return ""
    if isinstance(completion, Sequence) and not isinstance(completion, (bytes, bytearray)):
        chunks = [completion_to_text(item) for item in completion]
        return "".join(chunk for chunk in chunks if chunk)
    return str(completion)


def parse_forecast_points(text: str, issue_time: datetime) -> list[ParsedForecastPoint]:
    """Parse generated forecast lines into structured points."""
    points: list[ParsedForecastPoint] = []
    for match in FORECAST_LINE_RE.finditer(text or ""):
        valid_time = _resolve_valid_time(
            issue_time=issue_time,
            day=int(match.group("day")),
            hour=int(match.group("hh")),
            minute=int(match.group("mm")),
        )
        if valid_time is None:
            continue

        lat = _hemi_signed(float(match.group("lat")), match.group("lat_hemi"))
        lon = _hemi_signed(float(match.group("lon")), match.group("lon_hemi"))
        delta_h = (valid_time - issue_time).total_seconds() / 3600.0
        points.append(
            ParsedForecastPoint(
                valid_time=valid_time,
                lead_from_issue_h=delta_h,
                lat=lat,
                lon=lon,
                vmax_kt=float(match.group("vmax")),
            )
        )
    points.sort(key=lambda item: item.valid_time)
    return points


def _is_header_line(line: str) -> bool:
    return bool(HEADER_LINE_RE.match(line.strip()))


def inspect_forecast_schema(text: str) -> ForecastSchemaStats:
    lines = [line.strip() for line in (text or "").strip().splitlines() if line.strip()]
    if not lines:
        return ForecastSchemaStats(
            has_header=False,
            parsed_line_count=0,
            invalid_line_count=0,
        )

    has_header = False
    parsed_line_count = 0
    invalid_line_count = 0
    for line in lines:
        if _is_header_line(line):
            has_header = True
            continue
        if FORECAST_LINE_RE.match(line):
            parsed_line_count += 1
        else:
            invalid_line_count += 1
    return ForecastSchemaStats(
        has_header=has_header,
        parsed_line_count=parsed_line_count,
        invalid_line_count=invalid_line_count,
    )


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute great-circle distance in kilometers."""
    radius_km = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    )
    return 2 * radius_km * math.atan2(math.sqrt(a), math.sqrt(max(1e-12, 1 - a)))


def _truth_point_time(point: Mapping[str, Any]) -> datetime:
    return _to_datetime(str(point["valid_time_utc"]))


def _resolve_slot_time(raw_slot: Mapping[str, Any], issue_time: datetime) -> datetime | None:
    valid_time_utc = raw_slot.get("valid_time_utc")
    if valid_time_utc:
        return _to_datetime(str(valid_time_utc))

    day_text = str(raw_slot.get("valid_day") or "").strip()
    hhmm_text = str(raw_slot.get("valid_hhmmz") or "").strip().upper().removesuffix("Z")
    if not day_text.isdigit() or len(hhmm_text) != 4 or not hhmm_text.isdigit():
        return None
    return _resolve_valid_time(
        issue_time=issue_time,
        day=int(day_text),
        hour=int(hhmm_text[:2]),
        minute=int(hhmm_text[2:]),
    )


def parse_target_forecast_slots(
    verification: Mapping[str, Any],
    issue_time: datetime,
) -> list[TargetForecastSlot]:
    raw_slots = verification.get("forecast_slots") or []
    slots: list[TargetForecastSlot] = []
    for raw_slot in raw_slots:
        if not isinstance(raw_slot, Mapping):
            continue
        valid_time = _resolve_slot_time(raw_slot, issue_time)
        if valid_time is None:
            continue
        slots.append(TargetForecastSlot(valid_time=valid_time))
    slots.sort(key=lambda item: item.valid_time)
    return slots


def _closest_truth_point(
    valid_time: datetime,
    truth_points: Sequence[Mapping[str, Any]],
    tolerance_hours: int,
) -> Mapping[str, Any] | None:
    best_truth: Mapping[str, Any] | None = None
    best_delta_h: float | None = None
    for truth in truth_points:
        delta_h = abs((_truth_point_time(truth) - valid_time).total_seconds()) / 3600.0
        if delta_h > tolerance_hours:
            continue
        if best_truth is None or delta_h < best_delta_h:
            best_truth = truth
            best_delta_h = delta_h
    return best_truth


def _closest_unmatched_forecast_point(
    valid_time: datetime,
    forecast_points: Sequence[ParsedForecastPoint],
    *,
    used_indices: set[int],
    tolerance_hours: int,
) -> tuple[int | None, float | None]:
    best_index: int | None = None
    best_delta_h: float | None = None
    for index, forecast_point in enumerate(forecast_points):
        if index in used_indices:
            continue
        delta_h = abs((forecast_point.valid_time - valid_time).total_seconds()) / 3600.0
        if delta_h > tolerance_hours:
            continue
        if best_index is None or delta_h < best_delta_h:
            best_index = index
            best_delta_h = delta_h
    return best_index, best_delta_h


def _soft_slot_alignment_reward(
    effective_slots: Sequence[tuple[TargetForecastSlot, Mapping[str, Any]]],
    forecast_points: Sequence[ParsedForecastPoint],
    *,
    max_hours: float | None,
    time_scale_hours: float,
    track_error_scale_km: float,
    intensity_error_scale_kt: float,
    track_error_weight: float,
    intensity_error_weight: float,
) -> float:
    if not effective_slots or not forecast_points:
        return 0.0
    if max_hours is None or max_hours <= 0:
        return 0.0

    used_indices: set[int] = set()
    time_scale_h = max(1e-6, time_scale_hours)
    track_scale_km = max(1e-6, track_error_scale_km)
    intensity_scale_kt = max(1e-6, intensity_error_scale_kt)
    component_weight = track_error_weight + intensity_error_weight
    if component_weight <= 0:
        return 0.0
    scores: list[float] = []
    for target_slot, truth_point in effective_slots:
        forecast_index, forecast_delta_h = _closest_unmatched_forecast_point(
            target_slot.valid_time,
            forecast_points,
            used_indices=used_indices,
            tolerance_hours=int(math.ceil(max_hours)),
        )
        if forecast_index is None or forecast_delta_h is None or forecast_delta_h > max_hours:
            scores.append(0.0)
            continue
        used_indices.add(forecast_index)
        forecast_point = forecast_points[forecast_index]
        time_score = math.exp(-forecast_delta_h / time_scale_h)
        track_error = haversine_km(
            forecast_point.lat,
            forecast_point.lon,
            float(truth_point["lat"]),
            float(truth_point["lon"]),
        )
        intensity_error = abs(forecast_point.vmax_kt - float(truth_point["vmax_kt"]))
        track_score = math.exp(-track_error / track_scale_km)
        intensity_score = math.exp(-intensity_error / intensity_scale_kt)
        accuracy_score = (
            track_error_weight * track_score + intensity_error_weight * intensity_score
        ) / component_weight
        scores.append(time_score * accuracy_score)
    if not scores:
        return 0.0
    return float(sum(scores) / len(scores))


def match_forecast_to_truth_slots(
    forecast_points: Sequence[ParsedForecastPoint],
    target_slots: Sequence[TargetForecastSlot],
    truth_points: Sequence[Mapping[str, Any]],
    truth_slot_tolerance_hours: int,
    forecast_slot_tolerance_hours: int,
    forecast_slot_time_scale_hours: float,
) -> tuple[
    list[tuple[TargetForecastSlot, Mapping[str, Any]]],
    list[MatchedForecastSlot],
]:
    effective_slots: list[tuple[TargetForecastSlot, Mapping[str, Any]]] = []
    matched_slots: list[MatchedForecastSlot] = []
    used_forecast_indices: set[int] = set()
    time_scale_h = max(1e-6, forecast_slot_time_scale_hours)
    for target_slot in target_slots:
        truth_point = _closest_truth_point(
            target_slot.valid_time,
            truth_points,
            tolerance_hours=truth_slot_tolerance_hours,
        )
        if truth_point is None:
            continue
        effective_slots.append((target_slot, truth_point))
        forecast_index, forecast_delta_h = _closest_unmatched_forecast_point(
            target_slot.valid_time,
            forecast_points,
            used_indices=used_forecast_indices,
            tolerance_hours=forecast_slot_tolerance_hours,
        )
        if forecast_index is None or forecast_delta_h is None:
            continue
        used_forecast_indices.add(forecast_index)
        forecast_point = forecast_points[forecast_index]
        matched_slots.append(
            MatchedForecastSlot(
                target_slot=target_slot,
                forecast_point=forecast_point,
                truth_point=truth_point,
                forecast_time_delta_h=forecast_delta_h,
                time_alignment_score=math.exp(-forecast_delta_h / time_scale_h),
            )
        )
    return effective_slots, matched_slots


def _extract_truth_issue_time(verification: Mapping[str, Any]) -> datetime:
    issue = verification.get("best_track_at_issue") or {}
    matched_time = issue.get("matched_datetime_utc")
    if not matched_time:
        future_points = verification.get("future_best_track") or []
        if not future_points:
            raise ValueError("Verification block must contain best_track_at_issue or future_best_track.")
        first_time = _truth_point_time(future_points[0])
        first_lead = float(future_points[0].get("lead_from_issue_h") or 0.0)
        return first_time - timedelta(hours=first_lead)
    return _to_datetime(str(matched_time))


class CycloneRewardFunction:
    """Deterministic structured reward for GRPO training."""

    def __init__(self, config: RewardRuntimeConfig) -> None:
        self.config = config
        self.__name__ = self.__class__.__name__

    def __call__(
        self,
        prompts: Sequence[Any] | None = None,
        completions: Sequence[Any] | None = None,
        verification: Sequence[Mapping[str, Any]] | None = None,
        **_: Any,
    ) -> list[float]:
        if completions is None:
            return []
        if verification is None:
            raise ValueError("CycloneRewardFunction requires the 'verification' dataset column.")
        rewards: list[float] = []
        for completion, verify in zip(completions, verification):
            rewards.append(self.score_one(completion_to_text(completion), verify))
        return rewards

    def _format_score(
        self,
        schema: ForecastSchemaStats,
        *,
        target_slot_count: int,
        resolved_line_count: int,
    ) -> float:
        """Return partial credit for outputs that are parseable but incomplete."""
        if resolved_line_count <= 0:
            return 0.0
        total_lines = schema.parsed_line_count + schema.invalid_line_count
        valid_ratio = resolved_line_count / max(total_lines, 1)
        if target_slot_count > 0:
            line_coverage = min(resolved_line_count, target_slot_count) / target_slot_count
        else:
            line_coverage = 1.0 if resolved_line_count > 0 else 0.0
        return float(min(self.config.format_bonus, self.config.format_bonus * valid_ratio * line_coverage))

    def score_one(
        self,
        completion_text: str,
        verification: Mapping[str, Any],
    ) -> float:
        schema = inspect_forecast_schema(completion_text)

        issue_time = _extract_truth_issue_time(verification)
        target_slots = parse_target_forecast_slots(verification, issue_time)
        if not target_slots:
            raise ValueError("CycloneRewardFunction requires verification.forecast_slots.")
        forecast_points = parse_forecast_points(completion_text, issue_time)
        partial = self._format_score(
            schema,
            target_slot_count=len(target_slots),
            resolved_line_count=len(forecast_points),
        )

        # No time-resolvable forecast lines — no reward.
        if not forecast_points:
            return partial

        truth_points = list(verification.get("future_best_track") or [])
        effective_slots, matched_slots = match_forecast_to_truth_slots(
            forecast_points,
            target_slots,
            truth_points,
            truth_slot_tolerance_hours=self.config.truth_slot_tolerance_hours,
            forecast_slot_tolerance_hours=self.config.forecast_slot_tolerance_hours,
            forecast_slot_time_scale_hours=self.config.forecast_slot_time_scale_hours,
        )

        soft_slot_reward = 0.0
        if effective_slots and self.config.soft_slot_reward_weight > 0.0:
            soft_slot_alignment = _soft_slot_alignment_reward(
                effective_slots,
                forecast_points,
                max_hours=self.config.soft_slot_max_hours,
                time_scale_hours=self.config.forecast_slot_time_scale_hours,
                track_error_scale_km=self.config.track_error_scale_km,
                intensity_error_scale_kt=self.config.intensity_error_scale_kt,
                track_error_weight=self.config.track_error_weight,
                intensity_error_weight=self.config.intensity_error_weight,
            )
            total_lines = schema.parsed_line_count + schema.invalid_line_count
            valid_ratio = len(forecast_points) / total_lines if total_lines > 0 else 0.0
            parsed_line_coverage = min(len(forecast_points), len(effective_slots)) / max(
                len(effective_slots),
                1,
            )
            soft_slot_reward = (
                self.config.soft_slot_reward_weight
                * valid_ratio
                * parsed_line_coverage
                * soft_slot_alignment
            )

        # Format correct but no slot-aligned comparisons — keep structural credit,
        # and optionally add a small smooth reward for near-miss timing.
        if not effective_slots or not matched_slots:
            return float(max(partial, soft_slot_reward))

        track_score_total = 0.0
        intensity_score_total = 0.0
        time_alignment_total = 0.0
        for matched in matched_slots:
            forecast = matched.forecast_point
            truth = matched.truth_point
            track_error = haversine_km(
                forecast.lat,
                forecast.lon,
                float(truth["lat"]),
                float(truth["lon"]),
            )
            intensity_error = abs(forecast.vmax_kt - float(truth["vmax_kt"]))
            track_score_total += math.exp(-track_error / self.config.track_error_scale_km)
            intensity_score_total += math.exp(
                -intensity_error / self.config.intensity_error_scale_kt
            )
            time_alignment_total += matched.time_alignment_score

        matched_count = float(len(matched_slots))
        track_score = track_score_total / matched_count
        intensity_score = intensity_score_total / matched_count
        time_alignment_score = time_alignment_total / matched_count
        component_weight = self.config.track_error_weight + self.config.intensity_error_weight
        if component_weight <= 0:
            raise ValueError("Reward weights must sum to a positive value.")
        accuracy_score = (
            self.config.track_error_weight * track_score
            + self.config.intensity_error_weight * intensity_score
        ) / component_weight
        slot_factor = matched_count / max(len(effective_slots), len(forecast_points))
        accuracy_reward = slot_factor * time_alignment_score * accuracy_score
        # Discount outputs that contain extra text or lines whose timestamps
        # cannot be resolved into valid forecast points.
        if schema.invalid_line_count > 0 or len(forecast_points) < schema.parsed_line_count:
            total_lines = schema.parsed_line_count + schema.invalid_line_count
            valid_ratio = len(forecast_points) / max(total_lines, 1)
            accuracy_reward *= valid_ratio
        return float(max(partial, soft_slot_reward, min(1.0, accuracy_reward)))
