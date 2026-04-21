#!/usr/bin/env python3
"""Export training views from canonical v2 while preserving strict forecast compatibility."""

from __future__ import annotations

import json
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import dataset_formatter as df
from dataset_v2 import (
    CANONICAL_VERSION,
    DIAGNOSTIC_DERIVATION_VERSION,
    DIAGNOSTIC_CORE_FIELDS,
    DIAGNOSTIC_FIELDS,
    DIAGNOSTIC_SLOT_CORRECTION_FIELDS,
    DIAGNOSTIC_SLOT_TURN_CORRECTION_FIELDS,
    DIAGNOSTIC_TRACK_INFLECTION_FIELDS,
    DIAGNOSTIC_TRACK_CORRECTION_FIELDS,
    DIAGNOSTIC_TRACK_CORE_FIELDS,
    DIAGNOSTIC_TRACK_TURN_FIELDS,
    describe_diagnostic_field,
    SUPPORTED_SPLITS,
    iter_canonical_split_records,
    legacy_sample_from_canonical,
    rederive_diagnostics_from_canonical,
)


NULL_LABEL = "<null>"
SLOT_CORRECTION_LAT_BUCKETS = (
    "south_large",
    "south_small",
    "near_consensus",
    "north_small",
    "north_large",
)
SLOT_CORRECTION_LON_BUCKETS = (
    "east_large",
    "east_small",
    "near_consensus",
    "west_small",
    "west_large",
)


@dataclass(frozen=True)
class DiagnosticViewSpec:
    view_name: str
    field_names: tuple[str, ...]
    compat_prefix: str


def _build_diagnostic_system_prompt(field_names: tuple[str, ...]) -> str:
    keys = ", ".join(field_names)
    schema_lines = "\n".join(f"- {describe_diagnostic_field(field_name)}" for field_name in field_names)
    return (
        "You are a tropical cyclone forecaster. Use only the evidence and guidance "
        "provided in the prompt. Output exactly one JSON object for the structured "
        f"diagnostics with exactly these keys: {keys}. Keep every key in the schema. "
        "Use only the canonical value space below; do not paraphrase or restyle labels.\n"
        f"{schema_lines}\n"
        "Use null when the evidence does not support a field. The first character of "
        "the response must be '{' and the last character must be '}'. Do not output "
        "a forecast table, free-text reasoning, markdown fences, or additional prose."
    )


def _build_slot_correction_system_prompt(field_names: tuple[str, ...]) -> str:
    keys = ", ".join(field_names)
    lat_values = ", ".join(SLOT_CORRECTION_LAT_BUCKETS)
    lon_values = ", ".join(SLOT_CORRECTION_LON_BUCKETS)
    return (
        "You are a tropical cyclone forecaster. Return exactly one JSON object with exactly "
        f"these keys: {keys}. For every *_lat_* field use only {lat_values}, or null. "
        f"For every *_lon_* field use only {lon_values}, or null. Keep all keys present. "
        "Use null for missing later slots. The response must start with '{' and end with '}'. "
        "Output JSON only."
    )


def _build_slot_turn_correction_system_prompt(field_names: tuple[str, ...]) -> str:
    ordered_field_names = _ordered_slot_turn_field_names(field_names)
    keys = ", ".join(ordered_field_names)
    lat_values = ", ".join(SLOT_CORRECTION_LAT_BUCKETS)
    lon_values = ", ".join(SLOT_CORRECTION_LON_BUCKETS)
    turning_values = "steady, notable_turn, recurvature"
    return (
        "You are a tropical cyclone forecaster. Return exactly one minified JSON object on a "
        "single line with exactly these keys in this order: "
        f"{keys}. Emit turning_signal first, then slot_1..slot_6 fields in order. "
        f"For every *_lat_* field use only {lat_values}, or null. "
        f"For every *_lon_* field use only {lon_values}, or null. For turning_signal use only "
        f"{turning_values}, or null. Keep all keys present. Use null for missing later slots. "
        "Decide each slot field independently from the matching slot evidence. Within each slot, "
        "judge lat and lon independently as separate decisions. For *_lat_* fields, judge only "
        "north/south displacement relative to the matching ATCF point; east/west agreement does "
        "not imply near_consensus for lat. For *_lon_* fields, judge only east/west displacement "
        "relative to the matching ATCF point. Do not copy the same bucket across all slots unless "
        "the visible guidance supports that. Use "
        "near_consensus only when the slot-specific displacement is genuinely small or mixed; "
        "if the slot-specific evidence shows a consistent directional offset, emit the matching "
        "directional small/large bucket instead of defaulting to near_consensus. Do not let a "
        "clear north/south offset collapse to near_consensus just because longitude stays close "
        "to consensus. "
        "Do not pretty-print. Do not add spaces or line breaks outside the JSON syntax. "
        "The response must start with '{' and end with '}'. Output JSON only."
    )


def _ordered_slot_turn_field_names(field_names: tuple[str, ...]) -> tuple[str, ...]:
    ordered: List[str] = []
    if "turning_signal" in field_names:
        ordered.append("turning_signal")
    ordered.extend(field_name for field_name in field_names if field_name != "turning_signal")
    return tuple(ordered)


def _ordered_slot_turn_payload(
    payload: Dict[str, Any],
    field_names: tuple[str, ...],
) -> Dict[str, Any]:
    return {
        field_name: payload.get(field_name)
        for field_name in _ordered_slot_turn_field_names(field_names)
    }


def _safe_float_text(value: Any) -> str:
    try:
        return str(int(round(float(value))))
    except (TypeError, ValueError):
        return "NA"


def _build_slot_correction_slot_map(
    atcf_guidance: Dict[str, Any],
    *,
    issue_time_utc: Any,
    official_track_table: List[Dict[str, Any]],
) -> str:
    if atcf_guidance.get("status") != "available":
        return "- ATCF unavailable; leave unsupported slots null."

    target_leads = df.derive_guidance_target_leads(official_track_table, issue_time_utc)
    points = df.select_representative_guidance_points(
        atcf_guidance.get("consensus_spread_points_future", []) or [],
        target_leads=target_leads,
    )
    if not points:
        return "- No matching ATCF slots; leave unsupported slots null."

    lines: List[str] = [
        "- Keep slot count/order fixed; only predict lat/lon buckets relative to ATCF.",
        "- Do not change slot timing or intensity.",
    ]
    for slot_index, point in enumerate(points[: df.SLOT_LOCKED_CORRECTION_MAX_SLOTS], start=1):
        time_label = df.resolve_time_label(
            valid_day=point.get("valid_day"),
            valid_hhmmz=point.get("valid_hhmmz"),
            valid_time_utc=point.get("valid_time_utc"),
            issue_time_utc=issue_time_utc,
            lead_from_issue_h=point.get("lead_from_issue_h"),
        )
        lead_text = _safe_float_text(point.get("lead_from_issue_h"))
        lines.append(f"- slot_{slot_index} -> {time_label} | lead {lead_text}h")
    lines.append("- Missing later slots -> null.")
    return "\n".join(lines)


def _build_slot_correction_user_prompt(
    legacy_sample: Dict[str, Any],
    *,
    official_track_table: List[Dict[str, Any]],
) -> str:
    prompt_data = legacy_sample.get("prompt", {}) or {}
    storm_meta = prompt_data.get("storm_meta", {}) or {}
    now_inputs = prompt_data.get("now_inputs", {}) or {}
    guidance_inputs = prompt_data.get("guidance_inputs", {}) or {}
    guidance_time_reference = guidance_inputs.get("guidance_time_reference", {}) or {}
    issue_time_utc = (
        storm_meta.get("issue_time_utc")
        or guidance_time_reference.get("issue_time_utc")
        or None
    )

    current_state = now_inputs.get("current_state_from_noaa_forecast_advisory", {}) or {}
    center = current_state.get("center", {}) or {}
    motion = current_state.get("motion", {}) or {}
    intensity = current_state.get("intensity", {}) or {}
    issue_time_label = df.resolve_time_label(
        valid_day=center.get("obs_day"),
        valid_hhmmz=center.get("obs_hhmmz"),
        valid_time_utc=issue_time_utc,
    )

    state_parts = [f"Position {df.format_coord(center.get('lat'), center.get('lon'))}"]
    motion_text = df.normalize_text(motion.get("motion_text"))
    if motion_text:
        state_parts.append(
            f"Motion {motion_text} at {df.format_value_with_unit(motion.get('speed_kt'), 'kt')}"
        )
    if intensity:
        state_parts.append(
            "Intensity "
            f"{df.format_value_with_unit(intensity.get('max_wind_kt'), 'kt')} / "
            f"{df.format_value_with_unit(intensity.get('min_pressure_mb'), 'mb')}"
        )

    env_features = (
        (now_inputs.get("environment_now_ec_reanalysis", {}) or {}).get("features", {}) or {}
    )
    steering_lines = []
    for feature_key in ("subtropical_high", "westerly_trough", "vertical_wind_shear"):
        feature = env_features.get(feature_key)
        if feature:
            steering_lines.append(f"- {df.compress_cds_description(feature_key, feature)}")

    observation_block = now_inputs.get("observation_evidence_structured", {}) or {}
    available_components = df.get_available_observation_components(observation_block)
    observation_support = df.classify_observation_support(observation_block)
    observation_text = ", ".join(available_components) if available_components else "none"

    guidance_target_leads = df.derive_guidance_target_leads(official_track_table, issue_time_utc)
    atcf_text = df.format_atcf_guidance(
        guidance_inputs.get("multimodel_guidance_a_deck", {}) or {},
        issue_time_utc=issue_time_utc,
        target_leads=guidance_target_leads,
    )
    hres_text = df.format_hres_guidance(
        guidance_inputs.get("ec_single_model_guidance_hres", {}) or {},
        issue_time_utc=issue_time_utc,
        target_leads=guidance_target_leads,
    )

    user_parts: List[str] = []
    if issue_time_label:
        user_parts.append(f"## Time\n- Advisory issue {issue_time_label}")
    if state_parts:
        user_parts.append(f"## State\n- {' | '.join(state_parts)}")
    if steering_lines:
        user_parts.append("## Steering\n" + "\n".join(steering_lines))
    user_parts.append(
        "## Observations\n"
        f"- support {observation_support}; available {observation_text}"
    )
    guidance_parts = [text for text in (atcf_text, hres_text) if text]
    if guidance_parts:
        user_parts.append("## Guidance\n" + "\n\n".join(guidance_parts))
    user_parts.append(
        "## Slot Map\n"
        + _build_slot_correction_slot_map(
            guidance_inputs.get("multimodel_guidance_a_deck", {}) or {},
            issue_time_utc=issue_time_utc,
            official_track_table=official_track_table,
        )
    )
    return "\n\n".join(user_parts)


DIAGNOSTIC_VIEW_SPECS = (
    DiagnosticViewSpec(
        view_name="diagnostic_only",
        field_names=tuple(DIAGNOSTIC_FIELDS),
        compat_prefix="sft_diagnostic",
    ),
    DiagnosticViewSpec(
        view_name="diagnostic_track_core_only",
        field_names=tuple(DIAGNOSTIC_TRACK_CORE_FIELDS),
        compat_prefix="sft_diagnostic_track_core",
    ),
    DiagnosticViewSpec(
        view_name="diagnostic_track_turn_only",
        field_names=tuple(DIAGNOSTIC_TRACK_TURN_FIELDS),
        compat_prefix="sft_diagnostic_track_turn",
    ),
    DiagnosticViewSpec(
        view_name="diagnostic_track_inflection_only",
        field_names=tuple(DIAGNOSTIC_TRACK_INFLECTION_FIELDS),
        compat_prefix="sft_diagnostic_track_inflection",
    ),
    DiagnosticViewSpec(
        view_name="diagnostic_track_correction_only",
        field_names=tuple(DIAGNOSTIC_TRACK_CORRECTION_FIELDS),
        compat_prefix="sft_diagnostic_track_correction",
    ),
    DiagnosticViewSpec(
        view_name="diagnostic_slot_correction_only",
        field_names=tuple(DIAGNOSTIC_SLOT_CORRECTION_FIELDS),
        compat_prefix="sft_diagnostic_slot_correction",
    ),
    DiagnosticViewSpec(
        view_name="diagnostic_slot_turn_correction_only",
        field_names=tuple(DIAGNOSTIC_SLOT_TURN_CORRECTION_FIELDS),
        compat_prefix="sft_diagnostic_slot_turn_correction",
    ),
    DiagnosticViewSpec(
        view_name="diagnostic_core_only",
        field_names=tuple(DIAGNOSTIC_CORE_FIELDS),
        compat_prefix="sft_diagnostic_core",
    ),
)


def _copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        shutil.copyfile(src, dst)


def _load_canonical_records_by_split(canonical_dir: Path) -> Dict[str, List[Dict[str, Any]]]:
    split_records: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for split_name, _path, record in iter_canonical_split_records(canonical_dir, splits=SUPPORTED_SPLITS):
        split_records[split_name].append(record)
    return split_records


def _load_anonymizer(base_dir: Path):
    scripts_dir = base_dir / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    from data_leakage_prevention import Anonymizer

    return Anonymizer


def _summarize_diagnostic_value_counts(
    field_name: str,
    value_counts: Counter[str],
) -> Dict[str, Any]:
    if field_name == "expert_decision_notes":
        null_count = value_counts.get(NULL_LABEL, 0)
        non_null_count = sum(
            count
            for label, count in value_counts.items()
            if label != NULL_LABEL
        )
        distinct_non_null_value_count = sum(
            1
            for label in value_counts
            if label != NULL_LABEL
        )
        return {
            "value_counts": {
                NULL_LABEL: null_count,
                "<non_null_text>": non_null_count,
            },
            "distinct_non_null_value_count": distinct_non_null_value_count,
        }
    return {
        "value_counts": dict(sorted(value_counts.items())),
    }


def _diagnostic_payload(
    record: Dict[str, Any],
    *,
    field_names: tuple[str, ...],
) -> Dict[str, Any]:
    diagnostics = rederive_diagnostics_from_canonical(record)
    return {
        field_name: diagnostics.get(field_name)
        for field_name in field_names
    }


def format_diagnostic_sample(
    record: Dict[str, Any],
    *,
    view_spec: DiagnosticViewSpec,
) -> Optional[Dict[str, Any]]:
    """Convert one canonical record into one structured-diagnostic training view."""
    if not (record.get("flags", {}) or {}).get("diagnostic_view_eligible"):
        return None

    legacy_sample = legacy_sample_from_canonical(record)
    forecast_view = df.format_sft_sample(
        legacy_sample,
        view=df.SFT_VIEW_STRICT_FORECAST,
    )
    if forecast_view is None:
        return None

    prompt_data = legacy_sample.get("prompt", {}) or {}
    now_inputs = prompt_data.get("now_inputs", {}) or {}
    guidance_inputs = prompt_data.get("guidance_inputs", {}) or {}
    guidance_time_reference = guidance_inputs.get("guidance_time_reference", {}) or {}
    issue_time_utc = (
        (prompt_data.get("storm_meta", {}) or {}).get("issue_time_utc")
        or guidance_time_reference.get("issue_time_utc")
        or None
    )
    official_track_table = (
        (
            ((legacy_sample.get("target", {}) or {}).get("official_outputs", {}) or {})
            .get("track_intensity_table", {})
            .get("from_forecast_advisory", [])
        )
        or []
    )
    guidance_target_leads = df.derive_guidance_target_leads(official_track_table, issue_time_utc)
    is_track_correction_view = any(
        field_name in DIAGNOSTIC_TRACK_CORRECTION_FIELDS
        for field_name in view_spec.field_names
    )
    is_slot_correction_view = any(
        field_name in DIAGNOSTIC_SLOT_CORRECTION_FIELDS
        for field_name in view_spec.field_names
    )
    is_slot_turn_correction_view = (
        is_slot_correction_view and "turning_signal" in view_spec.field_names
    )
    if is_slot_correction_view:
        user_content = _build_slot_correction_user_prompt(
            legacy_sample,
            official_track_table=official_track_table,
        )
    else:
        user_content = next(
            (
                message.get("content", "")
                for message in forecast_view.get("messages", [])
                if message.get("role") == "user"
            ),
            "",
        )
    ridge_trough_cues = ""
    turning_cues = ""
    if not is_track_correction_view and not is_slot_correction_view:
        ridge_trough_cues = df.format_steering_competition_cues(
            ((now_inputs.get("environment_now_ec_reanalysis") or {}).get("features", {}) or {}),
            guidance_inputs.get("multimodel_guidance_a_deck", {}) or {},
            guidance_inputs.get("ec_single_model_guidance_hres", {}) or {},
            issue_time_utc=issue_time_utc,
            target_leads=guidance_target_leads,
        )
        if ridge_trough_cues:
            user_content = "\n\n".join(
                [user_content, "## Ridge/Trough Competition Cues\n" + ridge_trough_cues]
            )
        turning_cues = df.format_turning_guidance_cues(
            guidance_inputs.get("multimodel_guidance_a_deck", {}) or {},
            guidance_inputs.get("ec_single_model_guidance_hres", {}) or {},
            issue_time_utc=issue_time_utc,
            target_leads=guidance_target_leads,
        )
        if turning_cues:
            user_content = "\n\n".join([user_content, "## Turning Signal Cues\n" + turning_cues])
    elif is_slot_turn_correction_view:
        turning_cues = df.format_turning_guidance_cues(
            guidance_inputs.get("multimodel_guidance_a_deck", {}) or {},
            guidance_inputs.get("ec_single_model_guidance_hres", {}) or {},
            issue_time_utc=issue_time_utc,
            target_leads=guidance_target_leads,
        )
        if turning_cues:
            user_content = "\n\n".join([user_content, "## Turning Signal Cues\n" + turning_cues])
    track_inflection_cues = ""
    if any(field_name in DIAGNOSTIC_TRACK_INFLECTION_FIELDS for field_name in view_spec.field_names):
        track_inflection_cues = df.format_track_inflection_guidance_cues(
            guidance_inputs.get("multimodel_guidance_a_deck", {}) or {},
            guidance_inputs.get("ec_single_model_guidance_hres", {}) or {},
            issue_time_utc=issue_time_utc,
            target_leads=guidance_target_leads,
        )
        if track_inflection_cues:
            user_content = "\n\n".join(
                [user_content, "## Track Inflection Cues\n" + track_inflection_cues]
            )
    track_correction_cues = ""
    if is_track_correction_view:
        track_correction_cues = df.format_track_correction_guidance_cues(
            guidance_inputs.get("multimodel_guidance_a_deck", {}) or {},
            issue_time_utc=issue_time_utc,
        )
        if track_correction_cues:
            user_content = "\n\n".join(
                [user_content, "## Track Correction Anchors\n" + track_correction_cues]
            )
    slot_correction_cues = ""
    if is_slot_correction_view:
        slot_correction_cues = df.format_slot_locked_correction_guidance_cues(
            guidance_inputs.get("multimodel_guidance_a_deck", {}) or {},
            issue_time_utc=issue_time_utc,
            official_track_table=official_track_table,
        )
        if slot_correction_cues:
            user_content = "\n\n".join(
                [user_content, "## Slot-Locked Track Correction Cues\n" + slot_correction_cues]
            )
    diagnostic_payload = _diagnostic_payload(record, field_names=view_spec.field_names)
    if is_slot_correction_view:
        assistant_content = json.dumps(
            _ordered_slot_turn_payload(diagnostic_payload, view_spec.field_names)
            if is_slot_turn_correction_view
            else diagnostic_payload,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=False,
        )
    else:
        assistant_content = json.dumps(
            diagnostic_payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=False,
        )
    non_null_fields = [
        field_name
        for field_name, field_value in diagnostic_payload.items()
        if field_value not in (None, "", [], {})
    ]

    train_metadata = dict(forecast_view.get("train_metadata", {}) or {})
    train_metadata.update(
        {
            "sft_view": view_spec.view_name,
            "diagnostic_field_names": (
                list(_ordered_slot_turn_field_names(view_spec.field_names))
                if is_slot_turn_correction_view
                else list(view_spec.field_names)
            ),
            "diagnostic_non_null_fields": non_null_fields,
            "diagnostic_field_count": len(non_null_fields),
            "ridge_trough_cues_included": bool(ridge_trough_cues),
            "turning_cues_included": bool(turning_cues),
            "track_inflection_cues_included": bool(track_inflection_cues),
            "track_correction_cues_included": bool(track_correction_cues),
            "slot_correction_cues_included": bool(slot_correction_cues),
            "diagnostic_derivation_version": DIAGNOSTIC_DERIVATION_VERSION,
        }
    )

    return {
        "sample_id": record.get("sample_id", ""),
        "format_version": CANONICAL_VERSION,
        "train_metadata": train_metadata,
        "messages": [
            {
                "role": "system",
                "content": (
                    _build_slot_turn_correction_system_prompt(view_spec.field_names)
                    if is_slot_turn_correction_view
                    else (
                        _build_slot_correction_system_prompt(view_spec.field_names)
                        if is_slot_correction_view
                        else _build_diagnostic_system_prompt(view_spec.field_names)
                    )
                ),
            },
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ],
    }


def convert_diagnostic_split(
    records: List[Dict[str, Any]],
    output_dir: Path,
    split_name: str,
    output_name: str,
    *,
    view_spec: DiagnosticViewSpec,
) -> Dict[str, Any]:
    """Write one diagnostic split and return compact summary stats."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / output_name

    count = 0
    errors = 0
    skipped = 0
    skip_reasons = Counter()
    total_token_stats: List[int] = []
    user_token_stats: List[int] = []
    assistant_token_stats: List[int] = []
    observation_support_counts = Counter()
    observation_source_combo_counts = Counter()
    guidance_counts = Counter()
    leakage_counts = Counter()
    anchor_counts = Counter()
    diagnostic_non_null_counts = Counter()
    diagnostic_value_counts: Dict[str, Counter[str]] = {
        field_name: Counter() for field_name in view_spec.field_names
    }

    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            try:
                formatted = format_diagnostic_sample(record, view_spec=view_spec)
                if formatted is None:
                    skipped += 1
                    if not (record.get("flags", {}) or {}).get("diagnostic_view_eligible"):
                        skip_reasons["diagnostic_not_eligible"] += 1
                    else:
                        skip_reasons["missing_forecast_context"] += 1
                    continue

                legacy_sample = legacy_sample_from_canonical(record)
                messages = formatted.get("messages", [])
                user_content = next(
                    (message.get("content", "") for message in messages if message.get("role") == "user"),
                    "",
                )
                assistant_content = next(
                    (message.get("content", "") for message in messages if message.get("role") == "assistant"),
                    "",
                )
                total_text = " ".join(message.get("content", "") for message in messages)
                total_token_stats.append(df.estimate_tokens(total_text))
                user_token_stats.append(df.estimate_tokens(user_content))
                assistant_token_stats.append(df.estimate_tokens(assistant_content))

                train_metadata = formatted.get("train_metadata", {}) or {}
                observation_support = train_metadata.get("observation_support", "unknown")
                observation_support_counts[observation_support] += 1
                combo = "+".join(train_metadata.get("available_observation_components") or ["none"])
                observation_source_combo_counts[combo] += 1

                if train_metadata.get("issue_time_anchor_present"):
                    anchor_counts["issue_time_anchor_present"] += 1
                if train_metadata.get("track_time_labels_present"):
                    anchor_counts["track_time_labels_present"] += 1
                if train_metadata.get("atcf_guidance_included"):
                    guidance_counts["atcf_consensus"] += 1
                if train_metadata.get("hres_guidance_included"):
                    guidance_counts["hres_deterministic"] += 1
                if train_metadata.get("pre_issue_hres_included"):
                    guidance_counts["pre_issue_hres"] += 1

                for field_name in train_metadata.get("diagnostic_non_null_fields", []) or []:
                    diagnostic_non_null_counts[str(field_name)] += 1
                record_payload = _diagnostic_payload(record, field_names=view_spec.field_names)
                for field_name in view_spec.field_names:
                    raw_value = record_payload.get(field_name)
                    label = NULL_LABEL if raw_value in (None, "", [], {}) else str(raw_value)
                    diagnostic_value_counts[field_name][label] += 1

                leakage_flags = df.scan_train_view_leakage(
                    user_content=user_content,
                    assistant_content=assistant_content,
                    sample=legacy_sample,
                )
                for flag_name, is_present in leakage_flags.items():
                    if is_present:
                        leakage_counts[flag_name] += 1

                handle.write(json.dumps(formatted, ensure_ascii=False) + "\n")
                count += 1
            except Exception:
                errors += 1

    stats: Dict[str, Any] = {
        "count": count,
        "errors": errors,
        "skipped": skipped,
        "sft_view": view_spec.view_name,
        "diagnostic_fields": list(view_spec.field_names),
        "observation_support": {
            "none": observation_support_counts.get("none", 0),
            "single_source": observation_support_counts.get("single_source", 0),
            "multi_source": observation_support_counts.get("multi_source", 0),
        },
        "observation_source_combos": dict(sorted(observation_source_combo_counts.items())),
        "guidance_included": {
            "atcf_consensus": guidance_counts.get("atcf_consensus", 0),
            "hres_deterministic": guidance_counts.get("hres_deterministic", 0),
            "pre_issue_hres": guidance_counts.get("pre_issue_hres", 0),
        },
        "time_anchor": {
            "issue_time_anchor_present": anchor_counts.get("issue_time_anchor_present", 0),
            "track_time_labels_present": anchor_counts.get("track_time_labels_present", 0),
        },
        "diagnostic_non_null_fields": {
            field_name: diagnostic_non_null_counts.get(field_name, 0)
            for field_name in view_spec.field_names
        },
        "diagnostic_field_summary": {
            field_name: {
                "non_null_count": diagnostic_non_null_counts.get(field_name, 0),
                "non_null_rate": (
                    diagnostic_non_null_counts.get(field_name, 0) / count if count else 0.0
                ),
                **_summarize_diagnostic_value_counts(
                    field_name,
                    diagnostic_value_counts[field_name],
                ),
            }
            for field_name in view_spec.field_names
        },
        "train_view_leakage": {
            flag_name: leakage_counts.get(flag_name, 0)
            for flag_name in df.KNOWN_LEAKAGE_FLAGS
        },
    }

    total_summary = df.summarize_numeric_series(total_token_stats)
    user_summary = df.summarize_numeric_series(user_token_stats)
    assistant_summary = df.summarize_numeric_series(assistant_token_stats)
    if total_summary:
        stats["token_est_median"] = total_summary["median"]
        stats["token_est_p90"] = total_summary["p90"]
    if user_summary:
        stats["user_token_est_median"] = user_summary["median"]
        stats["user_token_est_p90"] = user_summary["p90"]
    if assistant_summary:
        stats["assistant_token_est_median"] = assistant_summary["median"]
        stats["assistant_token_est_p90"] = assistant_summary["p90"]
    if skip_reasons:
        stats["skipped_reasons"] = dict(sorted(skip_reasons.items()))
    return stats


def generate_test_variants_from_samples(
    base_dir: Path,
    legacy_test_samples: List[Dict[str, Any]],
    forecast_dir: Path,
    compat_root: Path,
) -> Dict[str, Any]:
    """Regenerate anonymous / structured-only / perturbation test variants."""
    if not legacy_test_samples:
        return {}

    Anonymizer = _load_anonymizer(base_dir)
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
                for sample in legacy_test_samples
            }
        )
        mapping = anonymizer.generate_mapping([storm_id for storm_id in storm_ids if storm_id])

    variant_sources = {
        "anonymous": anonymizer.generate_anonymous_test_set(legacy_test_samples, mapping),
        "structured_only": anonymizer.generate_structured_only_test_set(legacy_test_samples),
        "perturbation": anonymizer.generate_perturbation_test_set(legacy_test_samples, mapping),
    }

    stats_by_variant: Dict[str, Any] = {}
    for variant_name, variant_samples in variant_sources.items():
        stats = df.convert_split(
            raw_dir=None,
            output_dir=forecast_dir,
            split_name="test",
            fmt="sft",
            sft_view=df.SFT_VIEW_STRICT_FORECAST,
            samples=variant_samples,
            output_name=f"test_{variant_name}.jsonl",
            evaluation_variant=variant_name,
        )
        stats_by_variant[variant_name] = stats
        _copy_if_exists(
            forecast_dir / f"test_{variant_name}.jsonl",
            compat_root / f"sft_test_{variant_name}.jsonl",
        )
    return stats_by_variant


def export_views(
    base_dir: Path,
    canonical_dir: Path,
    output_dir: Path,
    include_test_variants: bool = True,
) -> Dict[str, Any]:
    """Export all training views plus strict forecast compatible root files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    compat_root = output_dir
    views_root = output_dir / "views"
    forecast_dir = views_root / "forecast_only"
    reasoning_dir = views_root / "reasoning_only"
    diagnostic_dirs = {
        spec.view_name: views_root / spec.view_name
        for spec in DIAGNOSTIC_VIEW_SPECS
    }
    for directory in [forecast_dir, reasoning_dir, *diagnostic_dirs.values()]:
        directory.mkdir(parents=True, exist_ok=True)

    split_records = _load_canonical_records_by_split(canonical_dir)
    forecast_report: Dict[str, Any] = {"view": "forecast_only"}
    reasoning_report: Dict[str, Any] = {"view": "reasoning_only"}
    diagnostic_reports: Dict[str, Dict[str, Any]] = {
        spec.view_name: {
            "view": spec.view_name,
            "diagnostic_fields": list(spec.field_names),
        }
        for spec in DIAGNOSTIC_VIEW_SPECS
    }

    legacy_test_samples: List[Dict[str, Any]] = []

    for split_name in [split for split in SUPPORTED_SPLITS if split != "unassigned"]:
        records = split_records.get(split_name, [])
        legacy_samples = [legacy_sample_from_canonical(record) for record in records]
        if split_name == "test":
            legacy_test_samples = legacy_samples

        forecast_stats = df.convert_split(
            raw_dir=None,
            output_dir=forecast_dir,
            split_name=split_name,
            fmt="sft",
            sft_view=df.SFT_VIEW_STRICT_FORECAST,
            samples=legacy_samples,
            output_name=f"{split_name}.jsonl",
        )
        forecast_report[f"sft_{split_name}"] = forecast_stats
        _copy_if_exists(
            forecast_dir / f"{split_name}.jsonl",
            compat_root / f"sft_{split_name}.jsonl",
        )

        rl_stats = df.convert_split(
            raw_dir=None,
            output_dir=forecast_dir,
            split_name=split_name,
            fmt="rl",
            samples=legacy_samples,
            output_name=f"rl_{split_name}.jsonl",
        )
        forecast_report[f"rl_{split_name}"] = rl_stats
        _copy_if_exists(
            forecast_dir / f"rl_{split_name}.jsonl",
            compat_root / f"rl_{split_name}.jsonl",
        )

        reasoning_stats = df.convert_split(
            raw_dir=None,
            output_dir=reasoning_dir,
            split_name=split_name,
            fmt="sft",
            sft_view=df.SFT_VIEW_REASONING,
            samples=legacy_samples,
            output_name=f"{split_name}.jsonl",
        )
        reasoning_report[f"sft_{split_name}"] = reasoning_stats
        _copy_if_exists(
            reasoning_dir / f"{split_name}.jsonl",
            compat_root / f"sft_reasoning_{split_name}.jsonl",
        )

        for view_spec in DIAGNOSTIC_VIEW_SPECS:
            diagnostic_dir = diagnostic_dirs[view_spec.view_name]
            diagnostic_stats = convert_diagnostic_split(
                records=records,
                output_dir=diagnostic_dir,
                split_name=split_name,
                output_name=f"{split_name}.jsonl",
                view_spec=view_spec,
            )
            diagnostic_reports[view_spec.view_name][f"sft_{split_name}"] = diagnostic_stats
            _copy_if_exists(
                diagnostic_dir / f"{split_name}.jsonl",
                compat_root / f"{view_spec.compat_prefix}_{split_name}.jsonl",
            )

    variant_report = {}
    if include_test_variants:
        variant_report = generate_test_variants_from_samples(
            base_dir=base_dir,
            legacy_test_samples=legacy_test_samples,
            forecast_dir=forecast_dir,
            compat_root=compat_root,
        )
        if variant_report:
            forecast_report["sft_test_variants"] = variant_report

    (forecast_dir / "report.json").write_text(
        json.dumps(forecast_report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (reasoning_dir / "report.json").write_text(
        json.dumps(reasoning_report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    for view_spec in DIAGNOSTIC_VIEW_SPECS:
        diagnostic_dir = diagnostic_dirs[view_spec.view_name]
        diagnostic_report = diagnostic_reports[view_spec.view_name]
        (diagnostic_dir / "report.json").write_text(
            json.dumps(diagnostic_report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    aggregate_report = {
        "canonical_version": CANONICAL_VERSION,
        "forecast_only": forecast_report,
        "reasoning_only": reasoning_report,
        "compatibility_root_files": sorted(path.name for path in compat_root.glob("*.jsonl")),
    }
    for view_spec in DIAGNOSTIC_VIEW_SPECS:
        aggregate_report[view_spec.view_name] = diagnostic_reports[view_spec.view_name]
    format_report_path = compat_root / "format_report.json"
    format_report_path.write_text(
        json.dumps(aggregate_report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    ready_report = {
        "canonical_dir": str(canonical_dir),
        "views_dir": str(views_root),
        "forecast_root_compatible": all(
            (compat_root / filename).exists()
            for filename in [
                "sft_train.jsonl",
                "sft_val.jsonl",
                "sft_test.jsonl",
                "rl_train.jsonl",
                "rl_val.jsonl",
                "rl_test.jsonl",
            ]
        ),
        "reasoning_view_ready": any(
            (reasoning_report.get(f"sft_{split_name}", {}) or {}).get("count", 0) > 0
            for split_name in ("train", "val", "test")
        ),
        "diagnostic_view_ready": any(
            (diagnostic_reports["diagnostic_only"].get(f"sft_{split_name}", {}) or {}).get("count", 0) > 0
            for split_name in ("train", "val", "test")
        ),
        "diagnostic_track_core_view_ready": any(
            (
                diagnostic_reports["diagnostic_track_core_only"].get(f"sft_{split_name}", {})
                or {}
            ).get("count", 0) > 0
            for split_name in ("train", "val", "test")
        ),
        "diagnostic_track_turn_view_ready": any(
            (
                diagnostic_reports["diagnostic_track_turn_only"].get(f"sft_{split_name}", {})
                or {}
            ).get("count", 0) > 0
            for split_name in ("train", "val", "test")
        ),
        "diagnostic_track_inflection_view_ready": any(
            (
                diagnostic_reports["diagnostic_track_inflection_only"].get(
                    f"sft_{split_name}",
                    {},
                )
                or {}
            ).get("count", 0) > 0
            for split_name in ("train", "val", "test")
        ),
        "diagnostic_track_correction_view_ready": any(
            (
                diagnostic_reports["diagnostic_track_correction_only"].get(
                    f"sft_{split_name}",
                    {},
                )
                or {}
            ).get("count", 0) > 0
            for split_name in ("train", "val", "test")
        ),
        "diagnostic_slot_correction_view_ready": any(
            (
                diagnostic_reports["diagnostic_slot_correction_only"].get(
                    f"sft_{split_name}",
                    {},
                )
                or {}
            ).get("count", 0) > 0
            for split_name in ("train", "val", "test")
        ),
        "diagnostic_slot_turn_correction_view_ready": any(
            (
                diagnostic_reports["diagnostic_slot_turn_correction_only"].get(
                    f"sft_{split_name}",
                    {},
                )
                or {}
            ).get("count", 0) > 0
            for split_name in ("train", "val", "test")
        ),
        "diagnostic_core_view_ready": any(
            (
                diagnostic_reports["diagnostic_core_only"].get(f"sft_{split_name}", {})
                or {}
            ).get("count", 0) > 0
            for split_name in ("train", "val", "test")
        ),
    }
    (compat_root / "dataset_ready_report.json").write_text(
        json.dumps(ready_report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return aggregate_report


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Export training views from canonical v2")
    parser.add_argument("--base-dir", type=str, default=".")
    parser.add_argument(
        "--canonical-dir",
        type=str,
        required=True,
        help="Canonical v2 directory containing train/val/test JSONL files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Dataset root that will receive /views plus strict-compatible root files",
    )
    parser.add_argument(
        "--skip-test-variants",
        action="store_true",
        help="Do not regenerate anonymous / structured-only / perturbation forecast test variants",
    )
    args = parser.parse_args()

    base_dir = Path(args.base_dir).resolve()
    canonical_dir = (base_dir / args.canonical_dir).resolve()
    output_dir = (base_dir / args.output_dir).resolve()
    report = export_views(
        base_dir=base_dir,
        canonical_dir=canonical_dir,
        output_dir=output_dir,
        include_test_variants=not args.skip_test_variants,
    )
    forecast_train = ((report.get("forecast_only", {}) or {}).get("sft_train", {}) or {}).get("count", 0)
    reasoning_train = ((report.get("reasoning_only", {}) or {}).get("sft_train", {}) or {}).get("count", 0)
    diagnostic_train = ((report.get("diagnostic_only", {}) or {}).get("sft_train", {}) or {}).get("count", 0)
    diagnostic_track_core_train = (
        ((report.get("diagnostic_track_core_only", {}) or {}).get("sft_train", {}) or {}).get("count", 0)
    )
    diagnostic_track_turn_train = (
        ((report.get("diagnostic_track_turn_only", {}) or {}).get("sft_train", {}) or {}).get("count", 0)
    )
    diagnostic_track_inflection_train = (
        (
            (report.get("diagnostic_track_inflection_only", {}) or {}).get("sft_train", {})
            or {}
        ).get("count", 0)
    )
    diagnostic_track_correction_train = (
        (
            (report.get("diagnostic_track_correction_only", {}) or {}).get("sft_train", {})
            or {}
        ).get("count", 0)
    )
    diagnostic_slot_correction_train = (
        (
            (report.get("diagnostic_slot_correction_only", {}) or {}).get("sft_train", {})
            or {}
        ).get("count", 0)
    )
    diagnostic_slot_turn_correction_train = (
        (
            (report.get("diagnostic_slot_turn_correction_only", {}) or {}).get("sft_train", {})
            or {}
        ).get("count", 0)
    )
    diagnostic_core_train = (
        ((report.get("diagnostic_core_only", {}) or {}).get("sft_train", {}) or {}).get("count", 0)
    )
    print(
        f"views: forecast train {forecast_train} | "
        f"reasoning train {reasoning_train} | "
        f"diagnostic train {diagnostic_train} | "
        f"diagnostic_track_core train {diagnostic_track_core_train} | "
        f"diagnostic_track_turn train {diagnostic_track_turn_train} | "
        f"diagnostic_track_inflection train {diagnostic_track_inflection_train} | "
        f"diagnostic_track_correction train {diagnostic_track_correction_train} | "
        f"diagnostic_slot_correction train {diagnostic_slot_correction_train} | "
        f"diagnostic_slot_turn_correction train {diagnostic_slot_turn_correction_train} | "
        f"diagnostic_core train {diagnostic_core_train}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
