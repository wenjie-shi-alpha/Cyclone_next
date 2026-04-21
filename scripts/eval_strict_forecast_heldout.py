#!/usr/bin/env python3
"""Evaluate one adapter on held-out strict-forecast prompts and verification targets."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import dataset_formatter as df
from cyclone_training.config import load_grpo_config, load_sft_config
from cyclone_training.datasets import render_chat
from cyclone_training.modeling import load_adapter_weights, load_model_and_tokenizer
from cyclone_training.rewards import (
    CycloneRewardFunction,
    _extract_truth_issue_time,
    completion_to_text,
    haversine_km,
    inspect_forecast_schema,
    match_forecast_to_truth_slots,
    parse_forecast_points,
    parse_target_forecast_slots,
)


def _read_jsonl(
    path: Path,
    max_samples: int | None,
    *,
    sample_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    if sample_ids is not None:
        wanted = {str(sample_id): index for index, sample_id in enumerate(sample_ids)}
        rows_by_id: dict[str, dict[str, Any]] = {}
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                sample_id = str(row.get("sample_id", ""))
                if sample_id not in wanted:
                    continue
                rows_by_id[sample_id] = row
                if len(rows_by_id) >= len(wanted):
                    break
        missing_ids = [sample_id for sample_id in sample_ids if sample_id not in rows_by_id]
        if missing_ids:
            preview = ", ".join(missing_ids[:5])
            raise ValueError(f"Missing {len(missing_ids)} sample_ids in {path}: {preview}")
        return [rows_by_id[sample_id] for sample_id in sample_ids]

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if max_samples is not None and len(rows) >= max_samples:
                break
    return rows


def _load_target_map(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    target_map: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            sample_id = str(row.get("sample_id", ""))
            messages = row.get("messages") or []
            target_text = ""
            for message in messages:
                if message.get("role") == "assistant":
                    target_text = str(message.get("content", "")).strip()
                    break
            if sample_id and target_text:
                target_map[sample_id] = target_text
    return target_map


def _load_prompt_override_map(path: Path) -> dict[str, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    candidate = payload.get("overrides") if isinstance(payload, dict) and "overrides" in payload else payload
    if not isinstance(candidate, dict):
        raise ValueError(
            f"Prompt override file must be a JSON object or contain an 'overrides' object: {path}"
        )

    overrides: dict[str, str] = {}
    for sample_id, prompt_text in candidate.items():
        if not isinstance(prompt_text, str):
            raise ValueError(f"Prompt override for sample_id={sample_id!r} must be a string.")
        overrides[str(sample_id)] = prompt_text
    return overrides


def _apply_user_prompt_override(
    messages: list[dict[str, Any]],
    override_prompt: str,
) -> list[dict[str, Any]]:
    updated = [dict(message) for message in messages]
    for index in range(len(updated) - 1, -1, -1):
        if updated[index].get("role") == "user":
            updated[index]["content"] = override_prompt
            return updated
    raise ValueError("Prompt override requested for a sample without a user message.")


def _load_inference_stack(config_path: Path, adapter_path: Path | None):
    config = load_sft_config(config_path)
    model, tokenizer = load_model_and_tokenizer(
        config.model,
        config.lora,
        stage="sft",
        attach_lora=adapter_path is not None,
    )
    if adapter_path is not None:
        model = load_adapter_weights(model, adapter_path, is_trainable=False)
    model.eval()
    tokenizer.padding_side = "left"
    if hasattr(tokenizer, "truncation_side"):
        tokenizer.truncation_side = "left"

    from unsloth import FastLanguageModel

    FastLanguageModel.for_inference(model)
    return config, model, tokenizer


def _generate_batch(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    *,
    max_prompt_tokens: int,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
) -> list[str]:
    device = getattr(model, "device", None)
    if device is None:
        device = next(model.parameters()).device
    encoded = tokenizer(
        text=prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_prompt_tokens,
    )
    encoded = {name: tensor.to(device) for name, tensor in encoded.items()}
    prompt_width = int(encoded["input_ids"].shape[1])
    generation_kwargs = {
        **encoded,
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "use_cache": True,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        generation_kwargs["temperature"] = temperature
        generation_kwargs["top_p"] = top_p
    with torch.inference_mode():
        outputs = model.generate(**generation_kwargs)
    completions = outputs[:, prompt_width:]
    return tokenizer.batch_decode(completions, skip_special_tokens=True)


def _safe_mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def _safe_p50(values: list[float]) -> float | None:
    if not values:
        return None
    return float(statistics.median(values))


def _safe_std(values: list[float]) -> float | None:
    if len(values) < 2:
        return 0.0 if values else None
    return float(statistics.pstdev(values))


def _reward_details(
    reward_function: CycloneRewardFunction,
    completion_text: str,
    verification: Mapping[str, Any],
) -> dict[str, Any]:
    issue_time = _extract_truth_issue_time(verification)
    forecast_points = parse_forecast_points(completion_to_text(completion_text), issue_time)
    target_slots = parse_target_forecast_slots(verification, issue_time)
    truth_points = list(verification.get("future_best_track") or [])
    effective_slots, matched_slots = match_forecast_to_truth_slots(
        forecast_points,
        target_slots,
        truth_points,
        truth_slot_tolerance_hours=reward_function.config.truth_slot_tolerance_hours,
        forecast_slot_tolerance_hours=reward_function.config.forecast_slot_tolerance_hours,
        forecast_slot_time_scale_hours=reward_function.config.forecast_slot_time_scale_hours,
    )

    track_errors_km: list[float] = []
    intensity_errors_kt: list[float] = []
    time_alignment_scores: list[float] = []
    for matched in matched_slots:
        forecast = matched.forecast_point
        truth = matched.truth_point
        track_errors_km.append(
            haversine_km(
                forecast.lat,
                forecast.lon,
                float(truth["lat"]),
                float(truth["lon"]),
            )
        )
        intensity_errors_kt.append(abs(forecast.vmax_kt - float(truth["vmax_kt"])))
        time_alignment_scores.append(float(matched.time_alignment_score))

    reward = reward_function.score_one(completion_text, verification)
    return {
        "reward": reward,
        "forecast_point_count": len(forecast_points),
        "target_track_point_count": len(target_slots),
        "matched_point_count": len(matched_slots),
        "match_coverage_vs_target": (
            len(matched_slots) / len(effective_slots) if effective_slots else None
        ),
        "mean_time_alignment_score": _safe_mean(time_alignment_scores),
        "mean_track_error_km": _safe_mean(track_errors_km),
        "mean_intensity_error_kt": _safe_mean(intensity_errors_kt),
    }


def _relaxed_schema_metrics(text: str) -> dict[str, int]:
    schema = inspect_forecast_schema(text)
    relaxed_parseable = int(schema.parsed_line_count > 0)
    relaxed_clean = int(schema.parsed_line_count > 0 and schema.invalid_line_count == 0)
    return {
        "relaxed_forecast_parseable": relaxed_parseable,
        "relaxed_forecast_clean": relaxed_clean,
        "relaxed_parseable_with_header": int(relaxed_parseable and schema.has_header),
        "relaxed_parseable_without_header": int(relaxed_parseable and not schema.has_header),
    }


def _inject_forecast_slots_from_target(
    verification: dict[str, Any],
    target_text: str | None,
) -> dict[str, Any]:
    if verification.get("forecast_slots") or not target_text:
        return verification
    issue_time = _extract_truth_issue_time(verification)
    target_points = parse_forecast_points(target_text, issue_time)
    if not target_points:
        return verification

    enriched = dict(verification)
    enriched["forecast_slots"] = [
        {
            "valid_time_utc": point.valid_time.isoformat().replace("+00:00", "Z"),
        }
        for point in target_points
    ]
    return enriched


def _reference_truth_points_from_forecast_text(
    reference_text: str | None,
    *,
    verification: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if not reference_text:
        return []
    issue_time = _extract_truth_issue_time(verification)
    reference_points = parse_forecast_points(reference_text, issue_time)
    return [
        {
            "valid_time_utc": point.valid_time.isoformat().replace("+00:00", "Z"),
            "lat": point.lat,
            "lon": point.lon,
            "vmax_kt": point.vmax_kt,
        }
        for point in reference_points
    ]


def _compare_against_reference_forecast(
    completion_text: str,
    *,
    reference_text: str | None,
    verification: Mapping[str, Any],
    reward_function: CycloneRewardFunction,
) -> dict[str, Any]:
    reference_truth_points = _reference_truth_points_from_forecast_text(
        reference_text,
        verification=verification,
    )
    if not reference_truth_points:
        return {
            "reference_point_count": 0,
            "matched_point_count": 0,
            "slot_time_match_rate": None,
            "mean_time_alignment_score": None,
            "mean_track_diff_km": None,
            "mean_intensity_diff_kt": None,
        }

    issue_time = _extract_truth_issue_time(verification)
    forecast_points = parse_forecast_points(completion_to_text(completion_text), issue_time)
    reference_slots = parse_target_forecast_slots(
        {
            "forecast_slots": [
                {"valid_time_utc": truth_point["valid_time_utc"]}
                for truth_point in reference_truth_points
            ]
        },
        issue_time,
    )
    effective_slots, matched_slots = match_forecast_to_truth_slots(
        forecast_points,
        reference_slots,
        reference_truth_points,
        truth_slot_tolerance_hours=reward_function.config.forecast_slot_tolerance_hours,
        forecast_slot_tolerance_hours=reward_function.config.forecast_slot_tolerance_hours,
        forecast_slot_time_scale_hours=reward_function.config.forecast_slot_time_scale_hours,
    )

    track_diffs_km: list[float] = []
    intensity_diffs_kt: list[float] = []
    time_alignment_scores: list[float] = []
    for matched in matched_slots:
        forecast = matched.forecast_point
        reference = matched.truth_point
        track_diffs_km.append(
            haversine_km(
                forecast.lat,
                forecast.lon,
                float(reference["lat"]),
                float(reference["lon"]),
            )
        )
        intensity_diffs_kt.append(abs(forecast.vmax_kt - float(reference["vmax_kt"])))
        time_alignment_scores.append(float(matched.time_alignment_score))

    return {
        "reference_point_count": len(reference_truth_points),
        "matched_point_count": len(matched_slots),
        "slot_time_match_rate": (
            len(matched_slots) / len(effective_slots) if effective_slots else None
        ),
        "mean_time_alignment_score": _safe_mean(time_alignment_scores),
        "mean_track_diff_km": _safe_mean(track_diffs_km),
        "mean_intensity_diff_kt": _safe_mean(intensity_diffs_kt),
    }


def load_eval_samples(
    *,
    rl_dataset_path: Path,
    sft_dataset_path: Path | None,
    max_samples: int | None,
    sample_ids: list[str] | None = None,
    prompt_overrides: Mapping[str, str] | None = None,
) -> list[dict[str, Any]]:
    rl_rows = _read_jsonl(
        rl_dataset_path,
        max_samples=max_samples if sample_ids is None else None,
        sample_ids=sample_ids,
    )
    target_map = _load_target_map(sft_dataset_path)

    prepared_samples: list[dict[str, Any]] = []
    for row in rl_rows:
        messages = [dict(message) for message in row["messages"]]
        sample_id = str(row.get("sample_id", ""))
        prompt_overridden = bool(prompt_overrides is not None and sample_id in prompt_overrides)
        if prompt_overridden:
            messages = _apply_user_prompt_override(messages, prompt_overrides[sample_id])
        prepared_samples.append(
            {
                "sample_id": sample_id,
                "messages": messages,
                "verification": dict(row["verification"]),
                "target_text": target_map.get(sample_id),
                "prompt_overridden": prompt_overridden,
            }
        )
    return prepared_samples


def evaluate_outputs(
    *,
    config_path: Path | None,
    reward_config_path: Path,
    adapter_path: Path | None,
    rl_dataset_path: Path,
    sft_dataset_path: Path | None,
    prepared_samples: list[dict[str, Any]],
    outputs: list[str],
    batch_size: int,
    max_prompt_tokens: int,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    include_samples: bool = False,
) -> dict[str, Any]:
    if len(outputs) != len(prepared_samples):
        raise ValueError(
            f"Output count mismatch: got {len(outputs)} outputs for {len(prepared_samples)} samples."
        )

    reward_config = load_grpo_config(reward_config_path)
    reward_function = CycloneRewardFunction(reward_config.reward)
    prompt_override_count = sum(int(sample["prompt_overridden"]) for sample in prepared_samples)

    schema_totals = {
        "strict_forecast_parseable": 0,
        "strict_forecast_with_extra_text": 0,
        "no_track_forecast": 0,
        "reasoning_only": 0,
    }
    relaxed_schema_totals = {
        "relaxed_forecast_parseable": 0,
        "relaxed_forecast_clean": 0,
        "relaxed_parseable_with_header": 0,
        "relaxed_parseable_without_header": 0,
    }
    exact_match_count = 0
    exact_match_eligible = 0
    rewards: list[float] = []
    parseable_rewards: list[float] = []
    nonparseable_rewards: list[float] = []
    relaxed_parseable_rewards: list[float] = []
    relaxed_nonparseable_rewards: list[float] = []
    matched_counts: list[float] = []
    coverage_values: list[float] = []
    time_alignment_values: list[float] = []
    track_errors_km: list[float] = []
    intensity_errors_kt: list[float] = []
    slot_time_match_rates_vs_official: list[float] = []
    time_alignment_scores_vs_official: list[float] = []
    track_diffs_vs_official_km: list[float] = []
    intensity_diffs_vs_official_kt: list[float] = []
    expert_rewards: list[float] = []
    expert_coverages: list[float] = []
    expert_track_errors_km: list[float] = []
    expert_intensity_errors_kt: list[float] = []
    samples: list[dict[str, Any]] = []

    for sample, generated_text in zip(prepared_samples, outputs):
        sample_id = str(sample["sample_id"])
        target_text = sample["target_text"]
        verification = _inject_forecast_slots_from_target(
            dict(sample["verification"]),
            target_text,
        )
        generated_text = generated_text.strip()
        schema = df.analyze_assistant_schema(generated_text)
        relaxed_schema = _relaxed_schema_metrics(generated_text)
        for key in schema_totals:
            schema_totals[key] += int(schema.get(key, 0))
        for key in relaxed_schema_totals:
            relaxed_schema_totals[key] += int(relaxed_schema.get(key, 0))

        exact_match = 0
        if target_text is not None:
            exact_match_eligible += 1
            exact_match = int(generated_text == target_text)
            exact_match_count += exact_match

        details = _reward_details(reward_function, generated_text, verification)
        reward = float(details["reward"])
        rewards.append(reward)
        matched_counts.append(float(details["matched_point_count"]))
        if schema["strict_forecast_parseable"]:
            parseable_rewards.append(reward)
        else:
            nonparseable_rewards.append(reward)
        if relaxed_schema["relaxed_forecast_parseable"]:
            relaxed_parseable_rewards.append(reward)
        else:
            relaxed_nonparseable_rewards.append(reward)
        if details["match_coverage_vs_target"] is not None:
            coverage_values.append(float(details["match_coverage_vs_target"]))
        if details["mean_time_alignment_score"] is not None:
            time_alignment_values.append(float(details["mean_time_alignment_score"]))
        if details["mean_track_error_km"] is not None:
            track_errors_km.append(float(details["mean_track_error_km"]))
        if details["mean_intensity_error_kt"] is not None:
            intensity_errors_kt.append(float(details["mean_intensity_error_kt"]))

        vs_official = _compare_against_reference_forecast(
            generated_text,
            reference_text=target_text,
            verification=verification,
            reward_function=reward_function,
        )
        if vs_official["slot_time_match_rate"] is not None:
            slot_time_match_rates_vs_official.append(float(vs_official["slot_time_match_rate"]))
        if vs_official["mean_time_alignment_score"] is not None:
            time_alignment_scores_vs_official.append(
                float(vs_official["mean_time_alignment_score"])
            )
        if vs_official["mean_track_diff_km"] is not None:
            track_diffs_vs_official_km.append(float(vs_official["mean_track_diff_km"]))
        if vs_official["mean_intensity_diff_kt"] is not None:
            intensity_diffs_vs_official_kt.append(float(vs_official["mean_intensity_diff_kt"]))

        expert_details = (
            _reward_details(reward_function, target_text, verification)
            if target_text is not None
            else None
        )
        if expert_details is not None:
            expert_rewards.append(float(expert_details["reward"]))
            if expert_details["match_coverage_vs_target"] is not None:
                expert_coverages.append(float(expert_details["match_coverage_vs_target"]))
            if expert_details["mean_track_error_km"] is not None:
                expert_track_errors_km.append(float(expert_details["mean_track_error_km"]))
            if expert_details["mean_intensity_error_kt"] is not None:
                expert_intensity_errors_kt.append(float(expert_details["mean_intensity_error_kt"]))

        sample_record = {
            "sample_id": sample_id,
            "generated": generated_text,
            "target": target_text,
            "schema": schema,
            "relaxed_schema": relaxed_schema,
            "exact_match": exact_match,
            "reward_details": details,
            "vs_official": vs_official,
            "expert_official_truth_details": expert_details,
            "prompt_overridden": bool(sample["prompt_overridden"]),
        }
        samples.append(sample_record)

    total = len(samples)
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path) if config_path is not None else None,
        "reward_config_path": str(reward_config.config_path),
        "adapter_path": str(adapter_path) if adapter_path is not None else None,
        "uses_adapter": adapter_path is not None,
        "rl_dataset_path": str(rl_dataset_path),
        "sft_dataset_path": str(sft_dataset_path) if sft_dataset_path is not None else None,
        "sample_count": total,
        "batch_size": batch_size,
        "max_prompt_tokens": max_prompt_tokens,
        "max_new_tokens": max_new_tokens,
        "prompt_override_count": prompt_override_count,
        "decode": {
            "do_sample": do_sample,
            "temperature": temperature,
            "top_p": top_p,
        },
        "schema_counts": schema_totals,
        "schema_rates": {
            key: (value / total if total else 0.0) for key, value in schema_totals.items()
        },
        "relaxed_schema_counts": relaxed_schema_totals,
        "relaxed_schema_rates": {
            key: (value / total if total else 0.0) for key, value in relaxed_schema_totals.items()
        },
        "exact_match_count": exact_match_count,
        "exact_match_eligible": exact_match_eligible,
        "exact_match_rate": (
            exact_match_count / exact_match_eligible if exact_match_eligible else None
        ),
        "reward": {
            "mean": _safe_mean(rewards),
            "std": _safe_std(rewards),
            "p50": _safe_p50(rewards),
            "mean_when_parseable": _safe_mean(parseable_rewards),
            "mean_when_nonparseable": _safe_mean(nonparseable_rewards),
            "mean_when_relaxed_parseable": _safe_mean(relaxed_parseable_rewards),
            "mean_when_non_relaxed_parseable": _safe_mean(relaxed_nonparseable_rewards),
        },
        "matches": {
            "mean_matched_points": _safe_mean(matched_counts),
            "mean_match_coverage_vs_target": _safe_mean(coverage_values),
            "mean_time_alignment_score": _safe_mean(time_alignment_values),
        },
        "errors": {
            "mean_track_error_km": _safe_mean(track_errors_km),
            "p50_track_error_km": _safe_p50(track_errors_km),
            "mean_intensity_error_kt": _safe_mean(intensity_errors_kt),
            "p50_intensity_error_kt": _safe_p50(intensity_errors_kt),
        },
        "vs_official": {
            "mean_track_diff_km": _safe_mean(track_diffs_vs_official_km),
            "p50_track_diff_km": _safe_p50(track_diffs_vs_official_km),
            "mean_intensity_diff_kt": _safe_mean(intensity_diffs_vs_official_kt),
            "p50_intensity_diff_kt": _safe_p50(intensity_diffs_vs_official_kt),
            "slot_time_match_rate": _safe_mean(slot_time_match_rates_vs_official),
            "mean_time_alignment_score": _safe_mean(time_alignment_scores_vs_official),
        },
        "expert_official": {
            "reward_mean": _safe_mean(expert_rewards),
            "coverage_mean": _safe_mean(expert_coverages),
            "track_error_km_mean": _safe_mean(expert_track_errors_km),
            "intensity_error_kt_mean": _safe_mean(expert_intensity_errors_kt),
        },
        "examples": {
            "worst_rewards": sorted(samples, key=lambda item: item["reward_details"]["reward"])[:10],
            "format_failures": [
                sample for sample in samples if not sample["schema"]["strict_forecast_parseable"]
            ][:10],
        },
    }
    if include_samples:
        report["samples"] = samples
    return report


def evaluate(
    *,
    config_path: Path,
    reward_config_path: Path,
    adapter_path: Path | None,
    rl_dataset_path: Path,
    sft_dataset_path: Path | None,
    max_samples: int | None,
    batch_size: int,
    max_prompt_tokens: int,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    prompt_overrides: Mapping[str, str] | None = None,
    sample_ids: list[str] | None = None,
    include_samples: bool = False,
) -> dict[str, Any]:
    config, model, tokenizer = _load_inference_stack(config_path, adapter_path)
    prepared_samples = load_eval_samples(
        rl_dataset_path=rl_dataset_path,
        sft_dataset_path=sft_dataset_path,
        max_samples=max_samples,
        sample_ids=sample_ids,
        prompt_overrides=prompt_overrides,
    )

    rendered_prompts = [
        render_chat(tokenizer, sample["messages"], add_generation_prompt=True)
        for sample in prepared_samples
    ]

    outputs: list[str] = []
    for start in range(0, len(rendered_prompts), batch_size):
        batch_outputs = _generate_batch(
            model,
            tokenizer,
            rendered_prompts[start : start + batch_size],
            max_prompt_tokens=max_prompt_tokens,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
        )
        outputs.extend(text.strip() for text in batch_outputs)

    return evaluate_outputs(
        config_path=config.config_path,
        reward_config_path=reward_config_path,
        adapter_path=adapter_path,
        rl_dataset_path=rl_dataset_path,
        sft_dataset_path=sft_dataset_path,
        prepared_samples=prepared_samples,
        outputs=outputs,
        batch_size=batch_size,
        max_prompt_tokens=max_prompt_tokens,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        include_samples=include_samples,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate one adapter on held-out strict forecast prompts and verification targets."
    )
    parser.add_argument("--config", required=True, help="Path to the SFT YAML config.")
    parser.add_argument(
        "--reward-config",
        required=True,
        help="Path to the GRPO YAML config used to instantiate the reward function.",
    )
    parser.add_argument(
        "--adapter",
        help="Optional path to one saved LoRA adapter directory. Omit to evaluate the base model without any adapter.",
    )
    parser.add_argument("--rl-dataset", required=True, help="Path to held-out RL JSONL with verification.")
    parser.add_argument(
        "--sft-dataset",
        help="Optional held-out SFT JSONL used to recover gold assistant targets for exact-match scoring.",
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-prompt-tokens", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument(
        "--prompt-overrides",
        help="Optional JSON file containing sample_id -> overridden forecast user prompt.",
    )
    parser.add_argument("--output", help="Optional JSON report path.")
    args = parser.parse_args(argv)

    prompt_overrides = (
        _load_prompt_override_map(Path(args.prompt_overrides).resolve())
        if args.prompt_overrides
        else None
    )
    report = evaluate(
        config_path=Path(args.config).resolve(),
        reward_config_path=Path(args.reward_config).resolve(),
        adapter_path=Path(args.adapter).resolve() if args.adapter else None,
        rl_dataset_path=Path(args.rl_dataset).resolve(),
        sft_dataset_path=Path(args.sft_dataset).resolve() if args.sft_dataset else None,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        max_prompt_tokens=args.max_prompt_tokens,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
        prompt_overrides=prompt_overrides,
    )
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).resolve().write_text(payload, encoding="utf-8")
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
