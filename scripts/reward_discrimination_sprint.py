#!/usr/bin/env python3
"""Offline reward-discrimination sprint for one held-out adapter."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import statistics
import sys
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import dataset_formatter as df
from cyclone_training.config import RewardRuntimeConfig, load_grpo_config, load_sft_config
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


def _read_jsonl(path: Path, max_rows: int | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if max_rows is not None and len(rows) >= max_rows:
                break
    return rows


def _select_evenly_spaced_rows(rows: list[dict[str, Any]], prompt_count: int) -> list[dict[str, Any]]:
    if prompt_count >= len(rows):
        return rows
    if prompt_count <= 1:
        return rows[:1]

    indices = {
        round(index * (len(rows) - 1) / (prompt_count - 1))
        for index in range(prompt_count)
    }
    return [rows[index] for index in sorted(indices)]


def _safe_mean(values: Iterable[float]) -> float | None:
    values = list(values)
    if not values:
        return None
    return float(sum(values) / len(values))


def _rank_values(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(indexed):
        end = cursor
        while end + 1 < len(indexed) and indexed[end + 1][1] == indexed[cursor][1]:
            end += 1
        avg_rank = (cursor + end) / 2.0 + 1.0
        for rank_index in range(cursor, end + 1):
            original_index = indexed[rank_index][0]
            ranks[original_index] = avg_rank
        cursor = end + 1
    return ranks


def _pearson_correlation(x_values: list[float], y_values: list[float]) -> float | None:
    if len(x_values) != len(y_values) or len(x_values) < 2:
        return None
    mean_x = sum(x_values) / len(x_values)
    mean_y = sum(y_values) / len(y_values)
    centered_x = [value - mean_x for value in x_values]
    centered_y = [value - mean_y for value in y_values]
    denom_x = math.sqrt(sum(value * value for value in centered_x))
    denom_y = math.sqrt(sum(value * value for value in centered_y))
    if denom_x <= 0 or denom_y <= 0:
        return None
    numerator = sum(x_value * y_value for x_value, y_value in zip(centered_x, centered_y))
    return float(numerator / (denom_x * denom_y))


def _spearman_correlation(x_values: list[float], y_values: list[float]) -> float | None:
    if len(x_values) != len(y_values) or len(x_values) < 2:
        return None
    return _pearson_correlation(_rank_values(x_values), _rank_values(y_values))


def _load_inference_stack(config_path: Path, adapter_path: Path):
    config = load_sft_config(config_path)
    model, tokenizer = load_model_and_tokenizer(config.model, config.lora, stage="sft")
    model = load_adapter_weights(model, adapter_path)
    tokenizer.padding_side = "left"
    if hasattr(tokenizer, "truncation_side"):
        tokenizer.truncation_side = "left"

    from unsloth import FastLanguageModel

    FastLanguageModel.for_inference(model)
    return config, model, tokenizer


def _generate_samples_for_prompt(
    model: Any,
    tokenizer: Any,
    prompt: str,
    *,
    generations_per_prompt: int,
    generation_batch_size: int,
    max_prompt_tokens: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
) -> list[str]:
    device = getattr(model, "device", None)
    if device is None:
        device = next(model.parameters()).device

    outputs: list[str] = []
    while len(outputs) < generations_per_prompt:
        current_batch_size = min(generation_batch_size, generations_per_prompt - len(outputs))
        prompts = [prompt] * current_batch_size
        encoded = tokenizer(
            text=prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_prompt_tokens,
        )
        encoded = {name: tensor.to(device) for name, tensor in encoded.items()}
        prompt_width = int(encoded["input_ids"].shape[1])
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        completions = generated[:, prompt_width:]
        outputs.extend(text.strip() for text in tokenizer.batch_decode(completions, skip_special_tokens=True))
    return outputs


def _reward_details(
    reward_function: CycloneRewardFunction,
    completion_text: str,
    verification: dict[str, Any],
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
    strict_schema = df.analyze_assistant_schema(completion_text)
    relaxed_schema = inspect_forecast_schema(completion_text)
    relaxed_parseable = relaxed_schema.parsed_line_count > 0
    return {
        "reward": float(reward),
        "strict_parseable": bool(strict_schema["strict_forecast_parseable"]),
        "relaxed_parseable": bool(relaxed_parseable),
        "relaxed_clean": bool(relaxed_parseable and relaxed_schema.invalid_line_count == 0),
        "has_header": bool(relaxed_schema.has_header),
        "headerless_parseable": bool(relaxed_parseable and not relaxed_schema.has_header),
        "matched_point_count": len(matched_slots),
        "match_coverage_vs_target": (
            len(matched_slots) / len(effective_slots) if effective_slots else None
        ),
        "mean_time_alignment_score": _safe_mean(time_alignment_scores),
        "mean_track_error_km": _safe_mean(track_errors_km),
        "mean_intensity_error_kt": _safe_mean(intensity_errors_kt),
    }


def _pairwise_abs_margin(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    diffs = [
        abs(left - right)
        for left_index, left in enumerate(values)
        for right in values[left_index + 1 :]
    ]
    return float(sum(diffs) / len(diffs)) if diffs else 0.0


def _variant_name(
    *,
    track_scale_km: float,
    intensity_scale_kt: float,
    soft_slot_reward_weight: float,
    soft_slot_max_hours: float | None,
) -> str:
    if soft_slot_reward_weight <= 0.0:
        return f"legacy_track{int(track_scale_km)}_int{int(intensity_scale_kt)}"
    max_hours_text = "none" if soft_slot_max_hours is None else str(int(soft_slot_max_hours))
    return (
        f"smooth_track{int(track_scale_km)}_int{int(intensity_scale_kt)}"
        f"_softw{soft_slot_reward_weight:.2f}_maxh{max_hours_text}"
    )


def _build_variants(
    base_reward_config: RewardRuntimeConfig,
    *,
    track_scales_km: list[float],
    intensity_scales_kt: list[float],
    soft_slot_reward_weights: list[float],
    soft_slot_max_hours: float | None,
) -> list[tuple[str, RewardRuntimeConfig]]:
    variants: list[tuple[str, RewardRuntimeConfig]] = []
    seen_names: set[str] = set()
    for soft_weight, track_scale_km, intensity_scale_kt in itertools.product(
        soft_slot_reward_weights,
        track_scales_km,
        intensity_scales_kt,
    ):
        reward_config = replace(
            base_reward_config,
            track_error_scale_km=track_scale_km,
            intensity_error_scale_kt=intensity_scale_kt,
            soft_slot_reward_weight=soft_weight,
            soft_slot_max_hours=soft_slot_max_hours if soft_weight > 0.0 else None,
        )
        variant_name = _variant_name(
            track_scale_km=track_scale_km,
            intensity_scale_kt=intensity_scale_kt,
            soft_slot_reward_weight=soft_weight,
            soft_slot_max_hours=reward_config.soft_slot_max_hours,
        )
        if variant_name in seen_names:
            continue
        seen_names.add(variant_name)
        variants.append((variant_name, reward_config))
    return variants


def _summarize_variant(
    variant_name: str,
    reward_config: RewardRuntimeConfig,
    sampled_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    reward_function = CycloneRewardFunction(reward_config)

    prompt_reward_groups: list[list[float]] = []
    prompt_unique_reward_ratios: list[float] = []
    prompt_pairwise_margins: list[float] = []
    completion_rewards: list[float] = []
    strict_parseable_values: list[float] = []
    relaxed_parseable_values: list[float] = []
    relaxed_clean_values: list[float] = []
    header_present_values: list[float] = []
    headerless_parseable_count = 0
    headerless_parseable_nonzero_count = 0
    zero_reward_count = 0
    reward_track_pairs: list[tuple[float, float]] = []
    reward_intensity_pairs: list[tuple[float, float]] = []
    reward_coverage_pairs: list[tuple[float, float]] = []
    headerless_nonzero_examples: list[dict[str, Any]] = []
    zero_reward_examples: list[dict[str, Any]] = []
    zero_variance_group_examples: list[dict[str, Any]] = []

    for sampled_row in sampled_rows:
        group_rewards: list[float] = []
        group_completion_records: list[dict[str, Any]] = []
        for completion_text in sampled_row["completions"]:
            details = _reward_details(
                reward_function,
                completion_text,
                sampled_row["verification"],
            )
            reward_value = float(details["reward"])
            group_rewards.append(reward_value)
            group_completion_records.append(
                {
                    "reward": reward_value,
                    "strict_parseable": details["strict_parseable"],
                    "relaxed_parseable": details["relaxed_parseable"],
                    "has_header": details["has_header"],
                    "completion_preview": completion_text[:240],
                }
            )
            completion_rewards.append(reward_value)
            strict_parseable_values.append(1.0 if details["strict_parseable"] else 0.0)
            relaxed_parseable_values.append(1.0 if details["relaxed_parseable"] else 0.0)
            relaxed_clean_values.append(1.0 if details["relaxed_clean"] else 0.0)
            header_present_values.append(1.0 if details["has_header"] else 0.0)
            if reward_value <= 1e-12:
                zero_reward_count += 1
                if len(zero_reward_examples) < 5:
                    zero_reward_examples.append(
                        {
                            "sample_id": sampled_row["sample_id"],
                            "reward": reward_value,
                            "strict_parseable": details["strict_parseable"],
                            "relaxed_parseable": details["relaxed_parseable"],
                            "has_header": details["has_header"],
                            "completion_preview": completion_text[:240],
                        }
                    )
            if details["headerless_parseable"]:
                headerless_parseable_count += 1
                if reward_value > 1e-12:
                    headerless_parseable_nonzero_count += 1
                    if len(headerless_nonzero_examples) < 5:
                        headerless_nonzero_examples.append(
                            {
                                "sample_id": sampled_row["sample_id"],
                                "reward": reward_value,
                                "strict_parseable": details["strict_parseable"],
                                "relaxed_parseable": details["relaxed_parseable"],
                                "has_header": details["has_header"],
                                "completion_preview": completion_text[:240],
                            }
                        )
            if details["mean_track_error_km"] is not None:
                reward_track_pairs.append((reward_value, -float(details["mean_track_error_km"])))
            if details["mean_intensity_error_kt"] is not None:
                reward_intensity_pairs.append((reward_value, -float(details["mean_intensity_error_kt"])))
            if details["match_coverage_vs_target"] is not None:
                reward_coverage_pairs.append((reward_value, float(details["match_coverage_vs_target"])))

        prompt_reward_groups.append(group_rewards)
        prompt_unique_reward_ratios.append(len(set(round(value, 12) for value in group_rewards)) / len(group_rewards))
        prompt_pairwise_margins.append(_pairwise_abs_margin(group_rewards))
        if statistics.pstdev(group_rewards) <= 1e-12 and len(zero_variance_group_examples) < 8:
            zero_variance_group_examples.append(
                {
                    "sample_id": sampled_row["sample_id"],
                    "unique_outputs": len(set(sampled_row["completions"])),
                    "unique_rewards": len(set(round(value, 12) for value in group_rewards)),
                    "reward_values": group_rewards,
                    "completions": group_completion_records,
                }
            )

    zero_variance_group_share = (
        sum(statistics.pstdev(group) <= 1e-12 for group in prompt_reward_groups) / len(prompt_reward_groups)
        if prompt_reward_groups
        else None
    )
    reward_std_mean = _safe_mean(statistics.pstdev(group) for group in prompt_reward_groups)

    def correlation_with(pairs: list[tuple[float, float]]) -> float | None:
        if len(pairs) < 2:
            return None
        reward_values = [pair[0] for pair in pairs]
        metric_values = [pair[1] for pair in pairs]
        return _spearman_correlation(reward_values, metric_values)

    coverage_corr = correlation_with(reward_coverage_pairs)
    track_corr = correlation_with(reward_track_pairs)
    intensity_corr = correlation_with(reward_intensity_pairs)
    strict_parseable_corr = _spearman_correlation(completion_rewards, strict_parseable_values)
    relaxed_parseable_corr = _spearman_correlation(completion_rewards, relaxed_parseable_values)

    positive_alignment_terms = [
        value
        for value in (coverage_corr, track_corr, intensity_corr, relaxed_parseable_corr)
        if value is not None
    ]
    alignment_score = _safe_mean(positive_alignment_terms) or 0.0
    discrimination_score = (
        0.35 * (_safe_mean(prompt_unique_reward_ratios) or 0.0)
        + 0.35 * (_safe_mean(prompt_pairwise_margins) or 0.0)
        + 0.15 * (1.0 - (zero_variance_group_share or 0.0))
        + 0.15 * max(0.0, alignment_score)
    )

    return {
        "variant_name": variant_name,
        "reward_config": asdict(reward_config),
        "prompt_count": len(sampled_rows),
        "completion_count": len(completion_rewards),
        "metrics": {
            "unique_reward_ratio": _safe_mean(prompt_unique_reward_ratios),
            "pairwise_reward_margin": _safe_mean(prompt_pairwise_margins),
            "zero_variance_group_share": zero_variance_group_share,
            "reward_std_mean": reward_std_mean,
            "strict_parseable_rate": _safe_mean(strict_parseable_values),
            "relaxed_parseable_rate": _safe_mean(relaxed_parseable_values),
            "relaxed_clean_rate": _safe_mean(relaxed_clean_values),
            "header_present_rate": _safe_mean(header_present_values),
            "headerless_parseable_rate": (
                headerless_parseable_count / len(completion_rewards) if completion_rewards else None
            ),
            "headerless_parseable_nonzero_reward_rate": (
                headerless_parseable_nonzero_count / headerless_parseable_count
                if headerless_parseable_count
                else None
            ),
            "zero_reward_rate": (
                zero_reward_count / len(completion_rewards) if completion_rewards else None
            ),
            "reward_vs_coverage_spearman": coverage_corr,
            "reward_vs_neg_track_error_spearman": track_corr,
            "reward_vs_neg_intensity_error_spearman": intensity_corr,
            "reward_vs_strict_parseable_spearman": strict_parseable_corr,
            "reward_vs_relaxed_parseable_spearman": relaxed_parseable_corr,
            "selection_score": discrimination_score,
        },
        "diagnostics": {
            "headerless_parseable_count": headerless_parseable_count,
            "headerless_parseable_nonzero_reward_count": headerless_parseable_nonzero_count,
            "zero_reward_count": zero_reward_count,
            "headerless_nonzero_examples": headerless_nonzero_examples,
            "zero_reward_examples": zero_reward_examples,
            "zero_variance_group_examples": zero_variance_group_examples,
        },
    }


def _parse_float_list(raw_text: str) -> list[float]:
    values: list[float] = []
    for chunk in raw_text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        values.append(float(chunk))
    if not values:
        raise ValueError("Expected at least one numeric value.")
    return values


def run_sprint(
    *,
    config_path: Path,
    reward_config_path: Path,
    adapter_path: Path,
    rl_dataset_path: Path,
    prompt_count: int,
    generations_per_prompt: int,
    generation_batch_size: int,
    max_prompt_tokens: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    track_scales_km: list[float],
    intensity_scales_kt: list[float],
    soft_slot_reward_weights: list[float],
    soft_slot_max_hours: float | None,
) -> dict[str, Any]:
    config, model, tokenizer = _load_inference_stack(config_path, adapter_path)
    grpo_config = load_grpo_config(reward_config_path)
    rl_rows = _read_jsonl(rl_dataset_path, max_rows=None)
    rl_rows = _select_evenly_spaced_rows(rl_rows, prompt_count=prompt_count)

    sampled_rows: list[dict[str, Any]] = []
    for row in rl_rows:
        prompt = render_chat(tokenizer, [dict(message) for message in row["messages"]], add_generation_prompt=True)
        completions = _generate_samples_for_prompt(
            model,
            tokenizer,
            prompt,
            generations_per_prompt=generations_per_prompt,
            generation_batch_size=generation_batch_size,
            max_prompt_tokens=max_prompt_tokens,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )
        sampled_rows.append(
            {
                "sample_id": str(row.get("sample_id", "")),
                "verification": dict(row["verification"]),
                "completions": completions,
            }
        )

    variants = _build_variants(
        grpo_config.reward,
        track_scales_km=track_scales_km,
        intensity_scales_kt=intensity_scales_kt,
        soft_slot_reward_weights=soft_slot_reward_weights,
        soft_slot_max_hours=soft_slot_max_hours,
    )
    variant_summaries = [
        _summarize_variant(variant_name, reward_config, sampled_rows)
        for variant_name, reward_config in variants
    ]
    variant_summaries.sort(
        key=lambda item: item["metrics"]["selection_score"] or float("-inf"),
        reverse=True,
    )

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config.config_path),
        "reward_config_path": str(grpo_config.config_path),
        "adapter_path": str(adapter_path),
        "rl_dataset_path": str(rl_dataset_path),
        "prompt_count": prompt_count,
        "generations_per_prompt": generations_per_prompt,
        "generation_batch_size": generation_batch_size,
        "max_prompt_tokens": max_prompt_tokens,
        "max_new_tokens": max_new_tokens,
        "decode": {
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
        },
        "variant_summaries": variant_summaries,
        "sample_preview": [
            {
                "sample_id": sampled_row["sample_id"],
                "unique_outputs": len(set(sampled_row["completions"])),
            }
            for sampled_row in sampled_rows[: min(5, len(sampled_rows))]
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one offline reward-discrimination sprint.")
    parser.add_argument("--config", required=True, help="Path to the SFT YAML config.")
    parser.add_argument(
        "--reward-config",
        required=True,
        help="Path to the GRPO YAML config used to seed reward variants.",
    )
    parser.add_argument("--adapter", required=True, help="Path to the LoRA adapter to probe.")
    parser.add_argument("--rl-dataset", required=True, help="Path to held-out RL JSONL with verification.")
    parser.add_argument("--prompt-count", type=int, default=16)
    parser.add_argument("--generations-per-prompt", type=int, default=16)
    parser.add_argument("--generation-batch-size", type=int, default=4)
    parser.add_argument("--max-prompt-tokens", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--track-scales-km", default="100,125,150")
    parser.add_argument("--intensity-scales-kt", default="8,10,12")
    parser.add_argument("--soft-slot-reward-weights", default="0,0.08")
    parser.add_argument("--soft-slot-max-hours", type=float, default=36.0)
    parser.add_argument("--output", help="Optional JSON report path.")
    args = parser.parse_args(argv)

    report = run_sprint(
        config_path=Path(args.config).resolve(),
        reward_config_path=Path(args.reward_config).resolve(),
        adapter_path=Path(args.adapter).resolve(),
        rl_dataset_path=Path(args.rl_dataset).resolve(),
        prompt_count=args.prompt_count,
        generations_per_prompt=args.generations_per_prompt,
        generation_batch_size=args.generation_batch_size,
        max_prompt_tokens=args.max_prompt_tokens,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        track_scales_km=_parse_float_list(args.track_scales_km),
        intensity_scales_kt=_parse_float_list(args.intensity_scales_kt),
        soft_slot_reward_weights=_parse_float_list(args.soft_slot_reward_weights),
        soft_slot_max_hours=args.soft_slot_max_hours,
    )
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).resolve().write_text(payload, encoding="utf-8")
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
