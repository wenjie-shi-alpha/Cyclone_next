#!/usr/bin/env python3
"""Evaluate one diagnostic adapter or synthetic baseline on held-out diagnostic prompts."""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from transformers import StoppingCriteria, StoppingCriteriaList


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from cyclone_training.config import load_sft_config
from cyclone_training.datasets import render_chat
from cyclone_training.modeling import load_adapter_weights, load_model_and_tokenizer
from dataset_v2 import (
    diagnostic_allowed_values,
    normalize_diagnostic_field_value,
    normalize_diagnostic_payload,
)


BASE_MODEL_SENTINELS = {"", "base", "none", "no_adapter", "null"}
NULL_LABEL = "<null>"
NON_NULL_LABEL = "<non_null>"
SYNTHETIC_BASELINES = {"majority_label", "rule_echo"}
DIAGNOSTIC_RESPONSE_PREFIX = "{"
DIAGNOSTIC_BAD_PHRASES = ("```", "`json", "`JSON", "`")


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


def _split_prompt_and_target(messages: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], str]:
    prompt: list[dict[str, Any]] = []
    for message in messages:
        if message.get("role") == "assistant":
            return prompt, str(message.get("content", ""))
        prompt.append(dict(message))
    raise ValueError("Diagnostic SFT sample does not contain an assistant message.")


def _extract_json_object(text: str) -> tuple[str | None, bool]:
    stripped = (text or "").strip()
    if not stripped:
        return None, False

    candidate = stripped
    if candidate.startswith("```"):
        lines = candidate.splitlines()
        if lines:
            lines = lines[1:]
        while lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        candidate = "\n".join(lines).strip()
        if candidate.startswith("{") and candidate.endswith("}"):
            return candidate, candidate != stripped

    start = stripped.find("{")
    if start < 0:
        return None, False

    in_string = False
    escape = False
    depth = 0
    end: int | None = None
    for index in range(start, len(stripped)):
        char = stripped[index]
        if escape:
            escape = False
            continue
        if char == "\\":
            escape = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                end = index
                break
    if end is None:
        return None, False

    candidate = stripped[start : end + 1]
    return candidate, stripped != candidate


def _parse_json_prediction(text: str) -> dict[str, Any]:
    extracted_text, had_extra_text = _extract_json_object(text)
    if extracted_text is None:
        return {
            "json_parseable": 0,
            "json_parseable_without_extra_text": 0,
            "json_with_extra_text": 0,
            "parsed": None,
            "raw_text": text,
        }

    parse_candidates = [extracted_text]
    if extracted_text.startswith("{{"):
        parse_candidates.append(extracted_text[1:])
    parsed = None
    for candidate in parse_candidates:
        try:
            parsed = json.loads(candidate)
            extracted_text = candidate
            break
        except json.JSONDecodeError:
            continue
    if parsed is None:
        return {
            "json_parseable": 0,
            "json_parseable_without_extra_text": 0,
            "json_with_extra_text": 0,
            "parsed": None,
            "raw_text": text,
        }
    if not isinstance(parsed, dict):
        return {
            "json_parseable": 0,
            "json_parseable_without_extra_text": 0,
            "json_with_extra_text": 0,
            "parsed": None,
            "raw_text": text,
        }
    return {
        "json_parseable": 1,
        "json_parseable_without_extra_text": int(not had_extra_text and extracted_text == (text or "").strip()),
        "json_with_extra_text": int(had_extra_text or extracted_text != (text or "").strip()),
        "parsed": parsed,
        "extracted_text": extracted_text,
        "raw_text": text,
    }


def _parse_line_based_prediction(
    text: str,
    *,
    field_names: list[str],
) -> dict[str, Any] | None:
    if not text.strip():
        return None
    field_pattern = "|".join(re.escape(field_name) for field_name in field_names)
    pattern = re.compile(
        rf'^\s*[-*]?\s*"?(?P<field>{field_pattern})"?\s*:\s*(?P<value>.+?)\s*,?\s*$'
    )
    recovered: dict[str, Any] = {}
    for raw_line in text.splitlines():
        match = pattern.match(raw_line)
        if match is None:
            continue
        field_name = str(match.group("field"))
        raw_value = str(match.group("value")).strip()
        if not raw_value:
            continue
        if raw_value.startswith('"') and raw_value.endswith('"') and len(raw_value) >= 2:
            raw_value = raw_value[1:-1].strip()
        normalized = normalize_diagnostic_field_value(field_name, raw_value)
        allowed = diagnostic_allowed_values(field_name)
        if normalized is None:
            recovered[field_name] = None
            continue
        if allowed is not None and normalized not in allowed:
            continue
        recovered[field_name] = normalized
    return recovered or None


def _parse_target_payload(target_text: str) -> dict[str, Any]:
    payload = json.loads(target_text)
    if not isinstance(payload, dict):
        raise ValueError("Diagnostic target must be one JSON object.")
    return payload


def _infer_field_names(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        raise ValueError("Diagnostic dataset is empty.")
    train_metadata = rows[0].get("train_metadata", {}) or {}
    field_names = train_metadata.get("diagnostic_field_names") or []
    if field_names:
        return [str(field_name) for field_name in field_names]
    prompt_messages, target_text = _split_prompt_and_target([dict(message) for message in rows[0]["messages"]])
    del prompt_messages
    payload = _parse_target_payload(target_text)
    return [str(field_name) for field_name in payload.keys()]


def _normalize_payload(
    payload: dict[str, Any] | None,
    *,
    field_names: list[str],
) -> dict[str, Any]:
    return normalize_diagnostic_payload(payload, field_names=field_names)


def _payload_label(payload: dict[str, Any], field_name: str) -> str:
    value = payload.get(field_name)
    if value is None:
        return NULL_LABEL
    return str(value)


def _has_complete_top_level_json_object(text: str) -> bool:
    extracted_text, _had_extra_text = _extract_json_object(text)
    return extracted_text is not None


def _load_majority_payload(
    train_dataset_path: Path,
    *,
    field_names: list[str],
) -> dict[str, Any]:
    train_rows = _read_jsonl(train_dataset_path, max_samples=None)
    counters = {field_name: Counter() for field_name in field_names}
    for row in train_rows:
        _prompt_messages, target_text = _split_prompt_and_target([dict(message) for message in row["messages"]])
        payload = _normalize_payload(_parse_target_payload(target_text), field_names=field_names)
        for field_name in field_names:
            counters[field_name][_payload_label(payload, field_name)] += 1
    majority_payload: dict[str, Any] = {}
    for field_name in field_names:
        label = counters[field_name].most_common(1)[0][0]
        majority_payload[field_name] = None if label == NULL_LABEL else label
    return majority_payload


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


def _f1_score_for_positive(truths: list[str], preds: list[str], positive_label: str) -> float:
    tp = sum(1 for truth, pred in zip(truths, preds) if truth == positive_label and pred == positive_label)
    fp = sum(1 for truth, pred in zip(truths, preds) if truth != positive_label and pred == positive_label)
    fn = sum(1 for truth, pred in zip(truths, preds) if truth == positive_label and pred != positive_label)
    if tp == 0 and fp == 0 and fn == 0:
        return 0.0
    return float((2 * tp) / max(1, 2 * tp + fp + fn))


def _macro_f1(truths: list[str], preds: list[str]) -> float:
    labels = sorted(set(truths) | set(preds))
    if not labels:
        return 0.0
    return float(sum(_f1_score_for_positive(truths, preds, label) for label in labels) / len(labels))


def _binary_non_null_f1(truths: list[str], preds: list[str]) -> float:
    binary_truths = [NON_NULL_LABEL if truth != NULL_LABEL else NULL_LABEL for truth in truths]
    binary_preds = [NON_NULL_LABEL if pred != NULL_LABEL else NULL_LABEL for pred in preds]
    return _f1_score_for_positive(binary_truths, binary_preds, NON_NULL_LABEL)


def _distribution(labels: list[str]) -> dict[str, int]:
    counter = Counter(labels)
    return dict(sorted(counter.items()))


def _rate_distribution(labels: list[str]) -> dict[str, float]:
    if not labels:
        return {}
    total = len(labels)
    return {
        label: count / total
        for label, count in sorted(Counter(labels).items())
    }


def _confusion_matrix(truths: list[str], preds: list[str]) -> dict[str, dict[str, int]]:
    matrix: dict[str, Counter[str]] = {}
    for truth, pred in zip(truths, preds):
        row = matrix.setdefault(truth, Counter())
        row[pred] += 1
    return {
        truth: dict(sorted(counter.items()))
        for truth, counter in sorted(matrix.items())
    }


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


def _build_bad_words_ids(
    tokenizer: Any,
    phrases: tuple[str, ...],
) -> list[list[int]]:
    encoding_tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
    bad_words_ids: list[list[int]] = []
    for phrase in phrases:
        token_ids = encoding_tokenizer.encode(phrase, add_special_tokens=False)
        if token_ids:
            bad_words_ids.append(token_ids)
    return bad_words_ids


class StopAfterFirstJSONObject(StoppingCriteria):
    def __init__(
        self,
        tokenizer: Any,
        *,
        prompt_width: int,
        response_prefix: str = "",
    ) -> None:
        self.decoding_tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
        self.prompt_width = int(prompt_width)
        self.response_prefix = response_prefix

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
        **kwargs,
    ) -> torch.BoolTensor:
        del scores, kwargs
        completion_ids = input_ids[:, self.prompt_width :]
        completion_texts = self.decoding_tokenizer.batch_decode(
            completion_ids,
            skip_special_tokens=True,
        )
        return torch.tensor(
            [
                _has_complete_top_level_json_object(f"{self.response_prefix}{text}")
                for text in completion_texts
            ],
            device=input_ids.device,
            dtype=torch.bool,
        )


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
    response_prefix: str = "",
    bad_phrases: tuple[str, ...] = (),
    stop_after_first_json_object: bool = False,
) -> list[str]:
    device = getattr(model, "device", None)
    if device is None:
        device = next(model.parameters()).device
    prompt_texts = [prompt + response_prefix for prompt in prompts] if response_prefix else prompts
    encoded = tokenizer(
        text=prompt_texts,
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
    bad_words_ids = _build_bad_words_ids(tokenizer, bad_phrases) if bad_phrases else []
    if bad_words_ids:
        generation_kwargs["bad_words_ids"] = bad_words_ids
    if stop_after_first_json_object:
        generation_kwargs["stopping_criteria"] = StoppingCriteriaList(
            [
                StopAfterFirstJSONObject(
                    tokenizer,
                    prompt_width=prompt_width,
                    response_prefix=response_prefix,
                )
            ]
        )
    if do_sample:
        generation_kwargs["temperature"] = temperature
        generation_kwargs["top_p"] = top_p
    with torch.inference_mode():
        outputs = model.generate(**generation_kwargs)
    completions = outputs[:, prompt_width:]
    decoded = tokenizer.batch_decode(completions, skip_special_tokens=True)
    if not response_prefix:
        return decoded
    return [response_prefix + text for text in decoded]


def predict_diagnostics(
    *,
    config_path: Path | None,
    adapter_path: Path | None,
    dataset_path: Path,
    field_names: list[str] | None = None,
    max_samples: int | None,
    batch_size: int,
    max_prompt_tokens: int,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    prediction_mode: str,
    train_dataset_path: Path | None = None,
    sample_ids: list[str] | None = None,
) -> dict[str, Any]:
    rows = _read_jsonl(
        dataset_path,
        max_samples=max_samples if sample_ids is None else None,
        sample_ids=sample_ids,
    )
    inferred_field_names = field_names or _infer_field_names(rows)

    targets: list[dict[str, Any]] = []
    prompts: list[str] = []
    rendered_prompt_messages: list[list[dict[str, Any]]] = []
    sample_id_list: list[str] = []
    raw_target_texts: list[str] = []

    tokenizer = None
    model = None
    config = None
    if prediction_mode not in SYNTHETIC_BASELINES:
        if config_path is None:
            raise ValueError("config_path is required for model-based diagnostic prediction.")
        config, model, tokenizer = _load_inference_stack(config_path, adapter_path)

    for row in rows:
        prompt_messages, target_text = _split_prompt_and_target([dict(message) for message in row["messages"]])
        target_payload = _normalize_payload(_parse_target_payload(target_text), field_names=inferred_field_names)
        targets.append(target_payload)
        rendered_prompt_messages.append(prompt_messages)
        sample_id_list.append(str(row.get("sample_id", "")))
        raw_target_texts.append(target_text)
        if tokenizer is not None:
            prompts.append(render_chat(tokenizer, prompt_messages, add_generation_prompt=True))

    prediction_records: list[dict[str, Any]] = []
    if prediction_mode == "rule_echo":
        for sample_id, target_payload, target_text in zip(sample_id_list, targets, raw_target_texts):
            prediction_records.append(
                {
                    "sample_id": sample_id,
                    "generated": target_text,
                    "raw_generated": target_text,
                    "target_text": target_text,
                    "target_payload": target_payload,
                    "prediction_payload": dict(target_payload),
                    "json_parseable": 1,
                    "json_parseable_without_extra_text": 1,
                    "json_with_extra_text": 0,
                    "raw_json_parseable_without_extra_text": 1,
                    "raw_json_with_extra_text": 0,
                    "json_exact_keyset": 1,
                    "missing_keys": [],
                    "unexpected_keys": [],
                    "prediction_mode": prediction_mode,
                    "recovery_mode": "json",
                    "line_recovered": 0,
                }
            )
    elif prediction_mode == "majority_label":
        if train_dataset_path is None:
            train_dataset_path = dataset_path.with_name("train.jsonl")
        majority_payload = _load_majority_payload(train_dataset_path, field_names=inferred_field_names)
        majority_text = json.dumps(majority_payload, ensure_ascii=False, indent=2, sort_keys=True)
        for sample_id, target_payload, target_text in zip(sample_id_list, targets, raw_target_texts):
            prediction_records.append(
                {
                    "sample_id": sample_id,
                    "generated": majority_text,
                    "raw_generated": majority_text,
                    "target_text": target_text,
                    "target_payload": target_payload,
                    "prediction_payload": dict(majority_payload),
                    "json_parseable": 1,
                    "json_parseable_without_extra_text": 1,
                    "json_with_extra_text": 0,
                    "raw_json_parseable_without_extra_text": 1,
                    "raw_json_with_extra_text": 0,
                    "json_exact_keyset": 1,
                    "missing_keys": [],
                    "unexpected_keys": [],
                    "prediction_mode": prediction_mode,
                    "recovery_mode": "json",
                    "line_recovered": 0,
                }
            )
    else:
        outputs: list[str] = []
        assert model is not None
        assert tokenizer is not None
        for start in range(0, len(prompts), batch_size):
            batch_outputs = _generate_batch(
                model,
                tokenizer,
                prompts[start : start + batch_size],
                max_prompt_tokens=max_prompt_tokens,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                response_prefix=DIAGNOSTIC_RESPONSE_PREFIX,
                bad_phrases=DIAGNOSTIC_BAD_PHRASES,
                stop_after_first_json_object=True,
            )
            outputs.extend(text.strip() for text in batch_outputs)

        for sample_id, generated_text, target_payload, target_text in zip(
            sample_id_list,
            outputs,
            targets,
            raw_target_texts,
        ):
            parsed = _parse_json_prediction(generated_text)
            recovered_payload = None
            recovery_mode = "json"
            if not parsed["json_parseable"]:
                recovered_payload = _parse_line_based_prediction(
                    generated_text,
                    field_names=inferred_field_names,
                )
                if recovered_payload is not None:
                    recovery_mode = "line_kv"
                else:
                    recovery_mode = "none"
            payload_source = parsed.get("parsed") if parsed["json_parseable"] else recovered_payload
            parsed_payload = _normalize_payload(payload_source, field_names=inferred_field_names)
            parsed_keys = set((payload_source or {}).keys()) if payload_source else set()
            expected_keys = set(inferred_field_names)
            missing_keys = sorted(expected_keys - parsed_keys)
            unexpected_keys = sorted(parsed_keys - expected_keys)
            normalized_generated = (
                json.dumps(parsed_payload, ensure_ascii=False, indent=2, sort_keys=False)
                if parsed["json_parseable"]
                else None
            )
            cleaned_generated = parsed.get("extracted_text") if parsed["json_parseable"] else generated_text
            prediction_records.append(
                {
                    "sample_id": sample_id,
                    "generated": cleaned_generated,
                    "raw_generated": generated_text,
                    "generated_normalized": normalized_generated,
                    "target_text": target_text,
                    "target_payload": target_payload,
                    "prediction_payload": parsed_payload,
                    "json_parseable": int(parsed["json_parseable"]),
                    "json_parseable_without_extra_text": int(parsed["json_parseable"]),
                    "json_with_extra_text": 0 if parsed["json_parseable"] else int(parsed["json_with_extra_text"]),
                    "raw_json_parseable_without_extra_text": int(parsed["json_parseable_without_extra_text"]),
                    "raw_json_with_extra_text": int(parsed["json_with_extra_text"]),
                    "json_exact_keyset": int(parsed["json_parseable"] and not missing_keys and not unexpected_keys),
                    "missing_keys": missing_keys,
                    "unexpected_keys": unexpected_keys,
                    "prediction_mode": prediction_mode,
                    "recovery_mode": recovery_mode,
                    "line_recovered": int(recovery_mode == "line_kv"),
                }
            )

    return {
        "config_path": str(config.config_path) if config is not None else None,
        "adapter_path": str(adapter_path) if adapter_path is not None else None,
        "dataset_path": str(dataset_path),
        "field_names": inferred_field_names,
        "prediction_mode": prediction_mode,
        "sample_count": len(prediction_records),
        "predictions": prediction_records,
    }


def evaluate_predictions(
    prediction_payload: dict[str, Any],
    *,
    include_samples: bool = False,
) -> dict[str, Any]:
    field_names = list(prediction_payload["field_names"])
    samples = list(prediction_payload["predictions"])
    total = len(samples)

    parseable_count = sum(int(sample["json_parseable"]) for sample in samples)
    strict_json_only_count = sum(int(sample["json_parseable_without_extra_text"]) for sample in samples)
    extra_text_count = sum(int(sample["json_with_extra_text"]) for sample in samples)
    raw_strict_json_only_count = sum(
        int(sample.get("raw_json_parseable_without_extra_text", sample["json_parseable_without_extra_text"]))
        for sample in samples
    )
    raw_extra_text_count = sum(
        int(sample.get("raw_json_with_extra_text", sample["json_with_extra_text"]))
        for sample in samples
    )
    exact_keyset_count = sum(int(sample["json_exact_keyset"]) for sample in samples)
    line_recovered_count = sum(int(sample.get("line_recovered", 0)) for sample in samples)
    joint_exact_count = 0

    field_reports: dict[str, Any] = {}
    field_accuracies: list[float] = []
    field_macro_f1s: list[float] = []
    field_non_null_f1s: list[float] = []

    for field_name in field_names:
        truths = [_payload_label(sample["target_payload"], field_name) for sample in samples]
        preds = [_payload_label(sample["prediction_payload"], field_name) for sample in samples]
        exact_accuracy = sum(int(truth == pred) for truth, pred in zip(truths, preds)) / total if total else 0.0
        macro_f1 = _macro_f1(truths, preds)
        non_null_f1 = _binary_non_null_f1(truths, preds)
        field_reports[field_name] = {
            "exact_accuracy": exact_accuracy,
            "macro_f1": macro_f1,
            "null_vs_non_null_f1": non_null_f1,
            "target_distribution": _distribution(truths),
            "target_distribution_rate": _rate_distribution(truths),
            "prediction_distribution": _distribution(preds),
            "prediction_distribution_rate": _rate_distribution(preds),
            "confusion_matrix": _confusion_matrix(truths, preds),
        }
        field_accuracies.append(exact_accuracy)
        field_macro_f1s.append(macro_f1)
        field_non_null_f1s.append(non_null_f1)

    sample_rows: list[dict[str, Any]] = []
    for sample in samples:
        field_matches = {
            field_name: int(
                _payload_label(sample["target_payload"], field_name)
                == _payload_label(sample["prediction_payload"], field_name)
            )
            for field_name in field_names
        }
        joint_exact = int(all(field_matches.values()))
        joint_exact_count += joint_exact
        sample_row = {
            **sample,
            "field_matches": field_matches,
            "joint_exact_match": joint_exact,
        }
        sample_rows.append(sample_row)

    parseable_joint_exact_rates = [
        sample["joint_exact_match"]
        for sample in sample_rows
        if sample["json_parseable"]
    ]

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": prediction_payload.get("config_path"),
        "adapter_path": prediction_payload.get("adapter_path"),
        "dataset_path": prediction_payload["dataset_path"],
        "prediction_mode": prediction_payload["prediction_mode"],
        "field_names": field_names,
        "sample_count": total,
        "json_parseable_count": parseable_count,
        "json_parseable_rate": parseable_count / total if total else 0.0,
        "json_parseable_without_extra_text_count": strict_json_only_count,
        "json_parseable_without_extra_text_rate": strict_json_only_count / total if total else 0.0,
        "json_with_extra_text_count": extra_text_count,
        "json_with_extra_text_rate": extra_text_count / total if total else 0.0,
        "raw_json_parseable_without_extra_text_count": raw_strict_json_only_count,
        "raw_json_parseable_without_extra_text_rate": raw_strict_json_only_count / total if total else 0.0,
        "raw_json_with_extra_text_count": raw_extra_text_count,
        "raw_json_with_extra_text_rate": raw_extra_text_count / total if total else 0.0,
        "json_exact_keyset_count": exact_keyset_count,
        "json_exact_keyset_rate": exact_keyset_count / total if total else 0.0,
        "line_recovered_count": line_recovered_count,
        "line_recovered_rate": line_recovered_count / total if total else 0.0,
        "joint_exact_match_count": joint_exact_count,
        "joint_exact_match_rate": joint_exact_count / total if total else 0.0,
        "joint_exact_match_rate_when_parseable": (
            sum(parseable_joint_exact_rates) / len(parseable_joint_exact_rates)
            if parseable_joint_exact_rates
            else 0.0
        ),
        "mean_field_exact_accuracy": _safe_mean(field_accuracies),
        "mean_field_macro_f1": _safe_mean(field_macro_f1s),
        "mean_field_null_vs_non_null_f1": _safe_mean(field_non_null_f1s),
        "field_exact_accuracy_p50": _safe_p50(field_accuracies),
        "field_macro_f1_p50": _safe_p50(field_macro_f1s),
        "field_reports": field_reports,
        "parseable_schema_failures": [
            {
                "sample_id": sample["sample_id"],
                "generated": sample["generated"],
                "missing_keys": sample["missing_keys"],
                "unexpected_keys": sample["unexpected_keys"],
            }
            for sample in sample_rows
            if not sample["json_parseable"] or not sample["json_exact_keyset"]
        ][:25],
    }
    if include_samples:
        report["samples"] = sample_rows
    return report


def evaluate(
    *,
    config_path: Path | None,
    adapter_path: Path | None,
    dataset_path: Path,
    max_samples: int | None,
    batch_size: int,
    max_prompt_tokens: int,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    prediction_mode: str,
    train_dataset_path: Path | None = None,
    sample_ids: list[str] | None = None,
    include_samples: bool = False,
) -> dict[str, Any]:
    prediction_payload = predict_diagnostics(
        config_path=config_path,
        adapter_path=adapter_path,
        dataset_path=dataset_path,
        max_samples=max_samples,
        batch_size=batch_size,
        max_prompt_tokens=max_prompt_tokens,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        prediction_mode=prediction_mode,
        train_dataset_path=train_dataset_path,
        sample_ids=sample_ids,
    )
    report = evaluate_predictions(prediction_payload, include_samples=include_samples)
    report["runtime"] = {
        "requested_batch_size": batch_size,
        "max_prompt_tokens": max_prompt_tokens,
        "max_new_tokens": max_new_tokens,
        "decode": {
            "do_sample": do_sample,
            "temperature": temperature,
            "top_p": top_p,
        },
        "train_dataset_path": str(train_dataset_path) if train_dataset_path is not None else None,
    }
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate one diagnostic adapter or synthetic baseline on held-out diagnostic prompts."
    )
    parser.add_argument("--config", help="Path to the diagnostic SFT YAML config.")
    parser.add_argument(
        "--adapter",
        help="Optional path to one saved LoRA adapter directory. Omit for base model or synthetic baselines.",
    )
    parser.add_argument("--dataset", required=True, help="Path to one diagnostic SFT JSONL split.")
    parser.add_argument(
        "--prediction-mode",
        choices=["adapter", "base_model", "majority_label", "rule_echo"],
        default="adapter",
        help="How predictions should be produced.",
    )
    parser.add_argument(
        "--train-dataset",
        help="Optional train split used to derive the majority-label synthetic baseline.",
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-prompt-tokens", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--include-samples", action="store_true")
    parser.add_argument("--output", help="Optional JSON report path.")
    args = parser.parse_args(argv)

    if args.prediction_mode in {"adapter", "base_model"} and not args.config:
        raise ValueError("--config is required for adapter/base_model diagnostic evaluation.")
    if args.prediction_mode == "adapter" and not args.adapter:
        raise ValueError("--adapter is required when --prediction-mode=adapter.")
    if args.prediction_mode == "base_model":
        adapter_path = None
    else:
        adapter_path = Path(args.adapter).resolve() if args.adapter else None

    report = evaluate(
        config_path=Path(args.config).resolve() if args.config else None,
        adapter_path=adapter_path,
        dataset_path=Path(args.dataset).resolve(),
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        max_prompt_tokens=args.max_prompt_tokens,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
        prediction_mode=args.prediction_mode,
        train_dataset_path=Path(args.train_dataset).resolve() if args.train_dataset else None,
        include_samples=args.include_samples,
    )
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).resolve().write_text(payload, encoding="utf-8")
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
