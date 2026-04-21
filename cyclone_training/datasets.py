"""Dataset loading helpers for SFT and GRPO."""

from __future__ import annotations

import json
import logging
import math
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Optional

from .config import DataConfig
from .utils import limit_records, read_jsonl


LOGGER = logging.getLogger("cyclone_training.datasets")
NULL_RESAMPLE_LABEL = "<null>"


def _import_datasets():
    try:
        from datasets import Dataset
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise ImportError("The 'datasets' package is required for training.") from exc
    return Dataset


def _is_processor_like(processing_class: Any) -> bool:
    try:
        from transformers import ProcessorMixin
    except ImportError:  # pragma: no cover - runtime dependency
        return hasattr(processing_class, "tokenizer")
    return isinstance(processing_class, ProcessorMixin)


def normalize_chat_messages(
    processing_class: Any,
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not _is_processor_like(processing_class):
        return [dict(message) for message in messages]

    normalized: list[dict[str, Any]] = []
    for message in messages:
        content = message.get("content", "")
        if isinstance(content, list):
            content_blocks = [
                dict(block) if isinstance(block, Mapping) else {"type": "text", "text": str(block)}
                for block in content
            ]
        else:
            content_blocks = [{"type": "text", "text": str(content)}]
        normalized.append({**message, "content": content_blocks})
    return normalized


def render_chat(tokenizer: Any, messages: list[dict[str, Any]], add_generation_prompt: bool) -> str:
    """Render one chat transcript with the tokenizer template when available."""
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            normalize_chat_messages(tokenizer, messages),
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )

    role_prefix = {
        "system": "[SYSTEM]",
        "user": "[USER]",
        "assistant": "[ASSISTANT]",
    }
    chunks: list[str] = []
    for message in messages:
        role = role_prefix.get(message.get("role", ""), "[MESSAGE]")
        content = str(message.get("content", "")).strip()
        chunks.append(f"{role}\n{content}")
    if add_generation_prompt:
        chunks.append("[ASSISTANT]")
    return "\n\n".join(chunks)


def _split_prompt_completion_messages(
    sample_id: str,
    messages: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    for idx in range(len(messages) - 1, -1, -1):
        message = messages[idx]
        if message.get("role") == "assistant":
            prompt = [dict(item) for item in messages[:idx]]
            completion = [dict(item) for item in messages[idx: idx + 1]]
            if prompt and completion:
                return prompt, completion
            break
    raise ValueError(
        f"SFT sample {sample_id} does not contain a usable assistant turn for prompt/completion training."
    )


def _build_sft_records(
    path: Optional[str | Path | Any],
    tokenizer: Any,
    max_samples: int | None,
    record_format: str,
    *,
    data_config: DataConfig,
    split_name: str,
) -> list[dict[str, Any]]:
    if path is None:
        return []

    records = list(limit_records(read_jsonl(path), max_samples))
    if split_name == "train":
        records = _apply_sft_resampling(records, data_config)
    prepared: list[dict[str, Any]] = []
    for record in records:
        messages = [dict(message) for message in record["messages"]]
        row = {
            "sample_id": record["sample_id"],
            "train_metadata": dict(record.get("train_metadata", {})),
            "format_version": record.get("format_version"),
        }
        if record_format == "text":
            row["messages"] = messages
            row["text"] = render_chat(tokenizer, row["messages"], add_generation_prompt=False)
        elif record_format == "messages":
            row["messages"] = messages
        elif record_format == "prompt_completion":
            prompt, completion = _split_prompt_completion_messages(record["sample_id"], messages)
            row["prompt"] = normalize_chat_messages(tokenizer, prompt)
            row["completion"] = normalize_chat_messages(tokenizer, completion)
        else:
            raise ValueError(f"Unsupported SFT record format: {record_format}")
        prepared.append(row)
    return prepared


def _extract_sft_target_payload(record: Mapping[str, Any]) -> dict[str, Any] | None:
    messages = [dict(message) for message in record.get("messages", []) or []]
    if not messages:
        return None
    try:
        _prompt, completion = _split_prompt_completion_messages(
            str(record.get("sample_id", "")),
            messages,
        )
    except ValueError:
        return None
    content = completion[0].get("content", "")
    if not isinstance(content, str):
        return None
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _resample_label_key(value: Any) -> str:
    if value is None:
        return NULL_RESAMPLE_LABEL
    if isinstance(value, (str, int, float, bool)):
        return str(value)
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _compute_sft_resample_multiplier(
    payload: Mapping[str, Any] | None,
    *,
    fields: list[str],
    field_counters: dict[str, Counter[str]],
    field_max_counts: dict[str, int],
    label_min_multipliers: dict[str, dict[str, int]],
    power: float,
    max_multiplier: int,
) -> int:
    if not payload:
        return 1

    rarity_weight = 1.0
    for field_name in fields:
        if field_name not in payload:
            continue
        label_key = _resample_label_key(payload.get(field_name))
        label_count = field_counters[field_name].get(label_key, 0)
        max_count = field_max_counts.get(field_name, 0)
        if label_count <= 0 or max_count <= 0:
            continue
        rarity_weight = max(
            rarity_weight,
            (max_count / label_count) ** power,
        )
        rarity_weight = max(
            rarity_weight,
            float(label_min_multipliers.get(field_name, {}).get(label_key, 1)),
        )

    return max(1, min(max_multiplier, int(math.ceil(rarity_weight))))


def _apply_sft_resampling(
    records: list[Mapping[str, Any]],
    data_config: DataConfig,
) -> list[Mapping[str, Any]]:
    fields = [
        str(field_name).strip()
        for field_name in data_config.sft_resample_fields
        if str(field_name).strip()
    ]
    power = float(data_config.sft_resample_power)
    max_multiplier = int(data_config.sft_resample_max_multiplier)
    label_min_multipliers = {
        str(field_name): {
            str(label): max(1, int(multiplier))
            for label, multiplier in (value_map or {}).items()
            if str(label).strip()
        }
        for field_name, value_map in (data_config.sft_resample_label_min_multipliers or {}).items()
        if str(field_name).strip()
    }
    if not records or not fields or max_multiplier <= 1:
        return records
    if power <= 0.0 and not label_min_multipliers:
        return records

    payloads: list[dict[str, Any] | None] = []
    field_counters = {field_name: Counter() for field_name in fields}
    invalid_payload_count = 0
    for record in records:
        payload = _extract_sft_target_payload(record)
        payloads.append(payload)
        if payload is None:
            invalid_payload_count += 1
            continue
        for field_name in fields:
            if field_name not in payload:
                continue
            field_counters[field_name][_resample_label_key(payload.get(field_name))] += 1

    field_max_counts = {
        field_name: max(counter.values(), default=0)
        for field_name, counter in field_counters.items()
    }
    expanded_records: list[Mapping[str, Any]] = []
    multiplier_histogram: Counter[int] = Counter()
    duplicated_sample_count = 0

    for record, payload in zip(records, payloads):
        multiplier = _compute_sft_resample_multiplier(
            payload,
            fields=fields,
            field_counters=field_counters,
            field_max_counts=field_max_counts,
            label_min_multipliers=label_min_multipliers,
            power=power,
            max_multiplier=max_multiplier,
        )
        multiplier_histogram[multiplier] += 1
        if multiplier > 1:
            duplicated_sample_count += 1

        for copy_index in range(multiplier):
            cloned = dict(record)
            train_metadata = dict(cloned.get("train_metadata", {}) or {})
            train_metadata.update(
                {
                    "sft_resample_fields": list(fields),
                    "sft_resample_multiplier": multiplier,
                    "sft_resample_copy_index": copy_index,
                }
            )
            cloned["train_metadata"] = train_metadata
            expanded_records.append(cloned)

    LOGGER.info(
        "Applied SFT resampling | source=%s | expanded=%s | duplicated=%s | invalid_payloads=%s | fields=%s | power=%.3f | max_multiplier=%s | label_min_multipliers=%s | multiplier_hist=%s",
        len(records),
        len(expanded_records),
        duplicated_sample_count,
        invalid_payload_count,
        fields,
        power,
        max_multiplier,
        label_min_multipliers,
        dict(sorted(multiplier_histogram.items())),
    )
    return expanded_records


def load_sft_datasets(
    data_config: DataConfig,
    tokenizer: Any,
    record_format: str = "text",
):
    """Load train/eval SFT splits as Hugging Face datasets."""
    Dataset = _import_datasets()
    train_records = _build_sft_records(
        data_config.sft_train_file,
        tokenizer=tokenizer,
        max_samples=data_config.max_train_samples,
        record_format=record_format,
        data_config=data_config,
        split_name="train",
    )
    eval_records = _build_sft_records(
        data_config.sft_eval_file,
        tokenizer=tokenizer,
        max_samples=data_config.max_eval_samples,
        record_format=record_format,
        data_config=data_config,
        split_name="eval",
    )
    train_dataset = Dataset.from_list(train_records)
    eval_dataset = Dataset.from_list(eval_records) if eval_records else None
    return train_dataset, eval_dataset


def _build_grpo_records(
    path: Optional[str | Any],
    tokenizer: Any,
    max_samples: int | None,
) -> list[dict[str, Any]]:
    if path is None:
        return []

    records = limit_records(read_jsonl(path), max_samples)
    prepared: list[dict[str, Any]] = []
    for record in records:
        messages = normalize_chat_messages(tokenizer, [dict(message) for message in record["messages"]])
        prepared.append(
            {
                "sample_id": record["sample_id"],
                "prompt": messages,
                "verification": dict(record["verification"]),
                "train_metadata": dict(record.get("train_metadata", {})),
                "format_version": record.get("format_version"),
            }
        )
    return prepared


def load_grpo_datasets(data_config: DataConfig, tokenizer: Any):
    """Load train/eval GRPO splits as Hugging Face datasets."""
    Dataset = _import_datasets()
    train_records = _build_grpo_records(
        data_config.rl_train_file,
        tokenizer=tokenizer,
        max_samples=data_config.max_train_samples,
    )
    eval_records = _build_grpo_records(
        data_config.rl_eval_file,
        tokenizer=tokenizer,
        max_samples=data_config.max_eval_samples,
    )
    train_dataset = Dataset.from_list(train_records)
    eval_dataset = Dataset.from_list(eval_records) if eval_records else None
    return train_dataset, eval_dataset
