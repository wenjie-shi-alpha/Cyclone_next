#!/usr/bin/env python3
"""Probe whether one SFT adapter learned the strict forecast output format."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import dataset_formatter as df
from cyclone_training.config import load_sft_config
from cyclone_training.datasets import render_chat
from cyclone_training.modeling import load_adapter_weights, load_model_and_tokenizer


def _read_jsonl(path: Path, max_samples: int | None) -> list[dict[str, Any]]:
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
    raise ValueError("SFT sample does not contain an assistant message.")


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


def _generate_batch(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    *,
    max_prompt_tokens: int,
    max_new_tokens: int,
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
    with torch.inference_mode():
        outputs = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    completions = outputs[:, prompt_width:]
    return tokenizer.batch_decode(completions, skip_special_tokens=True)


def evaluate(
    *,
    config_path: Path,
    adapter_path: Path,
    dataset_path: Path,
    max_samples: int | None,
    batch_size: int,
    max_prompt_tokens: int,
    max_new_tokens: int,
) -> dict[str, Any]:
    config, model, tokenizer = _load_inference_stack(config_path, adapter_path)
    rows = _read_jsonl(dataset_path, max_samples=max_samples)

    rendered_prompts: list[str] = []
    targets: list[str] = []
    sample_ids: list[str] = []
    for row in rows:
        prompt_messages, target_text = _split_prompt_and_target(list(row["messages"]))
        rendered_prompts.append(render_chat(tokenizer, prompt_messages, add_generation_prompt=True))
        targets.append(target_text.strip())
        sample_ids.append(str(row.get("sample_id", "")))

    outputs: list[str] = []
    for start in range(0, len(rendered_prompts), batch_size):
        batch_prompts = rendered_prompts[start : start + batch_size]
        batch_outputs = _generate_batch(
            model,
            tokenizer,
            batch_prompts,
            max_prompt_tokens=max_prompt_tokens,
            max_new_tokens=max_new_tokens,
        )
        outputs.extend(text.strip() for text in batch_outputs)

    schema_totals = {
        "strict_forecast_parseable": 0,
        "strict_forecast_with_extra_text": 0,
        "no_track_forecast": 0,
        "reasoning_only": 0,
    }
    exact_match_count = 0
    failure_examples: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []

    for sample_id, target_text, generated_text in zip(sample_ids, targets, outputs):
        schema = df.analyze_assistant_schema(generated_text)
        for key in schema_totals:
            schema_totals[key] += int(schema.get(key, 0))
        exact_match = int(generated_text == target_text)
        exact_match_count += exact_match
        sample_record = {
            "sample_id": sample_id,
            "generated": generated_text,
            "target": target_text,
            "schema": schema,
            "exact_match": exact_match,
        }
        samples.append(sample_record)
        if not schema.get("strict_forecast_parseable"):
            failure_examples.append(sample_record)

    total = len(samples)
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config.config_path),
        "adapter_path": str(adapter_path),
        "dataset_path": str(dataset_path),
        "sample_count": total,
        "batch_size": batch_size,
        "max_prompt_tokens": max_prompt_tokens,
        "max_new_tokens": max_new_tokens,
        "schema_counts": schema_totals,
        "schema_rates": {
            key: (value / total if total else 0.0) for key, value in schema_totals.items()
        },
        "exact_match_count": exact_match_count,
        "exact_match_rate": exact_match_count / total if total else 0.0,
        "failure_examples": failure_examples[:20],
        "samples": samples,
    }
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate strict-format adherence for one SFT adapter.")
    parser.add_argument("--config", required=True, help="Path to the SFT YAML config.")
    parser.add_argument("--adapter", required=True, help="Path to one saved LoRA adapter directory.")
    parser.add_argument("--dataset", help="Path to an SFT JSONL split. Defaults to the config eval split.")
    parser.add_argument("--max-samples", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-prompt-tokens", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--output", help="Optional JSON report path.")
    args = parser.parse_args(argv)

    config = load_sft_config(args.config)
    dataset_path = Path(args.dataset) if args.dataset else config.data.sft_eval_file
    if dataset_path is None:
        raise ValueError("No dataset path provided and the SFT config does not define sft_eval_file.")

    report = evaluate(
        config_path=Path(args.config).resolve(),
        adapter_path=Path(args.adapter).resolve(),
        dataset_path=Path(dataset_path).resolve(),
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        max_prompt_tokens=args.max_prompt_tokens,
        max_new_tokens=args.max_new_tokens,
    )
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).resolve().write_text(payload, encoding="utf-8")
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
