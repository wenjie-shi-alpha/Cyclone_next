#!/usr/bin/env python3
"""Build per-sample forecast prompt overrides with injected diagnostic JSON."""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import eval_diagnostic_heldout as diagnostic_eval
from dataset_v2 import DIAGNOSTIC_TRACK_CORRECTION_FIELDS


ORACLE_MODE = "oracle"
TRACK_CORRECTION_FIELD_NAMES = frozenset(DIAGNOSTIC_TRACK_CORRECTION_FIELDS)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def _read_jsonl_sample_ids(path: Path) -> list[str]:
    sample_ids: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            sample_id = str(row.get("sample_id", ""))
            if not sample_id:
                raise ValueError(f"Missing sample_id in {path}")
            sample_ids.append(sample_id)
    return sample_ids


def _select_sample_ids(
    dataset_path: Path,
    sample_count: int | None,
    sample_seed: int,
) -> list[str]:
    sample_ids = _read_jsonl_sample_ids(dataset_path)
    if sample_count is None or sample_count >= len(sample_ids):
        return sample_ids

    rng = random.Random(sample_seed)
    selected = set(rng.sample(sample_ids, sample_count))
    return [sample_id for sample_id in sample_ids if sample_id in selected]


def _read_rows_by_sample_id(
    path: Path,
    sample_ids: list[str],
) -> list[dict[str, Any]]:
    wanted = {sample_id: index for index, sample_id in enumerate(sample_ids)}
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
    missing = [sample_id for sample_id in sample_ids if sample_id not in rows_by_id]
    if missing:
        preview = ", ".join(missing[:5])
        raise ValueError(f"Missing {len(missing)} sample_ids in {path}: {preview}")
    return [rows_by_id[sample_id] for sample_id in sample_ids]


def _extract_user_prompt(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages):
        if message.get("role") == "user":
            return str(message.get("content", ""))
    raise ValueError("Forecast RL sample does not contain a user prompt.")


def _is_track_correction_payload(diagnostic_payload: dict[str, Any]) -> bool:
    return any(field_name in TRACK_CORRECTION_FIELD_NAMES for field_name in diagnostic_payload)


def _format_injected_prompt(
    original_prompt: str,
    *,
    diagnostic_payload: dict[str, Any],
    source_label: str,
    section_title: str,
) -> str:
    diagnostic_text = json.dumps(
        diagnostic_payload,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
    instruction_lines = [f"- Source: {source_label}"]
    if _is_track_correction_payload(diagnostic_payload):
        instruction_lines.extend(
            [
                (
                    "- These fixed 48h/72h track corrections are relative to the ATCF consensus "
                    "guidance already shown above, using the nearest visible 48h-ish and 72h-ish anchors."
                ),
                "- Use this structured diagnostic assessment only as auxiliary track guidance when producing the official forecast table.",
                "- Do not change forecast Day/HHMMZ slots because of this block.",
                "- Do not change intensity because of this block.",
                "- Do not repeat this diagnostic block in the answer; return only the forecast table.",
            ]
        )
    else:
        instruction_lines.extend(
            [
                "- Use this structured diagnostic assessment as auxiliary guidance when producing the official forecast table.",
                "- Do not repeat this diagnostic block in the answer; return only the forecast table.",
            ]
        )
    injection_block = "\n\n".join(
        [f"## {section_title}", *instruction_lines, diagnostic_text]
    )
    return original_prompt.rstrip() + "\n\n" + injection_block + "\n"


def _build_source_label(
    *,
    injection_mode: str,
    prediction_mode: str | None,
    adapter_path: Path | None,
) -> str:
    if injection_mode == ORACLE_MODE:
        return "oracle_reference"
    if prediction_mode == "adapter":
        if adapter_path is None:
            return "predicted_diagnostic_adapter"
        return f"predicted_diagnostic_adapter:{adapter_path.name}"
    if prediction_mode == "base_model":
        return "predicted_base_model"
    if prediction_mode == "majority_label":
        return "predicted_majority_label"
    if prediction_mode == "rule_echo":
        return "predicted_rule_echo"
    return "predicted_diagnostics"


def build_prompt_overrides(
    *,
    forecast_rl_dataset_path: Path,
    diagnostic_dataset_path: Path,
    injection_mode: str,
    sample_count: int | None,
    sample_seed: int,
    diagnostic_config_path: Path | None,
    diagnostic_adapter_path: Path | None,
    diagnostic_prediction_mode: str,
    diagnostic_train_dataset_path: Path | None,
    batch_size: int,
    max_prompt_tokens: int,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    section_title: str,
) -> dict[str, Any]:
    sample_ids = _select_sample_ids(forecast_rl_dataset_path, sample_count, sample_seed)
    forecast_rows = _read_rows_by_sample_id(forecast_rl_dataset_path, sample_ids)

    prediction_mode = "rule_echo" if injection_mode == ORACLE_MODE else diagnostic_prediction_mode
    prediction_payload = diagnostic_eval.predict_diagnostics(
        config_path=diagnostic_config_path,
        adapter_path=diagnostic_adapter_path,
        dataset_path=diagnostic_dataset_path,
        field_names=None,
        max_samples=None,
        batch_size=batch_size,
        max_prompt_tokens=max_prompt_tokens,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        prediction_mode=prediction_mode,
        train_dataset_path=diagnostic_train_dataset_path,
        sample_ids=sample_ids,
    )
    prediction_by_id = {
        str(record["sample_id"]): record for record in prediction_payload["predictions"]
    }

    source_label = _build_source_label(
        injection_mode=injection_mode,
        prediction_mode=prediction_mode,
        adapter_path=diagnostic_adapter_path,
    )
    overrides: dict[str, str] = {}
    diagnostics_by_id: dict[str, dict[str, Any]] = {}
    records: list[dict[str, Any]] = []

    for row in forecast_rows:
        sample_id = str(row["sample_id"])
        original_prompt = _extract_user_prompt(list(row.get("messages") or []))
        prediction_record = prediction_by_id[sample_id]
        diagnostic_payload = dict(prediction_record["prediction_payload"])
        overridden_prompt = _format_injected_prompt(
            original_prompt,
            diagnostic_payload=diagnostic_payload,
            source_label=source_label,
            section_title=section_title,
        )
        overrides[sample_id] = overridden_prompt
        diagnostics_by_id[sample_id] = diagnostic_payload
        records.append(
            {
                "sample_id": sample_id,
                "source_label": source_label,
                "diagnostic_payload": diagnostic_payload,
                "original_prompt_chars": len(original_prompt),
                "overridden_prompt_chars": len(overridden_prompt),
            }
        )

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "forecast_rl_dataset_path": str(forecast_rl_dataset_path),
        "diagnostic_dataset_path": str(diagnostic_dataset_path),
        "diagnostic_train_dataset_path": (
            str(diagnostic_train_dataset_path)
            if diagnostic_train_dataset_path is not None
            else None
        ),
        "diagnostic_config_path": (
            str(diagnostic_config_path) if diagnostic_config_path is not None else None
        ),
        "diagnostic_adapter_path": (
            str(diagnostic_adapter_path) if diagnostic_adapter_path is not None else None
        ),
        "injection_mode": injection_mode,
        "diagnostic_prediction_mode": prediction_mode,
        "sample_count": len(sample_ids),
        "requested_sample_count": sample_count,
        "sample_seed": sample_seed,
        "sample_ids": sample_ids,
        "section_title": section_title,
        "source_label": source_label,
        "field_names": list(prediction_payload["field_names"]),
        "runtime": {
            "requested_batch_size": batch_size,
            "max_prompt_tokens": max_prompt_tokens,
            "max_new_tokens": max_new_tokens,
            "decode": {
                "do_sample": do_sample,
                "temperature": temperature,
                "top_p": top_p,
            },
        },
        "overrides": overrides,
        "diagnostics": diagnostics_by_id,
        "records": records,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build sample_id -> forecast prompt overrides with injected diagnostic JSON."
    )
    parser.add_argument("--forecast-rl-dataset", required=True, help="Forecast RL JSONL split.")
    parser.add_argument("--diagnostic-dataset", required=True, help="Diagnostic SFT JSONL split.")
    parser.add_argument(
        "--injection-mode",
        choices=[ORACLE_MODE, "predicted"],
        required=True,
        help="Whether to inject oracle labels or model-predicted diagnostics.",
    )
    parser.add_argument("--diagnostic-config", help="Diagnostic SFT YAML config.")
    parser.add_argument("--diagnostic-adapter", help="Diagnostic adapter path.")
    parser.add_argument(
        "--diagnostic-prediction-mode",
        choices=["adapter", "base_model", "majority_label", "rule_echo"],
        default="adapter",
        help="How predicted diagnostics should be produced when --injection-mode=predicted.",
    )
    parser.add_argument(
        "--diagnostic-train-dataset",
        help="Optional train split used for majority-label diagnostic predictions.",
    )
    parser.add_argument("--sample-count", type=int, default=None)
    parser.add_argument("--sample-seed", type=int, default=3407)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-prompt-tokens", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument(
        "--section-title",
        default="Structured Diagnostic Assessment",
        help="Heading used for the injected prompt block.",
    )
    parser.add_argument("--output", required=True, help="JSON output path.")
    args = parser.parse_args(argv)

    if args.injection_mode == "predicted":
        if args.diagnostic_prediction_mode in {"adapter", "base_model"} and not args.diagnostic_config:
            raise ValueError("--diagnostic-config is required for model-based predicted diagnostics.")
        if args.diagnostic_prediction_mode == "adapter" and not args.diagnostic_adapter:
            raise ValueError("--diagnostic-adapter is required when prediction mode is 'adapter'.")

    payload = build_prompt_overrides(
        forecast_rl_dataset_path=Path(args.forecast_rl_dataset).resolve(),
        diagnostic_dataset_path=Path(args.diagnostic_dataset).resolve(),
        injection_mode=args.injection_mode,
        sample_count=args.sample_count,
        sample_seed=args.sample_seed,
        diagnostic_config_path=Path(args.diagnostic_config).resolve() if args.diagnostic_config else None,
        diagnostic_adapter_path=(
            Path(args.diagnostic_adapter).resolve() if args.diagnostic_adapter else None
        ),
        diagnostic_prediction_mode=args.diagnostic_prediction_mode,
        diagnostic_train_dataset_path=(
            Path(args.diagnostic_train_dataset).resolve()
            if args.diagnostic_train_dataset
            else None
        ),
        batch_size=args.batch_size,
        max_prompt_tokens=args.max_prompt_tokens,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
        section_title=args.section_title,
    )
    _write_json(Path(args.output).resolve(), payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
