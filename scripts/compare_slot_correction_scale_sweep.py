#!/usr/bin/env python3
"""Evaluate track calibration scale variants for slot-correction rendering."""

from __future__ import annotations

import argparse
import json
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

import eval_strict_forecast_heldout as forecast_eval
from compare_slot_correction_forecast_integration import (
    _build_variant_report_path,
    _model_dir,
    _summary_markdown,
    _write_json,
    _write_text,
    _build_report,
    _load_canonical_rows_by_sample_id,
)
from slot_locked_forecast_correction import render_slot_correction_forecast_text


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sample_ids_from_prepared(prepared_samples: list[dict[str, Any]]) -> list[str]:
    return [str(sample["sample_id"]) for sample in prepared_samples]


def _render_scaled_outputs(
    *,
    sample_ids: list[str],
    canonical_rows_by_id: dict[str, dict[str, Any]],
    payload_by_id: dict[str, dict[str, Any]],
    calibration: dict[str, Any],
    baseline_text_by_id: dict[str, str],
    offset_scale: float,
) -> list[str]:
    outputs: list[str] = []
    for sample_id in sample_ids:
        outputs.append(
            render_slot_correction_forecast_text(
                canonical_rows_by_id[sample_id],
                payload_by_id[sample_id],
                calibration=calibration,
                intensity_source="baseline_forecast",
                intensity_reference_text=baseline_text_by_id[sample_id],
                offset_scale=offset_scale,
            )
        )
    return outputs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Sweep scale variants for slot-correction offsets.")
    parser.add_argument("--reward-config", required=True)
    parser.add_argument("--forecast-rl-dataset", required=True)
    parser.add_argument("--forecast-sft-dataset", required=True)
    parser.add_argument("--canonical-test", required=True)
    parser.add_argument("--baseline-report", required=True)
    parser.add_argument("--predicted-payload", required=True)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--official-report", required=True)
    parser.add_argument(
        "--scales",
        required=True,
        help="Comma-separated scale values, e.g. 0.9,1.0,1.1",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    reward_config_path = Path(args.reward_config).resolve()
    forecast_rl_dataset_path = Path(args.forecast_rl_dataset).resolve()
    forecast_sft_dataset_path = Path(args.forecast_sft_dataset).resolve()
    canonical_test_path = Path(args.canonical_test).resolve()
    baseline_report_path = Path(args.baseline_report).resolve()
    predicted_payload_path = Path(args.predicted_payload).resolve()
    calibration_path = Path(args.calibration).resolve()
    official_report_path = Path(args.official_report).resolve()
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model_dir = _model_dir(output_path)
    model_dir.mkdir(parents=True, exist_ok=True)

    baseline_report = _load_json(baseline_report_path)
    official_report = _load_json(official_report_path)
    predicted_payload = _load_json(predicted_payload_path)
    calibration = _load_json(calibration_path)
    prepared_samples = forecast_eval.load_eval_samples(
        rl_dataset_path=forecast_rl_dataset_path,
        sft_dataset_path=forecast_sft_dataset_path,
        max_samples=None,
        sample_ids=[str(sample["sample_id"]) for sample in baseline_report.get("samples", [])],
        prompt_overrides=None,
    )
    sample_ids = _sample_ids_from_prepared(prepared_samples)
    canonical_rows_by_id = _load_canonical_rows_by_sample_id(canonical_test_path, sample_ids)
    baseline_text_by_id = {
        str(sample["sample_id"]): str(sample["generated"] or "")
        for sample in baseline_report.get("samples", []) or []
    }
    payload_by_id = {
        str(record["sample_id"]): dict(record["prediction_payload"])
        for record in predicted_payload["predictions"]
    }

    baseline_copy_path = _build_variant_report_path(model_dir, 1, baseline_report["variant_label"])
    _write_json(baseline_copy_path, baseline_report)
    official_copy_path = _build_variant_report_path(model_dir, 2, official_report["variant_label"])
    _write_json(official_copy_path, official_report)
    model_reports: list[dict[str, Any]] = [baseline_report, official_report]
    for index, raw_scale in enumerate(args.scales.split(","), start=3):
        scale = float(raw_scale.strip())
        outputs = _render_scaled_outputs(
            sample_ids=sample_ids,
            canonical_rows_by_id=canonical_rows_by_id,
            payload_by_id=payload_by_id,
            calibration=calibration,
            baseline_text_by_id=baseline_text_by_id,
            offset_scale=scale,
        )
        report = forecast_eval.evaluate_outputs(
            config_path=None,
            reward_config_path=reward_config_path,
            adapter_path=None,
            rl_dataset_path=forecast_rl_dataset_path,
            sft_dataset_path=forecast_sft_dataset_path,
            prepared_samples=prepared_samples,
            outputs=outputs,
            batch_size=4,
            max_prompt_tokens=1792,
            max_new_tokens=160,
            do_sample=False,
            temperature=0.0,
            top_p=1.0,
            include_samples=True,
        )
        label = f"predicted_slot_locked_track_plus_baseline_intensity_scale_{scale:.2f}"
        report["variant_label"] = label
        report["variant_kind"] = "predicted_rendered_baseline_intensity_scaled"
        report["forecast_adapter_path"] = baseline_report.get("forecast_adapter_path")
        report["diagnostic_adapter_path"] = predicted_payload.get("adapter_path")
        report["diagnostic_prediction_mode"] = predicted_payload.get("prediction_mode")
        report["slot_correction_payload_path"] = str(predicted_payload_path)
        report["offset_scale"] = scale
        report_path = _build_variant_report_path(model_dir, index, label)
        _write_json(report_path, report)
        model_reports.append(report)

    artifact_paths = {
        "baseline_report": str(baseline_report_path),
        "official_report": str(official_report_path),
        "predicted_payload": str(predicted_payload_path),
        "calibration": str(calibration_path),
    }
    report = _build_report(
        output_path=output_path,
        artifact_paths=artifact_paths,
        args=argparse.Namespace(
            forecast_config=baseline_report.get("config_path") or str(ROOT / "configs/training/sft_gemma_4_e4b_unsloth_forecast_phase1_baseline_v2.yaml"),
            reward_config=str(reward_config_path),
            forecast_adapter=baseline_report.get("forecast_adapter_path"),
            forecast_rl_dataset=str(forecast_rl_dataset_path),
            forecast_sft_dataset=str(forecast_sft_dataset_path),
            diagnostic_dataset=str(predicted_payload_path),
            diagnostic_config=None,
            diagnostic_adapter=predicted_payload.get("adapter_path"),
            diagnostic_prediction_mode=predicted_payload.get("prediction_mode"),
            canonical_train=str(canonical_test_path),
            canonical_test=str(canonical_test_path),
            sample_count=len(sample_ids),
            sample_seed=3407,
            baseline_label="baseline_forecast_sft_v2",
            official_label="expert_official",
        ),
        model_reports=model_reports,
    )
    report["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    _write_json(output_path, report)
    _write_text(output_path.with_suffix(".summary.md"), _summary_markdown(report))
    print(f"[compare_slot_correction_scale_sweep] completed output={output_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
