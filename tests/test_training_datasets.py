from __future__ import annotations

import json
import unittest
from collections import Counter
from pathlib import Path

from cyclone_training.config import DataConfig
from cyclone_training.datasets import _apply_sft_resampling


def _diagnostic_record(sample_id: str, payload: dict[str, object]) -> dict[str, object]:
    return {
        "sample_id": sample_id,
        "messages": [
            {"role": "system", "content": "Return one JSON object."},
            {"role": "user", "content": "Prompt."},
            {"role": "assistant", "content": json.dumps(payload, ensure_ascii=False)},
        ],
        "train_metadata": {},
    }


class TrainingDatasetResamplingTests(unittest.TestCase):
    def test_apply_sft_resampling_upsamples_rare_diagnostic_labels(self) -> None:
        records = [
            _diagnostic_record(
                "majority_a",
                {
                    "track_control_signal": "subtropical_high",
                    "turning_signal": "steady",
                    "model_agreement_level": "high",
                    "forecast_confidence_level": "high",
                },
            ),
            _diagnostic_record(
                "majority_b",
                {
                    "track_control_signal": "subtropical_high",
                    "turning_signal": "steady",
                    "model_agreement_level": "high",
                    "forecast_confidence_level": "high",
                },
            ),
            _diagnostic_record(
                "rare_turn",
                {
                    "track_control_signal": "midlatitude_trough",
                    "turning_signal": "recurvature",
                    "model_agreement_level": "low",
                    "forecast_confidence_level": "low",
                },
            ),
        ]
        config = DataConfig(
            dataset_root=Path("."),
            sft_resample_fields=[
                "track_control_signal",
                "turning_signal",
                "model_agreement_level",
                "forecast_confidence_level",
            ],
            sft_resample_power=0.5,
            sft_resample_max_multiplier=4,
        )

        expanded = _apply_sft_resampling(records, config)
        sample_counts = Counter(str(record["sample_id"]) for record in expanded)

        self.assertEqual(sample_counts["majority_a"], 1)
        self.assertEqual(sample_counts["majority_b"], 1)
        self.assertGreaterEqual(sample_counts["rare_turn"], 2)
        self.assertLessEqual(sample_counts["rare_turn"], 4)

        rare_record = next(record for record in expanded if record["sample_id"] == "rare_turn")
        self.assertEqual(
            rare_record["train_metadata"]["sft_resample_fields"],
            config.sft_resample_fields,
        )
        self.assertEqual(
            rare_record["train_metadata"]["sft_resample_multiplier"],
            sample_counts["rare_turn"],
        )

    def test_apply_sft_resampling_honors_label_min_multiplier_overrides(self) -> None:
        records = [
            _diagnostic_record(
                "competing",
                {
                    "track_control_signal": "competing_ridge_and_trough",
                    "turning_signal": "steady",
                },
            ),
            _diagnostic_record(
                "subtropical_a",
                {
                    "track_control_signal": "subtropical_high",
                    "turning_signal": "steady",
                },
            ),
            _diagnostic_record(
                "subtropical_b",
                {
                    "track_control_signal": "subtropical_high",
                    "turning_signal": "steady",
                },
            ),
        ]
        config = DataConfig(
            dataset_root=Path("."),
            sft_resample_fields=["track_control_signal"],
            sft_resample_power=0.0,
            sft_resample_max_multiplier=5,
            sft_resample_label_min_multipliers={
                "track_control_signal": {
                    "competing_ridge_and_trough": 4,
                }
            },
        )

        expanded = _apply_sft_resampling(records, config)
        sample_counts = Counter(str(record["sample_id"]) for record in expanded)

        self.assertEqual(sample_counts["competing"], 4)
        self.assertEqual(sample_counts["subtropical_a"], 1)
        self.assertEqual(sample_counts["subtropical_b"], 1)


if __name__ == "__main__":
    unittest.main()
