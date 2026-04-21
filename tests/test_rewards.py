from __future__ import annotations

import math
import unittest

from cyclone_training.config import RewardRuntimeConfig
from cyclone_training.rewards import (
    CycloneRewardFunction,
    _extract_truth_issue_time,
    match_forecast_to_truth_slots,
    parse_forecast_points,
    parse_target_forecast_slots,
)


def _build_verification(*slot_times: str) -> dict:
    future_best_track = []
    for slot_time in slot_times:
        future_best_track.append(
            {
                "valid_time_utc": slot_time,
                "lat": 10.0,
                "lon": -20.0,
                "vmax_kt": 50.0,
            }
        )
    return {
        "best_track_at_issue": {
            "matched_datetime_utc": "2024-01-01T03:00:00Z",
        },
        "forecast_slots": [{"valid_time_utc": slot_time} for slot_time in slot_times],
        "future_best_track": future_best_track,
    }


class RewardFunctionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = RewardRuntimeConfig()
        self.reward = CycloneRewardFunction(self.config)

    def test_exact_slot_match_without_header_keeps_full_accuracy_reward(self) -> None:
        verification = _build_verification("2024-01-01T09:00:00Z")
        completion = "- Day01 0900Z | 10.0°N 20.0°W | 50 kt"

        reward = self.reward.score_one(completion, verification)

        self.assertAlmostEqual(reward, 1.0, places=6)

    def test_near_slot_match_gets_partial_credit(self) -> None:
        verification = _build_verification("2024-01-01T15:00:00Z")
        completion = "- Day01 0900Z | 10.0°N 20.0°W | 50 kt"

        reward = self.reward.score_one(completion, verification)

        self.assertGreater(reward, 0.02)
        self.assertLess(reward, 1.0)
        self.assertAlmostEqual(reward, math.exp(-1.0), places=6)

    def test_beyond_slot_tolerance_falls_back_to_format_bonus(self) -> None:
        verification = _build_verification("2024-01-01T09:00:00Z")
        completion = "- Day01 2100Z | 10.0°N 20.0°W | 50 kt"

        reward = self.reward.score_one(completion, verification)

        self.assertAlmostEqual(reward, 0.02, places=6)

    def test_header_only_gets_no_credit(self) -> None:
        verification = _build_verification("2024-01-01T09:00:00Z")
        completion = "Official forecast:"

        reward = self.reward.score_one(completion, verification)

        self.assertAlmostEqual(reward, 0.0, places=6)

    def test_reasoning_prefix_still_allows_accuracy_credit(self) -> None:
        verification = _build_verification("2024-01-01T09:00:00Z")
        completion = "Reasoning: confidence is low\n- Day01 0900Z | 10.0°N 20.0°W | 50 kt"

        reward = self.reward.score_one(completion, verification)

        self.assertAlmostEqual(reward, 0.5, places=6)

    def test_unresolvable_time_line_does_not_collapse_to_partial_only(self) -> None:
        verification = _build_verification(
            "2024-01-01T09:00:00Z",
            "2024-01-01T21:00:00Z",
        )
        completion = (
            "- Day01 0900Z | 10.0°N 20.0°W | 50 kt\n"
            "- Day32 0900Z | 10.0°N 20.0°W | 50 kt"
        )

        reward = self.reward.score_one(completion, verification)

        self.assertGreater(reward, 0.02)

    def test_soft_matching_is_one_to_one(self) -> None:
        verification = _build_verification(
            "2024-01-01T09:00:00Z",
            "2024-01-01T21:00:00Z",
        )
        completion = "Official forecast:\n- Day01 1500Z | 10.0°N 20.0°W | 50 kt"
        issue_time = _extract_truth_issue_time(verification)
        forecast_points = parse_forecast_points(completion, issue_time)
        target_slots = parse_target_forecast_slots(verification, issue_time)

        _, matched_slots = match_forecast_to_truth_slots(
            forecast_points,
            target_slots,
            verification["future_best_track"],
            truth_slot_tolerance_hours=self.config.truth_slot_tolerance_hours,
            forecast_slot_tolerance_hours=self.config.forecast_slot_tolerance_hours,
            forecast_slot_time_scale_hours=self.config.forecast_slot_time_scale_hours,
        )

        self.assertEqual(len(matched_slots), 1)

    def test_soft_slot_reward_uses_accuracy_when_enabled(self) -> None:
        config = RewardRuntimeConfig(
            track_error_scale_km=100.0,
            intensity_error_scale_kt=8.0,
            soft_slot_max_hours=24.0,
            soft_slot_reward_weight=0.1,
        )
        reward = CycloneRewardFunction(config)
        verification = _build_verification("2024-01-01T09:00:00Z")

        close_completion = "- Day01 1500Z | 10.0°N 20.0°W | 50 kt"
        far_completion = "- Day01 1500Z | 20.0°N 40.0°W | 80 kt"

        close_reward = reward.score_one(close_completion, verification)
        far_reward = reward.score_one(far_completion, verification)

        self.assertGreater(close_reward, far_reward)
        self.assertGreater(close_reward, 0.02)


if __name__ == "__main__":
    unittest.main()
