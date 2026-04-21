from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from slot_locked_forecast_correction import (
    SLOT_CORRECTION_FIELD_NAMES,
    build_slot_correction_calibration,
    derive_slot_correction_payload,
    render_slot_correction_forecast_text,
    render_visible_consensus_forecast_text,
)


class SlotLockedForecastCorrectionTests(unittest.TestCase):
    def test_derive_slot_correction_payload_uses_visible_slot_matching(self) -> None:
        payload = derive_slot_correction_payload(
            "2024-11-16T15:00:00Z",
            [
                {"valid_day": 17, "valid_hhmmz": "0000", "lat": 10.8, "lon": -61.3},
                {"valid_day": 17, "valid_hhmmz": "1200", "lat": 11.0, "lon": -60.4},
            ],
            [
                {
                    "lead_from_issue_h": 9,
                    "valid_time_utc": "2024-11-17T00:00:00Z",
                    "consensus_lat": 10.0,
                    "consensus_lon": -60.0,
                },
                {
                    "lead_from_issue_h": 21,
                    "valid_time_utc": "2024-11-17T12:00:00Z",
                    "consensus_lat": 10.9,
                    "consensus_lon": -60.1,
                },
            ],
        )
        self.assertEqual(payload["slot_1_lat_bias_vs_consensus_bucket"], "north_small")
        self.assertEqual(payload["slot_1_lon_bias_vs_consensus_bucket"], "west_small")
        self.assertEqual(payload["slot_2_lat_bias_vs_consensus_bucket"], "near_consensus")
        self.assertEqual(payload["slot_2_lon_bias_vs_consensus_bucket"], "near_consensus")
        for field_name in SLOT_CORRECTION_FIELD_NAMES[4:]:
            self.assertIsNone(payload[field_name])

    def test_build_slot_correction_calibration_uses_observed_medians(self) -> None:
        records = [
            {
                "issue_time": "2024-11-16T15:00:00Z",
                "targets": {
                    "official_forecast_points": [
                        {"valid_day": 17, "valid_hhmmz": "0000", "lat": 11.0, "lon": -61.0},
                    ]
                },
                "inputs": {
                    "model_guidance": {
                        "atcf_consensus": {
                            "consensus_spread_points_future": [
                                {
                                    "lead_from_issue_h": 9,
                                    "valid_time_utc": "2024-11-17T00:00:00Z",
                                    "consensus_lat": 10.0,
                                    "consensus_lon": -60.0,
                                }
                            ]
                        }
                    }
                },
            },
            {
                "issue_time": "2024-11-16T15:00:00Z",
                "targets": {
                    "official_forecast_points": [
                        {"valid_day": 17, "valid_hhmmz": "0000", "lat": 11.2, "lon": -61.2},
                    ]
                },
                "inputs": {
                    "model_guidance": {
                        "atcf_consensus": {
                            "consensus_spread_points_future": [
                                {
                                    "lead_from_issue_h": 9,
                                    "valid_time_utc": "2024-11-17T00:00:00Z",
                                    "consensus_lat": 10.0,
                                    "consensus_lon": -60.0,
                                }
                            ]
                        }
                    }
                },
            },
        ]
        calibration = build_slot_correction_calibration(records)
        self.assertAlmostEqual(
            calibration["slot_bucket_offsets_km"]["1"]["lat"]["north_small"],
            122.452,
            places=3,
        )
        self.assertLess(
            calibration["slot_bucket_offsets_km"]["1"]["lon"]["west_small"],
            0.0,
        )
        self.assertEqual(
            calibration["slot_bucket_offsets_km"]["1"]["lat"]["near_consensus"],
            0.0,
        )

    def test_renderers_keep_official_slots_and_apply_track_only_offsets(self) -> None:
        record = {
            "issue_time": "2024-11-16T15:00:00Z",
            "targets": {
                "official_forecast_points": [
                    {"valid_day": 17, "valid_hhmmz": "0000", "lat": 99.0, "lon": -99.0},
                ]
            },
            "inputs": {
                "model_guidance": {
                    "atcf_consensus": {
                        "consensus_spread_points_future": [
                            {
                                "lead_from_issue_h": 9,
                                "valid_time_utc": "2024-11-17T00:00:00Z",
                                "consensus_lat": 0.0,
                                "consensus_lon": -60.0,
                                "consensus_vmax_kt": 40,
                            }
                        ]
                    }
                }
            },
        }
        visible_text = render_visible_consensus_forecast_text(record)
        self.assertIn("- Day17 0000Z | 0.0°N 60.0°W | 40 kt", visible_text)

        corrected_text = render_slot_correction_forecast_text(
            record,
            {
                "slot_1_lat_bias_vs_consensus_bucket": "north_small",
                "slot_1_lon_bias_vs_consensus_bucket": "west_small",
            },
            calibration={
                "slot_bucket_offsets_km": {
                    "1": {
                        "lat": {"north_small": 111.32},
                        "lon": {"west_small": -111.32},
                    }
                }
            },
        )
        self.assertIn("- Day17 0000Z | 1.0°N 61.0°W | 40 kt", corrected_text)
        self.assertNotIn("99.0°N 99.0°W", corrected_text)


if __name__ == "__main__":
    unittest.main()
