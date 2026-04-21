from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import dataset_formatter as df
from dataset_v2 import legacy_sample_from_canonical


class DatasetFormatterGuidanceTests(unittest.TestCase):
    def test_classify_turning_signal_from_points_detects_recurvature(self) -> None:
        points = [
            {"consensus_lat": 18.0, "consensus_lon": -60.0},
            {"consensus_lat": 18.4, "consensus_lon": -61.5},
            {"consensus_lat": 19.4, "consensus_lon": -61.0},
            {"consensus_lat": 21.0, "consensus_lon": -59.5},
        ]
        self.assertEqual(
            df.classify_turning_signal_from_points(
                points,
                lat_key="consensus_lat",
                lon_key="consensus_lon",
            ),
            "recurvature",
        )

    def test_derive_guidance_target_leads_from_official_forecast_slots(self) -> None:
        track_table = [
            {"valid_day": 17, "valid_hhmmz": "0000"},
            {"valid_day": 17, "valid_hhmmz": "1200"},
            {"valid_day": 18, "valid_hhmmz": "0000"},
            {"valid_day": 18, "valid_hhmmz": "1200"},
        ]
        leads = df.derive_guidance_target_leads(track_table, "2024-11-16T15:00:00Z")
        self.assertEqual(leads, [9, 21, 33, 45])

    def test_select_representative_guidance_points_prefers_official_target_leads(self) -> None:
        points = [
            {"lead_from_issue_h": 9, "model_count": 200},
            {"lead_from_issue_h": 12, "model_count": 1},
            {"lead_from_issue_h": 21, "model_count": 180},
            {"lead_from_issue_h": 24, "model_count": 1},
            {"lead_from_issue_h": 33, "model_count": 160},
            {"lead_from_issue_h": 36, "model_count": 1},
            {"lead_from_issue_h": 45, "model_count": 140},
            {"lead_from_issue_h": 48, "model_count": 1},
        ]
        selected = df.select_representative_guidance_points(points, target_leads=[9, 21, 33, 45])
        self.assertEqual([int(point["lead_from_issue_h"]) for point in selected], [9, 21, 33, 45])

    def test_select_representative_guidance_points_does_not_backfill_earlier_leads(self) -> None:
        points = [
            {"lead_from_issue_h": 3, "model_count": 33},
            {"lead_from_issue_h": 9, "model_count": 26},
            {"lead_from_issue_h": 15, "model_count": 4},
        ]
        selected = df.select_representative_guidance_points(points, target_leads=[9, 21, 33, 45])
        self.assertEqual([int(point["lead_from_issue_h"]) for point in selected], [9, 15])

    def test_format_atcf_guidance_uses_truthful_header_and_per_line_model_count(self) -> None:
        atcf = {
            "status": "available",
            "model_count": 248,
            "consensus_spread_points_future": [
                {
                    "lead_from_issue_h": 9,
                    "valid_time_utc": "2024-11-17T00:00:00Z",
                    "model_count": 248,
                    "consensus_lat": 16.4,
                    "consensus_lon": -86.9,
                    "consensus_vmax_kt": 40,
                    "consensus_mslp_hpa": 1000,
                    "track_spread_km": 1490,
                    "wind_spread_kt": 8,
                },
                {
                    "lead_from_issue_h": 21,
                    "valid_time_utc": "2024-11-17T12:00:00Z",
                    "model_count": 1,
                    "consensus_lat": 16.9,
                    "consensus_lon": -88.0,
                    "consensus_vmax_kt": 40,
                    "consensus_mslp_hpa": 1001,
                    "track_spread_km": 0,
                    "wind_spread_kt": 0,
                },
            ],
        }
        text = df.format_atcf_guidance(
            atcf,
            issue_time_utc="2024-11-16T15:00:00Z",
            target_leads=[9, 21],
        )
        self.assertTrue(text.startswith("ATCF guidance:"))
        self.assertNotIn("Consensus guidance", text)
        self.assertIn("| 248 models", text)
        self.assertIn("| 1 model", text)

    def test_format_turning_guidance_cues_emits_compact_turn_summary(self) -> None:
        atcf = {
            "status": "available",
            "consensus_spread_points_future": [
                {"lead_from_issue_h": 12, "consensus_lat": 18.0, "consensus_lon": -60.0, "track_spread_km": 120},
                {"lead_from_issue_h": 24, "consensus_lat": 18.5, "consensus_lon": -61.4, "track_spread_km": 180},
                {"lead_from_issue_h": 36, "consensus_lat": 19.5, "consensus_lon": -60.8, "track_spread_km": 320},
                {"lead_from_issue_h": 48, "consensus_lat": 21.0, "consensus_lon": -59.2, "track_spread_km": 620},
            ],
        }
        hres = {
            "track_intensity_points_future": [
                {"lead_from_issue_h": 12, "lat": 18.1, "lon": -60.1},
                {"lead_from_issue_h": 24, "lat": 18.6, "lon": -61.2},
                {"lead_from_issue_h": 36, "lat": 19.4, "lon": -60.5},
                {"lead_from_issue_h": 48, "lat": 20.8, "lon": -58.9},
            ]
        }
        text = df.format_turning_guidance_cues(
            atcf,
            hres,
            issue_time_utc="2024-11-16T15:00:00Z",
            target_leads=[12, 24, 36, 48],
        )
        self.assertIn("- ATCF:", text)
        self.assertIn("recurvature signal", text)
        self.assertIn("- HRES:", text)

    def test_format_track_inflection_guidance_cues_emits_candidate_summary(self) -> None:
        atcf = {
            "status": "available",
            "consensus_spread_points_future": [
                {"lead_from_issue_h": 12, "consensus_lat": 18.0, "consensus_lon": -60.0, "track_spread_km": 120},
                {"lead_from_issue_h": 24, "consensus_lat": 18.5, "consensus_lon": -61.4, "track_spread_km": 180},
                {"lead_from_issue_h": 36, "consensus_lat": 19.5, "consensus_lon": -60.8, "track_spread_km": 320},
                {"lead_from_issue_h": 48, "consensus_lat": 21.0, "consensus_lon": -59.2, "track_spread_km": 620},
            ],
        }
        hres = {
            "track_intensity_points_future": [
                {"lead_from_issue_h": 12, "lat": 18.1, "lon": -60.1},
                {"lead_from_issue_h": 24, "lat": 18.6, "lon": -61.2},
                {"lead_from_issue_h": 36, "lat": 19.4, "lon": -60.5},
                {"lead_from_issue_h": 48, "lat": 20.8, "lon": -58.9},
            ]
        }
        text = df.format_track_inflection_guidance_cues(
            atcf,
            hres,
            issue_time_utc="2024-11-16T15:00:00Z",
            target_leads=[12, 24, 36, 48],
        )
        self.assertIn("regime recurving", text)
        self.assertIn("turn window 24 to 48h", text)
        self.assertIn("direction eastward escape", text)
        self.assertIn("magnitude sharp", text)
        self.assertIn("Cross-model cue", text)

    def test_format_track_correction_guidance_cues_emits_fixed_lead_anchors(self) -> None:
        atcf = {
            "status": "available",
            "consensus_spread_points_future": [
                {
                    "lead_from_issue_h": 45,
                    "valid_time_utc": "2024-11-18T12:00:00Z",
                    "consensus_lat": 20.0,
                    "consensus_lon": -60.0,
                    "track_spread_km": 180,
                    "model_count": 32,
                },
                {
                    "lead_from_issue_h": 69,
                    "valid_time_utc": "2024-11-19T12:00:00Z",
                    "consensus_lat": 22.5,
                    "consensus_lon": -58.2,
                    "track_spread_km": 260,
                    "model_count": 28,
                },
            ],
        }
        text = df.format_track_correction_guidance_cues(
            atcf,
            issue_time_utc="2024-11-16T15:00:00Z",
        )
        self.assertIn("48h anchor", text)
        self.assertIn("72h anchor", text)
        self.assertIn("Day18 1200Z", text)
        self.assertIn("lead 45h", text)
        self.assertIn("track-only reference points", text)

    def test_format_slot_locked_correction_guidance_cues_emits_slot_locked_reference_lines(self) -> None:
        atcf = {
            "status": "available",
            "consensus_spread_points_future": [
                {
                    "lead_from_issue_h": 9,
                    "valid_time_utc": "2024-11-17T00:00:00Z",
                    "consensus_lat": 16.4,
                    "consensus_lon": -86.9,
                    "track_spread_km": 149,
                },
                {
                    "lead_from_issue_h": 21,
                    "valid_time_utc": "2024-11-17T12:00:00Z",
                    "consensus_lat": 16.9,
                    "consensus_lon": -88.0,
                    "track_spread_km": 212,
                },
                {
                    "lead_from_issue_h": 33,
                    "valid_time_utc": "2024-11-18T00:00:00Z",
                    "consensus_lat": 17.5,
                    "consensus_lon": -89.4,
                    "track_spread_km": 265,
                },
            ],
        }
        official_track_table = [
            {"valid_day": 17, "valid_hhmmz": "0000"},
            {"valid_day": 17, "valid_hhmmz": "1200"},
            {"valid_day": 18, "valid_hhmmz": "0000"},
            {"valid_day": 18, "valid_hhmmz": "1200"},
        ]
        text = df.format_slot_locked_correction_guidance_cues(
            atcf,
            issue_time_utc="2024-11-16T15:00:00Z",
            official_track_table=official_track_table,
        )
        self.assertIn("predict only the lat/lon correction bucket", text)
        self.assertIn("Keep slot count and slot order fixed", text)
        self.assertIn("slot_1: Day17 0000Z", text)
        self.assertIn("slot_2: Day17 1200Z", text)
        self.assertIn("slot_3: Day18 0000Z", text)
        self.assertIn("If a later slot field has no matching ATCF point above, leave that field null.", text)

    def test_format_steering_competition_cues_emits_ridge_trough_summary(self) -> None:
        env_features = {
            "subtropical_high": {"level": "强"},
            "westerly_trough": {"level": "中等"},
            "monsoon_trough": {"level": "弱"},
        }
        atcf = {
            "status": "available",
            "consensus_spread_points_future": [
                {"lead_from_issue_h": 12, "consensus_lat": 18.0, "consensus_lon": -60.0},
                {"lead_from_issue_h": 24, "consensus_lat": 18.4, "consensus_lon": -61.4},
                {"lead_from_issue_h": 36, "consensus_lat": 19.4, "consensus_lon": -60.7},
                {"lead_from_issue_h": 48, "consensus_lat": 20.8, "consensus_lon": -59.1},
            ],
        }
        hres = {
            "track_intensity_points_future": [
                {"lead_from_issue_h": 12, "lat": 18.1, "lon": -60.1},
                {"lead_from_issue_h": 24, "lat": 18.5, "lon": -61.2},
                {"lead_from_issue_h": 36, "lat": 19.6, "lon": -60.5},
                {"lead_from_issue_h": 48, "lat": 21.0, "lon": -59.0},
            ]
        }
        text = df.format_steering_competition_cues(
            env_features,
            atcf,
            hres,
            issue_time_utc="2024-11-16T15:00:00Z",
            target_leads=[12, 24, 36, 48],
        )
        self.assertIn("subtropical high strong", text)
        self.assertIn("westerly trough moderate", text)
        self.assertIn("competition regime", text)
        self.assertIn("Guidance regime cue", text)

    def test_real_sample_guidance_aligns_to_official_slots(self) -> None:
        canonical_path = ROOT / "data" / "training_rebuilt_v2_20260414_guidancefix" / "canonical_v2" / "test.jsonl"
        if not canonical_path.exists():
            self.skipTest(f"Missing canonical dataset: {canonical_path}")

        sample_id = "Atlantic_2024319N16282_2024-11-16T15:00:00Z_012"
        record = None
        with canonical_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                candidate = json.loads(line)
                if candidate.get("sample_id") == sample_id:
                    record = candidate
                    break
        if record is None:
            self.skipTest(f"Missing regression sample: {sample_id}")

        formatted = df.format_sft_sample(
            legacy_sample_from_canonical(record),
            view=df.SFT_VIEW_STRICT_FORECAST,
        )
        self.assertIsNotNone(formatted)
        user_content = next(
            message["content"]
            for message in formatted["messages"]
            if message["role"] == "user"
        )
        self.assertIn("ATCF guidance:", user_content)
        self.assertIn("- Day17 0000Z", user_content)
        self.assertIn("- Day17 1200Z", user_content)
        self.assertIn("- Day18 0000Z", user_content)
        self.assertIn("- Day18 1200Z", user_content)
        self.assertNotIn("- Day17 0300Z", user_content)


if __name__ == "__main__":
    unittest.main()
