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
from dataset_v2 import (
    DIAGNOSTIC_FIELDS,
    DIAGNOSTIC_SLOT_CORRECTION_FIELDS,
    DIAGNOSTIC_TRACK_CORRECTION_FIELDS,
    _derive_track_correction_candidate,
    canonicalize_legacy_sample,
    infer_latest_legacy_raw_dir,
    legacy_sample_from_canonical,
    normalize_diagnostic_field_value,
    read_json_with_retry,
)
from export_views_v2 import (
    DIAGNOSTIC_VIEW_SPECS,
    _build_diagnostic_system_prompt,
    format_diagnostic_sample,
)


class DatasetV2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        raw_base_dir = infer_latest_legacy_raw_dir(ROOT)
        if raw_base_dir is None:
            raise unittest.SkipTest("No legacy raw dataset directory found for tests.")
        train_dir = raw_base_dir / "train"
        sample_path = next(iter(sorted(train_dir.glob("*.json"))))
        cls.legacy_sample = read_json_with_retry(sample_path)
        cls.canonical_sample = canonicalize_legacy_sample(cls.legacy_sample, source_split="train")

    def test_canonical_schema_has_expected_top_level_keys(self) -> None:
        sample = self.canonical_sample
        for key in [
            "sample_id",
            "storm_id",
            "basin",
            "issue_time",
            "lead_times",
            "source_split",
            "time_anchor_complete",
            "input_window_spec",
            "inputs",
            "targets",
            "diagnostics",
            "flags",
            "metadata",
        ]:
            self.assertIn(key, sample)
        self.assertEqual(sample["metadata"]["canonical_version"], "tc_canonical_v2.0.0")

    def test_forecast_roundtrip_matches_legacy_formatter(self) -> None:
        expected = df.format_sft_sample(
            self.legacy_sample,
            view=df.SFT_VIEW_STRICT_FORECAST,
        )
        roundtrip = legacy_sample_from_canonical(self.canonical_sample)
        actual = df.format_sft_sample(
            roundtrip,
            view=df.SFT_VIEW_STRICT_FORECAST,
        )
        self.assertEqual(expected, actual)

    def test_diagnostic_view_exports_full_schema(self) -> None:
        view_spec = next(spec for spec in DIAGNOSTIC_VIEW_SPECS if spec.view_name == "diagnostic_only")
        formatted = format_diagnostic_sample(self.canonical_sample, view_spec=view_spec)
        self.assertIsNotNone(formatted)
        self.assertEqual(formatted["train_metadata"]["sft_view"], "diagnostic_only")
        self.assertIn("turning_cues_included", formatted["train_metadata"])
        assistant = next(
            message["content"]
            for message in formatted["messages"]
            if message["role"] == "assistant"
        )
        payload = json.loads(assistant)
        self.assertEqual(sorted(payload.keys()), sorted(DIAGNOSTIC_FIELDS))
        self.assertTrue(any(payload[field] is not None for field in DIAGNOSTIC_FIELDS))

    def test_diagnostic_view_appends_turning_signal_cues(self) -> None:
        view_spec = next(spec for spec in DIAGNOSTIC_VIEW_SPECS if spec.view_name == "diagnostic_track_core_only")
        formatted = format_diagnostic_sample(self.canonical_sample, view_spec=view_spec)
        self.assertIsNotNone(formatted)
        user_content = next(
            message["content"]
            for message in formatted["messages"]
            if message["role"] == "user"
        )
        self.assertIn("## Turning Signal Cues", user_content)
        self.assertTrue(formatted["train_metadata"]["turning_cues_included"])

    def test_track_turn_view_exports_two_field_schema_with_steering_cues(self) -> None:
        view_spec = next(spec for spec in DIAGNOSTIC_VIEW_SPECS if spec.view_name == "diagnostic_track_turn_only")
        formatted = format_diagnostic_sample(self.canonical_sample, view_spec=view_spec)
        self.assertIsNotNone(formatted)
        assistant = next(
            message["content"]
            for message in formatted["messages"]
            if message["role"] == "assistant"
        )
        payload = json.loads(assistant)
        self.assertEqual(
            sorted(payload.keys()),
            ["track_control_signal", "turning_signal"],
        )
        user_content = next(
            message["content"]
            for message in formatted["messages"]
            if message["role"] == "user"
        )
        self.assertIn("## Ridge/Trough Competition Cues", user_content)
        self.assertTrue(formatted["train_metadata"]["ridge_trough_cues_included"])

    def test_track_inflection_view_exports_candidate_schema_with_inflection_cues(self) -> None:
        view_spec = next(
            spec for spec in DIAGNOSTIC_VIEW_SPECS if spec.view_name == "diagnostic_track_inflection_only"
        )
        formatted = format_diagnostic_sample(self.canonical_sample, view_spec=view_spec)
        self.assertIsNotNone(formatted)
        assistant = next(
            message["content"]
            for message in formatted["messages"]
            if message["role"] == "assistant"
        )
        payload = json.loads(assistant)
        self.assertEqual(
            sorted(payload.keys()),
            [
                "steering_regime_phase",
                "turn_direction_family",
                "turn_magnitude_bucket",
                "turn_timing_bucket",
            ],
        )
        user_content = next(
            message["content"]
            for message in formatted["messages"]
            if message["role"] == "user"
        )
        self.assertIn("## Track Inflection Cues", user_content)
        self.assertTrue(formatted["train_metadata"]["track_inflection_cues_included"])

    def test_track_correction_view_exports_candidate_schema_with_anchor_cues(self) -> None:
        view_spec = next(
            spec for spec in DIAGNOSTIC_VIEW_SPECS if spec.view_name == "diagnostic_track_correction_only"
        )
        formatted = format_diagnostic_sample(self.canonical_sample, view_spec=view_spec)
        self.assertIsNotNone(formatted)
        assistant = next(
            message["content"]
            for message in formatted["messages"]
            if message["role"] == "assistant"
        )
        payload = json.loads(assistant)
        self.assertEqual(
            sorted(payload.keys()),
            sorted(DIAGNOSTIC_TRACK_CORRECTION_FIELDS),
        )
        user_content = next(
            message["content"]
            for message in formatted["messages"]
            if message["role"] == "user"
        )
        self.assertIn("## Track Correction Anchors", user_content)
        self.assertNotIn("## Ridge/Trough Competition Cues", user_content)
        self.assertNotIn("## Turning Signal Cues", user_content)
        self.assertTrue(formatted["train_metadata"]["track_correction_cues_included"])

    def test_slot_correction_view_exports_slot_locked_schema_with_slot_cues(self) -> None:
        view_spec = next(
            spec for spec in DIAGNOSTIC_VIEW_SPECS if spec.view_name == "diagnostic_slot_correction_only"
        )
        formatted = format_diagnostic_sample(self.canonical_sample, view_spec=view_spec)
        self.assertIsNotNone(formatted)
        assistant = next(
            message["content"]
            for message in formatted["messages"]
            if message["role"] == "assistant"
        )
        payload = json.loads(assistant)
        self.assertEqual(
            sorted(payload.keys()),
            sorted(DIAGNOSTIC_SLOT_CORRECTION_FIELDS),
        )
        user_content = next(
            message["content"]
            for message in formatted["messages"]
            if message["role"] == "user"
        )
        self.assertIn("## Slot-Locked Track Correction Cues", user_content)
        self.assertNotIn("## Ridge/Trough Competition Cues", user_content)
        self.assertNotIn("## Turning Signal Cues", user_content)
        self.assertTrue(formatted["train_metadata"]["slot_correction_cues_included"])

class DiagnosticNormalizationUnitTests(unittest.TestCase):
    def test_track_core_prompt_lists_canonical_values(self) -> None:
        system_prompt = _build_diagnostic_system_prompt(
            ("track_control_signal", "turning_signal", "model_agreement_level", "forecast_confidence_level")
        )
        self.assertIn(
            '"track_control_signal": one of "competing_ridge_and_trough", "subtropical_high", "midlatitude_trough", or null',
            system_prompt,
        )
        self.assertIn(
            '"turning_signal": one of "steady", "notable_turn", "recurvature", or null',
            system_prompt,
        )
        self.assertIn("markdown fences", system_prompt)

    def test_track_inflection_prompt_lists_canonical_values(self) -> None:
        system_prompt = _build_diagnostic_system_prompt(
            (
                "steering_regime_phase",
                "turn_timing_bucket",
                "turn_direction_family",
                "turn_magnitude_bucket",
            )
        )
        self.assertIn(
            '"steering_regime_phase": one of "locked", "transition", "recurving", or null',
            system_prompt,
        )
        self.assertIn(
            '"turn_timing_bucket": one of "none_or_locked", "24_to_48h", "48_to_72h", or null',
            system_prompt,
        )

    def test_track_correction_prompt_lists_canonical_values(self) -> None:
        system_prompt = _build_diagnostic_system_prompt(DIAGNOSTIC_TRACK_CORRECTION_FIELDS)
        self.assertIn(
            '"lat_bias_vs_consensus_48h_bucket": one of "south_large", "south_small", "near_consensus", "north_small", "north_large", or null',
            system_prompt,
        )
        self.assertIn(
            '"lon_bias_vs_consensus_72h_bucket": one of "east_large", "east_small", "near_consensus", "west_small", "west_large", or null',
            system_prompt,
        )

    def test_slot_correction_prompt_lists_canonical_values(self) -> None:
        system_prompt = _build_diagnostic_system_prompt(DIAGNOSTIC_SLOT_CORRECTION_FIELDS)
        self.assertIn(
            '"slot_1_lat_bias_vs_consensus_bucket": one of "south_large", "south_small", "near_consensus", "north_small", "north_large", or null',
            system_prompt,
        )
        self.assertIn(
            '"slot_6_lon_bias_vs_consensus_bucket": one of "east_large", "east_small", "near_consensus", "west_small", "west_large", or null',
            system_prompt,
        )

    def test_track_correction_derivation_uses_signed_fixed_lead_bias_buckets(self) -> None:
        candidate = _derive_track_correction_candidate(
            "2024-11-16T15:00:00Z",
            [
                {"lead_from_issue_h": 48, "consensus_lat": 20.0, "consensus_lon": -60.0},
                {"lead_from_issue_h": 72, "consensus_lat": 22.0, "consensus_lon": -58.0},
            ],
            [
                {"valid_day": 18, "valid_hhmmz": "1200", "lat": 20.6, "lon": -61.1},
                {"valid_day": 19, "valid_hhmmz": "1200", "lat": 24.0, "lon": -55.8},
            ],
        )
        self.assertEqual(candidate["lat_bias_vs_consensus_48h_bucket"], "north_small")
        self.assertEqual(candidate["lon_bias_vs_consensus_48h_bucket"], "west_small")
        self.assertEqual(candidate["lat_bias_vs_consensus_72h_bucket"], "north_large")
        self.assertEqual(candidate["lon_bias_vs_consensus_72h_bucket"], "east_large")

    def test_normalize_diagnostic_field_value_maps_surface_forms(self) -> None:
        self.assertEqual(
            normalize_diagnostic_field_value("track_control_signal", "Subtropical high"),
            "subtropical_high",
        )
        self.assertEqual(
            normalize_diagnostic_field_value("model_agreement_level", "Moderate"),
            "medium",
        )
        self.assertEqual(
            normalize_diagnostic_field_value("forecast_confidence_level", "High"),
            "high",
        )
        self.assertEqual(
            normalize_diagnostic_field_value("turn_direction_family", "Eastward escape"),
            "eastward_escape",
        )
        self.assertEqual(
            normalize_diagnostic_field_value("turn_magnitude_bucket", "Moderate"),
            "modest",
        )
        self.assertEqual(
            normalize_diagnostic_field_value("lat_bias_vs_consensus_48h_bucket", "North small"),
            "north_small",
        )
        self.assertEqual(
            normalize_diagnostic_field_value("lon_bias_vs_consensus_72h_bucket", "Near"),
            "near_consensus",
        )
        self.assertIsNone(normalize_diagnostic_field_value("turning_signal", "null"))


if __name__ == "__main__":
    unittest.main()
