from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import build_forecast_prompt_overrides as prompt_builder


class ForecastPromptOverrideTests(unittest.TestCase):
    def test_track_correction_injection_adds_track_only_guardrails(self) -> None:
        prompt = prompt_builder._format_injected_prompt(
            "Forecast base prompt",
            diagnostic_payload={
                "lat_bias_vs_consensus_48h_bucket": "north_small",
                "lon_bias_vs_consensus_48h_bucket": "west_small",
            },
            source_label="oracle_reference",
            section_title="Track-Correction Candidate v0 Assessment",
        )
        self.assertIn("fixed 48h/72h track corrections", prompt)
        self.assertIn("Do not change forecast Day/HHMMZ slots", prompt)
        self.assertIn("Do not change intensity because of this block.", prompt)

    def test_generic_injection_keeps_generic_guidance_text(self) -> None:
        prompt = prompt_builder._format_injected_prompt(
            "Forecast base prompt",
            diagnostic_payload={"track_control_signal": "subtropical_high"},
            source_label="oracle_reference",
            section_title="Structured Diagnostic Assessment",
        )
        self.assertIn(
            "Use this structured diagnostic assessment as auxiliary guidance when producing the official forecast table.",
            prompt,
        )
        self.assertNotIn("Do not change intensity because of this block.", prompt)


if __name__ == "__main__":
    unittest.main()
