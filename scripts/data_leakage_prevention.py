#!/usr/bin/env python3
"""Data Leakage Prevention Pipeline for TC fine-tuning dataset.

Implements the anti-leakage protocol specified in data_leakage.md:
  1. Event-level temporal splitting (no random splits)
  2. Input temporal leakage auditing
  3. Reversible anonymization (storm names, dates, locations)
  4. Train/val/test deduplication scanning
  5. Split manifest with frozen test set
  6. Contamination audit with re-identification attack

Execution order follows data_leakage.md Section 10:
  Freeze test set → event-level split → define issuance time →
  anonymize → dedup → then train.
"""

from __future__ import annotations

import argparse
import csv
import difflib
import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class SplitConfig:
    """Configuration for temporal event-level splitting."""

    train_years: Tuple[int, int] = (2016, 2020)
    val_years: Tuple[int, int] = (2021, 2022)
    test_years: Tuple[int, int] = (2023, 2025)
    atcf_era_only: bool = True
    basin_subset: Optional[List[str]] = None
    min_advisories_per_storm: int = 1
    respect_existing_manifest: bool = True


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class StormRecord:
    storm_id: str
    storm_name: str
    season_year: int
    basin: str
    first_valid_time: Optional[datetime]
    last_valid_time: Optional[datetime]
    total_rows: int
    atcf_available: bool = False


@dataclass
class ValidationResult:
    passed: bool
    violations: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# StormSplitter
# ---------------------------------------------------------------------------

class StormSplitter:
    """Event-level temporal splitting: same storm_id belongs to exactly one split."""

    def __init__(
        self,
        config: SplitConfig,
        groundtruth_csv: Path,
        crosswalk_csv: Optional[Path] = None,
        by_storm_dir: Optional[Path] = None,
    ):
        self.config = config
        self.groundtruth_csv = groundtruth_csv
        self.crosswalk_csv = crosswalk_csv
        self.by_storm_dir = by_storm_dir
        self._registry: Optional[Dict[str, StormRecord]] = None

    def load_storm_registry(self) -> Dict[str, StormRecord]:
        """Load all storms from ground truth, keyed by storm_id."""
        if self._registry is not None:
            return self._registry

        # Load ATCF-available storm IDs from crosswalk or by_storm dir
        atcf_storm_ids: Set[str] = set()
        if self.config.atcf_era_only:
            if self.crosswalk_csv and self.crosswalk_csv.exists():
                with self.crosswalk_csv.open("r", encoding="utf-8", newline="") as f:
                    for row in csv.DictReader(f):
                        sid = (row.get("matched_storm_id") or "").strip()
                        if sid:
                            atcf_storm_ids.add(sid)
            elif self.by_storm_dir and self.by_storm_dir.exists():
                for d in self.by_storm_dir.iterdir():
                    if d.is_dir():
                        atcf_storm_ids.add(d.name)

        registry: Dict[str, StormRecord] = {}
        with self.groundtruth_csv.open("r", encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                sid = (row.get("storm_id") or "").strip()
                if not sid or len(sid) < 4:
                    continue
                season_year = int(sid[:4])
                basin = (row.get("noaa_basin") or "unknown").strip()
                name = (row.get("storm_name") or "").strip()
                dt_str = (row.get("datetime") or "").strip()[:19]
                try:
                    dt = datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
                except ValueError:
                    dt = None

                if sid not in registry:
                    registry[sid] = StormRecord(
                        storm_id=sid,
                        storm_name=name,
                        season_year=season_year,
                        basin=basin,
                        first_valid_time=dt,
                        last_valid_time=dt,
                        total_rows=1,
                        atcf_available=sid in atcf_storm_ids,
                    )
                else:
                    rec = registry[sid]
                    rec.total_rows += 1
                    if dt:
                        if rec.first_valid_time is None or dt < rec.first_valid_time:
                            rec.first_valid_time = dt
                        if rec.last_valid_time is None or dt > rec.last_valid_time:
                            rec.last_valid_time = dt

        self._registry = registry
        return registry

    def _year_to_split(self, year: int) -> Optional[str]:
        cfg = self.config
        if cfg.train_years[0] <= year <= cfg.train_years[1]:
            return "train"
        if cfg.val_years[0] <= year <= cfg.val_years[1]:
            return "val"
        if cfg.test_years[0] <= year <= cfg.test_years[1]:
            return "test"
        return None

    def assign_splits(
        self, existing_manifest_path: Optional[Path] = None
    ) -> Dict[str, List[str]]:
        """Assign each storm_id to train/val/test by season_year."""
        registry = self.load_storm_registry()
        splits: Dict[str, List[str]] = {"train": [], "val": [], "test": []}

        # Load existing frozen manifest assignments
        frozen_assignments: Dict[str, str] = {}
        if self.config.respect_existing_manifest and existing_manifest_path:
            frozen_assignments = self._load_frozen_assignments(existing_manifest_path)

        for sid, rec in registry.items():
            # Basin filter
            if self.config.basin_subset:
                basin_code = self._basin_to_code(rec.basin)
                if basin_code not in self.config.basin_subset:
                    continue

            # ATCF era filter
            if self.config.atcf_era_only and not rec.atcf_available:
                continue

            # Min advisories filter
            if rec.total_rows < self.config.min_advisories_per_storm:
                continue

            # Check frozen assignment first
            if sid in frozen_assignments:
                splits[frozen_assignments[sid]].append(sid)
                continue

            # Assign by year
            split_name = self._year_to_split(rec.season_year)
            if split_name:
                splits[split_name].append(sid)

        # Sort for determinism
        for k in splits:
            splits[k].sort()

        return splits

    def get_split_statistics(
        self, split_assignment: Dict[str, List[str]]
    ) -> Dict[str, Any]:
        """Compute summary statistics for the split."""
        registry = self.load_storm_registry()
        stats: Dict[str, Any] = {}
        for split_name, storm_ids in split_assignment.items():
            records = [registry[sid] for sid in storm_ids if sid in registry]
            basin_dist: Dict[str, int] = {}
            year_range: List[int] = []
            total_rows = 0
            for rec in records:
                total_rows += rec.total_rows
                bc = self._basin_to_code(rec.basin)
                basin_dist[bc] = basin_dist.get(bc, 0) + 1
                year_range.append(rec.season_year)
            stats[split_name] = {
                "storm_count": len(storm_ids),
                "row_count": total_rows,
                "basin_distribution": basin_dist,
                "year_range": (
                    [min(year_range), max(year_range)] if year_range else []
                ),
            }
        return stats

    def validate_no_cross_contamination(
        self, split_assignment: Dict[str, List[str]]
    ) -> ValidationResult:
        """Verify no storm_id appears in more than one split."""
        seen: Dict[str, str] = {}
        violations: List[str] = []
        for split_name, storm_ids in split_assignment.items():
            for sid in storm_ids:
                if sid in seen:
                    violations.append(
                        f"storm_id {sid} in both {seen[sid]} and {split_name}"
                    )
                seen[sid] = split_name

        # Check no test-year storm in train
        registry = self.load_storm_registry()
        for sid in split_assignment.get("train", []):
            rec = registry.get(sid)
            if rec and rec.season_year >= self.config.test_years[0]:
                violations.append(
                    f"test-year storm {sid} (year={rec.season_year}) in train split"
                )

        return ValidationResult(passed=len(violations) == 0, violations=violations)

    def _load_frozen_assignments(self, path: Path) -> Dict[str, str]:
        """Load storm-to-split assignments from a frozen manifest."""
        if not path.exists():
            return {}
        data = json.loads(path.read_text(encoding="utf-8"))
        if not data.get("frozen", False):
            return {}
        assignments: Dict[str, str] = {}
        for split_name, storm_dict in data.get("splits", {}).items():
            if isinstance(storm_dict, dict):
                for sid in storm_dict:
                    assignments[sid] = split_name
            elif isinstance(storm_dict, list):
                for sid in storm_dict:
                    assignments[sid] = split_name
        return assignments

    @staticmethod
    def _basin_to_code(basin: str) -> str:
        mapping = {
            "Atlantic": "AL",
            "E_Pacific": "EP",
            "C_Pacific": "CP",
        }
        return mapping.get(basin, basin)


# ---------------------------------------------------------------------------
# InputLeakageAuditor
# ---------------------------------------------------------------------------

class InputLeakageAuditor:
    """Three-dimensional leakage auditor: temporal, identity, verification."""

    RETROSPECTIVE_PHRASES: List[str] = [
        "后来证明", "最终在", "最终登陆", "事后", "回顾性",
        "最终等级", "最终结果", "事后分析",
        "later proved", "eventually made landfall", "in hindsight",
        "retrospective", "post-landfall", "post-hoc",
        "final category", "ultimately",
    ]

    # Keys that belong to verification_targets only
    VERIFICATION_KEYS: List[str] = [
        "max_wind_wmo", "min_pressure_wmo",
        "max_wind_usa", "min_pressure_usa",
        "storm_speed", "storm_direction",
        "best_track_point_near_issue", "future_best_track_series",
    ]

    def __init__(self, groundtruth_csv: Optional[Path] = None):
        self.groundtruth_csv = groundtruth_csv

    def audit_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Run all leakage checks on a single sample."""
        temporal = self.check_temporal_leakage(sample)
        identity = self.check_identity_leakage(sample)
        verification = self.check_verification_leakage(sample)
        return {
            "sample_id": sample.get("sample_id", "unknown"),
            "temporal_leakage": temporal,
            "identity_leakage": identity,
            "verification_leakage": verification,
        }

    def check_temporal_leakage(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Check that prompt only contains pre-issuance-time information."""
        violations: List[str] = []
        prompt = sample.get("prompt", {})
        issue_time_str = (
            prompt.get("storm_meta", {}).get("issue_time_utc") or ""
        )
        issue_dt = self._parse_time(issue_time_str)
        if issue_dt is None:
            return {"passed": False, "violations": ["cannot parse issue_time_utc"]}

        tolerance = timedelta(hours=3)

        # Check guidance inputs have valid_time > issue_time
        guidance = prompt.get("guidance_inputs", {})
        for key in ["ec_single_model_guidance_hres"]:
            block = guidance.get(key, {})
            for pts_key in ["track_intensity_points_future", "environment_points_future"]:
                for pt in block.get(pts_key, []):
                    vt = self._parse_time(pt.get("valid_time_utc", ""))
                    if vt and vt <= issue_dt:
                        violations.append(
                            f"guidance {key}.{pts_key} valid_time {pt.get('valid_time_utc')} "
                            f"<= issue_time {issue_time_str}"
                        )

        # Check observation times <= issue_time + tolerance
        now_inputs = prompt.get("now_inputs", {})
        obs = now_inputs.get("observation_evidence_structured", {})
        for obs_key in ["goes_ir_structured", "ascat_surface_wind_structured", "recon_structured"]:
            obs_block = obs.get(obs_key, {})
            if obs_block.get("status") != "available":
                continue
            obs_time = self._parse_time(obs_block.get("obs_time_utc", ""))
            if obs_time and obs_time > issue_dt + tolerance:
                violations.append(
                    f"observation {obs_key} obs_time {obs_block.get('obs_time_utc')} "
                    f"> issue_time + 3h tolerance"
                )

        # Check for retrospective language in prompt text
        prompt_json = json.dumps(prompt, ensure_ascii=False).lower()
        for phrase in self.RETROSPECTIVE_PHRASES:
            if phrase.lower() in prompt_json:
                violations.append(f"retrospective phrase found in prompt: '{phrase}'")

        # Check guidance lead_from_issue_h
        for pt in guidance.get("ec_single_model_guidance_hres", {}).get(
            "track_intensity_points_future", []
        ):
            if pt.get("lead_from_issue_h", 0) <= 0:
                violations.append(
                    f"guidance point has non-positive lead_from_issue_h: {pt.get('lead_from_issue_h')}"
                )

        return {"passed": len(violations) == 0, "violations": violations}

    def check_identity_leakage(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Check for identity-revealing information in prompt."""
        findings: List[str] = []
        severity = 0.0
        prompt = sample.get("prompt", {})
        prompt_json = json.dumps(prompt, ensure_ascii=False)
        meta = prompt.get("storm_meta", {})

        # Storm name in prompt
        storm_name = meta.get("storm_name", "")
        if storm_name and storm_name in prompt_json:
            findings.append(f"storm_name '{storm_name}' present in prompt")
            severity += 0.3

        # Storm_id in prompt (contains year)
        storm_id = meta.get("storm_id", "")
        if storm_id and len(storm_id) >= 4 and storm_id[:4].isdigit():
            findings.append(f"storm_id '{storm_id}' contains identifying year prefix")
            severity += 0.2

        # Absolute dates in prompt
        absolute_date_pattern = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z")
        date_matches = absolute_date_pattern.findall(prompt_json)
        if date_matches:
            findings.append(f"{len(date_matches)} absolute ISO timestamps in prompt")
            severity += min(0.3, len(date_matches) * 0.02)

        # Source file paths containing storm names
        source_trace = sample.get("source_trace", {})
        for key, path_val in source_trace.items():
            if isinstance(path_val, str) and storm_name and storm_name in path_val:
                findings.append(f"source_trace.{key} contains storm name")
                severity += 0.1
                break

        severity = min(1.0, severity)
        return {"severity": round(severity, 3), "findings": findings}

    def check_verification_leakage(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Check that verification_targets are isolated from prompt."""
        violations: List[str] = []
        prompt = sample.get("prompt", {})
        verification = sample.get("verification_targets", {})
        prompt_json = json.dumps(prompt, ensure_ascii=False)

        # Check verification marker keys
        for key in self.VERIFICATION_KEYS:
            if f'"{key}"' in prompt_json:
                violations.append(f"verification key '{key}' found in prompt")

        # Check that verification values don't leak
        gt_near = verification.get("best_track_point_near_issue", {})
        gt_value = gt_near.get("value")
        if gt_value and isinstance(gt_value, dict):
            for vk, vv in gt_value.items():
                if vv is not None and str(vv) in prompt_json:
                    # Only flag if it's a specific numeric value match
                    if isinstance(vv, (int, float)):
                        # Check if this value appears in a suspicious context
                        now_state = prompt.get("now_inputs", {}).get(
                            "current_state_from_noaa_forecast_advisory", {}
                        )
                        intensity = now_state.get("intensity", {})
                        # Allow same values in official forecast (that's the input)
                        if vk in ("max_wind_wmo", "max_wind_usa"):
                            if intensity.get("max_wind_kt") == vv:
                                continue  # This is the advisory value, not a leak
                        if vk in ("min_pressure_wmo", "min_pressure_usa"):
                            if intensity.get("min_pressure_mb") == vv:
                                continue
                        violations.append(
                            f"verification value {vk}={vv} may appear in prompt"
                        )

        return {"passed": len(violations) == 0, "violations": violations}

    def audit_batch(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Run audit_sample on a batch and aggregate results."""
        results = [self.audit_sample(s) for s in samples]
        temporal_pass = sum(1 for r in results if r["temporal_leakage"]["passed"])
        verification_pass = sum(1 for r in results if r["verification_leakage"]["passed"])
        avg_severity = (
            sum(r["identity_leakage"]["severity"] for r in results) / len(results)
            if results
            else 0
        )
        return {
            "total_samples": len(samples),
            "temporal_pass_count": temporal_pass,
            "temporal_pass_rate": round(temporal_pass / max(len(samples), 1), 4),
            "verification_pass_count": verification_pass,
            "verification_pass_rate": round(verification_pass / max(len(samples), 1), 4),
            "avg_identity_severity": round(avg_severity, 4),
            "failing_samples": [
                {
                    "sample_id": r["sample_id"],
                    "temporal": r["temporal_leakage"]["violations"][:3],
                    "identity_severity": r["identity_leakage"]["severity"],
                    "verification": r["verification_leakage"]["violations"][:3],
                }
                for r in results
                if not r["temporal_leakage"]["passed"]
                or not r["verification_leakage"]["passed"]
                or r["identity_leakage"]["severity"] > 0.5
            ],
        }

    @staticmethod
    def _parse_time(s: str) -> Optional[datetime]:
        if not s:
            return None
        s = s.strip()
        fmts = [
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%dT%H:%M:%S",
        ]
        for fmt in fmts:
            try:
                return datetime.strptime(s, fmt)
            except ValueError:
                continue
        return None


# ---------------------------------------------------------------------------
# Anonymizer
# ---------------------------------------------------------------------------

class Anonymizer:
    """Reversible anonymization: storm names, dates, locations."""

    LOCATION_LEVELS: Dict[str, Dict[str, Any]] = {
        "fine": {"lat_res": 0.1, "lon_res": 0.1, "coast_proximity": True},
        "medium": {"lat_res": 1.0, "lon_res": 1.0, "coast_proximity": True},
        "coarse": {"lat_res": 5.0, "lon_res": 5.0, "coast_proximity": False},
    }

    def __init__(
        self,
        groundtruth_csv: Path,
        mapping_file: Optional[Path] = None,
        location_level: str = "medium",
    ):
        self.groundtruth_csv = groundtruth_csv
        self.location_level = location_level
        self._name_index: Dict[str, str] = {}  # storm_name -> storm_id
        self._genesis_times: Dict[str, datetime] = {}  # storm_id -> first time
        self._load_ground_truth()
        if mapping_file and mapping_file.exists():
            self._mapping = self.load_mapping(mapping_file)
        else:
            self._mapping: Optional[Dict[str, Any]] = None

    def _load_ground_truth(self) -> None:
        with self.groundtruth_csv.open("r", encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                sid = (row.get("storm_id") or "").strip()
                name = (row.get("storm_name") or "").strip()
                dt_str = (row.get("datetime") or "").strip()[:19]
                if name and sid:
                    self._name_index[name] = sid
                if sid and dt_str:
                    try:
                        dt = datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
                        if sid not in self._genesis_times or dt < self._genesis_times[sid]:
                            self._genesis_times[sid] = dt
                    except ValueError:
                        pass

    def generate_mapping(self, storm_ids: List[str]) -> Dict[str, Any]:
        """Generate reversible anonymization mapping."""
        name_to_anon: Dict[str, str] = {}
        id_to_anon: Dict[str, str] = {}
        genesis: Dict[str, str] = {}

        sorted_ids = sorted(storm_ids)
        for idx, sid in enumerate(sorted_ids, start=1):
            anon_id = f"STORM_{idx:03d}"
            # Find storm name
            for name, name_sid in self._name_index.items():
                if name_sid == sid:
                    name_to_anon[name] = anon_id
                    break
            id_to_anon[sid] = anon_id
            gen_dt = self._genesis_times.get(sid)
            genesis[anon_id] = gen_dt.strftime("%Y-%m-%dT%H:%M:%SZ") if gen_dt else ""

        mapping = {
            "version": "1.0",
            "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "name_to_anon": name_to_anon,
            "id_to_anon": id_to_anon,
            "genesis_times": genesis,
            "location_level": self.location_level,
        }
        self._mapping = mapping
        return mapping

    def anonymize_sample(
        self, sample: Dict[str, Any], mapping: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Apply anonymization to a single sample. Returns a new dict."""
        m = mapping or self._mapping
        if m is None:
            return sample

        import copy
        result = copy.deepcopy(sample)

        prompt = result.get("prompt", {})
        meta = prompt.get("storm_meta", {})

        # 1. Replace storm_name with anonymous ID
        storm_name = meta.get("storm_name", "")
        storm_id = meta.get("storm_id", "")
        anon_id = m["id_to_anon"].get(storm_id, m["name_to_anon"].get(storm_name, "STORM_???"))
        meta["storm_name"] = anon_id
        meta["storm_id"] = anon_id

        # 2. Convert absolute dates to relative time indices
        genesis_str = m["genesis_times"].get(anon_id, "")
        genesis_dt = self._parse_time(genesis_str) if genesis_str else None

        self._anonymize_timestamps(prompt, genesis_dt)

        # 3. Generalize locations
        self._generalize_locations(prompt)

        # 4. Sanitize source paths
        source_trace = result.get("source_trace", {})
        for key in source_trace:
            if isinstance(source_trace[key], str):
                source_trace[key] = self._sanitize_path(source_trace[key], m)

        # Also sanitize observation source_file fields
        now_inputs = prompt.get("now_inputs", {})
        obs = now_inputs.get("observation_evidence_structured", {})
        for obs_key in ["goes_ir_structured", "ascat_surface_wind_structured", "recon_structured"]:
            obs_block = obs.get(obs_key, {})
            if obs_block.get("source_file"):
                obs_block["source_file"] = self._sanitize_path(obs_block["source_file"], m)

        # 5. Rewrite description text in CDS features
        env = now_inputs.get("environment_now_ec_reanalysis", {})
        features = env.get("features", {})
        if isinstance(features, dict):
            for feat_key in features:
                feat = features[feat_key]
                if isinstance(feat, dict) and feat.get("description"):
                    feat["description"] = self._rewrite_description(
                        feat["description"], m
                    )

        return result

    def _anonymize_timestamps(self, obj: Any, genesis_dt: Optional[datetime]) -> None:
        """Recursively replace absolute timestamps with relative indices in prompt."""
        if isinstance(obj, dict):
            for key in list(obj.keys()):
                val = obj[key]
                if isinstance(val, str) and self._is_iso_timestamp(val):
                    if genesis_dt:
                        dt = self._parse_time(val)
                        if dt:
                            delta_h = int(round((dt - genesis_dt).total_seconds() / 3600))
                            obj[key] = f"T+{delta_h}h"
                    else:
                        obj[key] = "T_RELATIVE"
                elif isinstance(val, (dict, list)):
                    self._anonymize_timestamps(val, genesis_dt)
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                if isinstance(item, str) and self._is_iso_timestamp(item):
                    if genesis_dt:
                        dt = self._parse_time(item)
                        if dt:
                            delta_h = int(round((dt - genesis_dt).total_seconds() / 3600))
                            obj[i] = f"T+{delta_h}h"
                    else:
                        obj[i] = "T_RELATIVE"
                elif isinstance(item, (dict, list)):
                    self._anonymize_timestamps(item, genesis_dt)

    def _generalize_locations(self, obj: Any) -> None:
        """Recursively generalize lat/lon coordinates in prompt."""
        level = self.LOCATION_LEVELS.get(self.location_level, self.LOCATION_LEVELS["medium"])
        lat_res = level["lat_res"]
        lon_res = level["lon_res"]

        if isinstance(obj, dict):
            # Check for lat/lon pairs
            if "lat" in obj and "lon" in obj:
                lat_val = obj["lat"]
                lon_val = obj["lon"]
                if isinstance(lat_val, (int, float)) and isinstance(lon_val, (int, float)):
                    obj["lat"] = round(lat_val / lat_res) * lat_res
                    obj["lon"] = round(lon_val / lon_res) * lon_res
            for val in obj.values():
                if isinstance(val, (dict, list)):
                    self._generalize_locations(val)
        elif isinstance(obj, list):
            for item in obj:
                if isinstance(item, (dict, list)):
                    self._generalize_locations(item)

    @staticmethod
    def _sanitize_path(path_str: str, mapping: Dict[str, Any]) -> str:
        """Replace storm names and identifying patterns in source_file paths."""
        result = path_str
        for name, anon in mapping.get("name_to_anon", {}).items():
            if name and len(name) > 2:
                result = result.replace(name, anon)
        for sid, anon in mapping.get("id_to_anon", {}).items():
            if sid:
                result = result.replace(sid, anon)
        # Replace ATCF IDs like AL052020
        result = re.sub(r"(AL|EP|CP)\d{4}", r"\1XXXX", result)
        # Replace year directories
        result = re.sub(r"/(\d{4})/", "/YYYY/", result)
        return result

    @staticmethod
    def _rewrite_description(text: str, mapping: Dict[str, Any]) -> str:
        """Template-rewrite free-text descriptions, removing identity cues."""
        result = text
        for name in mapping.get("name_to_anon", {}):
            if name and len(name) > 2:
                result = result.replace(name, "[STORM]")
        # Remove absolute dates
        result = re.sub(r"\d{4}-\d{2}-\d{2}", "YYYY-MM-DD", result)
        # Remove specific coordinates
        result = re.sub(
            r"(\d+\.?\d*)\s*(°|degrees?)\s*([NS])",
            r"[lat]°\3",
            result,
        )
        result = re.sub(
            r"(\d+\.?\d*)\s*(°|degrees?)\s*([EW])",
            r"[lon]°\3",
            result,
        )
        return result

    def save_mapping(self, mapping: Dict[str, Any], path: Path) -> None:
        """Save mapping with SHA256 integrity hash."""
        path.parent.mkdir(parents=True, exist_ok=True)
        json_bytes = json.dumps(mapping, ensure_ascii=False, indent=2).encode("utf-8")
        path.write_bytes(json_bytes)
        sha = hashlib.sha256(json_bytes).hexdigest()
        sha_path = path.parent / (path.name + ".sha256")
        sha_path.write_text(sha, encoding="utf-8")

    def load_mapping(self, path: Path) -> Dict[str, Any]:
        """Load and verify integrity of mapping file."""
        json_bytes = path.read_bytes()
        sha_path = path.parent / (path.name + ".sha256")
        if sha_path.exists():
            expected = sha_path.read_text(encoding="utf-8").strip()
            actual = hashlib.sha256(json_bytes).hexdigest()
            if expected != actual:
                raise IntegrityError(
                    f"SHA256 mismatch for {path}: expected {expected}, got {actual}"
                )
        return json.loads(json_bytes)

    def generate_anonymous_test_set(
        self, samples: List[Dict[str, Any]], mapping: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate Test Set B (fully anonymous) from the main test set."""
        old_level = self.location_level
        self.location_level = "coarse"
        results = [self.anonymize_sample(s, mapping) for s in samples]
        self.location_level = old_level
        return results

    def generate_structured_only_test_set(
        self, samples: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Generate Test Set C (structured-only, no free text)."""
        import copy
        results = []
        for s in samples:
            anon = copy.deepcopy(s)
            prompt = anon.get("prompt", {})
            # Remove text fields that could trigger memory
            meta = prompt.get("storm_meta", {})
            meta.pop("storm_name", None)
            # Remove environment descriptions (keep numeric values)
            env = prompt.get("now_inputs", {}).get(
                "environment_now_ec_reanalysis", {}
            )
            features = env.get("features", {})
            if isinstance(features, dict):
                for feat_key in features:
                    feat = features[feat_key]
                    if isinstance(feat, dict):
                        feat.pop("description", None)
                        feat.pop("level", None)
            # Remove target reasoning text (keep structured output only)
            target = anon.get("target", {})
            official = target.get("official_outputs", {})
            official.pop("reasoning_text", None)
            official.pop("risk_messages", None)
            results.append(anon)
        return results

    def generate_perturbation_test_set(
        self, samples: List[Dict[str, Any]], mapping: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate Test Set D (lightly perturbed prompt-visible fields)."""
        import copy
        results = []
        for s in samples:
            perturbed = copy.deepcopy(s)
            prompt = perturbed.get("prompt", {})
            meta = prompt.get("storm_meta", {})

            # Perturb storm name with a different anonymous label
            storm_id = meta.get("storm_id", "")
            anon_id = mapping.get("id_to_anon", {}).get(storm_id, "")
            if anon_id:
                # Add noise to the anonymous ID
                num = int(anon_id.split("_")[1]) if "_" in anon_id else 0
                perturbed_id = f"STORM_{(num + 500) % 999:03d}"
                meta["storm_name"] = perturbed_id
                meta["storm_id"] = perturbed_id

            # Perturb dates by adding deterministic offsets.
            sample_id = perturbed.get("sample_id", "")
            offset_seed = self._stable_int(sample_id, "time_shift", -24, 24)
            self._perturb_timestamps(prompt, offset_seed)
            self._perturb_prompt_visible_fields(prompt, sample_id)

            results.append(perturbed)
        return results

    def _perturb_prompt_visible_fields(self, prompt: Dict[str, Any], sample_id: str) -> None:
        """Apply light deterministic perturbations to prompt-visible numeric fields."""
        now_inputs = prompt.get("now_inputs", {})
        current_state = now_inputs.get("current_state_from_noaa_forecast_advisory", {})
        center = current_state.get("center", {})
        motion = current_state.get("motion", {})
        intensity = current_state.get("intensity", {})

        self._perturb_numeric(center, "lat", sample_id, "center_lat", 0.35, -89.9, 89.9)
        self._perturb_numeric(center, "lon", sample_id, "center_lon", 0.35, -179.9, 179.9)
        self._perturb_numeric(motion, "speed_kt", sample_id, "motion_speed", 1.5, 0.0, 120.0)
        self._perturb_numeric(intensity, "max_wind_kt", sample_id, "current_wind", 3.0, 0.0, 220.0)
        self._perturb_numeric(intensity, "min_pressure_mb", sample_id, "current_pressure", 3.0, 850.0, 1050.0)

        features = now_inputs.get("environment_now_ec_reanalysis", {}).get("features", {}) or {}
        env_scales = {
            "vertical_wind_shear": 1.5,
            "upper_level_divergence": 0.35,
            "ocean_heat_content_or_sst": 0.6,
            "subtropical_high": 80.0,
            "westerly_trough": 80.0,
            "monsoon_trough": 0.6,
            "low_level_flow": 1.5,
        }
        for feature_key, feature in features.items():
            if isinstance(feature, dict):
                scale = env_scales.get(feature_key, 1.0)
                self._perturb_numeric(feature, "value", sample_id, f"env_{feature_key}", scale)

        pre_issue = now_inputs.get("pre_issue_guidance_context", {}) or {}
        for key in [
            "ec_hres_latest_point_at_or_before_issue_track",
            "ec_hres_latest_point_at_or_before_issue_environment",
        ]:
            point = pre_issue.get(key)
            if isinstance(point, dict):
                self._perturb_track_point(point, sample_id, f"pre_issue_{key}")

        guidance = prompt.get("guidance_inputs", {}) or {}
        hres_points = (guidance.get("ec_single_model_guidance_hres", {}) or {}).get("track_intensity_points_future", []) or []
        for idx, point in enumerate(hres_points):
            if isinstance(point, dict):
                self._perturb_track_point(point, sample_id, f"hres_{idx}")

        consensus_points = (guidance.get("multimodel_guidance_a_deck", {}) or {}).get("consensus_spread_points_future", []) or []
        for idx, point in enumerate(consensus_points):
            if not isinstance(point, dict):
                continue
            self._perturb_numeric(point, "consensus_lat", sample_id, f"atcf_lat_{idx}", 0.35, -89.9, 89.9)
            self._perturb_numeric(point, "consensus_lon", sample_id, f"atcf_lon_{idx}", 0.35, -179.9, 179.9)
            self._perturb_numeric(point, "consensus_vmax_kt", sample_id, f"atcf_vmax_{idx}", 3.0, 0.0, 220.0)
            self._perturb_numeric(point, "consensus_mslp_hpa", sample_id, f"atcf_mslp_{idx}", 3.0, 850.0, 1050.0)
            self._perturb_numeric(point, "track_spread_km", sample_id, f"atcf_spread_km_{idx}", 15.0, 0.0, None)
            self._perturb_numeric(point, "wind_spread_kt", sample_id, f"atcf_spread_kt_{idx}", 1.5, 0.0, None)

    def _perturb_track_point(self, point: Dict[str, Any], sample_id: str, salt: str) -> None:
        """Perturb one guidance track point lightly and deterministically."""
        self._perturb_numeric(point, "lat", sample_id, f"{salt}_lat", 0.35, -89.9, 89.9)
        self._perturb_numeric(point, "lon", sample_id, f"{salt}_lon", 0.35, -179.9, 179.9)
        self._perturb_numeric(point, "wind_kt", sample_id, f"{salt}_wind", 3.0, 0.0, 220.0)
        self._perturb_numeric(point, "mslp_hpa", sample_id, f"{salt}_mslp", 3.0, 850.0, 1050.0)

    def _perturb_numeric(
        self,
        obj: Dict[str, Any],
        key: str,
        sample_id: str,
        salt: str,
        max_delta: float,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
    ) -> None:
        """Apply a bounded deterministic perturbation to one numeric field."""
        value = obj.get(key)
        if not isinstance(value, (int, float)):
            return
        delta = self._stable_float(sample_id, salt, -max_delta, max_delta)
        new_value = float(value) + delta
        if min_value is not None:
            new_value = max(min_value, new_value)
        if max_value is not None:
            new_value = min(max_value, new_value)
        obj[key] = new_value

    @staticmethod
    def _stable_float(sample_id: str, salt: str, low: float, high: float) -> float:
        """Return a deterministic float in [low, high]."""
        digest = hashlib.sha256(f"{sample_id}:{salt}".encode("utf-8")).hexdigest()
        fraction = int(digest[:12], 16) / float(16 ** 12 - 1)
        return low + (high - low) * fraction

    @staticmethod
    def _stable_int(sample_id: str, salt: str, low: int, high: int) -> int:
        """Return a deterministic integer in [low, high]."""
        if low > high:
            low, high = high, low
        digest = hashlib.sha256(f"{sample_id}:{salt}".encode("utf-8")).hexdigest()
        span = high - low + 1
        return low + (int(digest[:8], 16) % span)

    def _perturb_timestamps(self, obj: Any, offset_hours: int) -> None:
        """Add perturbation to timestamps for Test Set D."""
        if isinstance(obj, dict):
            for key in list(obj.keys()):
                val = obj[key]
                if isinstance(val, str) and self._is_iso_timestamp(val):
                    dt = self._parse_time(val)
                    if dt:
                        perturbed = dt + timedelta(hours=offset_hours)
                        obj[key] = perturbed.strftime("%Y-%m-%dT%H:%M:%SZ")
                elif isinstance(val, (dict, list)):
                    self._perturb_timestamps(val, offset_hours)
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                if isinstance(item, str) and self._is_iso_timestamp(item):
                    dt = self._parse_time(item)
                    if dt:
                        perturbed = dt + timedelta(hours=offset_hours)
                        obj[i] = perturbed.strftime("%Y-%m-%dT%H:%M:%SZ")
                elif isinstance(item, (dict, list)):
                    self._perturb_timestamps(item, offset_hours)

    @staticmethod
    def _is_iso_timestamp(s: str) -> bool:
        return bool(re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$", s.strip()))

    @staticmethod
    def _parse_time(s: str) -> Optional[datetime]:
        if not s:
            return None
        s = s.strip()
        for fmt in ["%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d %H:%M:%S"]:
            try:
                return datetime.strptime(s, fmt)
            except ValueError:
                continue
        return None


class IntegrityError(Exception):
    pass


# ---------------------------------------------------------------------------
# DeduplicationScanner
# ---------------------------------------------------------------------------

class DeduplicationScanner:
    """Train/val/test overlap detection using stdlib only."""

    def __init__(
        self,
        ngram_size: int = 5,
        jaccard_threshold: float = 0.8,
    ):
        self.ngram_size = ngram_size
        self.jaccard_threshold = jaccard_threshold

    def scan_text_overlap(
        self,
        train_samples: List[Dict[str, Any]],
        test_samples: List[Dict[str, Any]],
        max_pairs: int = 1000,
    ) -> Dict[str, Any]:
        """Scan for text overlap between train and test prompt JSONs."""
        exact_matches: List[Dict[str, str]] = []
        near_duplicates: List[Dict[str, Any]] = []

        train_blobs: List[Tuple[str, str]] = []
        for s in train_samples[:max_pairs]:
            blob = json.dumps(s.get("prompt", {}), ensure_ascii=False, sort_keys=True)
            train_blobs.append((s.get("sample_id", ""), blob))

        test_blobs: List[Tuple[str, str]] = []
        for s in test_samples[:max_pairs]:
            blob = json.dumps(s.get("prompt", {}), ensure_ascii=False, sort_keys=True)
            test_blobs.append((s.get("sample_id", ""), blob))

        # Exact match check
        train_blob_set = {blob: sid for sid, blob in train_blobs}
        for test_sid, test_blob in test_blobs:
            if test_blob in train_blob_set:
                exact_matches.append({
                    "train_sample_id": train_blob_set[test_blob],
                    "test_sample_id": test_sid,
                })

        # Near-duplicate via n-gram Jaccard
        train_ngrams = [
            (sid, self._char_ngrams(blob)) for sid, blob in train_blobs
        ]
        for test_sid, test_blob in test_blobs:
            test_ng = self._char_ngrams(test_blob)
            for train_sid, train_ng in train_ngrams:
                jaccard = self._jaccard(train_ng, test_ng)
                if jaccard >= self.jaccard_threshold:
                    near_duplicates.append({
                        "train_sample_id": train_sid,
                        "test_sample_id": test_sid,
                        "jaccard": round(jaccard, 4),
                    })

        return {
            "exact_match_count": len(exact_matches),
            "exact_matches": exact_matches[:50],
            "near_duplicate_count": len(near_duplicates),
            "near_duplicates": sorted(
                near_duplicates, key=lambda x: -x["jaccard"]
            )[:50],
            "pairs_scanned": len(train_blobs) * len(test_blobs),
        }

    def scan_storm_id_overlap(
        self,
        split_assignment: Dict[str, List[str]],
    ) -> Dict[str, Any]:
        """Check that no storm_id appears in multiple splits."""
        all_ids: Dict[str, List[str]] = {}
        for split_name, storm_ids in split_assignment.items():
            for sid in storm_ids:
                if sid not in all_ids:
                    all_ids[sid] = []
                all_ids[sid].append(split_name)

        overlaps = {sid: splits for sid, splits in all_ids.items() if len(splits) > 1}
        return {
            "overlap_count": len(overlaps),
            "overlaps": overlaps,
        }

    def scan_structured_overlap(
        self,
        train_samples: List[Dict[str, Any]],
        test_samples: List[Dict[str, Any]],
        max_pairs: int = 500,
    ) -> Dict[str, Any]:
        """Check for structural overlap in numeric guidance data."""
        overlaps: List[Dict[str, Any]] = []

        for ts in test_samples[:max_pairs]:
            test_guidance = ts.get("prompt", {}).get("guidance_inputs", {}).get(
                "ec_single_model_guidance_hres", {}
            )
            test_track = test_guidance.get("track_intensity_points_future", [])

            for trs in train_samples[:max_pairs]:
                train_guidance = trs.get("prompt", {}).get("guidance_inputs", {}).get(
                    "ec_single_model_guidance_hres", {}
                )
                train_track = train_guidance.get("track_intensity_points_future", [])

                # Check if first 3 track points are nearly identical
                if len(test_track) >= 3 and len(train_track) >= 3:
                    match_count = 0
                    for i in range(3):
                        t_pt = test_track[i]
                        tr_pt = train_track[i]
                        lat_diff = abs(
                            (t_pt.get("lat") or 0) - (tr_pt.get("lat") or 0)
                        )
                        lon_diff = abs(
                            (t_pt.get("lon") or 0) - (tr_pt.get("lon") or 0)
                        )
                        wind_diff = abs(
                            (t_pt.get("wind_kt") or 0) - (tr_pt.get("wind_kt") or 0)
                        )
                        if lat_diff < 0.1 and lon_diff < 0.1 and wind_diff < 1:
                            match_count += 1
                    if match_count == 3:
                        overlaps.append({
                            "train_sample_id": trs.get("sample_id", ""),
                            "test_sample_id": ts.get("sample_id", ""),
                            "match_type": "first_3_track_points_nearly_identical",
                        })

        return {
            "structural_overlap_count": len(overlaps),
            "overlaps": overlaps[:50],
        }

    def _char_ngrams(self, text: str) -> Counter:
        return Counter(
            text[i : i + self.ngram_size]
            for i in range(len(text) - self.ngram_size + 1)
        )

    @staticmethod
    def _jaccard(a: Counter, b: Counter) -> float:
        intersection = sum((a & b).values())
        union = sum((a | b).values())
        return intersection / union if union > 0 else 0.0


# ---------------------------------------------------------------------------
# SplitManifest
# ---------------------------------------------------------------------------

class SplitManifest:
    """Frozen test set manifest with SHA256 integrity."""

    VERSION = "1.0"

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir

    def create_manifest(
        self,
        split_assignment: Dict[str, List[str]],
        config: SplitConfig,
        storm_registry: Dict[str, StormRecord],
    ) -> Dict[str, Any]:
        """Create a new split manifest with storm metadata."""
        splits_data: Dict[str, Dict[str, Dict[str, Any]]] = {}
        for split_name, storm_ids in split_assignment.items():
            splits_data[split_name] = {}
            for sid in storm_ids:
                rec = storm_registry.get(sid)
                splits_data[split_name][sid] = {
                    "basin": StormSplitter._basin_to_code(rec.basin) if rec else "unknown",
                    "year": rec.season_year if rec else None,
                    "name_hash": hashlib.sha256(
                        (rec.storm_name or "").encode("utf-8")
                    ).hexdigest()[:12] if rec else "",
                }

        # Compute SHA256 of the assignment
        assignment_str = json.dumps(
            {k: sorted(v) for k, v in split_assignment.items()}, sort_keys=True
        )
        sha = hashlib.sha256(assignment_str.encode("utf-8")).hexdigest()

        manifest = {
            "version": self.VERSION,
            "created_at_utc": datetime.now(timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            ),
            "config": {
                "train_years": list(config.train_years),
                "val_years": list(config.val_years),
                "test_years": list(config.test_years),
                "atcf_era_only": config.atcf_era_only,
                "basin_subset": config.basin_subset,
            },
            "splits": splits_data,
            "frozen": False,
            "sha256": sha,
        }
        return manifest

    def freeze_manifest(self, manifest: Dict[str, Any]) -> Dict[str, Any]:
        """Permanently freeze the manifest (test set locked)."""
        manifest["frozen"] = True
        # Recompute SHA256 with frozen=True
        assignment_str = json.dumps(manifest["splits"], sort_keys=True)
        manifest["sha256"] = hashlib.sha256(assignment_str.encode("utf-8")).hexdigest()
        return manifest

    def is_frozen(self, manifest: Dict[str, Any]) -> bool:
        return manifest.get("frozen", False)

    def verify_integrity(self, manifest: Dict[str, Any]) -> bool:
        """Verify SHA256 of split assignment."""
        stored_sha = manifest.get("sha256", "")
        assignment_str = json.dumps(manifest.get("splits", {}), sort_keys=True)
        computed_sha = hashlib.sha256(assignment_str.encode("utf-8")).hexdigest()
        return stored_sha == computed_sha

    def get_storm_split(self, storm_id: str, manifest: Dict[str, Any]) -> Optional[str]:
        """Look up which split a storm belongs to."""
        for split_name, storm_dict in manifest.get("splits", {}).items():
            if storm_id in storm_dict:
                return split_name
        return None

    def save_manifest(self, manifest: Dict[str, Any], path: Optional[Path] = None) -> Path:
        """Save manifest to disk."""
        if path is None:
            path = self.output_dir / "split_manifest_v1.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return path

    def load_manifest(self, path: Optional[Path] = None) -> Dict[str, Any]:
        """Load manifest from disk."""
        if path is None:
            path = self.output_dir / "split_manifest_v1.json"
        return json.loads(path.read_text(encoding="utf-8"))

    def export_split_lists(
        self, manifest: Dict[str, Any], output_dir: Optional[Path] = None
    ) -> Dict[str, Path]:
        """Export train/val/test storm_id lists as CSV files."""
        out = output_dir or self.output_dir
        out.mkdir(parents=True, exist_ok=True)
        paths: Dict[str, Path] = {}
        for split_name in ["train", "val", "test"]:
            storm_dict = manifest.get("splits", {}).get(split_name, {})
            fp = out / f"split_{split_name}_storms.csv"
            with fp.open("w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["storm_id", "basin", "season_year"])
                for sid, meta in storm_dict.items():
                    writer.writerow([sid, meta.get("basin", ""), meta.get("year", "")])
            paths[split_name] = fp
        return paths


# ---------------------------------------------------------------------------
# ContaminationAuditor
# ---------------------------------------------------------------------------

class ContaminationAuditor:
    """Contamination audit: overlap, ablation, re-identification, rewrite."""

    def __init__(self, groundtruth_csv: Path, dedup_scanner: DeduplicationScanner):
        self.groundtruth_csv = groundtruth_csv
        self.dedup_scanner = dedup_scanner
        self._storm_profiles: Optional[Dict[str, Dict[str, Any]]] = None

    def _load_storm_profiles(self) -> Dict[str, Dict[str, Any]]:
        """Build lightweight storm profiles for re-identification attack."""
        if self._storm_profiles is not None:
            return self._storm_profiles

        profiles: Dict[str, Dict[str, Any]] = {}
        with self.groundtruth_csv.open("r", encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                sid = (row.get("storm_id") or "").strip()
                basin = (row.get("noaa_basin") or "").strip()
                lat = row.get("latitude")
                vmax = row.get("max_wind_wmo")
                if sid not in profiles:
                    profiles[sid] = {
                        "basin": basin,
                        "lats": [],
                        "max_vmax": None,
                        "first_time": None,
                        "last_time": None,
                    }
                if lat:
                    try:
                        profiles[sid]["lats"].append(float(lat))
                    except ValueError:
                        pass
                if vmax:
                    try:
                        v = float(vmax)
                        cur = profiles[sid]["max_vmax"]
                        if cur is None or v > cur:
                            profiles[sid]["max_vmax"] = v
                    except ValueError:
                        pass
                dt_str = (row.get("datetime") or "").strip()[:19]
                if dt_str:
                    try:
                        dt = datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
                        p = profiles[sid]
                        if p["first_time"] is None or dt < p["first_time"]:
                            p["first_time"] = dt
                        if p["last_time"] is None or dt > p["last_time"]:
                            p["last_time"] = dt
                    except ValueError:
                        pass

        self._storm_profiles = profiles
        return profiles

    def run_overlap_scan(
        self,
        train_samples: List[Dict[str, Any]],
        test_samples: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Audit Item 1: Overlap scanning."""
        return self.dedup_scanner.scan_text_overlap(train_samples, test_samples)

    def run_clue_ablation(
        self, samples: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Audit Item 2: Clue ablation test."""
        ablation_fields = ["storm_name", "storm_id", "issue_time_utc"]
        change_ratios: Dict[str, float] = {}

        for field in ablation_fields:
            changed = 0
            for s in samples:
                prompt = s.get("prompt", {})
                meta = prompt.get("storm_meta", {})
                if field in meta and meta[field]:
                    changed += 1
            ratio = changed / max(len(samples), 1)
            change_ratios[field] = round(ratio, 4)

        # Cumulative
        total_chars = 0
        redacted_chars = 0
        import copy
        for s in samples:
            original = json.dumps(s.get("prompt", {}), ensure_ascii=False)
            redacted = copy.deepcopy(s.get("prompt", {}))
            meta = redacted.get("storm_meta", {})
            for field in ablation_fields:
                if field in meta:
                    meta[field] = "[REDACTED]"
            redacted_str = json.dumps(redacted, ensure_ascii=False)
            total_chars += len(original)
            redacted_chars += len(redacted_str)

        change_ratio = 1 - (redacted_chars / max(total_chars, 1))
        return {
            "per_field_change_ratio": change_ratios,
            "cumulative_content_change_ratio": round(change_ratio, 4),
            "note": "ratio > 0.3 suggests significant identity cues in prompt",
        }

    def run_reidentification_attack(
        self,
        anonymized_samples: List[Dict[str, Any]],
        mapping: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Audit Item 3: Re-identification attack."""
        profiles = self._load_storm_profiles()
        id_to_anon = mapping.get("id_to_anon", {})

        reidentified = 0
        ambiguous = 0
        unmatched = 0

        for s in anonymized_samples[:500]:
            prompt = s.get("prompt", {})
            meta = prompt.get("storm_meta", {})

            # Try to match using available features
            basin = meta.get("basin", "")
            # Look for intensity and latitude clues in the sample
            now_state = prompt.get("now_inputs", {}).get(
                "current_state_from_noaa_forecast_advisory", {}
            )
            center = now_state.get("center", {})
            sample_lat = center.get("lat")

            candidates: List[str] = []
            for sid, profile in profiles.items():
                # Match basin
                p_basin = StormSplitter._basin_to_code(profile.get("basin", ""))
                if basin and p_basin != basin:
                    continue
                # Match approximate latitude
                if sample_lat is not None and profile.get("lats"):
                    avg_lat = sum(profile["lats"]) / len(profile["lats"])
                    if abs(avg_lat - sample_lat) > 15:
                        continue
                candidates.append(sid)

            if len(candidates) == 1:
                reidentified += 1
            elif len(candidates) > 1 and len(candidates) <= 5:
                ambiguous += 1
            else:
                unmatched += 1

        total = reidentified + ambiguous + unmatched
        reid_rate = reidentified / max(total, 1)
        return {
            "reidentification_rate": round(reid_rate, 4),
            "ambiguous_rate": round(ambiguous / max(total, 1), 4),
            "anonymity_score": round(1 - reid_rate, 4),
            "recommendation": (
                "anonymization sufficient" if reid_rate < 0.1
                else "consider stronger anonymization"
            ),
        }

    def run_adversarial_rewrite_retest(
        self,
        samples: List[Dict[str, Any]],
        anonymizer: Anonymizer,
        mapping: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Audit Item 4: Adversarial rewrite and retest."""
        auditor = InputLeakageAuditor()

        # Apply anonymization
        anonymized = [anonymizer.anonymize_sample(s, mapping) for s in samples[:100]]

        # Re-audit for identity leakage
        results = [auditor.check_identity_leakage(a) for a in anonymized]
        avg_severity = (
            sum(r["severity"] for r in results) / len(results) if results else 0
        )
        high_severity = sum(1 for r in results if r["severity"] > 0.3)

        return {
            "samples_tested": len(anonymized),
            "avg_identity_severity_post_anonymization": round(avg_severity, 4),
            "high_severity_count": high_severity,
            "passed": avg_severity < 0.3 and high_severity == 0,
        }

    def generate_audit_report(
        self,
        overlap_result: Dict[str, Any],
        ablation_result: Dict[str, Any],
        reid_result: Dict[str, Any],
        rewrite_result: Dict[str, Any],
        manifest: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Generate comprehensive contamination audit report."""
        return {
            "report_version": "1.0",
            "generated_at_utc": datetime.now(timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            ),
            "split_method": "event_level_temporal",
            "train_val_test_time_ranges": manifest.get("config", {}),
            "input_temporal_compliance": "checked_via_InputLeakageAuditor",
            "anonymization_status": "applied",
            "audit_results": {
                "overlap_scan": {
                    "exact_match_count": overlap_result.get("exact_match_count", 0),
                    "near_duplicate_count": overlap_result.get(
                        "near_duplicate_count", 0
                    ),
                },
                "clue_ablation": ablation_result,
                "reidentification_attack": reid_result,
                "adversarial_rewrite": rewrite_result,
            },
            "residual_risks": [
                "base_model_pretraining_contamination_cannot_be_fully_excluded",
                "high_intensity_unique_storms_may_be_identifiable_from_features_alone",
                "CDS_description_text_may_contain_location_identifiers",
            ],
            "model_level_contamination_note": (
                "CoDeC/KDS/CDD methods can provide supplementary evidence "
                "but cannot replace strict data protocol. Limitations per "
                "COLING 2025 research apply."
            ),
        }


# ---------------------------------------------------------------------------
# LeakagePreventionPipeline
# ---------------------------------------------------------------------------

class LeakagePreventionPipeline:
    """Orchestrator enforcing data_leakage.md Section 10 execution order."""

    def __init__(self, base_dir: Path, config: Optional[SplitConfig] = None):
        self.base_dir = base_dir
        self.config = config or SplitConfig()
        self.output_dir = base_dir / "data" / "interim" / "leakage_prevention"

        # Initialize components
        gt_csv = base_dir / "GroundTruth_Cyclones" / "matched_cyclone_tracks.csv"
        cw_csv = base_dir / "data" / "interim" / "atcf" / "storm_id_crosswalk.csv"
        by_storm = base_dir / "data" / "interim" / "atcf" / "by_storm"

        self.splitter = StormSplitter(
            config=self.config,
            groundtruth_csv=gt_csv,
            crosswalk_csv=cw_csv if cw_csv.exists() else None,
            by_storm_dir=by_storm if by_storm.exists() else None,
        )
        self.auditor = InputLeakageAuditor(groundtruth_csv=gt_csv)
        self.anonymizer = Anonymizer(groundtruth_csv=gt_csv)
        self.dedup = DeduplicationScanner()
        self.manifest_mgr = SplitManifest(output_dir=self.output_dir)
        self.contamination = ContaminationAuditor(
            groundtruth_csv=gt_csv, dedup_scanner=self.dedup
        )

    def run_full_pipeline(
        self,
        existing_manifest_path: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """Execute the full leakage prevention pipeline (split + audit only).

        Steps 1-2: Freeze test set + event-level split
        Steps 3-6 require pre-built samples; they run via run_with_samples().
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Step 1-2: Create and freeze split manifest
        split_assignment = self.splitter.assign_splits(
            existing_manifest_path=existing_manifest_path
        )
        registry = self.splitter.load_storm_registry()

        # Validate
        validation = self.splitter.validate_no_cross_contamination(split_assignment)
        if not validation.passed:
            print(f"WARNING: Cross-contamination detected: {validation.violations}")

        # Create manifest
        manifest = self.manifest_mgr.create_manifest(
            split_assignment=split_assignment,
            config=self.config,
            storm_registry=registry,
        )

        # Freeze
        manifest = self.manifest_mgr.freeze_manifest(manifest)

        # Save manifest
        manifest_path = self.manifest_mgr.save_manifest(manifest)

        # Export split lists
        split_paths = self.manifest_mgr.export_split_lists(manifest)

        # Statistics
        stats = self.splitter.get_split_statistics(split_assignment)

        # Generate anonymization mapping
        all_storm_ids = []
        for storm_ids in split_assignment.values():
            all_storm_ids.extend(storm_ids)
        mapping = self.anonymizer.generate_mapping(all_storm_ids)

        # Save mapping
        mapping_path = self.output_dir / "anonymization_mapping_v1.json"
        self.anonymizer.save_mapping(mapping, mapping_path)

        # Save statistics
        stats_path = self.output_dir / "split_statistics_v1.json"
        stats_path.write_text(
            json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        return {
            "manifest": manifest,
            "manifest_path": str(manifest_path),
            "mapping": mapping,
            "mapping_path": str(mapping_path),
            "split_assignment": split_assignment,
            "statistics": stats,
            "validation": {
                "passed": validation.passed,
                "violations": validation.violations,
            },
            "split_list_paths": {k: str(v) for k, v in split_paths.items()},
        }

    def run_with_samples(
        self,
        samples: List[Dict[str, Any]],
        manifest: Dict[str, Any],
        mapping: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Run Steps 4-6 (anonymize, dedup, audit) on pre-built samples."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Split samples by split assignment
        split_assignment = {
            split_name: list(storm_dict.keys())
            for split_name, storm_dict in manifest.get("splits", {}).items()
        }
        id_to_anon = mapping.get("id_to_anon", {})

        split_samples: Dict[str, List[Dict[str, Any]]] = {
            "train": [], "val": [], "test": []
        }
        for s in samples:
            sid = s.get("prompt", {}).get("storm_meta", {}).get("storm_id", "")
            for split_name, storm_ids in split_assignment.items():
                if sid in storm_ids:
                    split_samples[split_name].append(s)
                    break

        # Step 4: Pre-anonymization audit
        pre_audit = self.auditor.audit_batch(samples)

        # Step 5: Anonymize
        anonymized_samples = {
            k: [self.anonymizer.anonymize_sample(s, mapping) for s in v]
            for k, v in split_samples.items()
        }

        # Post-anonymization audit
        all_anonymized = []
        for v in anonymized_samples.values():
            all_anonymized.extend(v)
        post_audit = self.auditor.audit_batch(all_anonymized)

        # Step 6: Dedup scan
        dedup_result = self.dedup.scan_text_overlap(
            anonymized_samples.get("train", []),
            anonymized_samples.get("test", []),
        )
        struct_overlap = self.dedup.scan_structured_overlap(
            anonymized_samples.get("train", []),
            anonymized_samples.get("test", []),
        )
        storm_overlap = self.dedup.scan_storm_id_overlap(split_assignment)

        # Contamination audit
        overlap_audit = self.contamination.run_overlap_scan(
            anonymized_samples.get("train", []),
            anonymized_samples.get("test", []),
        )
        ablation_audit = self.contamination.run_clue_ablation(samples)
        reid_audit = self.contamination.run_reidentification_attack(
            all_anonymized, mapping
        )
        rewrite_audit = self.contamination.run_adversarial_rewrite_retest(
            samples, self.anonymizer, mapping
        )

        contamination_report = self.contamination.generate_audit_report(
            overlap_result=overlap_audit,
            ablation_result=ablation_audit,
            reid_result=reid_audit,
            rewrite_result=rewrite_audit,
            manifest=manifest,
        )

        # Generate test set variants
        test_samples = split_samples.get("test", [])
        test_variants = {
            "main": anonymized_samples.get("test", []),
            "anonymous": self.anonymizer.generate_anonymous_test_set(
                test_samples, mapping
            ),
            "structured_only": self.anonymizer.generate_structured_only_test_set(
                test_samples
            ),
            "perturbation": self.anonymizer.generate_perturbation_test_set(
                test_samples, mapping
            ),
        }

        # Save reports
        self._save_json("leakage_audit_report_v1.json", {
            "pre_anonymization": pre_audit,
            "post_anonymization": post_audit,
        })
        self._save_json("dedup_report_v1.json", {
            "text_overlap": dedup_result,
            "structural_overlap": struct_overlap,
            "storm_id_overlap": storm_overlap,
        })
        self._save_json("contamination_audit_report_v1.json", contamination_report)

        return {
            "pre_audit": pre_audit,
            "post_audit": post_audit,
            "dedup": dedup_result,
            "structural_overlap": struct_overlap,
            "contamination_report": contamination_report,
            "test_set_variants": {
                k: len(v) for k, v in test_variants.items()
            },
        }

    def _save_json(self, filename: str, data: Any) -> Path:
        path = self.output_dir / filename
        path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Data Leakage Prevention Pipeline"
    )
    parser.add_argument(
        "--mode",
        choices=["split", "full", "audit-only"],
        default="split",
        help="Pipeline mode: split (split only), full (split+audit), audit-only",
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default=".",
        help="Project root directory",
    )
    parser.add_argument(
        "--train-years",
        nargs=2,
        type=int,
        default=[2016, 2020],
        help="Train year range (inclusive)",
    )
    parser.add_argument(
        "--val-years",
        nargs=2,
        type=int,
        default=[2021, 2022],
        help="Validation year range (inclusive)",
    )
    parser.add_argument(
        "--test-years",
        nargs=2,
        type=int,
        default=[2023, 2025],
        help="Test year range (inclusive)",
    )
    parser.add_argument(
        "--basin-subset",
        nargs="*",
        default=None,
        help="Basin codes to include (e.g., AL EP). Default: all",
    )
    parser.add_argument(
        "--existing-manifest",
        type=str,
        default=None,
        help="Path to existing frozen manifest",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: data/interim/leakage_prevention)",
    )

    args = parser.parse_args()
    base_dir = Path(args.base_dir)

    config = SplitConfig(
        train_years=tuple(args.train_years),
        val_years=tuple(args.val_years),
        test_years=tuple(args.test_years),
        basin_subset=args.basin_subset,
    )

    pipeline = LeakagePreventionPipeline(base_dir=base_dir, config=config)

    if args.output_dir:
        pipeline.output_dir = Path(args.output_dir)

    existing_manifest = (
        Path(args.existing_manifest) if args.existing_manifest else None
    )

    if args.mode in ("split", "full"):
        print("=== Step 1-2: Event-level temporal split + freeze test set ===")
        result = pipeline.run_full_pipeline(
            existing_manifest_path=existing_manifest,
        )

        manifest = result["manifest"]
        stats = result["statistics"]

        print(f"\nSplit statistics:")
        for split_name, split_stats in stats.items():
            print(
                f"  {split_name}: {split_stats['storm_count']} storms, "
                f"{split_stats['row_count']} rows, "
                f"years {split_stats['year_range']}, "
                f"basins {split_stats['basin_distribution']}"
            )

        print(f"\nManifest: {result['manifest_path']}")
        print(f"Mapping: {result['mapping_path']}")
        print(f"Validation: {'PASSED' if result['validation']['passed'] else 'FAILED'}")

        if not result["validation"]["passed"]:
            for v in result["validation"]["violations"]:
                print(f"  VIOLATION: {v}")

        print(f"\nManifest frozen: {manifest.get('frozen', False)}")
        print(f"Manifest SHA256: {manifest.get('sha256', '')[:16]}...")

        integrity = pipeline.manifest_mgr.verify_integrity(manifest)
        print(f"Integrity check: {'PASSED' if integrity else 'FAILED'}")

    if args.mode == "audit-only":
        print("=== Audit-only mode: load existing manifest ===")
        manifest_path = existing_manifest or pipeline.output_dir / "split_manifest_v1.json"
        manifest = pipeline.manifest_mgr.load_manifest(manifest_path)

        print(f"Manifest loaded: frozen={manifest.get('frozen', False)}")
        print(f"Integrity: {'PASSED' if pipeline.manifest_mgr.verify_integrity(manifest) else 'FAILED'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
