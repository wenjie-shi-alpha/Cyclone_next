#!/usr/bin/env python3
"""Batch dataset builder: iterate advisories from manifest, build full samples.

Takes the parameterized logic from build_dataset_sample_preview_v0_1.py and
applies it to all advisories in the ASCAT manifest, producing raw JSON samples
split into train/val/test per the frozen leakage prevention manifest.
"""

from __future__ import annotations

import csv
import json
import sys
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


BASE = Path(".")


# ---------------------------------------------------------------------------
# Reuse from build_dataset_sample_preview_v0_1.py
# ---------------------------------------------------------------------------

def _import_build_module():
    """Import the existing build module's functions."""
    # Add scripts/ to path so we can import the module
    scripts_dir = BASE / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    import build_dataset_sample_preview_v0_1 as bm
    return bm


# ---------------------------------------------------------------------------
# Path derivation
# ---------------------------------------------------------------------------

def derive_noaa_paths(source_file: str) -> Dict[str, Optional[Path]]:
    """Derive all NOAA file paths from the forecast_advisory source_file.

    source_file example:
        noaa/2020/Atlantic/EDOUARD/forecast_advisory/al052020.fstadv.007.txt

    Derives:
        forecast_discussion: .../forecast_discussion/al052020.discus.007.txt
        public_advisory:     .../public_advisory/al052020.public.007.txt
        wind_probabilities:  .../wind_speed_probabilities/al052020.wndprb.007.txt
    """
    sf = Path(source_file)
    result: Dict[str, Optional[Path]] = {"forecast_advisory": BASE / sf}

    # Extract parts
    # sf = noaa/2020/Atlantic/EDOUARD/forecast_advisory/al052020.fstadv.007.txt
    parts = sf.parts  # ('noaa', '2020', 'Atlantic', 'EDOUARD', 'forecast_advisory', 'al052020.fstadv.007.txt')
    if len(parts) < 6:
        return {k: None for k in result}

    noaa_root = Path(*parts[:4])  # noaa/2020/Atlantic/EDOUARD
    fname = parts[-1]  # al052020.fstadv.007.txt

    # Parse: {storm_code}.{type}.{number}.txt
    fname_parts = fname.rsplit(".", 2)  # ['al052020.fstadv', '007', 'txt'] -- nope
    # Better: split on first dot and last dot
    base_name = fname  # al052020.fstadv.007.txt
    # Pattern: {code}.{type_ext}.{num}.txt
    m_dot = base_name.rsplit(".", 2)  # ['al052020.fstadv', '007', 'txt'] -- wrong for 'fstadv'
    # Actually: al052020.fstadv.007.txt -> code=al052020, ext=fstadv, num=007
    # Split: remove .txt, then split by last dot for number, rest is code.ext
    stem = base_name
    if stem.endswith(".txt"):
        stem = stem[:-4]
    # stem = al052020.fstadv.007
    last_dot = stem.rfind(".")
    if last_dot > 0:
        number = stem[last_dot + 1:]  # 007
        code_ext = stem[:last_dot]  # al052020.fstadv
        ext_dot = code_ext.find(".")
        if ext_dot > 0:
            code = code_ext[:ext_dot]  # al052020
        else:
            code = code_ext
    else:
        number = "001"
        code = stem

    path_map = {
        "forecast_discussion": f"{code}.discus.{number}.txt",
        "public_advisory": f"{code}.public.{number}.txt",
        "wind_probabilities": f"{code}.wndprb.{number}.txt",
    }

    for key, filename in path_map.items():
        p = BASE / noaa_root / key / filename
        result[key] = p if p.exists() else None

    return result


def derive_hres_init_time(issue_dt: datetime) -> datetime:
    """Derive HRES model initialization time: round down to nearest 6h."""
    hour = (issue_dt.hour // 6) * 6
    return issue_dt.replace(hour=hour, minute=0, second=0, microsecond=0)


# ---------------------------------------------------------------------------
# Batch builder
# ---------------------------------------------------------------------------

class BatchBuilder:
    """Build dataset samples from the advisory manifest."""

    def __init__(
        self,
        manifest_csv: Path,
        split_manifest_path: Path,
        output_dir: Path,
        max_samples: Optional[int] = None,
        year_start: int = 2016,
        year_end: int = 2025,
        enable_raw_audit: bool = False,
    ):
        self.manifest_csv = manifest_csv
        self.split_manifest_path = split_manifest_path
        self.output_dir = output_dir
        self.max_samples = max_samples
        self.year_start = year_start
        self.year_end = year_end
        self.enable_raw_audit = enable_raw_audit

        self.bm = _import_build_module()
        self.split_manifest = self._load_split_manifest()
        self.storm_splits = self._build_storm_split_index()
        self.raw_auditor = self._init_raw_auditor() if enable_raw_audit else None

        # Pre-load caches for performance (avoid re-reading CSVs per sample)
        self._gt_cache: Dict[str, List[Dict[str, str]]] = {}
        self._obs_cache: Dict[str, Dict[str, Dict[str, str]]] = {}
        self._preload_groundtruth()
        self._preload_observations()

    def _init_raw_auditor(self) -> Optional[Any]:
        """Instantiate the raw-sample auditor once for the whole batch run."""
        try:
            from data_leakage_prevention import InputLeakageAuditor
        except ImportError:
            return None
        return InputLeakageAuditor(
            groundtruth_csv=BASE / "GroundTruth_Cyclones" / "matched_cyclone_tracks.csv"
        )

    def _load_split_manifest(self) -> Dict[str, Any]:
        if self.split_manifest_path.exists():
            return json.loads(self.split_manifest_path.read_text(encoding="utf-8"))
        return {"splits": {}}

    def _build_storm_split_index(self) -> Dict[str, str]:
        """Build storm_id -> split_name index."""
        index: Dict[str, str] = {}
        for split_name, storm_dict in self.split_manifest.get("splits", {}).items():
            if isinstance(storm_dict, dict):
                for sid in storm_dict:
                    index[sid] = split_name
        return index

    def _preload_groundtruth(self) -> None:
        """Pre-load and index ground truth CSV by storm_id."""
        gt_path = BASE / "GroundTruth_Cyclones" / "matched_cyclone_tracks.csv"
        if not gt_path.exists():
            return
        with gt_path.open("r", encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                sid = (row.get("storm_id") or "").strip()
                if sid:
                    if sid not in self._gt_cache:
                        self._gt_cache[sid] = []
                    self._gt_cache[sid].append(row)
        print(f"  Ground truth cache: {len(self._gt_cache)} storms, {sum(len(v) for v in self._gt_cache.values())} rows")

    def _preload_observations(self) -> None:
        """Pre-load GOES/ASCAT/Recon observation features into lookup dict."""
        for obs_name, obs_path in [
            ("goes", BASE / "data" / "interim" / "goes" / "goes_observation_features.csv"),
            ("ascat", BASE / "data" / "interim" / "ascat" / "ascat_observation_features.csv"),
            ("recon", BASE / "data" / "interim" / "recon" / "recon_observation_features.csv"),
        ]:
            if not obs_path.exists():
                continue
            cache: Dict[str, Dict[str, str]] = {}
            with obs_path.open("r", encoding="utf-8", newline="") as f:
                for row in csv.DictReader(f):
                    sid = (row.get("storm_id") or "").strip()
                    issue_t = (row.get("issue_time_utc") or "").strip()
                    if sid and issue_t:
                        cache[f"{sid}|{issue_t}"] = row
            self._obs_cache[obs_name] = cache
            print(f"  {obs_name} obs cache: {len(cache)} entries")

    def _cached_load_goes_obs(
        self, storm_id: str, issue_dt: datetime, max_time_diff_hours: int = 3
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Load GOES observation using pre-loaded cache."""
        goes_fp = BASE / "data" / "interim" / "goes" / "goes_observation_features.csv"
        trace: Dict[str, Any] = {"goes_feature_file": str(goes_fp)}

        # Try cache lookup with exact match first
        cache = self._obs_cache.get("goes", {})
        issue_str = issue_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        exact = cache.get(f"{storm_id}|{issue_str}")
        if exact:
            # Found exact match, build block from row
            return self._build_goes_block_from_row(exact, goes_fp, trace, 0.0)

        # Try nearest match within tolerance
        best_row = None
        best_abs_h = None
        for key, row in cache.items():
            if not key.startswith(f"{storm_id}|"):
                continue
            row_issue_dt = self.bm.parse_iso_utc_flexible(row.get("issue_time_utc") or "")
            if row_issue_dt is None:
                continue
            dt_abs_h = abs((row_issue_dt - issue_dt).total_seconds()) / 3600.0
            if best_abs_h is None or dt_abs_h < best_abs_h:
                best_abs_h = dt_abs_h
                best_row = row

        if best_row is not None and best_abs_h is not None and best_abs_h <= max_time_diff_hours:
            return self._build_goes_block_from_row(best_row, goes_fp, trace, best_abs_h)

        # Fall back to original function
        return self.bm.load_goes_observation_structured(storm_id, issue_dt, max_time_diff_hours)

    def _build_goes_block_from_row(
        self, row: Dict[str, str], goes_fp: Path, trace: Dict[str, Any], time_delta: float
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Build GOES observation block from a cached row."""
        goes_status = (row.get("goes_status") or "").strip()
        if goes_status != "available":
            missing_reason = (row.get("missing_reason") or "").strip() or "GOES row marked missing"
            return self.bm.build_observation_placeholder(missing_reason=missing_reason, source_file=str(goes_fp)), trace

        obs_time_utc = (row.get("obs_time_utc") or "").strip() or (row.get("issue_time_utc") or "").strip()
        source_platform = (row.get("goes_source_collection") or "").strip() or "GOES_MCMIPC"
        qc_has_image = str(row.get("qc_has_image") or "").strip() in {"1", "true", "True", "TRUE"}
        qc_flag = "ok" if qc_has_image else "warn"

        metric_specs = [
            ("cloud_top_temp_min_k", "c13_min_k", "K"),
            ("cloud_top_temp_p10_k", "c13_p10_k", "K"),
            ("cloud_top_temp_mean_k", "c13_mean_k", "K"),
            ("cloud_top_temp_std_k", "c13_std_k", "K"),
            ("cold_cloud_area_inner_km2", "cold_area_inner_km2", "km2"),
            ("cold_cloud_fraction_inner", "cold_fraction_inner", "ratio"),
            ("cold_cloud_area_ring_km2", "cold_area_ring_km2", "km2"),
            ("cold_cloud_fraction_ring", "cold_fraction_ring", "ratio"),
            ("eye_ring_temp_contrast_k", "eye_ring_temp_contrast_k", "K"),
        ]

        value_rows = []
        for signal_name, col_name, unit in metric_specs:
            val = self.bm.sanitize_goes_metric(signal_name, self.bm.parse_float(row.get(col_name)))
            if val is None:
                continue
            value_rows.append({
                "obs_time_utc": obs_time_utc,
                "obs_type": "goes_ir_structured",
                "signal": signal_name,
                "value": round(val, 4),
                "unit": unit,
                "qc_flag": qc_flag,
                "source_platform": source_platform,
            })

        if not value_rows:
            return self.bm.build_observation_placeholder(missing_reason="No numeric features parsed", source_file=str(goes_fp)), trace

        trace.update({
            "goes_request_id": (row.get("request_id") or "").strip() or None,
            "goes_issue_time_selected_utc": (row.get("issue_time_utc") or "").strip() or None,
            "goes_obs_time_selected_utc": obs_time_utc or None,
        })

        block = {
            "status": "available",
            "source_file": str(goes_fp),
            "match_rule": "storm_id + nearest_issue_time_within_3h",
            "time_match_delta_hours": round(time_delta, 3),
            "obs_time_utc": obs_time_utc,
            "source_platform": source_platform,
            "value": value_rows,
            "qc": {
                "has_image": 1 if qc_has_image else 0,
                "time_within_window": str(row.get("qc_time_within_window") or ""),
                "obs_offset_minutes": self.bm.parse_float(row.get("obs_offset_minutes")),
                "obs_offset_abs_minutes": self.bm.parse_float(row.get("obs_offset_abs_minutes")),
            },
            "expected_real_fields": ["obs_time_utc", "obs_type", "signal", "value", "qc_flag", "source_platform"],
            "policy_note": "GOES metrics from satellite radiances only, no forecast text leakage",
        }
        return block, trace

    def _cached_load_ascat_obs(
        self, storm_id: str, issue_dt: datetime, max_time_diff_hours: int = 3
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Load ASCAT observation using pre-loaded cache."""
        fp = BASE / "data" / "interim" / "ascat" / "ascat_observation_features.csv"
        trace: Dict[str, Any] = {"ascat_feature_file": str(fp)}

        cache = self._obs_cache.get("ascat", {})
        if not cache:
            return self.bm.load_ascat_observation_structured(storm_id, issue_dt, max_time_diff_hours)

        row, dt_abs_h, err = self._find_nearest_obs_row(cache, storm_id, issue_dt, max_time_diff_hours)
        if row is None:
            return self.bm.build_observation_placeholder(missing_reason=f"ASCAT {err}", source_file=str(fp)), trace

        # Build block using original function's logic with the cached row
        return self.bm.load_ascat_observation_structured(storm_id, issue_dt, max_time_diff_hours)

    def _cached_load_recon_obs(
        self, storm_id: str, issue_dt: datetime, max_time_diff_hours: int = 6
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Load Recon observation using pre-loaded cache."""
        cache = self._obs_cache.get("recon", {})
        if not cache:
            return self.bm.load_recon_observation_structured(storm_id, issue_dt, max_time_diff_hours)

        # Recon still uses the original scan-based function for now
        return self.bm.load_recon_observation_structured(storm_id, issue_dt, max_time_diff_hours)

    def _find_nearest_obs_row(
        self, cache: Dict[str, Dict[str, str]],
        storm_id: str, issue_dt: datetime, max_time_diff_hours: int
    ) -> Tuple[Optional[Dict[str, str]], Optional[float], str]:
        """Find nearest observation row in cache by storm_id and issue_time."""
        best_row = None
        best_abs_h = None
        for key, row in cache.items():
            if not key.startswith(f"{storm_id}|"):
                continue
            row_issue_dt = self.bm.parse_iso_utc_flexible(row.get("issue_time_utc") or "")
            if row_issue_dt is None:
                continue
            dt_abs_h = abs((row_issue_dt - issue_dt).total_seconds()) / 3600.0
            if best_abs_h is None or dt_abs_h < best_abs_h:
                best_abs_h = dt_abs_h
                best_row = row

        if best_row is None:
            return None, None, "No rows matched storm_id and issue_time window"
        if best_abs_h > max_time_diff_hours:
            return None, best_abs_h, f"Nearest row exceeds time tolerance: {best_abs_h:.2f}h > {max_time_diff_hours}h"
        return best_row, best_abs_h, ""

    def _cached_load_groundtruth_state(
        self, storm_id: str, issue_dt: datetime
    ) -> Dict[str, Any]:
        """Load ground truth state using pre-loaded cache."""
        rows = self._gt_cache.get(storm_id, [])
        if not rows:
            return {}

        best = None
        best_dt = None
        for row in rows:
            dt_str = (row.get("datetime") or "")[:19]
            try:
                dt = datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
            except ValueError:
                continue
            if best is None or abs((dt - issue_dt).total_seconds()) < abs((best_dt - issue_dt).total_seconds()):
                best = row
                best_dt = dt

        if best is None or best_dt is None:
            return {}

        return {
            "matched_datetime_utc": best_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "lat": self.bm.parse_float(best.get("latitude")),
            "lon": self.bm.parse_float(best.get("longitude")),
            "max_wind_wmo": self.bm.parse_float(best.get("max_wind_wmo")),
            "min_pressure_wmo": self.bm.parse_float(best.get("min_pressure_wmo")),
            "max_wind_usa": self.bm.parse_float(best.get("max_wind_usa")),
            "min_pressure_usa": self.bm.parse_float(best.get("min_pressure_usa")),
            "storm_speed": self.bm.parse_float(best.get("storm_speed")),
            "storm_direction": self.bm.parse_float(best.get("storm_direction")),
        }

    def _load_manifest_rows(self) -> List[Dict[str, str]]:
        """Load and filter manifest rows."""
        rows = []
        with self.manifest_csv.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                sid = (row.get("storm_id") or "").strip()
                match_status = (row.get("storm_id_match_status") or "").strip()
                if match_status != "matched":
                    continue
                yr = int(sid[:4]) if len(sid) >= 4 else 0
                if yr < self.year_start or yr > self.year_end:
                    continue
                rows.append(row)
        return rows

    def build_all(self) -> Dict[str, Any]:
        """Build all samples from manifest."""
        rows = self._load_manifest_rows()
        if self.max_samples:
            rows = rows[:self.max_samples]

        stats = {
            "total_manifest_rows": len(rows),
            "built": 0,
            "skipped": 0,
            "skip_reasons": {},
            "split_counts": {"train": 0, "val": 0, "test": 0, "unassigned": 0},
            "raw_sample_audit_summary": {
                "enabled": self.enable_raw_audit,
                "temporal_pass": 0,
                "verification_pass": 0,
                "identity_avg": 0.0,
                "note": (
                    "Raw canonical-sample audit is optional and runs before SFT/RL formatting. "
                    "Use format_report.json for final train-view leakage checks."
                ),
            },
        }

        # Prepare output dirs
        raw_dirs = {}
        existing_raw_files = set()
        for split_name in ["train", "val", "test", "unassigned"]:
            d = self.output_dir / "raw" / split_name
            d.mkdir(parents=True, exist_ok=True)
            raw_dirs[split_name] = d
            existing_raw_files.update(d.glob("*.json"))

        identity_severities = []
        build_manifest_entries = []
        written_raw_files = set()

        for i, row in enumerate(rows):
            storm_id = (row.get("storm_id") or "").strip()
            storm_name = (row.get("storm_name") or "").strip()
            basin = (row.get("basin") or "").strip()
            advisory_no = int(row.get("advisory_no") or 0)
            issue_time_str = (row.get("issue_time_utc") or "").strip()
            source_file = (row.get("source_file") or "").strip()

            if not issue_time_str or not source_file:
                self._record_skip(stats, "missing_issue_time_or_source")
                continue

            try:
                issue_dt = datetime.strptime(issue_time_str, "%Y-%m-%dT%H:%M:%SZ")
            except ValueError:
                self._record_skip(stats, "invalid_issue_time_format")
                continue

            # Determine split
            split_name = self.storm_splits.get(storm_id, "unassigned")

            # Build sample
            try:
                sample = self._build_one_sample(
                    storm_id=storm_id,
                    storm_name=storm_name,
                    basin=basin,
                    issue_dt=issue_dt,
                    advisory_no=advisory_no,
                    source_file=source_file,
                )
            except Exception as e:
                self._record_skip(stats, f"build_error:{type(e).__name__}")
                if i < 20:
                    print(f"  ERROR [{storm_id} adv#{advisory_no}]: {e}")
                continue

            if sample is None:
                self._record_skip(stats, "build_returned_none")
                continue

            # Run leakage audit
            if self.raw_auditor is not None:
                audit = self.raw_auditor.audit_sample(sample)
                if audit["temporal_leakage"]["passed"]:
                    stats["raw_sample_audit_summary"]["temporal_pass"] += 1
                if audit["verification_leakage"]["passed"]:
                    stats["raw_sample_audit_summary"]["verification_pass"] += 1
                identity_severities.append(audit["identity_leakage"]["severity"])

            # Save raw sample
            sample_id = sample.get("sample_id", f"{storm_id}_{issue_time_str}_{advisory_no:03d}")
            raw_path = raw_dirs[split_name] / f"{sample_id}.json"
            raw_path.write_text(
                json.dumps(sample, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            written_raw_files.add(raw_path)

            # Record
            stats["built"] += 1
            stats["split_counts"][split_name] = stats["split_counts"].get(split_name, 0) + 1
            build_manifest_entries.append({
                "sample_id": sample_id,
                "storm_id": storm_id,
                "split": split_name,
                "issue_time_utc": issue_time_str,
                "advisory_no": advisory_no,
            })

            if (i + 1) % 500 == 0:
                print(f"  Progress: {i+1}/{len(rows)} built={stats['built']} skipped={stats['skipped']}")

        # Average identity severity
        if identity_severities:
            stats["raw_sample_audit_summary"]["identity_avg"] = round(
                sum(identity_severities) / len(identity_severities), 4
            )
        stats["raw_sample_audit_summary"]["total_audited"] = len(identity_severities)

        # Remove stale raw files so full rebuilds cannot mix old and new samples.
        stale_raw_files = existing_raw_files - written_raw_files
        for stale_path in stale_raw_files:
            stale_path.unlink()
        stats["stale_raw_removed"] = len(stale_raw_files)

        # Save build manifest
        manifest_path = self.output_dir / "build_manifest.json"
        manifest_path.write_text(
            json.dumps(build_manifest_entries, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        # Save report
        report_path = self.output_dir / "build_report.json"
        report_path.write_text(
            json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        return stats

    def _build_one_sample(
        self,
        storm_id: str,
        storm_name: str,
        basin: str,
        issue_dt: datetime,
        advisory_no: int,
        source_file: str,
    ) -> Optional[Dict[str, Any]]:
        """Build one sample using the existing build module's functions."""
        bm = self.bm

        # Derive file paths
        paths = derive_noaa_paths(source_file)
        adv_path = paths.get("forecast_advisory")
        dis_path = paths.get("forecast_discussion")
        pub_path = paths.get("public_advisory")
        wnd_path = paths.get("wind_probabilities")

        # P0 files must exist
        if adv_path is None or not adv_path.exists():
            return None
        if dis_path is None or not dis_path.exists():
            return None

        # Parse NOAA text products
        adv_lines = bm.read_text(adv_path)
        dis_lines = bm.read_text(dis_path)

        adv_summary = bm.extract_advisory_summary(adv_lines)
        dis_summary = bm.extract_discussion_sections(dis_lines)

        pub_summary = {}
        if pub_path and pub_path.exists():
            pub_summary = bm.extract_public_summary(bm.read_text(pub_path))

        wnd_summary = {}
        if wnd_path and wnd_path.exists():
            wnd_summary = bm.extract_wind_prob_summary(bm.read_text(wnd_path))

        # Current state
        center = adv_summary.get("center", {})
        if not center:
            return None

        target_lat = center.get("lat", 0)
        target_lon = center.get("lon", 0)
        motion = adv_summary.get("motion", {})
        intensity = adv_summary.get("intensity", {})

        # CDS environment
        cds_row = None
        cds_file = None
        try:
            cds_row, cds_file = bm.pick_cds_row(issue_dt, target_lat, target_lon)
        except Exception:
            pass

        if cds_row is None:
            return None  # CDS is P0

        cds_features = bm.extract_cds_features(cds_row)

        # HRES guidance
        hres_init = derive_hres_init_time(issue_dt)
        hres_track = []
        hres_env = []
        hres_track_file = None
        hres_system_file = None
        try:
            hres_track, hres_track_file = bm.extract_hres_track(storm_id, hres_init)
            hres_env, hres_system_file = bm.extract_hres_system(storm_id, hres_init)
        except Exception:
            pass

        hres_track_with_lead = bm.add_issue_lead_fields(hres_track, issue_dt, "tau_from_model_init_h")
        hres_env_with_lead = bm.add_issue_lead_fields(hres_env, issue_dt, "tau_from_model_init_h")

        pre_issue_track = bm.latest_pre_issue_point(hres_track_with_lead, issue_dt)
        pre_issue_env = bm.latest_pre_issue_point(hres_env_with_lead, issue_dt)
        future_track = bm.future_only(hres_track_with_lead, issue_dt)
        future_env = bm.future_only(hres_env_with_lead, issue_dt)

        # Observations (use cached data when available, fall back to original scan)
        goes_obs_block, goes_trace = self._cached_load_goes_obs(storm_id, issue_dt)
        ascat_obs_block, ascat_trace = self._cached_load_ascat_obs(storm_id, issue_dt)
        recon_obs_block, recon_trace = self._cached_load_recon_obs(storm_id, issue_dt)

        obs_blocks = [goes_obs_block, ascat_obs_block, recon_obs_block]
        obs_any_available = any(b.get("status") == "available" for b in obs_blocks)
        obs_all_available = all(b.get("status") == "available" for b in obs_blocks)

        # Merge observation blocks
        merged_obs = bm.merge_observation_blocks(goes_obs_block, ascat_obs_block, recon_obs_block)

        # ATCF guidance
        atcf_guidance_available = False
        atcf_guidance = {}
        atcf_guidance_trace = {}
        try:
            atcf_result, atcf_trace = bm.load_atcf_multimodel_guidance(storm_id, issue_dt)
            if atcf_result is not None:
                atcf_guidance = atcf_result
                atcf_guidance_available = atcf_result.get("status") == "available"
            atcf_guidance_trace = {
                "guidance_file": atcf_trace.get("source_track_file") if atcf_trace else None,
                "spread_file": atcf_trace.get("source_spread_file") if atcf_trace else None,
            }
        except Exception:
            pass

        if not atcf_guidance_available:
            atcf_guidance = bm.build_multimodel_proxy(future_track)
            atcf_guidance_trace = {"guidance_file": None, "spread_file": None}

        # Guidance model IDs
        guidance_model_ids = set()
        if atcf_guidance_available and "model_ids" in atcf_guidance:
            guidance_model_ids = set(atcf_guidance.get("model_ids", []))

        # Verification
        preferred_verify_available = False
        preferred_verify = {}
        preferred_verify_trace = {}
        try:
            verify_result, verify_trace = bm.load_preferred_verification_future_series(storm_id, issue_dt)
            if verify_result is not None:
                preferred_verify = verify_result
                preferred_verify_available = verify_result.get("status") == "available"
            preferred_verify_trace = {
                "preferred_verification_file": verify_trace.get("preferred_verification_file"),
            }
        except Exception:
            pass

        # Ground truth state (use cached version)
        gt_state = self._cached_load_groundtruth_state(storm_id, issue_dt)

        # Build prompt
        prompt = {
            "storm_meta": {
                "storm_id": storm_id,
                "storm_name": storm_name,
                "basin": basin,
                "issue_time_utc": issue_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "advisory_no": advisory_no,
                "time_match_rule": "nearest_within_3h",
            },
            "now_inputs": {
                "current_state_from_noaa_forecast_advisory": {
                    "center": center,
                    "motion": motion,
                    "intensity": intensity,
                },
                "environment_now_ec_reanalysis": {
                    "source_file": str(cds_file) if cds_file else None,
                    "source_time": cds_row.get("time", "") if cds_row else None,
                    "tc_position": cds_row.get("tc_position") if cds_row else None,
                    "features": cds_features,
                },
                "observation_evidence_structured": merged_obs,
                "pre_issue_guidance_context": {
                    "ec_hres_latest_point_at_or_before_issue_track": pre_issue_track,
                    "ec_hres_latest_point_at_or_before_issue_environment": pre_issue_env,
                },
            },
            "guidance_inputs": {
                "guidance_time_reference": {
                    "issue_time_utc": issue_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "model_init_time_utc": hres_init.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "tau_reference": "tau_from_model_init_h",
                    "lead_reference": "lead_from_issue_h",
                    "rule": "future_guidance_blocks_include_only_valid_time_after_issue",
                },
                "ec_single_model_guidance_hres": {
                    "model": "HRES",
                    "init_time_utc": hres_init.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "source_track_file": str(hres_track_file) if hres_track_file else None,
                    "source_environment_file": str(hres_system_file) if hres_system_file else None,
                    "track_intensity_points_future": future_track,
                    "environment_points_future": future_env,
                },
                "multimodel_guidance_a_deck": atcf_guidance,
            },
        }

        # Build target
        target = {
            "official_outputs": {
                "track_intensity_table": {
                    "from_forecast_advisory": adv_summary.get("forecast_table", []),
                },
                "risk_messages": {
                    "watch_warning_text": adv_summary.get("watch_warning_text"),
                    "public_advisory_summary": pub_summary,
                    "wind_speed_probabilities": wnd_summary,
                },
                "reasoning_text": {
                    "sections": {
                        "current_analysis_text": dis_summary.get("current_analysis_text", ""),
                        "forecast_reasoning_text": dis_summary.get("forecast_reasoning_text", ""),
                        "additional_context_text": dis_summary.get("additional_context_text", ""),
                    }
                },
            }
        }

        # Build verification targets
        future_best_track_series = preferred_verify if preferred_verify_available else {
            "status": "missing_real_data",
            "policy": "prefer_atcf_b_deck_then_ibtracs_matched_groundtruth",
            "points_future": [],
        }

        verification_targets = {
            "policy": "offline_evaluation_or_rft_reward_only_never_in_prompt",
            "groundtruth_source_policy": {
                "preferred_order": ["atcf_b_deck", "ibtracs_matched_groundtruth"],
                "selection_rule": "use storm-level preferred verification table",
            },
            "best_track_point_near_issue": {
                "status": "available" if gt_state else "missing",
                "source_file": "GroundTruth_Cyclones/matched_cyclone_tracks.csv",
                "value": gt_state,
                "note": "kept outside prompt to avoid leakage",
            },
            "future_best_track_series": future_best_track_series,
        }

        # Leakage audit
        prompt_blob = json.dumps(prompt, ensure_ascii=False)
        gt_marker_keys = [
            "max_wind_wmo", "min_pressure_wmo",
            "max_wind_usa", "min_pressure_usa",
            "storm_speed", "storm_direction",
        ]
        guidance_future_ok = all(
            p.get("lead_from_issue_h", 0) > 0
            for p in future_track + future_env
        )

        leakage_audit = {
            "checks": {
                "full_discussion_text_in_prompt": bool(dis_summary.get("full_reasoning_text")) and dis_summary.get("full_reasoning_text", "") in prompt_blob,
                "forecast_reasoning_text_in_prompt": bool(dis_summary.get("forecast_reasoning_text")) and dis_summary.get("forecast_reasoning_text", "") in prompt_blob,
                "public_advisory_summary_in_prompt": bool(pub_summary.get("summary_lines")) and all(line in prompt_blob for line in pub_summary.get("summary_lines", [])),
                "groundtruth_verification_fields_in_prompt": any(f'"{k}"' in prompt_blob for k in gt_marker_keys),
                "future_guidance_points_only": guidance_future_ok,
                "current_analysis_text_in_prompt": bool(dis_summary.get("current_analysis_text")) and dis_summary.get("current_analysis_text", "") in prompt_blob,
                "ofcl_not_leaked_into_guidance": not bool(guidance_model_ids & bm.ATCF_BLOCKED_GUIDANCE_MODELS),
            },
            "excluded_from_prompt": [
                "current_analysis_text_from_discussion",
                "forecast_reasoning_text_from_discussion",
                "public_advisory_summary_lines",
                "groundtruth_verification_fields",
                "official_track_from_discussion",
            ],
        }

        sample_id = f"{basin}_{storm_id}_{issue_dt.strftime('%Y-%m-%dT%H:%M:%SZ')}_{advisory_no:03d}"

        sample = {
            "sample_id": sample_id,
            "schema_version": "tc_sft_dataset_v0.1.3_strict",
            "task_type": "tc_forecast_sft",
            "time_window": f"{self.year_start}-{self.year_end}",
            "build_info": {
                "assembled_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "design_policy": "strict_anti_leak_with_bdeck_primary_ibtracs_fallback_verify",
            },
            "prompt": prompt,
            "target": target,
            "verification_targets": verification_targets,
            "leakage_audit": leakage_audit,
            "quality_flags": {
                "guidance_qc_pass": 1 if atcf_guidance_available else 0,
                "observation_status": "partial_available" if obs_any_available and not obs_all_available else ("available" if obs_all_available else "missing_real_data"),
                "atcf_guidance_available": atcf_guidance_available,
                "verification_available": preferred_verify_available,
            },
            "source_trace": {
                "forecast_advisory": str(adv_path) if adv_path else None,
                "forecast_discussion": str(dis_path) if dis_path else None,
                "public_advisory": str(pub_path) if pub_path else None,
                "wind_probabilities": str(wnd_path) if wnd_path else None,
                "cds_real": str(cds_file) if cds_file else None,
                "hres_track": str(hres_track_file) if hres_track_file else None,
                "hres_system": str(hres_system_file) if hres_system_file else None,
            },
        }

        return sample

    @staticmethod
    def _record_skip(stats: Dict[str, Any], reason: str) -> None:
        stats["skipped"] += 1
        stats["skip_reasons"][reason] = stats["skip_reasons"].get(reason, 0) + 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Batch dataset builder")
    parser.add_argument("--base-dir", type=str, default=".", help="Project root")
    parser.add_argument("--output-dir", type=str, default="data/training", help="Output directory")
    parser.add_argument("--manifest", type=str, default="data/interim/ascat/ascat_request_manifest_full.csv")
    parser.add_argument("--split-manifest", type=str, default="data/interim/leakage_prevention/split_manifest_v1.json")
    parser.add_argument("--year-start", type=int, default=2016)
    parser.add_argument("--year-end", type=int, default=2025)
    parser.add_argument("--max-samples", type=int, default=None, help="Limit samples (for testing)")
    parser.add_argument(
        "--enable-raw-audit",
        action="store_true",
        help="Run slower raw canonical-sample leakage audit during build",
    )

    args = parser.parse_args()

    global BASE
    BASE = Path(args.base_dir)

    output_dir = BASE / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    builder = BatchBuilder(
        manifest_csv=BASE / args.manifest,
        split_manifest_path=BASE / args.split_manifest,
        output_dir=output_dir,
        max_samples=args.max_samples,
        year_start=args.year_start,
        year_end=args.year_end,
        enable_raw_audit=args.enable_raw_audit,
    )

    print("=== Batch Dataset Build ===")
    print(f"Manifest: {args.manifest}")
    print(f"Output: {args.output_dir}")
    print()

    stats = builder.build_all()

    print()
    print("=== Build Summary ===")
    print(f"Total rows: {stats['total_manifest_rows']}")
    print(f"Built: {stats['built']}")
    print(f"Skipped: {stats['skipped']}")
    print(f"Split counts: {stats['split_counts']}")
    print(f"Raw sample audit: {stats['raw_sample_audit_summary']}")
    print(f"Stale raw removed: {stats['stale_raw_removed']}")
    if stats["skip_reasons"]:
        print("Skip reasons:")
        for reason, count in sorted(stats["skip_reasons"].items(), key=lambda x: -x[1]):
            print(f"  {reason}: {count}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
