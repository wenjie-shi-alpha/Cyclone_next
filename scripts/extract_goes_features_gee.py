#!/usr/bin/env python3
"""Extract GOES structured features with server-side aggregation on GEE.

Design goals:
1. Keep heavy raster compute on Google Earth Engine.
2. Transfer only compact per-request feature rows (CSV).
3. Support chunked execution for stable large-batch runs on CDS JupyterLab.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import time
import urllib.request
from urllib.error import HTTPError
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


DEFAULT_DATASET_IDS = [
    "NOAA/GOES/16/MCMIPC",
    "NOAA/GOES/17/MCMIPC",
    "NOAA/GOES/18/MCMIPC",
    "NOAA/GOES/19/MCMIPC",
]

OUTPUT_FIELDS = [
    "request_id",
    "storm_id",
    "storm_id_match_status",
    "atcf_storm_id",
    "basin",
    "storm_name",
    "advisory_no",
    "issue_time_utc",
    "lat",
    "lon",
    "source_file",
    "goes_status",
    "missing_reason",
    "obs_time_utc",
    "obs_offset_minutes",
    "obs_offset_abs_minutes",
    "goes_source_collection",
    "goes_platform",
    "goes_band",
    "scale_m",
    "inner_radius_km",
    "outer_radius_km",
    "cold_threshold_k",
    "c13_min_k",
    "c13_p10_k",
    "c13_mean_k",
    "c13_std_k",
    "c13_ring_mean_k",
    "cold_area_inner_km2",
    "cold_fraction_inner",
    "cold_area_ring_km2",
    "cold_fraction_ring",
    "eye_ring_temp_contrast_k",
    "qc_has_image",
    "qc_time_within_window",
]


@dataclass
class RequestRow:
    request_id: str
    issue_time_utc: str
    issue_ms: int
    lat: float
    lon: float
    raw: Dict[str, str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract GOES compact feature table from request manifest via GEE."
    )
    parser.add_argument(
        "--manifest-csv",
        type=Path,
        default=Path("data/interim/goes/goes_request_manifest.csv"),
        help="Input request manifest csv.",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path("data/interim/goes/goes_observation_features.csv"),
        help="Output feature csv.",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=Path("data/interim/goes/goes_observation_features_summary.json"),
        help="Output summary json.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=200,
        help="Rows per GEE batch call.",
    )
    parser.add_argument(
        "--window-before-min",
        type=int,
        default=90,
        help="Look-back window before issue time.",
    )
    parser.add_argument(
        "--window-after-min",
        type=int,
        default=30,
        help="Look-forward window after issue time.",
    )
    parser.add_argument(
        "--inner-radius-km",
        type=float,
        default=200.0,
        help="Inner disk radius for core stats.",
    )
    parser.add_argument(
        "--outer-radius-km",
        type=float,
        default=500.0,
        help="Outer disk radius for ring stats.",
    )
    parser.add_argument(
        "--scale-m",
        type=float,
        default=4000.0,
        help="ReduceRegion pixel scale in meters.",
    )
    parser.add_argument(
        "--cold-threshold-k",
        type=float,
        default=235.0,
        help="Cold cloud threshold (K) on CMI_C13.",
    )
    parser.add_argument(
        "--c13-band",
        type=str,
        default="CMI_C13",
        help="GOES IR band used for structured features.",
    )
    parser.add_argument(
        "--dataset-id",
        action="append",
        dest="dataset_ids",
        default=[],
        help="Repeatable GEE dataset id. Defaults to GOES16/17/18/19 MCMIPC.",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="",
        help="Optional GEE project id.",
    )
    parser.add_argument(
        "--service-account-key-json",
        type=Path,
        default=None,
        help="Optional service account key file for headless auth.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="If >0, process only first N manifest rows.",
    )
    parser.add_argument(
        "--only-with-storm-id",
        action="store_true",
        help="Keep only rows with non-empty storm_id.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Retries for batch download url calls.",
    )
    parser.add_argument(
        "--sleep-sec",
        type=float,
        default=0.5,
        help="Sleep between batches to avoid throttling.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print resolved config and row counts.",
    )
    return parser.parse_args()


def parse_iso_utc(value: str) -> datetime:
    s = (value or "").strip()
    fmts = [
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S.%f",
    ]
    for fmt in fmts:
        try:
            dt = datetime.strptime(s, fmt)
            return dt.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    raise ValueError(f"unsupported datetime format: {value}")


def chunked(seq: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def parse_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        txt = str(value).strip()
        if not txt:
            return None
        return float(txt)
    except Exception:
        return None


def sanitize_goes_numeric_value(field: str, raw_value: Any) -> Optional[float]:
    value = parse_float(raw_value)
    if value is None or not math.isfinite(value):
        return None
    if value <= -9000:
        return None

    if field in {"c13_min_k", "c13_p10_k", "c13_mean_k", "c13_ring_mean_k"}:
        if value < 120 or value > 380:
            return None
        return value

    if field == "c13_std_k":
        if value < 0 or value > 150:
            return None
        return value

    if field == "eye_ring_temp_contrast_k":
        if value < -200 or value > 200:
            return None
        return value

    if field in {"cold_fraction_inner", "cold_fraction_ring"}:
        if value < 0 or value > 1:
            return None
        return value

    if field in {"cold_area_inner_km2", "cold_area_ring_km2"}:
        if value < 0:
            return None
        return value

    return value


def has_scaled_temperature_signature(row: Dict[str, Any]) -> bool:
    temp_fields = ("c13_min_k", "c13_p10_k", "c13_mean_k", "c13_ring_mean_k")
    values: List[float] = []
    for field in temp_fields:
        value = parse_float(row.get(field))
        if value is None or not math.isfinite(value) or value <= -9000:
            continue
        values.append(value)
    if len(values) < 2:
        return False
    scaled_like = sum(1 for v in values if 400 < abs(v) < 10000)
    return scaled_like >= 2


def sanitize_goes_output_row(row: Dict[str, Any]) -> Dict[str, Any]:
    status = (str(row.get("goes_status") or "")).strip()
    if status != "available":
        return row

    metric_fields = [
        "c13_min_k",
        "c13_p10_k",
        "c13_mean_k",
        "c13_std_k",
        "c13_ring_mean_k",
        "cold_area_inner_km2",
        "cold_fraction_inner",
        "cold_area_ring_km2",
        "cold_fraction_ring",
        "eye_ring_temp_contrast_k",
    ]
    # Guardrail: extraction should already apply C13 scale (DN*0.1). If raw values
    # still look like Kelvin*10, mark the row missing instead of silently fixing it.
    if has_scaled_temperature_signature(row):
        for field in metric_fields:
            row[field] = ""
        row["goes_status"] = "missing_real_data"
        prev_reason = (str(row.get("missing_reason") or "")).strip()
        unit_reason = "invalid_goes_units_c13_scaled_x10_signature"
        row["missing_reason"] = f"{prev_reason};{unit_reason}" if prev_reason else unit_reason
        return row

    for field in metric_fields:
        cleaned = sanitize_goes_numeric_value(field, row.get(field))
        row[field] = "" if cleaned is None else round(cleaned, 6)

    primary_temp_count = sum(
        1 for field in ("c13_min_k", "c13_p10_k", "c13_mean_k") if parse_float(row.get(field)) is not None
    )
    if primary_temp_count == 0:
        row["goes_status"] = "missing_real_data"
        prev_reason = (str(row.get("missing_reason") or "")).strip()
        row["missing_reason"] = prev_reason or "invalid_goes_temperature_metrics_after_qc"
    return row


def load_manifest(args: argparse.Namespace) -> List[RequestRow]:
    if not args.manifest_csv.exists():
        raise FileNotFoundError(f"manifest not found: {args.manifest_csv}")

    out: List[RequestRow] = []
    with args.manifest_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            issue_time = (row.get("issue_time_utc") or "").strip()
            lat = parse_float(row.get("lat"))
            lon = parse_float(row.get("lon"))
            if not issue_time or lat is None or lon is None:
                continue
            if args.only_with_storm_id and not (row.get("storm_id") or "").strip():
                continue
            try:
                dt = parse_iso_utc(issue_time)
            except ValueError:
                continue
            request_id = (row.get("request_id") or "").strip()
            if not request_id:
                request_id = f"REQ_{dt.strftime('%Y%m%dT%H%M%S')}_{len(out):06d}"
            out.append(
                RequestRow(
                    request_id=request_id,
                    issue_time_utc=dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    issue_ms=int(dt.timestamp() * 1000),
                    lat=float(lat),
                    lon=float(lon),
                    raw={k: (v or "").strip() for k, v in row.items()},
                )
            )

    out.sort(key=lambda x: (x.issue_time_utc, x.request_id))
    if args.max_rows > 0:
        out = out[: args.max_rows]
    return out


def init_ee(args: argparse.Namespace):
    import ee  # type: ignore

    if args.service_account_key_json:
        key_path = args.service_account_key_json
        info = json.loads(key_path.read_text(encoding="utf-8"))
        client_email = info.get("client_email")
        if not client_email:
            raise RuntimeError(f"service account key missing client_email: {key_path}")
        creds = ee.ServiceAccountCredentials(client_email, str(key_path))
        if args.project:
            ee.Initialize(credentials=creds, project=args.project)
        else:
            ee.Initialize(credentials=creds)
    else:
        if args.project:
            ee.Initialize(project=args.project)
        else:
            ee.Initialize()
    return ee


def build_merged_collection(ee, dataset_ids: List[str]):
    def with_source(ds: str):
        return ee.ImageCollection(ds).map(lambda img: img.set("source_collection", ds))

    merged = None
    for ds in dataset_ids:
        coll = with_source(ds)
        merged = coll if merged is None else merged.merge(coll)
    if merged is None:
        raise RuntimeError("no dataset id available for merged collection")
    return merged


def validate_dataset_ids(ee, dataset_ids: List[str]) -> List[str]:
    valid: List[str] = []
    for ds in dataset_ids:
        try:
            _ = ee.ImageCollection(ds).limit(1).size().getInfo()
            valid.append(ds)
        except Exception:
            continue
    return valid


def build_batch_feature_collection(ee, rows: Sequence[RequestRow]):
    features = []
    for r in rows:
        props = dict(r.raw)
        props["request_id"] = r.request_id
        props["issue_time_utc"] = r.issue_time_utc
        props["issue_ms"] = r.issue_ms
        props["lat"] = r.lat
        props["lon"] = r.lon
        features.append(
            ee.Feature(
                ee.Geometry.Point([r.lon, r.lat]),
                props,
            )
        )
    return ee.FeatureCollection(features)


def build_mapper(
    ee,
    merged_collection,
    window_before_min: int,
    window_after_min: int,
    inner_radius_km: float,
    outer_radius_km: float,
    scale_m: float,
    cold_threshold_k: float,
    c13_band: str,
):
    inner_m = ee.Number(inner_radius_km).multiply(1000.0)
    outer_m = ee.Number(outer_radius_km).multiply(1000.0)
    before = ee.Number(window_before_min)
    after = ee.Number(window_after_min)
    max_abs_window = ee.Number(max(window_before_min, window_after_min))
    scale = ee.Number(scale_m)
    cold_k = ee.Number(cold_threshold_k)
    reducer = (
        ee.Reducer.min()
        .combine(ee.Reducer.mean(), sharedInputs=True)
        .combine(ee.Reducer.stdDev(), sharedInputs=True)
        .combine(ee.Reducer.percentile([10]), sharedInputs=True)
    )

    def safe_fraction(numer, denom):
        d = ee.Number(denom)
        return ee.Number(ee.Algorithms.If(d.gt(0), ee.Number(numer).divide(d), -1))

    def add_time_diff(issue_date):
        def _inner(img):
            diff = ee.Number(img.date().difference(issue_date, "minute")).abs()
            return img.set("time_diff_abs_min", diff)

        return _inner

    def compute_available(feature, issue_date, inner_geom, outer_geom, nearest):
        def dict_number(dct, key: str, default_value: float):
            raw = dct.get(key)
            return ee.Number(
                ee.Algorithms.If(
                    ee.Algorithms.IsEqual(raw, None),
                    default_value,
                    raw,
                )
            )

        # CMI_C13 in NOAA/GOES/*/MCMIPC is stored in 0.1 K units (raw DN × 0.1 = actual K).
        # Without this scale the reducers return ×10 temperature values (e.g. 3109 instead of
        # 310.9 K) AND the cold-cloud mask `c13.lt(cold_threshold_k)` compares raw ×10 values
        # against the actual-K threshold, making every pixel appear "warm" (cold_area = 0).
        c13 = ee.Image(nearest).select(c13_band).multiply(0.1)

        inner_stats = ee.Dictionary(
            c13.reduceRegion(
                reducer=reducer,
                geometry=inner_geom,
                scale=scale,
                crs="EPSG:4326",
                bestEffort=True,
                maxPixels=1e8,
            )
        )
        outer_stats = ee.Dictionary(
            c13.reduceRegion(
                reducer=reducer,
                geometry=outer_geom,
                scale=scale,
                crs="EPSG:4326",
                bestEffort=True,
                maxPixels=1e8,
            )
        )

        area_band = ee.Image.pixelArea().divide(1e6).rename("px_area_km2")
        inner_area_total = ee.Number(
            area_band.reduceRegion(
                reducer=ee.Reducer.sum(),
                geometry=inner_geom,
                scale=scale,
                crs="EPSG:4326",
                bestEffort=True,
                maxPixels=1e8,
            ).get("px_area_km2", 0)
        )
        ring_area_total = ee.Number(
            area_band.reduceRegion(
                reducer=ee.Reducer.sum(),
                geometry=outer_geom,
                scale=scale,
                crs="EPSG:4326",
                bestEffort=True,
                maxPixels=1e8,
            ).get("px_area_km2", 0)
        ).subtract(inner_area_total).max(0)

        inner_cold_area = ee.Number(
            area_band.updateMask(c13.lt(cold_k))
            .reduceRegion(
                reducer=ee.Reducer.sum(),
                geometry=inner_geom,
                scale=scale,
                crs="EPSG:4326",
                bestEffort=True,
                maxPixels=1e8,
            )
            .get("px_area_km2", 0)
        )
        ring_cold_area = ee.Number(
            area_band.updateMask(c13.lt(cold_k))
            .reduceRegion(
                reducer=ee.Reducer.sum(),
                geometry=outer_geom,
                scale=scale,
                crs="EPSG:4326",
                bestEffort=True,
                maxPixels=1e8,
            )
            .get("px_area_km2", 0)
        ).subtract(inner_cold_area).max(0)

        obs_dt = ee.Date(nearest.get("system:time_start"))
        obs_offset = ee.Number(obs_dt.difference(issue_date, "minute"))
        obs_offset_abs = obs_offset.abs()

        band_min = dict_number(inner_stats, f"{c13_band}_min", -9999.0)
        band_p10 = dict_number(inner_stats, f"{c13_band}_p10", -9999.0)
        band_mean = dict_number(inner_stats, f"{c13_band}_mean", -9999.0)
        band_std = dict_number(inner_stats, f"{c13_band}_stdDev", -9999.0)
        outer_mean = dict_number(outer_stats, f"{c13_band}_mean", -9999.0)
        ring_mean = ee.Number(
            ee.Algorithms.If(
                ring_area_total.gt(0),
                outer_mean.multiply(ring_area_total.add(inner_area_total))
                .subtract(band_mean.multiply(inner_area_total))
                .divide(ring_area_total),
                outer_mean,
            )
        )
        temp_contrast = ee.Number(ring_mean.subtract(band_mean))

        has_platform = ee.Image(nearest).propertyNames().contains("platform")
        platform = ee.String(ee.Algorithms.If(has_platform, nearest.get("platform"), ""))

        return ee.Dictionary(
            {
                "goes_status": "available",
                "missing_reason": "",
                "obs_time_utc": obs_dt.format("YYYY-MM-dd'T'HH:mm:ss'Z'"),
                "obs_offset_minutes": obs_offset,
                "obs_offset_abs_minutes": obs_offset_abs,
                "goes_source_collection": nearest.get("source_collection"),
                "goes_platform": platform,
                "goes_band": c13_band,
                "scale_m": scale_m,
                "inner_radius_km": inner_radius_km,
                "outer_radius_km": outer_radius_km,
                "cold_threshold_k": cold_threshold_k,
                "c13_min_k": band_min,
                "c13_p10_k": band_p10,
                "c13_mean_k": band_mean,
                "c13_std_k": band_std,
                "c13_ring_mean_k": ring_mean,
                "cold_area_inner_km2": inner_cold_area,
                "cold_fraction_inner": safe_fraction(inner_cold_area, inner_area_total),
                "cold_area_ring_km2": ring_cold_area,
                "cold_fraction_ring": safe_fraction(ring_cold_area, ring_area_total),
                "eye_ring_temp_contrast_k": temp_contrast,
                "qc_has_image": 1,
                "qc_time_within_window": obs_offset_abs.lte(max_abs_window),
            }
        )

    def compute_missing():
        return ee.Dictionary(
            {
                "goes_status": "missing_real_data",
                "missing_reason": "no_goes_image_in_window",
                "obs_time_utc": "",
                "obs_offset_minutes": "",
                "obs_offset_abs_minutes": "",
                "goes_source_collection": "",
                "goes_platform": "",
                "goes_band": c13_band,
                "scale_m": scale_m,
                "inner_radius_km": inner_radius_km,
                "outer_radius_km": outer_radius_km,
                "cold_threshold_k": cold_threshold_k,
                "c13_min_k": "",
                "c13_p10_k": "",
                "c13_mean_k": "",
                "c13_std_k": "",
                "c13_ring_mean_k": "",
                "cold_area_inner_km2": "",
                "cold_fraction_inner": "",
                "cold_area_ring_km2": "",
                "cold_fraction_ring": "",
                "eye_ring_temp_contrast_k": "",
                "qc_has_image": 0,
                "qc_time_within_window": 0,
            }
        )

    def mapper(feature):
        issue_date = ee.Date(ee.Number(feature.get("issue_ms")))
        lat = ee.Number(feature.get("lat"))
        lon = ee.Number(feature.get("lon"))
        point = ee.Geometry.Point([lon, lat])
        inner_geom = point.buffer(inner_m)
        outer_geom = point.buffer(outer_m)

        start = issue_date.advance(before.multiply(-1), "minute")
        end = issue_date.advance(after, "minute")
        coll = merged_collection.filterDate(start, end).filterBounds(outer_geom)
        coll_with_dt = coll.map(add_time_diff(issue_date))
        has_img = coll_with_dt.size().gt(0)
        nearest = ee.Image(coll_with_dt.sort("time_diff_abs_min").first())

        computed = ee.Dictionary(
            ee.Algorithms.If(
                has_img,
                compute_available(feature, issue_date, inner_geom, outer_geom, nearest),
                compute_missing(),
            )
        )
        return feature.set(computed)

    return mapper


def download_batch_csv(url: str, retries: int) -> str:
    last_exc: Optional[Exception] = None
    for _ in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=120) as resp:
                return resp.read().decode("utf-8")
        except HTTPError as exc:
            # Do not retry deterministic client-side errors except 429.
            if 400 <= exc.code < 500 and exc.code != 429:
                detail = ""
                try:
                    detail = exc.read().decode("utf-8", errors="ignore")
                except Exception:
                    detail = ""
                detail = (detail or "").strip().replace("\n", " ")
                if len(detail) > 600:
                    detail = detail[:600] + "..."
                raise RuntimeError(f"HTTP {exc.code} client error while downloading batch CSV: {detail}")
            last_exc = exc
            time.sleep(1.0)
        except Exception as exc:
            last_exc = exc
            time.sleep(1.0)
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("download failed without exception")


def run_batches(args: argparse.Namespace, ee, rows: List[RequestRow], dataset_ids: List[str]) -> Dict[str, Any]:
    merged = build_merged_collection(ee, dataset_ids)
    mapper = build_mapper(
        ee=ee,
        merged_collection=merged,
        window_before_min=args.window_before_min,
        window_after_min=args.window_after_min,
        inner_radius_km=args.inner_radius_km,
        outer_radius_km=args.outer_radius_km,
        scale_m=args.scale_m,
        cold_threshold_k=args.cold_threshold_k,
        c13_band=args.c13_band,
    )

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    summary: Dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "manifest_csv": str(args.manifest_csv),
        "out_csv": str(args.out_csv),
        "summary_json": str(args.summary_json),
        "dataset_ids_requested": args.dataset_ids if args.dataset_ids else DEFAULT_DATASET_IDS,
        "dataset_ids_used": dataset_ids,
        "requests_total": len(rows),
        "rows_written": 0,
        "available_rows": 0,
        "missing_rows": 0,
        "mean_abs_offset_min_available": None,
        "p50_abs_offset_min_available": None,
        "p90_abs_offset_min_available": None,
        "coverage_by_year": {},
    }
    offsets: List[float] = []
    by_year: Dict[str, Dict[str, int]] = {}

    def update_summary_with_row(out_row: Dict[str, Any]) -> None:
        summary["rows_written"] += 1
        year = (str(out_row.get("issue_time_utc") or ""))[:4]
        if year:
            year_bucket = by_year.setdefault(year, {"total": 0, "available": 0, "missing": 0})
            year_bucket["total"] += 1

        status = (str(out_row.get("goes_status") or "")).strip()
        if status == "available":
            summary["available_rows"] += 1
            if year:
                by_year[year]["available"] += 1
            off = parse_float(out_row.get("obs_offset_abs_minutes"))
            if off is not None:
                offsets.append(off)
        else:
            summary["missing_rows"] += 1
            if year:
                by_year[year]["missing"] += 1

    def failed_row(r: RequestRow, reason: str) -> Dict[str, Any]:
        out = {k: "" for k in OUTPUT_FIELDS}
        # Preserve keys used by downstream matching and lineage.
        for k, v in r.raw.items():
            if k in out:
                out[k] = v
        out["request_id"] = r.request_id
        out["issue_time_utc"] = r.issue_time_utc
        out["lat"] = r.lat
        out["lon"] = r.lon
        out["goes_status"] = "missing_real_data"
        out["missing_reason"] = reason
        out["goes_band"] = args.c13_band
        out["scale_m"] = args.scale_m
        out["inner_radius_km"] = args.inner_radius_km
        out["outer_radius_km"] = args.outer_radius_km
        out["cold_threshold_k"] = args.cold_threshold_k
        out["qc_has_image"] = 0
        out["qc_time_within_window"] = 0
        return out

    with args.out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()

        def execute_chunk(chunk_rows: Sequence[RequestRow], level: int = 0) -> None:
            try:
                fc = build_batch_feature_collection(ee, chunk_rows)
                mapped = fc.map(mapper)
                # earthengine-api 1.7.x expects filetype/selectors kwargs instead of an options dict.
                url = mapped.getDownloadURL(filetype="csv", selectors=OUTPUT_FIELDS)
                csv_text = download_batch_csv(url, retries=max(1, args.max_retries))
                reader = csv.DictReader(io.StringIO(csv_text))
                out_count = 0
                for row in reader:
                    out_row = sanitize_goes_output_row({k: row.get(k, "") for k in OUTPUT_FIELDS})
                    writer.writerow(out_row)
                    update_summary_with_row(out_row)
                    out_count += 1
                if out_count != len(chunk_rows):
                    print(
                        f"[WARN] chunk output count mismatch: expected={len(chunk_rows)} got={out_count}; "
                        "writing fallback rows for missing records"
                    )
                    # Fallback rows ensure stable row count for downstream joins.
                    got_ids = set()
                    reader2 = csv.DictReader(io.StringIO(csv_text))
                    for row in reader2:
                        rid = (row.get("request_id") or "").strip()
                        if rid:
                            got_ids.add(rid)
                    for r in chunk_rows:
                        if r.request_id in got_ids:
                            continue
                        fb = failed_row(r, "missing_row_from_gee_response")
                        writer.writerow(fb)
                        update_summary_with_row(fb)
            except Exception as exc:
                msg = str(exc)
                code = exc.code if isinstance(exc, HTTPError) else None

                if code in {429, 500, 503} and level <= 4:
                    wait_s = min(60.0, (2 ** level) * 2.0)
                    print(f"[WARN] transient HTTP {code}, retry after {wait_s:.1f}s for chunk_size={len(chunk_rows)}")
                    time.sleep(wait_s)
                    execute_chunk(chunk_rows, level=level + 1)
                    return

                if len(chunk_rows) <= 25 and len(chunk_rows) > 1:
                    print(
                        f"[WARN] chunk_size={len(chunk_rows)} failed ({type(exc).__name__}); "
                        "switch to per-request fallback mode"
                    )
                    for r in chunk_rows:
                        execute_chunk([r], level=level + 1)
                    return

                if len(chunk_rows) > 1:
                    mid = len(chunk_rows) // 2
                    left = chunk_rows[:mid]
                    right = chunk_rows[mid:]
                    print(
                        f"[WARN] chunk_size={len(chunk_rows)} failed ({type(exc).__name__}: {msg[:180]}), "
                        f"split -> {len(left)} + {len(right)}"
                    )
                    execute_chunk(left, level=level + 1)
                    execute_chunk(right, level=level + 1)
                    return

                # Single-row fallback to prevent pipeline abort.
                r = chunk_rows[0]
                reason = f"gee_request_failed_{type(exc).__name__}"
                fb = failed_row(r, reason)
                writer.writerow(fb)
                update_summary_with_row(fb)
                print(
                    f"[WARN] single request fallback written: request_id={r.request_id} "
                    f"reason={reason} detail={msg[:220]}"
                )

        for idx, chunk in enumerate(chunked(rows, args.batch_size), start=1):
            execute_chunk(chunk_rows=chunk, level=0)
            print(f"batch {idx}: {len(chunk)} requests processed")
            time.sleep(max(0.0, args.sleep_sec))

    if offsets:
        offsets_sorted = sorted(offsets)
        n = len(offsets_sorted)

        def quantile(p: float) -> float:
            if n == 1:
                return offsets_sorted[0]
            pos = p * (n - 1)
            lo = int(pos)
            hi = min(lo + 1, n - 1)
            w = pos - lo
            return offsets_sorted[lo] * (1 - w) + offsets_sorted[hi] * w

        summary["mean_abs_offset_min_available"] = round(sum(offsets_sorted) / n, 3)
        summary["p50_abs_offset_min_available"] = round(quantile(0.5), 3)
        summary["p90_abs_offset_min_available"] = round(quantile(0.9), 3)

    coverage_by_year: Dict[str, Dict[str, Any]] = {}
    for y in sorted(by_year.keys()):
        total = by_year[y]["total"]
        available = by_year[y]["available"]
        missing = by_year[y]["missing"]
        coverage = (available / total) if total > 0 else 0.0
        coverage_by_year[y] = {
            "total": total,
            "available": available,
            "missing": missing,
            "coverage_rate": round(coverage, 6),
        }
    summary["coverage_by_year"] = coverage_by_year

    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    args = parse_args()
    dataset_ids = args.dataset_ids if args.dataset_ids else list(DEFAULT_DATASET_IDS)

    rows = load_manifest(args)
    if not rows:
        raise RuntimeError("no valid request rows found in manifest")

    print("manifest:", args.manifest_csv)
    print("rows_to_process:", len(rows))
    print("batch_size:", args.batch_size)
    print("dataset_ids_requested:", dataset_ids)

    if args.dry_run:
        return 0

    ee = init_ee(args)
    valid_ids = validate_dataset_ids(ee, dataset_ids)
    if not valid_ids:
        raise RuntimeError("none of the dataset ids are accessible in current GEE account")

    print("dataset_ids_used:", valid_ids)
    summary = run_batches(args=args, ee=ee, rows=rows, dataset_ids=valid_ids)

    print(args.out_csv)
    print(args.summary_json)
    print("rows_written:", summary["rows_written"])
    print("available_rows:", summary["available_rows"])
    print("missing_rows:", summary["missing_rows"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
