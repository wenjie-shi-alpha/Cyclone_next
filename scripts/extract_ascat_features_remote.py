#!/usr/bin/env python3
"""Extract ASCAT structured wind features with remote subsetting on Copernicus Marine.

Design goals:
1. Keep raw data request and heavy handling on remote compute nodes.
2. Save only compact per-request feature rows to local CSV.
3. Keep failure-tolerant behavior (one failed request should not stop the run).
"""

import argparse
import csv
import inspect
import json
import math
import os
import signal
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


DEFAULT_DATASET_IDS = [
    "cmems_obs-wind_glo_phy-ascat-metop_a-l3-pt1h",
    "cmems_obs-wind_glo_phy-ascat-metop_b-l3-pt1h",
    "cmems_obs-wind_glo_phy-ascat-metop_c-l3-pt1h",
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
    "ascat_status",
    "missing_reason",
    "obs_time_utc",
    "obs_offset_minutes",
    "obs_offset_abs_minutes",
    "ascat_dataset_id",
    "ascat_platform",
    "ascat_variable",
    "ascat_units",
    "inner_radius_km",
    "outer_radius_km",
    "wind_mean_inner_kt",
    "wind_p90_inner_kt",
    "wind_max_inner_kt",
    "wind_mean_ring_kt",
    "wind_p90_ring_kt",
    "wind_max_ring_kt",
    "wind_area_ge34kt_inner_km2",
    "wind_area_ge50kt_inner_km2",
    "valid_cell_count",
    "qc_has_data",
    "qc_time_within_window",
]


class RequestRow:
    def __init__(
        self,
        request_id: str,
        issue_time_utc: str,
        issue_dt: datetime,
        lat: float,
        lon: float,
        raw: Dict[str, str],
    ) -> None:
        self.request_id = request_id
        self.issue_time_utc = issue_time_utc
        self.issue_dt = issue_dt
        self.lat = lat
        self.lon = lon
        self.raw = raw


def log_progress(message: str) -> None:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"[{stamp}] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract ASCAT compact feature table from request manifest via Copernicus Marine."
    )
    parser.add_argument(
        "--manifest-csv",
        type=Path,
        default=Path("data/interim/ascat/ascat_request_manifest.csv"),
        help="Input request manifest csv.",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path("data/interim/ascat/ascat_observation_features.csv"),
        help="Output feature csv.",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=Path("data/interim/ascat/ascat_observation_features_summary.json"),
        help="Output summary json.",
    )
    parser.add_argument(
        "--dataset-id",
        action="append",
        dest="dataset_ids",
        default=[],
        help="Repeatable Copernicus Marine dataset id.",
    )
    parser.add_argument(
        "--variable",
        action="append",
        dest="variables",
        default=[],
        help="Optional variable filters passed to subset call.",
    )
    parser.add_argument(
        "--window-before-min",
        type=int,
        default=180,
        help="Look-back window before issue time.",
    )
    parser.add_argument(
        "--window-after-min",
        type=int,
        default=180,
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
        "--bbox-margin-deg",
        type=float,
        default=0.5,
        help="Extra bbox degree margin around outer-radius projection.",
    )
    parser.add_argument(
        "--assumed-cell-km",
        type=float,
        default=25.0,
        help="Fallback grid-cell edge size for area estimation when resolution is unknown.",
    )
    parser.add_argument(
        "--tmp-dir",
        type=Path,
        default=Path("/tmp/ascat_subset_tmp"),
        help="Temporary folder for subset netcdf files.",
    )
    parser.add_argument(
        "--keep-temp-files",
        action="store_true",
        help="Keep temporary subset files.",
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
        help="Retries for subset calls per dataset request.",
    )
    parser.add_argument(
        "--sleep-sec",
        type=float,
        default=0.2,
        help="Sleep between requests to reduce throttling.",
    )
    parser.add_argument(
        "--subset-timeout-sec",
        type=float,
        default=300.0,
        help="Hard timeout for each Copernicus Marine subset call; <=0 disables the timeout.",
    )
    parser.add_argument(
        "--username",
        type=str,
        default="",
        help="Copernicus Marine username (optional, fallback to env/config).",
    )
    parser.add_argument(
        "--password",
        type=str,
        default="",
        help="Copernicus Marine password (optional, fallback to env/config).",
    )
    parser.add_argument(
        "--credentials-file",
        type=str,
        default="",
        help="Optional credentials file path passed to subset call if supported.",
    )
    parser.add_argument(
        "--service",
        type=str,
        default="",
        help="Optional service endpoint passed to subset call if supported.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print resolved config and row counts.",
    )
    return parser.parse_args()


def first_nonempty(*values: str) -> str:
    for value in values:
        txt = (value or "").strip()
        if txt:
            return txt
    return ""


def env_first(*names: str) -> str:
    for name in names:
        value = os.environ.get(name, "")
        if value.strip():
            return value.strip()
    return ""


def env_first_float(*names: str) -> Optional[float]:
    raw = env_first(*names)
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"invalid float environment value: {raw}") from exc


def env_first_int(*names: str) -> Optional[int]:
    raw = env_first(*names)
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"invalid integer environment value: {raw}") from exc


def resolve_runtime_config(args: argparse.Namespace) -> argparse.Namespace:
    args.username = first_nonempty(
        args.username,
        env_first(
            "ASCAT_USERNAME",
            "CMEMS_USERNAME",
            "COPERNICUSMARINE_USERNAME",
            "COPERNICUSMARINE_SERVICE_USERNAME",
        ),
    )
    args.password = first_nonempty(
        args.password,
        env_first(
            "ASCAT_PASSWORD",
            "CMEMS_PASSWORD",
            "COPERNICUSMARINE_PASSWORD",
            "COPERNICUSMARINE_SERVICE_PASSWORD",
        ),
    )
    args.credentials_file = first_nonempty(
        args.credentials_file,
        env_first(
            "ASCAT_CREDENTIALS_FILE",
            "CMEMS_CREDENTIALS_FILE",
            "COPERNICUSMARINE_CREDENTIALS_FILE",
        ),
    )
    args.service = first_nonempty(
        args.service,
        env_first(
            "ASCAT_SERVICE",
            "CMEMS_SERVICE",
            "COPERNICUSMARINE_SERVICE",
        ),
    )
    pause_sec = env_first_float("ASCAT_REQUEST_PAUSE_SEC", "ASCAT_SLEEP_SEC")
    if pause_sec is not None:
        args.sleep_sec = pause_sec
    subset_timeout_sec = env_first_int("ASCAT_SUBSET_TIMEOUT_SEC")
    if subset_timeout_sec is not None:
        args.subset_timeout_sec = float(subset_timeout_sec)
    tmp_dir_env = env_first("ASCAT_TMP_DIR")
    if tmp_dir_env and str(args.tmp_dir) == "/tmp/ascat_subset_tmp":
        args.tmp_dir = Path(tmp_dir_env)
    return args


def parse_iso_utc(value: str) -> datetime:
    txt = (value or "").strip()
    fmts = [
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S.%f",
    ]
    for fmt in fmts:
        try:
            return datetime.strptime(txt, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    raise ValueError(f"unsupported datetime format: {value}")


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


def safe_request_id(request_id: str) -> str:
    out = []
    for ch in request_id:
        if ch.isalnum() or ch in {"-", "_"}:
            out.append(ch)
        else:
            out.append("_")
    return "".join(out)[:128]


def parse_lon_minus180_180(lon: float) -> float:
    x = lon
    while x >= 180:
        x -= 360
    while x < -180:
        x += 360
    return x


def build_bbox(lat: float, lon: float, outer_radius_km: float, margin_deg: float) -> Tuple[float, float, float, float]:
    lat_delta = (outer_radius_km / 111.0) + margin_deg
    cos_lat = max(0.15, abs(math.cos(math.radians(lat))))
    lon_delta = (outer_radius_km / (111.0 * cos_lat)) + margin_deg
    lon_c = parse_lon_minus180_180(lon)
    min_lon = lon_c - lon_delta
    max_lon = lon_c + lon_delta

    if min_lon < -180:
        min_lon += 360
        max_lon += 360
    if max_lon > 360:
        max_lon = 360.0
    if min_lon < -180:
        min_lon = -180.0

    min_lat = max(-89.9, lat - lat_delta)
    max_lat = min(89.9, lat + lat_delta)
    return min_lon, max_lon, min_lat, max_lat


def load_manifest(args: argparse.Namespace) -> List[RequestRow]:
    if not args.manifest_csv.exists():
        raise FileNotFoundError(f"manifest not found: {args.manifest_csv}")

    rows: List[RequestRow] = []
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
                issue_dt = parse_iso_utc(issue_time)
            except ValueError:
                continue
            request_id = (row.get("request_id") or "").strip()
            if not request_id:
                request_id = f"REQ_{issue_dt.strftime('%Y%m%dT%H%M%S')}_{len(rows):06d}"
            rows.append(
                RequestRow(
                    request_id=request_id,
                    issue_time_utc=issue_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    issue_dt=issue_dt,
                    lat=float(lat),
                    lon=float(lon),
                    raw={k: (v or "").strip() for k, v in row.items()},
                )
            )
    rows.sort(key=lambda x: (x.issue_time_utc, x.request_id))
    if args.max_rows > 0:
        rows = rows[: args.max_rows]
    return rows


def init_optional_deps() -> Tuple[Any, Any, Any]:
    try:
        import numpy as np  # type: ignore
    except Exception as exc:  # pragma: no cover - import guard
        raise RuntimeError("numpy is required for ASCAT feature extraction") from exc
    try:
        import xarray as xr  # type: ignore
    except Exception as exc:  # pragma: no cover - import guard
        raise RuntimeError("xarray is required for ASCAT feature extraction") from exc
    try:
        import copernicusmarine  # type: ignore
    except Exception as exc:  # pragma: no cover - import guard
        raise RuntimeError(
            "copernicusmarine package is required on remote node. "
            "Install via: pip install copernicusmarine"
        ) from exc
    return np, xr, copernicusmarine


def run_with_timeout(timeout_sec: float, func: Any, *args: Any, **kwargs: Any) -> Any:
    if timeout_sec <= 0 or os.name == "nt" or not hasattr(signal, "setitimer"):
        return func(*args, **kwargs)

    def _timeout_handler(signum: int, frame: Any) -> None:
        raise TimeoutError(f"subset call exceeded timeout_sec={timeout_sec}")

    previous_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.setitimer(signal.ITIMER_REAL, timeout_sec)
    try:
        return func(*args, **kwargs)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)


def call_subset(
    copernicusmarine: Any,
    dataset_id: str,
    out_file: Path,
    min_lon: float,
    max_lon: float,
    min_lat: float,
    max_lat: float,
    start_dt: datetime,
    end_dt: datetime,
    args: argparse.Namespace,
) -> Path:
    subset_sig = inspect.signature(copernicusmarine.subset)
    accepted = set(subset_sig.parameters.keys())

    kwargs: Dict[str, Any] = {
        "dataset_id": dataset_id,
        "minimum_longitude": float(min_lon),
        "maximum_longitude": float(max_lon),
        "minimum_latitude": float(min_lat),
        "maximum_latitude": float(max_lat),
        "start_datetime": start_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "end_datetime": end_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "output_directory": str(out_file.parent),
        "output_filename": out_file.name,
        "overwrite": True,
        "disable_progress_bar": True,
    }
    if args.variables:
        kwargs["variables"] = args.variables
    if args.username:
        kwargs["username"] = args.username
    if args.password:
        kwargs["password"] = args.password
    if args.credentials_file:
        kwargs["credentials_file"] = args.credentials_file
    if args.service:
        kwargs["service"] = args.service

    filtered_kwargs = {k: v for k, v in kwargs.items() if k in accepted and v is not None and v != ""}
    result = copernicusmarine.subset(**filtered_kwargs)

    if out_file.exists() and out_file.stat().st_size > 0:
        return out_file

    # Fallback: try to resolve file path from response object.
    if isinstance(result, dict):
        for key in ("output_path", "file_path", "path"):
            if key in result and result[key]:
                cand = Path(str(result[key]))
                if cand.exists() and cand.stat().st_size > 0:
                    return cand
        if "files" in result and isinstance(result["files"], list):
            for fp in result["files"]:
                cand = Path(str(fp))
                if cand.exists() and cand.stat().st_size > 0:
                    return cand

    raise RuntimeError(f"subset call finished but output file is missing: {out_file}")


def to_datetime_list(np: Any, values: Any) -> List[datetime]:
    out: List[datetime] = []
    arr = np.asarray(values)
    for v in arr.reshape(-1):
        try:
            if isinstance(v, datetime):
                out.append(v.replace(tzinfo=timezone.utc) if v.tzinfo is None else v.astimezone(timezone.utc))
                continue
            if isinstance(v, str):
                out.append(parse_iso_utc(v))
                continue
            if np.issubdtype(type(v), np.datetime64):
                sec = int(v.astype("datetime64[s]").astype("int64"))
                out.append(datetime.fromtimestamp(sec, tz=timezone.utc))
                continue
        except Exception:
            continue
    return out


def choose_time_index(
    np: Any,
    times: Sequence[datetime],
    issue_dt: datetime,
    before_min: int,
    after_min: int,
) -> Optional[Tuple[int, datetime, float]]:
    if not times:
        return None
    best_idx: Optional[int] = None
    best_abs_min: Optional[float] = None
    best_offset_min: Optional[float] = None
    for i, t in enumerate(times):
        offset_min = (t - issue_dt).total_seconds() / 60.0
        if offset_min < -before_min or offset_min > after_min:
            continue
        abs_min = abs(offset_min)
        if best_abs_min is None or abs_min < best_abs_min:
            best_idx = i
            best_abs_min = abs_min
            best_offset_min = offset_min
    if best_idx is None or best_offset_min is None:
        return None
    return best_idx, times[best_idx], best_offset_min


def infer_units_to_kt_factor(units: str) -> float:
    u = (units or "").strip().lower().replace(" ", "")
    if not u:
        return 1.94384
    if "knot" in u or "kt" in u:
        return 1.0
    if "m/s" in u or "ms-1" in u or "m*s-1" in u:
        return 1.94384
    return 1.94384


def extract_wind_speed_da(np: Any, xr: Any, ds: Any) -> Tuple[Any, str, str]:
    candidates_exact = [
        "wind_speed",
        "wind_speed_10m",
        "windspeed",
        "windspeed_10m",
        "wind_speed_mean",
    ]
    lower_to_name = {name.lower(): name for name in ds.data_vars}
    for c in candidates_exact:
        name = lower_to_name.get(c.lower())
        if name:
            da = ds[name]
            units = str(da.attrs.get("units") or "")
            factor = infer_units_to_kt_factor(units)
            return da.astype("float64") * factor, name, "kt"

    for name in ds.data_vars:
        lname = name.lower()
        if "wind" in lname and "speed" in lname:
            da = ds[name]
            units = str(da.attrs.get("units") or "")
            factor = infer_units_to_kt_factor(units)
            return da.astype("float64") * factor, name, "kt"

    u_names = [n for n in ds.data_vars if n.lower() in {"u10", "uwnd", "u_wind", "eastward_wind"}]
    v_names = [n for n in ds.data_vars if n.lower() in {"v10", "vwnd", "v_wind", "northward_wind"}]
    if u_names and v_names:
        u = ds[u_names[0]].astype("float64")
        v = ds[v_names[0]].astype("float64")
        units = str(u.attrs.get("units") or v.attrs.get("units") or "")
        factor = infer_units_to_kt_factor(units)
        da = np.sqrt(u * u + v * v) * factor
        da.name = "wind_speed_from_uv"
        return da, "wind_speed_from_uv", "kt"

    raise RuntimeError("no wind-speed variable found in subset dataset")


def pick_coord_name(da: Any, candidates: Sequence[str]) -> Optional[str]:
    names = list(da.coords.keys())
    lower_map = {x.lower(): x for x in names}
    for cand in candidates:
        found = lower_map.get(cand.lower())
        if found:
            return found
    for name in names:
        lname = name.lower()
        for cand in candidates:
            if cand.lower() in lname:
                return name
    return None


def haversine_km(np: Any, lat1: Any, lon1: Any, lat2: float, lon2: float) -> Any:
    r = 6371.0
    lat1_r = np.radians(lat1)
    lon1_r = np.radians(lon1)
    lat2_r = math.radians(lat2)
    lon2_r = math.radians(lon2)
    dlat = lat1_r - lat2_r
    dlon = lon1_r - lon2_r
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_r) * math.cos(lat2_r) * np.sin(dlon / 2.0) ** 2
    return 2.0 * r * np.arcsin(np.sqrt(a))


def infer_cell_area_km2(np: Any, lat_flat: Any, lon_flat: Any, assumed_cell_km: float) -> Any:
    if lat_flat.size < 4 or lon_flat.size < 4:
        return np.full(lat_flat.shape, assumed_cell_km * assumed_cell_km, dtype="float64")

    lat_unique = np.unique(np.round(lat_flat, 5))
    lon_unique = np.unique(np.round(lon_flat, 5))
    if lat_unique.size < 2 or lon_unique.size < 2:
        return np.full(lat_flat.shape, assumed_cell_km * assumed_cell_km, dtype="float64")

    dlat = np.median(np.abs(np.diff(lat_unique)))
    dlon = np.median(np.abs(np.diff(lon_unique)))
    if not np.isfinite(dlat) or not np.isfinite(dlon) or dlat <= 0 or dlon <= 0:
        return np.full(lat_flat.shape, assumed_cell_km * assumed_cell_km, dtype="float64")

    dlat_rad = math.radians(float(dlat))
    dlon_rad = math.radians(float(dlon))
    r = 6371.0
    area = (r * r) * dlat_rad * dlon_rad * np.cos(np.radians(lat_flat))
    area = np.abs(area)
    floor = max(1.0, assumed_cell_km * assumed_cell_km * 0.2)
    area = np.where(np.isfinite(area), np.maximum(area, floor), assumed_cell_km * assumed_cell_km)
    return area


def stat_mean(np: Any, values: Any) -> Optional[float]:
    if values.size == 0:
        return None
    return float(np.nanmean(values))


def stat_max(np: Any, values: Any) -> Optional[float]:
    if values.size == 0:
        return None
    return float(np.nanmax(values))


def stat_p90(np: Any, values: Any) -> Optional[float]:
    if values.size == 0:
        return None
    return float(np.nanpercentile(values, 90))


def round_or_blank(x: Optional[float], digits: int = 4) -> Any:
    if x is None or not math.isfinite(x):
        return ""
    return round(x, digits)


def compute_feature_row_from_dataset(
    np: Any,
    xr: Any,
    req: RequestRow,
    ds: Any,
    dataset_id: str,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    wind_da, var_name, out_unit = extract_wind_speed_da(np, xr, ds)

    time_name = pick_coord_name(wind_da, ["time", "valid_time", "observation_time"])
    obs_time_dt = req.issue_dt
    obs_offset_min = 0.0

    if time_name is not None:
        times = to_datetime_list(np, wind_da[time_name].values)
        selected = choose_time_index(
            np=np,
            times=times,
            issue_dt=req.issue_dt,
            before_min=args.window_before_min,
            after_min=args.window_after_min,
        )
        if selected is None:
            raise RuntimeError("no ASCAT time point in requested issue-time window")
        idx, obs_time_dt, obs_offset_min = selected
        wind_da = wind_da.isel({time_name: idx})

    lat_name = pick_coord_name(wind_da, ["latitude", "lat"])
    lon_name = pick_coord_name(wind_da, ["longitude", "lon"])
    if not lat_name or not lon_name:
        raise RuntimeError("latitude/longitude coordinates not found in ASCAT subset data")

    lat_da = wind_da[lat_name]
    lon_da = wind_da[lon_name]
    lon_da = ((lon_da + 180) % 360) - 180

    wind_b, lat_b, lon_b = xr.broadcast(wind_da, lat_da, lon_da)
    wind_arr = np.asarray(wind_b.values, dtype="float64").reshape(-1)
    lat_arr = np.asarray(lat_b.values, dtype="float64").reshape(-1)
    lon_arr = np.asarray(lon_b.values, dtype="float64").reshape(-1)

    valid = np.isfinite(wind_arr) & np.isfinite(lat_arr) & np.isfinite(lon_arr)
    if not valid.any():
        raise RuntimeError("ASCAT subset contains no valid wind cells")

    wind_arr = wind_arr[valid]
    lat_arr = lat_arr[valid]
    lon_arr = lon_arr[valid]

    dist_km = haversine_km(np, lat_arr, lon_arr, req.lat, parse_lon_minus180_180(req.lon))
    inner_mask = dist_km <= float(args.inner_radius_km)
    ring_mask = (dist_km > float(args.inner_radius_km)) & (dist_km <= float(args.outer_radius_km))
    if not inner_mask.any():
        raise RuntimeError("ASCAT subset has no valid inner-radius cells")

    inner_vals = wind_arr[inner_mask]
    ring_vals = wind_arr[ring_mask]
    cell_area = infer_cell_area_km2(np, lat_arr, lon_arr, float(args.assumed_cell_km))
    inner_area = cell_area[inner_mask]

    ge34_area = float(np.nansum(inner_area[inner_vals >= 34.0])) if inner_vals.size else 0.0
    ge50_area = float(np.nansum(inner_area[inner_vals >= 50.0])) if inner_vals.size else 0.0

    platform = (
        str(ds.attrs.get("platform") or "")
        or str(ds.attrs.get("platform_name") or "")
        or str(ds.attrs.get("satellite") or "")
        or dataset_id
    )

    row = {k: "" for k in OUTPUT_FIELDS}
    for k, v in req.raw.items():
        if k in row:
            row[k] = v
    row["request_id"] = req.request_id
    row["issue_time_utc"] = req.issue_time_utc
    row["lat"] = req.lat
    row["lon"] = req.lon
    row["ascat_status"] = "available"
    row["missing_reason"] = ""
    row["obs_time_utc"] = obs_time_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    row["obs_offset_minutes"] = round(obs_offset_min, 3)
    row["obs_offset_abs_minutes"] = round(abs(obs_offset_min), 3)
    row["ascat_dataset_id"] = dataset_id
    row["ascat_platform"] = platform
    row["ascat_variable"] = var_name
    row["ascat_units"] = out_unit
    row["inner_radius_km"] = args.inner_radius_km
    row["outer_radius_km"] = args.outer_radius_km
    row["wind_mean_inner_kt"] = round_or_blank(stat_mean(np, inner_vals))
    row["wind_p90_inner_kt"] = round_or_blank(stat_p90(np, inner_vals))
    row["wind_max_inner_kt"] = round_or_blank(stat_max(np, inner_vals))
    row["wind_mean_ring_kt"] = round_or_blank(stat_mean(np, ring_vals))
    row["wind_p90_ring_kt"] = round_or_blank(stat_p90(np, ring_vals))
    row["wind_max_ring_kt"] = round_or_blank(stat_max(np, ring_vals))
    row["wind_area_ge34kt_inner_km2"] = round_or_blank(ge34_area)
    row["wind_area_ge50kt_inner_km2"] = round_or_blank(ge50_area)
    row["valid_cell_count"] = int(inner_vals.size)
    row["qc_has_data"] = 1
    row["qc_time_within_window"] = 1
    return row


def failed_row(req: RequestRow, args: argparse.Namespace, reason: str) -> Dict[str, Any]:
    out = {k: "" for k in OUTPUT_FIELDS}
    for k, v in req.raw.items():
        if k in out:
            out[k] = v
    out["request_id"] = req.request_id
    out["issue_time_utc"] = req.issue_time_utc
    out["lat"] = req.lat
    out["lon"] = req.lon
    out["ascat_status"] = "missing_real_data"
    out["missing_reason"] = reason
    out["inner_radius_km"] = args.inner_radius_km
    out["outer_radius_km"] = args.outer_radius_km
    out["qc_has_data"] = 0
    out["qc_time_within_window"] = 0
    return out


def process_request(
    np: Any,
    xr: Any,
    copernicusmarine: Any,
    req: RequestRow,
    dataset_ids: List[str],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    start_dt = req.issue_dt - timedelta(minutes=args.window_before_min)
    end_dt = req.issue_dt + timedelta(minutes=args.window_after_min)
    min_lon, max_lon, min_lat, max_lat = build_bbox(
        lat=req.lat,
        lon=req.lon,
        outer_radius_km=args.outer_radius_km,
        margin_deg=args.bbox_margin_deg,
    )

    reasons: List[str] = []
    for ds_id in dataset_ids:
        req_token = safe_request_id(req.request_id)
        stamp = req.issue_dt.strftime("%Y%m%dT%H%M")
        tmp_file = args.tmp_dir / f"ascat_{req_token}_{stamp}_{ds_id.replace('/', '_')}.nc"

        last_err = ""
        for attempt in range(1, max(1, args.max_retries) + 1):
            try:
                log_progress(
                    "subset_start "
                    f"request_id={req.request_id} issue_time_utc={req.issue_time_utc} "
                    f"dataset_id={ds_id} attempt={attempt}/{max(1, args.max_retries)}"
                )
                path = run_with_timeout(
                    float(args.subset_timeout_sec),
                    call_subset,
                    copernicusmarine=copernicusmarine,
                    dataset_id=ds_id,
                    out_file=tmp_file,
                    min_lon=min_lon,
                    max_lon=max_lon,
                    min_lat=min_lat,
                    max_lat=max_lat,
                    start_dt=start_dt,
                    end_dt=end_dt,
                    args=args,
                )
                ds = xr.open_dataset(path)
                try:
                    row = compute_feature_row_from_dataset(
                        np=np,
                        xr=xr,
                        req=req,
                        ds=ds,
                        dataset_id=ds_id,
                        args=args,
                    )
                finally:
                    ds.close()
                if not args.keep_temp_files and path.exists():
                    try:
                        path.unlink()
                    except Exception:
                        pass
                log_progress(
                    "subset_done "
                    f"request_id={req.request_id} dataset_id={ds_id} "
                    f"status=available tmp_file={path}"
                )
                return row
            except Exception as exc:
                last_err = f"{type(exc).__name__}:{str(exc)[:200]}"
                log_progress(
                    "subset_failed "
                    f"request_id={req.request_id} dataset_id={ds_id} "
                    f"attempt={attempt}/{max(1, args.max_retries)} error={last_err}"
                )
                if attempt < args.max_retries:
                    time.sleep(min(5.0, 0.8 * attempt))
                continue
            finally:
                if not args.keep_temp_files and tmp_file.exists():
                    try:
                        tmp_file.unlink()
                    except Exception:
                        pass

        reasons.append(f"{ds_id}=>{last_err or 'unknown_error'}")

    return failed_row(req, args, " ; ".join(reasons)[:900] or "no_ascat_data_or_subset_failed")


def quantile(values: List[float], p: float) -> Optional[float]:
    if not values:
        return None
    values_sorted = sorted(values)
    if len(values_sorted) == 1:
        return values_sorted[0]
    pos = p * (len(values_sorted) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(values_sorted) - 1)
    w = pos - lo
    return values_sorted[lo] * (1 - w) + values_sorted[hi] * w


def run(args: argparse.Namespace, rows: List[RequestRow], dataset_ids: List[str]) -> Dict[str, Any]:
    args.tmp_dir.mkdir(parents=True, exist_ok=True)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)

    summary: Dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "manifest_csv": str(args.manifest_csv),
        "out_csv": str(args.out_csv),
        "summary_json": str(args.summary_json),
        "subset_timeout_sec": args.subset_timeout_sec,
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

    try:
        np, xr, copernicusmarine = init_optional_deps()
    except Exception as exc:
        log_progress(f"[WARN] dependency init failed: {type(exc).__name__}: {str(exc)[:260]}")
        with args.out_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS)
            writer.writeheader()
            f.flush()
            for req in rows:
                row = failed_row(req, args, f"runtime_dependency_error:{type(exc).__name__}")
                writer.writerow({k: row.get(k, "") for k in OUTPUT_FIELDS})
                summary["rows_written"] += 1
                summary["missing_rows"] += 1
            f.flush()
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        return summary

    def update_summary(row: Dict[str, Any]) -> None:
        summary["rows_written"] += 1
        year = (str(row.get("issue_time_utc") or ""))[:4]
        if year:
            bucket = by_year.setdefault(year, {"total": 0, "available": 0, "missing": 0})
            bucket["total"] += 1

        status = (str(row.get("ascat_status") or "")).strip()
        if status == "available":
            summary["available_rows"] += 1
            if year:
                by_year[year]["available"] += 1
            off = parse_float(row.get("obs_offset_abs_minutes"))
            if off is not None:
                offsets.append(off)
        else:
            summary["missing_rows"] += 1
            if year:
                by_year[year]["missing"] += 1

    with args.out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        f.flush()
        for i, req in enumerate(rows, start=1):
            log_progress(
                f"request_start {i}/{len(rows)} request_id={req.request_id} "
                f"issue_time_utc={req.issue_time_utc}"
            )
            row = process_request(
                np=np,
                xr=xr,
                copernicusmarine=copernicusmarine,
                req=req,
                dataset_ids=dataset_ids,
                args=args,
            )
            out_row = {k: row.get(k, "") for k in OUTPUT_FIELDS}
            writer.writerow(out_row)
            f.flush()
            update_summary(out_row)
            log_progress(
                f"request_done {i}/{len(rows)} request_id={req.request_id} "
                f"status={out_row.get('ascat_status', '')} dataset_id={out_row.get('ascat_dataset_id', '')}"
            )
            if i % 50 == 0 or i == len(rows):
                log_progress(f"processed {i}/{len(rows)}")
            time.sleep(max(0.0, args.sleep_sec))

    if offsets:
        summary["mean_abs_offset_min_available"] = round(sum(offsets) / len(offsets), 3)
        p50 = quantile(offsets, 0.5)
        p90 = quantile(offsets, 0.9)
        summary["p50_abs_offset_min_available"] = round(p50, 3) if p50 is not None else None
        summary["p90_abs_offset_min_available"] = round(p90, 3) if p90 is not None else None

    coverage_by_year: Dict[str, Dict[str, Any]] = {}
    for y in sorted(by_year.keys()):
        total = by_year[y]["total"]
        available = by_year[y]["available"]
        missing = by_year[y]["missing"]
        coverage_by_year[y] = {
            "total": total,
            "available": available,
            "missing": missing,
            "coverage_rate": round((available / total), 6) if total > 0 else 0.0,
        }
    summary["coverage_by_year"] = coverage_by_year

    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    args = resolve_runtime_config(parse_args())
    dataset_ids = args.dataset_ids if args.dataset_ids else list(DEFAULT_DATASET_IDS)
    rows = load_manifest(args)
    if not rows:
        raise RuntimeError("no valid request rows found in manifest")

    log_progress(f"manifest: {args.manifest_csv}")
    log_progress(f"rows_to_process: {len(rows)}")
    log_progress(f"dataset_ids_requested: {dataset_ids}")
    log_progress(f"tmp_dir: {args.tmp_dir}")
    log_progress(f"subset_timeout_sec: {args.subset_timeout_sec}")
    if args.dry_run:
        return 0

    summary = run(args=args, rows=rows, dataset_ids=dataset_ids)
    log_progress(str(args.out_csv))
    log_progress(str(args.summary_json))
    log_progress(f"rows_written: {summary['rows_written']}")
    log_progress(f"available_rows: {summary['available_rows']}")
    log_progress(f"missing_rows: {summary['missing_rows']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
