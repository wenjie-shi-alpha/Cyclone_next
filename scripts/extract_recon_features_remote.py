#!/usr/bin/env python3
"""Extract Recon structured features from NHC recon archive text products.

Design goals:
1. Remote side downloads/parses raw text products.
2. Local side stores compact row-wise structured features only.
3. Fault-tolerant execution for long historical windows.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import subprocess
import time
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib.error import HTTPError, URLError


DEFAULT_SUBDIRS = ["REPNT2", "REPNT3", "AHONT1", "AHOPN1"]

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
    "recon_status",
    "missing_reason",
    "recon_obs_time_utc",
    "obs_offset_minutes",
    "obs_offset_abs_minutes",
    "recon_message_type",
    "message_count",
    "recon_source_file",
    "recon_source_subdirs",
    "vdm_min_slp_mb",
    "vdm_max_flight_level_wind_kt",
    "vdm_center_lat",
    "vdm_center_lon",
    "hdob_max_sfmr_wind_kt",
    "hdob_max_flight_level_wind_kt",
    "dropsonde_min_slp_mb",
    "qc_has_data",
    "qc_time_within_window",
]

CATALOG_FILE_RE = re.compile(r'href="([A-Z0-9_-]+\.[0-9]{12}\.txt)"', re.IGNORECASE)
VDM_STORM_RE = re.compile(r"VORTEX DATA MESSAGE\s+([A-Z]{2}\d{2}\d{4})", re.IGNORECASE)
VDM_A_RE = re.compile(r"^\s*A\.\s*(\d{1,2})/(\d{2}):(\d{2}):(\d{2})Z", re.IGNORECASE | re.MULTILINE)
VDM_B_RE = re.compile(
    r"^\s*B\.\s*([0-9.]+)\s*deg\s*([NS])\s*([0-9.]+)\s*deg\s*([EW])",
    re.IGNORECASE | re.MULTILINE,
)
VDM_D_RE = re.compile(r"^\s*D\.\s*(\d{3,4})\s*mb", re.IGNORECASE | re.MULTILINE)
VDM_J_RE = re.compile(r"^\s*J\.\s*\d{3}\s*deg\s*(\d{1,3})\s*kt", re.IGNORECASE | re.MULTILINE)
MAX_FL_WIND_RE = re.compile(r"MAX FL WIND\s+(\d{1,3})\s*KT", re.IGNORECASE)

HDOB_HEAD_RE = re.compile(r"\bHDOB\b", re.IGNORECASE)
HDOB_LINE_RE = re.compile(r"^\s*(\d{6})\s+([0-9]{3,5}[NS])\s+([0-9]{4,6}[EW])\s+(.+)$")
SFMR_RE = re.compile(r"\bSFMR[^0-9]{0,10}(\d{2,3})\b", re.IGNORECASE)
DROPPSONDE_PRESS_RE = re.compile(
    r"(?:MIN(?:IMUM)?\s+)?(?:SFC|SURFACE)?\s*PRESS(?:URE)?[^0-9]{0,12}(\d{3,4})\s*MB",
    re.IGNORECASE,
)


@dataclass
class RequestRow:
    request_id: str
    issue_time_utc: str
    issue_dt: datetime
    lat: float
    lon: float
    raw: Dict[str, str]


@dataclass
class CatalogEntry:
    year: int
    subdir: str
    file_name: str
    file_dt: datetime
    url: str


@dataclass
class ParsedReconMessage:
    source_subdir: str
    source_file: str
    source_url: str
    message_type: str
    storm_id_extracted: str
    obs_time_utc: datetime
    center_lat: Optional[float]
    center_lon: Optional[float]
    vdm_min_slp_mb: Optional[float]
    vdm_max_flight_level_wind_kt: Optional[float]
    hdob_max_sfmr_wind_kt: Optional[float]
    hdob_max_flight_level_wind_kt: Optional[float]
    dropsonde_min_slp_mb: Optional[float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract Recon compact feature table from request manifest via NHC archive."
    )
    parser.add_argument(
        "--manifest-csv",
        type=Path,
        default=Path("data/interim/recon/recon_request_manifest.csv"),
        help="Input request manifest csv.",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path("data/interim/recon/recon_observation_features.csv"),
        help="Output feature csv.",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=Path("data/interim/recon/recon_observation_features_summary.json"),
        help="Output summary json.",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="https://www.nhc.noaa.gov/archive/recon",
        help="NHC recon archive base url.",
    )
    parser.add_argument(
        "--subdir",
        action="append",
        dest="subdirs",
        default=[],
        help="Recon message subdirectory under year folder (repeatable).",
    )
    parser.add_argument(
        "--window-before-hours",
        type=float,
        default=12.0,
        help="Look-back window before issue time.",
    )
    parser.add_argument(
        "--window-after-hours",
        type=float,
        default=3.0,
        help="Look-forward window after issue time.",
    )
    parser.add_argument(
        "--spatial-max-dist-km",
        type=float,
        default=450.0,
        help="When storm-id is unavailable, max center-distance for accepting a message.",
    )
    parser.add_argument(
        "--catalog-cache-dir",
        type=Path,
        default=Path("/tmp/recon_catalog_cache"),
        help="Cache directory for listing/message text.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable local cache for listing and message text.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="If >0, process first N manifest rows.",
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
        help="Retries for HTTP requests.",
    )
    parser.add_argument(
        "--sleep-sec",
        type=float,
        default=0.05,
        help="Sleep between manifest rows.",
    )
    parser.add_argument(
        "--http-sleep-sec",
        type=float,
        default=0.08,
        help="Sleep between HTTP requests to reduce request bursts.",
    )
    parser.add_argument(
        "--http-timeout-sec",
        type=int,
        default=20,
        help="HTTP timeout in seconds for listing/message fetch.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print resolved config and row counts.",
    )
    parser.add_argument(
        "--cache-only",
        action="store_true",
        help="Use local cache only and never fetch from network when cache file is missing.",
    )
    parser.add_argument(
        "--max-candidates-per-request",
        type=int,
        default=80,
        help="If >0, keep only nearest N catalog entries around each request time.",
    )
    return parser.parse_args()


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


def norm_lon_to_minus180_180(lon: float) -> float:
    out = lon
    while out >= 180:
        out -= 360
    while out < -180:
        out += 360
    return out


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


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dlat = p2 - p1
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2.0) ** 2) + math.cos(p1) * math.cos(p2) * (math.sin(dlon / 2.0) ** 2)
    return 2.0 * r * math.asin(math.sqrt(a))


def http_get_text(url: str, timeout_sec: int, max_retries: int, http_sleep_sec: float = 0.0) -> str:
    last_exc: Optional[Exception] = None
    for attempt in range(1, max(1, max_retries) + 1):
        # Prefer curl in this environment: generally more stable than urllib for repeated fetches.
        try:
            proc = subprocess.run(
                ["curl", "-fsSL", "--max-time", str(timeout_sec), url],
                check=True,
                capture_output=True,
                text=True,
                timeout=max(3, timeout_sec + 2),
            )
            if http_sleep_sec > 0:
                time.sleep(http_sleep_sec)
            return proc.stdout
        except Exception as exc:
            last_exc = exc

        try:
            with urllib.request.urlopen(url, timeout=timeout_sec) as resp:
                text = resp.read().decode("utf-8", errors="ignore")
                if http_sleep_sec > 0:
                    time.sleep(http_sleep_sec)
                return text
        except HTTPError as exc:
            if 400 <= exc.code < 500 and exc.code != 429:
                detail = ""
                try:
                    detail = exc.read().decode("utf-8", errors="ignore")
                except Exception:
                    detail = ""
                raise RuntimeError(f"HTTP {exc.code} client error for {url}: {detail[:300]}") from exc
            last_exc = exc
        except URLError as exc:
            last_exc = exc
        except Exception as exc:
            last_exc = exc
        time.sleep(min(8.0, 0.8 * attempt))
    if last_exc is not None:
        # Fallback for environments where Python DNS/network is restricted but curl works.
        try:
            proc = subprocess.run(
                ["curl", "-s", url],
                check=True,
                capture_output=True,
                text=True,
                timeout=timeout_sec,
            )
            if http_sleep_sec > 0:
                time.sleep(http_sleep_sec)
            return proc.stdout
        except Exception:
            raise last_exc
    raise RuntimeError(f"http_get_text failed without exception: {url}")


def list_catalog_for_subdir(
    args: argparse.Namespace,
    year: int,
    subdir: str,
) -> List[CatalogEntry]:
    cache_file = args.catalog_cache_dir / f"listing_{year}_{subdir}.html"
    url = f"{args.base_url.rstrip('/')}/{year}/{subdir}/"

    html = ""
    if not args.no_cache and cache_file.exists():
        html = cache_file.read_text(encoding="utf-8", errors="ignore")
    elif args.cache_only:
        raise FileNotFoundError(f"cache-only mode: missing listing cache file {cache_file}")
    else:
        html = http_get_text(
            url=url,
            timeout_sec=args.http_timeout_sec,
            max_retries=args.max_retries,
            http_sleep_sec=args.http_sleep_sec,
        )
        if not args.no_cache:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(html, encoding="utf-8")

    out: List[CatalogEntry] = []
    seen = set()
    for m in CATALOG_FILE_RE.finditer(html):
        fn = m.group(1).strip()
        if fn in seen:
            continue
        seen.add(fn)
        ts_match = re.search(r"\.(\d{12})\.txt$", fn)
        if not ts_match:
            continue
        try:
            file_dt = datetime.strptime(ts_match.group(1), "%Y%m%d%H%M").replace(tzinfo=timezone.utc)
        except ValueError:
            continue
        out.append(
            CatalogEntry(
                year=year,
                subdir=subdir,
                file_name=fn,
                file_dt=file_dt,
                url=f"{url}{fn}",
            )
        )
    out.sort(key=lambda x: x.file_dt)
    return out


def parse_latlon_token(token: str) -> Optional[float]:
    txt = (token or "").strip().upper()
    if not txt or txt[-1] not in {"N", "S", "E", "W"}:
        return None
    hemi = txt[-1]
    body = txt[:-1]
    if not body.isdigit():
        return None
    if hemi in {"N", "S"}:
        if len(body) not in {3, 4}:
            return None
    else:
        if len(body) not in {4, 5}:
            return None
    deg = int(body[:-2])
    minutes = int(body[-2:])
    value = deg + (minutes / 60.0)
    if hemi in {"S", "W"}:
        value = -value
    return value


def parse_vdm_message(text: str, file_dt: datetime, source_subdir: str, source_file: str, source_url: str) -> ParsedReconMessage:
    storm_id = ""
    m = VDM_STORM_RE.search(text)
    if m:
        storm_id = m.group(1).upper()

    obs_time = file_dt
    ma = VDM_A_RE.search(text)
    if ma:
        day = int(ma.group(1))
        hh = int(ma.group(2))
        mm = int(ma.group(3))
        ss = int(ma.group(4))
        try:
            obs_time = file_dt.replace(day=day, hour=hh, minute=mm, second=ss, microsecond=0)
        except ValueError:
            obs_time = file_dt.replace(hour=hh, minute=mm, second=ss, microsecond=0)

    center_lat = None
    center_lon = None
    mb = VDM_B_RE.search(text)
    if mb:
        lat = float(mb.group(1)) * (1.0 if mb.group(2).upper() == "N" else -1.0)
        lon = float(mb.group(3)) * (-1.0 if mb.group(4).upper() == "W" else 1.0)
        center_lat = lat
        center_lon = norm_lon_to_minus180_180(lon)

    min_slp = None
    md = VDM_D_RE.search(text)
    if md:
        min_slp = parse_float(md.group(1))

    max_fl = None
    mj = VDM_J_RE.search(text)
    if mj:
        max_fl = parse_float(mj.group(1))
    else:
        mmf = MAX_FL_WIND_RE.search(text)
        if mmf:
            max_fl = parse_float(mmf.group(1))

    return ParsedReconMessage(
        source_subdir=source_subdir,
        source_file=source_file,
        source_url=source_url,
        message_type="VDM",
        storm_id_extracted=storm_id,
        obs_time_utc=obs_time,
        center_lat=center_lat,
        center_lon=center_lon,
        vdm_min_slp_mb=min_slp,
        vdm_max_flight_level_wind_kt=max_fl,
        hdob_max_sfmr_wind_kt=None,
        hdob_max_flight_level_wind_kt=None,
        dropsonde_min_slp_mb=None,
    )


def parse_hdob_message(text: str, file_dt: datetime, source_subdir: str, source_file: str, source_url: str) -> ParsedReconMessage:
    center_lat = None
    center_lon = None
    max_fl_wind = None
    max_sfmr = None

    sfmr_vals = [parse_float(x) for x in SFMR_RE.findall(text)]
    sfmr_vals = [x for x in sfmr_vals if x is not None]
    if sfmr_vals:
        max_sfmr = max(sfmr_vals)

    for ln in text.splitlines():
        m = HDOB_LINE_RE.match(ln)
        if not m:
            continue
        lat_t = parse_latlon_token(m.group(2))
        lon_t = parse_latlon_token(m.group(3))
        if lat_t is not None and lon_t is not None and center_lat is None:
            center_lat = lat_t
            center_lon = norm_lon_to_minus180_180(lon_t)

        rest_tokens = m.group(4).split()
        speed_candidates: List[float] = []
        for tok in rest_tokens:
            if tok.isdigit():
                if len(tok) == 5:
                    spd = parse_float(tok[-2:])
                    if spd is not None and 0 <= spd <= 150:
                        speed_candidates.append(spd)
                elif len(tok) == 6:
                    spd = parse_float(tok[-3:])
                    if spd is not None and 0 <= spd <= 220:
                        speed_candidates.append(spd)
        if speed_candidates:
            cur = max(speed_candidates)
            max_fl_wind = cur if max_fl_wind is None else max(max_fl_wind, cur)

    mmf = MAX_FL_WIND_RE.search(text)
    if mmf:
        v = parse_float(mmf.group(1))
        if v is not None:
            max_fl_wind = v if max_fl_wind is None else max(max_fl_wind, v)

    return ParsedReconMessage(
        source_subdir=source_subdir,
        source_file=source_file,
        source_url=source_url,
        message_type="HDOB",
        storm_id_extracted="",
        obs_time_utc=file_dt,
        center_lat=center_lat,
        center_lon=center_lon,
        vdm_min_slp_mb=None,
        vdm_max_flight_level_wind_kt=None,
        hdob_max_sfmr_wind_kt=max_sfmr,
        hdob_max_flight_level_wind_kt=max_fl_wind,
        dropsonde_min_slp_mb=None,
    )


def parse_dropsonde_message(text: str, file_dt: datetime, source_subdir: str, source_file: str, source_url: str) -> ParsedReconMessage:
    press_vals = [parse_float(x) for x in DROPPSONDE_PRESS_RE.findall(text)]
    press_vals = [x for x in press_vals if x is not None]
    min_press = min(press_vals) if press_vals else None

    return ParsedReconMessage(
        source_subdir=source_subdir,
        source_file=source_file,
        source_url=source_url,
        message_type="DROPSONDE",
        storm_id_extracted="",
        obs_time_utc=file_dt,
        center_lat=None,
        center_lon=None,
        vdm_min_slp_mb=None,
        vdm_max_flight_level_wind_kt=None,
        hdob_max_sfmr_wind_kt=None,
        hdob_max_flight_level_wind_kt=None,
        dropsonde_min_slp_mb=min_press,
    )


def parse_message(text: str, entry: CatalogEntry) -> ParsedReconMessage:
    up = text.upper()
    if "VORTEX DATA MESSAGE" in up:
        return parse_vdm_message(
            text=text,
            file_dt=entry.file_dt,
            source_subdir=entry.subdir,
            source_file=entry.file_name,
            source_url=entry.url,
        )
    if HDOB_HEAD_RE.search(text):
        return parse_hdob_message(
            text=text,
            file_dt=entry.file_dt,
            source_subdir=entry.subdir,
            source_file=entry.file_name,
            source_url=entry.url,
        )
    if "XXAA" in up or "XXBB" in up or "UZNT13" in up:
        return parse_dropsonde_message(
            text=text,
            file_dt=entry.file_dt,
            source_subdir=entry.subdir,
            source_file=entry.file_name,
            source_url=entry.url,
        )
    return ParsedReconMessage(
        source_subdir=entry.subdir,
        source_file=entry.file_name,
        source_url=entry.url,
        message_type=entry.subdir,
        storm_id_extracted="",
        obs_time_utc=entry.file_dt,
        center_lat=None,
        center_lon=None,
        vdm_min_slp_mb=None,
        vdm_max_flight_level_wind_kt=None,
        hdob_max_sfmr_wind_kt=None,
        hdob_max_flight_level_wind_kt=None,
        dropsonde_min_slp_mb=None,
    )


def fetch_message_text(args: argparse.Namespace, entry: CatalogEntry) -> str:
    cache_file = args.catalog_cache_dir / "messages" / str(entry.year) / entry.subdir / entry.file_name
    if not args.no_cache and cache_file.exists():
        return cache_file.read_text(encoding="utf-8", errors="ignore")
    if args.cache_only:
        raise FileNotFoundError(f"cache-only mode: missing message cache file {cache_file}")
    text = http_get_text(
        url=entry.url,
        timeout_sec=args.http_timeout_sec,
        max_retries=args.max_retries,
        http_sleep_sec=args.http_sleep_sec,
    )
    if not args.no_cache:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(text, encoding="utf-8")
    return text


def build_catalog(args: argparse.Namespace, rows: Sequence[RequestRow], subdirs: List[str]) -> List[CatalogEntry]:
    years = set()
    before = timedelta(hours=args.window_before_hours)
    after = timedelta(hours=args.window_after_hours)
    for r in rows:
        years.add((r.issue_dt - before).year)
        years.add((r.issue_dt + after).year)

    catalog: List[CatalogEntry] = []
    for year in sorted(years):
        for subdir in subdirs:
            try:
                entries = list_catalog_for_subdir(args=args, year=year, subdir=subdir)
            except Exception as exc:
                print(f"[WARN] catalog fetch failed for {year}/{subdir}: {type(exc).__name__}: {str(exc)[:220]}")
                continue
            catalog.extend(entries)
            print(f"catalog {year}/{subdir}: {len(entries)} files")
    catalog.sort(key=lambda x: x.file_dt)
    return catalog


def pick_candidates(req: RequestRow, catalog: Sequence[CatalogEntry], args: argparse.Namespace) -> List[CatalogEntry]:
    start_dt = req.issue_dt - timedelta(hours=args.window_before_hours)
    end_dt = req.issue_dt + timedelta(hours=args.window_after_hours)
    cands = [e for e in catalog if start_dt <= e.file_dt <= end_dt]
    max_n = int(args.max_candidates_per_request or 0)
    if max_n > 0 and len(cands) > max_n:
        cands = sorted(cands, key=lambda e: abs((e.file_dt - req.issue_dt).total_seconds()))[:max_n]
    return cands


def is_message_match(req: RequestRow, msg: ParsedReconMessage, args: argparse.Namespace) -> bool:
    atcf_id = (req.raw.get("atcf_storm_id") or "").strip().upper()
    if msg.storm_id_extracted and atcf_id and msg.storm_id_extracted != atcf_id:
        return False

    if msg.center_lat is not None and msg.center_lon is not None and not (msg.storm_id_extracted and atcf_id):
        d = haversine_km(req.lat, norm_lon_to_minus180_180(req.lon), msg.center_lat, msg.center_lon)
        if d > args.spatial_max_dist_km:
            return False

    return True


def aggregate_messages(req: RequestRow, msgs: List[ParsedReconMessage], args: argparse.Namespace) -> Dict[str, Any]:
    if not msgs:
        out = {k: "" for k in OUTPUT_FIELDS}
        for k, v in req.raw.items():
            if k in out:
                out[k] = v
        out["request_id"] = req.request_id
        out["issue_time_utc"] = req.issue_time_utc
        out["lat"] = req.lat
        out["lon"] = req.lon
        out["recon_status"] = "missing_real_data"
        out["missing_reason"] = "no_recon_message_matched_time_and_filter"
        out["qc_has_data"] = 0
        out["qc_time_within_window"] = 0
        return out

    msgs_sorted = sorted(msgs, key=lambda x: abs((x.obs_time_utc - req.issue_dt).total_seconds()))
    nearest = msgs_sorted[0]
    offset_min = (nearest.obs_time_utc - req.issue_dt).total_seconds() / 60.0

    def min_opt(values: Sequence[Optional[float]]) -> Optional[float]:
        arr = [x for x in values if x is not None]
        return min(arr) if arr else None

    def max_opt(values: Sequence[Optional[float]]) -> Optional[float]:
        arr = [x for x in values if x is not None]
        return max(arr) if arr else None

    vdm_msgs = [m for m in msgs if m.message_type == "VDM"]
    nearest_vdm = None
    if vdm_msgs:
        nearest_vdm = sorted(vdm_msgs, key=lambda x: abs((x.obs_time_utc - req.issue_dt).total_seconds()))[0]

    out = {k: "" for k in OUTPUT_FIELDS}
    for k, v in req.raw.items():
        if k in out:
            out[k] = v
    out["request_id"] = req.request_id
    out["issue_time_utc"] = req.issue_time_utc
    out["lat"] = req.lat
    out["lon"] = req.lon
    out["recon_status"] = "available"
    out["missing_reason"] = ""
    out["recon_obs_time_utc"] = nearest.obs_time_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
    out["obs_offset_minutes"] = round(offset_min, 3)
    out["obs_offset_abs_minutes"] = round(abs(offset_min), 3)
    out["recon_message_type"] = ",".join(sorted({m.message_type for m in msgs}))
    out["message_count"] = len(msgs)
    out["recon_source_file"] = nearest.source_url
    out["recon_source_subdirs"] = ",".join(sorted({m.source_subdir for m in msgs}))
    out["vdm_min_slp_mb"] = min_opt([m.vdm_min_slp_mb for m in msgs]) or ""
    out["vdm_max_flight_level_wind_kt"] = max_opt([m.vdm_max_flight_level_wind_kt for m in msgs]) or ""
    out["vdm_center_lat"] = nearest_vdm.center_lat if nearest_vdm and nearest_vdm.center_lat is not None else ""
    out["vdm_center_lon"] = nearest_vdm.center_lon if nearest_vdm and nearest_vdm.center_lon is not None else ""
    out["hdob_max_sfmr_wind_kt"] = max_opt([m.hdob_max_sfmr_wind_kt for m in msgs]) or ""
    out["hdob_max_flight_level_wind_kt"] = max_opt([m.hdob_max_flight_level_wind_kt for m in msgs]) or ""
    out["dropsonde_min_slp_mb"] = min_opt([m.dropsonde_min_slp_mb for m in msgs]) or ""
    out["qc_has_data"] = 1
    out["qc_time_within_window"] = 1
    return out


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


def run(args: argparse.Namespace, rows: List[RequestRow], catalog: List[CatalogEntry], subdirs: List[str]) -> Dict[str, Any]:
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)

    summary: Dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "manifest_csv": str(args.manifest_csv),
        "out_csv": str(args.out_csv),
        "summary_json": str(args.summary_json),
        "base_url": args.base_url,
        "subdirs_used": subdirs,
        "cache_only": int(bool(args.cache_only)),
        "catalog_entries_total": len(catalog),
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
    parsed_cache: Dict[str, Optional[ParsedReconMessage]] = {}

    def update_summary(row: Dict[str, Any]) -> None:
        summary["rows_written"] += 1
        year = (str(row.get("issue_time_utc") or ""))[:4]
        if year:
            bucket = by_year.setdefault(year, {"total": 0, "available": 0, "missing": 0})
            bucket["total"] += 1

        status = (str(row.get("recon_status") or "")).strip()
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

        for i, req in enumerate(rows, start=1):
            candidates = pick_candidates(req=req, catalog=catalog, args=args)
            parsed_msgs: List[ParsedReconMessage] = []

            for entry in candidates:
                cache_key = f"{entry.year}/{entry.subdir}/{entry.file_name}"
                msg = parsed_cache.get(cache_key)
                if cache_key not in parsed_cache:
                    try:
                        text = fetch_message_text(args=args, entry=entry)
                        msg = parse_message(text=text, entry=entry)
                    except Exception:
                        msg = None
                    parsed_cache[cache_key] = msg
                if msg is None:
                    continue
                if is_message_match(req=req, msg=msg, args=args):
                    parsed_msgs.append(msg)

            row = aggregate_messages(req=req, msgs=parsed_msgs, args=args)
            out_row = {k: row.get(k, "") for k in OUTPUT_FIELDS}
            writer.writerow(out_row)
            update_summary(out_row)

            if i % 50 == 0 or i == len(rows):
                print(f"processed {i}/{len(rows)}")
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

    args.summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    args = parse_args()
    subdirs = args.subdirs if args.subdirs else list(DEFAULT_SUBDIRS)
    rows = load_manifest(args)
    if not rows:
        raise RuntimeError("no valid request rows found in manifest")

    print("manifest:", args.manifest_csv)
    print("rows_to_process:", len(rows))
    print("base_url:", args.base_url)
    print("subdirs:", subdirs)
    if args.dry_run:
        return 0

    if not args.no_cache:
        args.catalog_cache_dir.mkdir(parents=True, exist_ok=True)
    catalog = build_catalog(args=args, rows=rows, subdirs=subdirs)
    summary = run(args=args, rows=rows, catalog=catalog, subdirs=subdirs)

    print(args.out_csv)
    print(args.summary_json)
    print("rows_written:", summary["rows_written"])
    print("available_rows:", summary["available_rows"])
    print("missing_rows:", summary["missing_rows"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
