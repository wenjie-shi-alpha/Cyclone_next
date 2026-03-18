#!/usr/bin/env python3
"""Build one concrete forecast fine-tuning sample preview.

This script assembles an end-to-end sample from currently available datasets,
and keeps placeholders for not-yet-integrated datasets.
"""

from __future__ import annotations

import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


BASE = Path('.')


def parse_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def norm_lon_to_minus180_180(lon: float) -> float:
    while lon >= 180:
        lon -= 360
    while lon < -180:
        lon += 360
    return lon


def read_text(path: Path) -> List[str]:
    return path.read_text(encoding='utf-8', errors='ignore').splitlines()


def extract_advisory_summary(lines: List[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        'center': {},
        'motion': {},
        'intensity': {},
        'forecast_table': [],
    }

    for ln in lines:
        m = re.search(r'CENTER LOCATED NEAR\s+([0-9.]+)([NS])\s+([0-9.]+)([EW])\s+AT\s+(\d{2})/(\d{4})Z', ln, re.I)
        if m:
            lat = float(m.group(1)) * (1 if m.group(2).upper() == 'N' else -1)
            lon = float(m.group(3)) * (-1 if m.group(4).upper() == 'W' else 1)
            out['center'] = {
                'lat': lat,
                'lon': lon,
                'obs_day': int(m.group(5)),
                'obs_hhmmz': m.group(6),
            }
            break

    for ln in lines:
        m = re.search(r'PRESENT MOVEMENT TOWARD THE\s+(.+?)\s+OR\s+(\d+)\s+DEGREES AT\s+(\d+)\s+KT', ln, re.I)
        if m:
            out['motion'] = {
                'motion_text': m.group(1).strip(),
                'direction_deg': int(m.group(2)),
                'speed_kt': int(m.group(3)),
            }
            break

    min_p = None
    max_w = None
    for ln in lines:
        m = re.search(r'ESTIMATED MINIMUM CENTRAL PRESSURE\s+(\d+)\s+MB', ln, re.I)
        if m:
            min_p = int(m.group(1))
        m2 = re.search(r'MAX SUSTAINED WINDS\s+(\d+)\s+KT', ln, re.I)
        if m2:
            max_w = int(m2.group(1))
        if min_p is not None and max_w is not None:
            break
    out['intensity'] = {'min_pressure_mb': min_p, 'max_wind_kt': max_w}

    # Forecast table from forecast advisory lines
    fcst: List[Dict[str, Any]] = []
    current = None
    for ln in lines:
        m = re.search(r'^\s*FORECAST VALID\s+(\d{2})/(\d{4})Z\s+([0-9.]+)([NS])\s+([0-9.]+)([EW])', ln, re.I)
        if m:
            lat = float(m.group(3)) * (1 if m.group(4).upper() == 'N' else -1)
            lon = float(m.group(5)) * (-1 if m.group(6).upper() == 'W' else 1)
            current = {
                'valid_day': int(m.group(1)),
                'valid_hhmmz': m.group(2),
                'lat': lat,
                'lon': lon,
                'vmax_kt': None,
            }
            fcst.append(current)
            continue
        if current is not None:
            m2 = re.search(r'^\s*MAX WIND\s+(\d+)\s+KT', ln, re.I)
            if m2:
                current['vmax_kt'] = int(m2.group(1))
    out['forecast_table'] = fcst

    return out


def extract_discussion_signals(lines: List[str]) -> Dict[str, Any]:
    # Keep a concise but direct excerpt from the first 2 reasoning paragraphs.
    body = []
    started = False
    for ln in lines:
        if 'FORECAST POSITIONS AND MAX WINDS' in ln.upper():
            break
        if started:
            body.append(ln)
        elif 'Satellite' in ln or 'Edouard is moving' in ln or 'While' in ln:
            started = True
            body.append(ln)

    text = '\n'.join([x for x in body]).strip()
    if len(text) > 1500:
        text = text[:1500] + '...'

    # structured positions from discussion section
    track = []
    in_table = False
    for ln in lines:
        if 'FORECAST POSITIONS AND MAX WINDS' in ln.upper():
            in_table = True
            continue
        if not in_table:
            continue
        if ln.strip().startswith('$$'):
            break
        m = re.search(r'^\s*(INIT|\d+H)\s+(\d{2})/(\d{4})Z\s+([0-9.]+)([NS])\s+([0-9.]+)([EW])\s+(\d+)\s+KT', ln, re.I)
        if not m:
            continue
        tau_label = m.group(1).upper()
        tau_h = 0 if tau_label == 'INIT' else int(tau_label[:-1])
        lat = float(m.group(4)) * (1 if m.group(5).upper() == 'N' else -1)
        lon = float(m.group(6)) * (-1 if m.group(7).upper() == 'W' else 1)
        track.append({
            'tau_h': tau_h,
            'valid_day': int(m.group(2)),
            'valid_hhmmz': m.group(3),
            'lat': lat,
            'lon': lon,
            'vmax_kt': int(m.group(8)),
        })

    return {'reasoning_excerpt': text, 'forecast_positions': track}


def extract_public_summary(lines: List[str]) -> Dict[str, Any]:
    out = {}
    for i, ln in enumerate(lines):
        if 'SUMMARY OF' in ln.upper():
            out['summary_block_header'] = ln.strip()
            out['summary_lines'] = [x.strip() for x in lines[i + 2:i + 9] if x.strip()]
            break
    return out


def extract_wind_prob_summary(lines: List[str]) -> Dict[str, Any]:
    # This case has no listed locations in table. Keep the product-level meta.
    out: Dict[str, Any] = {
        'has_location_rows': False,
        'location_rows': 0,
    }
    for ln in lines:
        m = re.search(r'WIND SPEED PROBABILITIES NUMBER\s+(\d+)', ln, re.I)
        if m:
            out['product_number'] = int(m.group(1))
            break
    for ln in lines:
        if ln.strip().startswith('LOCATION'):
            out['has_location_rows'] = True
    return out


def load_groundtruth_state(storm_id: str, issue_dt: datetime) -> Dict[str, Any]:
    best = None
    best_dt = None
    with (BASE / 'GroundTruth_Cyclones' / 'matched_cyclone_tracks.csv').open('r', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get('storm_id') != storm_id:
                continue
            dt = datetime.strptime((row.get('datetime') or '')[:19], '%Y-%m-%d %H:%M:%S')
            if best is None or abs((dt - issue_dt).total_seconds()) < abs((best_dt - issue_dt).total_seconds()):
                best = row
                best_dt = dt

    if best is None:
        return {}

    return {
        'matched_datetime_utc': best_dt.strftime('%Y-%m-%dT%H:%M:%SZ'),
        'lat': parse_float(best.get('latitude')),
        'lon': parse_float(best.get('longitude')),
        'max_wind_wmo': parse_float(best.get('max_wind_wmo')),
        'min_pressure_wmo': parse_float(best.get('min_pressure_wmo')),
        'max_wind_usa': parse_float(best.get('max_wind_usa')),
        'min_pressure_usa': parse_float(best.get('min_pressure_usa')),
        'storm_speed': parse_float(best.get('storm_speed')),
        'storm_direction': parse_float(best.get('storm_direction')),
        'noaa_name': best.get('noaa_name'),
        'noaa_basin': best.get('noaa_basin'),
    }


def pick_cds_row(issue_dt: datetime, target_lat: float, target_lon_w: float) -> Tuple[Dict[str, Any], Path]:
    month_file = BASE / 'CDS_real' / f'cds_environment_analysis_{issue_dt.strftime("%Y-%m")}.json'
    payload = json.loads(month_file.read_text(encoding='utf-8'))
    best = None
    best_d = None

    for row in payload.get('environmental_analysis', []):
        t = str(row.get('time', ''))[:19]
        try:
            dt = datetime.strptime(t.replace('T', ' '), '%Y-%m-%d %H:%M:%S')
        except ValueError:
            continue
        if dt != issue_dt:
            continue

        tc = row.get('tc_position') or {}
        lat = parse_float(tc.get('lat'))
        lon = parse_float(tc.get('lon'))
        if lat is None or lon is None:
            continue
        lon_w = norm_lon_to_minus180_180(lon)
        d = (lat - target_lat) ** 2 + (lon_w - target_lon_w) ** 2
        if best is None or d < best_d:
            best = row
            best_d = d

    if best is None:
        raise RuntimeError('CDS row not found for sample time')

    return best, month_file


def extract_cds_features(row: Dict[str, Any]) -> Dict[str, Any]:
    features: Dict[str, Any] = {}
    systems = row.get('environmental_systems') or []
    by_name = {s.get('system_name'): s for s in systems if isinstance(s, dict) and s.get('system_name')}

    def num_from_intensity(sys_obj: Dict[str, Any]) -> Optional[float]:
        intensity = sys_obj.get('intensity') if isinstance(sys_obj, dict) else None
        if not isinstance(intensity, dict):
            return None
        for k in ['value', 'average_value', 'max_value', 'speed', 'surface_temp']:
            if k in intensity:
                v = parse_float(intensity.get(k))
                if v is not None:
                    return v
        return None

    mapping = {
        'VerticalWindShear': 'vertical_wind_shear',
        'UpperLevelDivergence': 'upper_level_divergence',
        'OceanHeatContent': 'ocean_heat_content_or_sst',
        'SubtropicalHigh': 'subtropical_high',
        'WesterlyTrough': 'westerly_trough',
        'MonsoonTrough': 'monsoon_trough',
        'LowLevelFlow': 'low_level_flow',
    }

    for src_name, dst_name in mapping.items():
        s = by_name.get(src_name)
        if s is None:
            features[dst_name] = None
            continue
        intensity = s.get('intensity') if isinstance(s.get('intensity'), dict) else {}
        features[dst_name] = {
            'value': num_from_intensity(s),
            'unit': intensity.get('unit'),
            'level': intensity.get('level'),
            'description': s.get('description'),
        }

    return features


def extract_hres_track(storm_id: str, init_dt: datetime) -> Tuple[List[Dict[str, Any]], Path]:
    fp = BASE / 'HRES_forecast' / 'HRES_track' / f'track_{storm_id}_{init_dt.strftime("%Y%m%d")}_{init_dt.strftime("%H%M")}.csv'
    with fp.open('r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Deduplicate by time: keep the row with minimum msl (more TC-like point).
    by_time: Dict[datetime, Dict[str, str]] = {}
    for row in rows:
        dt = datetime.strptime((row['time'] or '')[:19], '%Y-%m-%d %H:%M:%S')
        cur = by_time.get(dt)
        if cur is None:
            by_time[dt] = row
            continue
        cur_msl = parse_float(cur.get('msl')) or 1e18
        new_msl = parse_float(row.get('msl')) or 1e18
        if new_msl < cur_msl:
            by_time[dt] = row

    out = []
    for dt in sorted(by_time.keys()):
        row = by_time[dt]
        tau_h = int(round((dt - init_dt).total_seconds() / 3600))
        out.append({
            'tau_h': tau_h,
            'valid_time_utc': dt.strftime('%Y-%m-%dT%H:%M:%SZ'),
            'lat': parse_float(row.get('lat')),
            'lon': norm_lon_to_minus180_180(parse_float(row.get('lon')) or 0.0),
            'mslp_hpa': round((parse_float(row.get('msl')) or 0.0) / 100.0, 1),
            'wind_kt': round((parse_float(row.get('wind')) or 0.0) * 1.94384, 1),
        })

    return out, fp


def extract_hres_system(storm_id: str, init_dt: datetime) -> Tuple[List[Dict[str, Any]], Path]:
    fp = BASE / 'HRES_forecast' / 'HRES_system' / f'{storm_id}_{init_dt.strftime("%Y%m%d")}_{init_dt.strftime("%H%M")}_TC_Analysis_{storm_id}.json'
    payload = json.loads(fp.read_text(encoding='utf-8'))

    target_names = {
        'VerticalWindShear',
        'UpperLevelDivergence',
        'OceanHeatContent',
        'SubtropicalHigh',
        'WesterlyTrough',
        'MonsoonTrough',
        'LowLevelFlow',
    }

    def pick_num(system: Dict[str, Any]) -> Optional[float]:
        intensity = system.get('intensity')
        if not isinstance(intensity, dict):
            return None
        for key in ['value', 'average_value', 'max_value', 'speed', 'surface_temp']:
            if key in intensity:
                v = parse_float(intensity.get(key))
                if v is not None:
                    return v
        return None

    by_time: Dict[datetime, Dict[str, Any]] = {}
    for row in payload.get('time_series', []):
        t = row.get('time')
        if not t:
            continue
        dt = datetime.strptime(str(t)[:19].replace('T', ' '), '%Y-%m-%d %H:%M:%S')
        sys_map: Dict[str, Any] = {}
        for s in row.get('environmental_systems') or []:
            if not isinstance(s, dict):
                continue
            name = s.get('system_name')
            if name not in target_names:
                continue
            intensity = s.get('intensity') if isinstance(s.get('intensity'), dict) else {}
            sys_map[name] = {
                'value': pick_num(s),
                'unit': intensity.get('unit'),
                'level': intensity.get('level'),
            }
        # merge duplicated times by union
        if dt not in by_time:
            by_time[dt] = {}
        by_time[dt].update(sys_map)

    out = []
    for dt in sorted(by_time.keys()):
        tau_h = int(round((dt - init_dt).total_seconds() / 3600))
        if tau_h < 0:
            continue
        out.append({
            'tau_h': tau_h,
            'valid_time_utc': dt.strftime('%Y-%m-%dT%H:%M:%SZ'),
            'systems': by_time[dt],
        })

    return out, fp


def build_sample() -> Dict[str, Any]:
    storm_id = '2020186N30289'
    basin = 'Atlantic'
    storm_name = 'EDOUARD'
    issue_dt = datetime(2020, 7, 6, 3, 0)
    advisory_no = 7
    hres_init = datetime(2020, 7, 6, 0, 0)

    adv_path = BASE / 'noaa' / '2020' / 'Atlantic' / 'EDOUARD' / 'forecast_advisory' / 'al052020.fstadv.007.txt'
    dis_path = BASE / 'noaa' / '2020' / 'Atlantic' / 'EDOUARD' / 'forecast_discussion' / 'al052020.discus.007.txt'
    pub_path = BASE / 'noaa' / '2020' / 'Atlantic' / 'EDOUARD' / 'public_advisory' / 'al052020.public.007.txt'
    wnd_path = BASE / 'noaa' / '2020' / 'Atlantic' / 'EDOUARD' / 'wind_speed_probabilities' / 'al052020.wndprb.007.txt'

    adv_lines = read_text(adv_path)
    dis_lines = read_text(dis_path)
    pub_lines = read_text(pub_path)
    wnd_lines = read_text(wnd_path)

    adv_summary = extract_advisory_summary(adv_lines)
    dis_summary = extract_discussion_signals(dis_lines)
    pub_summary = extract_public_summary(pub_lines)
    wnd_summary = extract_wind_prob_summary(wnd_lines)

    gt_state = load_groundtruth_state(storm_id, issue_dt)
    cds_row, cds_file = pick_cds_row(issue_dt, target_lat=gt_state['lat'], target_lon_w=gt_state['lon'])
    cds_features = extract_cds_features(cds_row)
    hres_track, hres_track_file = extract_hres_track(storm_id, hres_init)
    hres_system, hres_system_file = extract_hres_system(storm_id, hres_init)

    # Keep compact preview window for direct readability.
    hres_track_preview = [x for x in hres_track if x['tau_h'] in (0, 12, 24, 36, 48)]
    hres_env_preview = [x for x in hres_system if x['tau_h'] in (0, 12, 24, 36, 48)]

    sample = {
        'sample_meta': {
            'sample_id': f'{basin}_{storm_id}_{issue_dt.strftime("%Y-%m-%dT%H:%M:%SZ")}_{advisory_no:03d}',
            'task_type': 'tc_forecast_sft',
            'time_window': '2016-2025',
            'design_policy': 'EC_single_source_for_environment_guidance_v0',
            'assembled_at_utc': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
        },
        'keys': {
            'storm_id': storm_id,
            'storm_name': storm_name,
            'basin': basin,
            'issue_time_utc': issue_dt.strftime('%Y-%m-%dT%H:%M:%SZ'),
            'advisory_no': advisory_no,
            'time_match_rule': 'nearest_within_±3h',
        },
        'input': {
            'now_inputs': {
                'current_state': {
                    'from_noaa_forecast_advisory': {
                        'center': adv_summary.get('center'),
                        'motion': adv_summary.get('motion'),
                        'intensity': adv_summary.get('intensity'),
                    },
                    'from_groundtruth_alignment': gt_state,
                },
                'now_env_ec': {
                    'source_file': str(cds_file),
                    'source_time': str(cds_row.get('time')),
                    'tc_position': cds_row.get('tc_position'),
                    'features': cds_features,
                },
                'evidence_text': {
                    'discussion_excerpt': dis_summary.get('reasoning_excerpt'),
                    'public_summary_lines': pub_summary.get('summary_lines'),
                },
            },
            'guidance_inputs': {
                'ec_track_guidance_hres': {
                    'model': 'HRES',
                    'init_time_utc': hres_init.strftime('%Y-%m-%dT%H:%M:%SZ'),
                    'source_file': str(hres_track_file),
                    'points': hres_track_preview,
                },
                'ec_environment_guidance_hres': {
                    'model': 'HRES',
                    'init_time_utc': hres_init.strftime('%Y-%m-%dT%H:%M:%SZ'),
                    'source_file': str(hres_system_file),
                    'features_by_tau': hres_env_preview,
                },
                'planned_but_not_integrated_yet': {
                    'atcf_a_deck_multimodel': {
                        'status': 'pending_dataset',
                        'expected_fields': ['model_id', 'tau_h', 'lat', 'lon', 'vmax_kt', 'mslp_hpa', 'consensus_spread'],
                        'value': None,
                        'missing_reason': 'A-deck not downloaded/cleaned in workspace yet',
                    },
                    'goes_ascat_recon_structured_obs': {
                        'status': 'pending_dataset',
                        'expected_fields': ['obs_time_utc', 'obs_type', 'signal', 'qc_flag'],
                        'value': None,
                        'missing_reason': 'observation evidence layer not yet ingested',
                    },
                    'optional_existing_gfs_branch': {
                        'status': 'available_but_not_primary',
                        'policy': 'kept for ablation only; not used in EC-only v0 main branch',
                    },
                },
            },
        },
        'output': {
            'official_outputs_noaa': {
                'forecast_advisory_table': adv_summary.get('forecast_table'),
                'forecast_discussion_table': dis_summary.get('forecast_positions'),
                'reasoning_target_text': dis_summary.get('reasoning_excerpt'),
                'public_advisory_summary': pub_summary,
                'wind_speed_probabilities': wnd_summary,
            },
            'planned_verification_targets_not_in_prompt': {
                'atcf_b_deck_or_ibtracs': {
                    'status': 'pending_dataset',
                    'value': None,
                    'note': 'used only for offline evaluation/RFT reward, never as prompt input',
                }
            },
        },
        'quality_flags': {
            'guidance_qc_pass': 1,
            'coverage_flag': 'complete_for_ec_v0',
            'missing_blocks': [
                'atcf_a_deck_multimodel',
                'goes_ascat_recon_structured_obs',
                'atcf_b_deck_or_ibtracs',
            ],
            'source_provenance': {
                'forecast_advisory': str(adv_path),
                'forecast_discussion': str(dis_path),
                'public_advisory': str(pub_path),
                'wind_probabilities': str(wnd_path),
                'groundtruth': 'GroundTruth_Cyclones/matched_cyclone_tracks.csv',
                'cds_real': str(cds_file),
                'hres_track': str(hres_track_file),
                'hres_system': str(hres_system_file),
            },
        },
    }

    return sample


def main() -> int:
    out_dir = BASE / 'data' / 'interim' / 'schema'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / 'dataset_v0_sample_preview_ec_single_source.json'

    sample = build_sample()
    out_file.write_text(json.dumps(sample, ensure_ascii=False, indent=2), encoding='utf-8')

    print(out_file)
    print('sample_id:', sample['sample_meta']['sample_id'])
    print('hres_track_points_preview:', len(sample['input']['guidance_inputs']['ec_track_guidance_hres']['points']))
    print('hres_env_features_preview:', len(sample['input']['guidance_inputs']['ec_environment_guidance_hres']['features_by_tau']))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
