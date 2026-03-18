#!/usr/bin/env python3
"""Convert structured sample preview to paragraph-style input/output sample."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


def format_num(x: Any, ndigits: int = 1) -> str:
    if x is None:
        return "未知"
    try:
        return str(round(float(x), ndigits))
    except Exception:
        return str(x)


def join_nonempty(parts: List[str], sep: str = "") -> str:
    return sep.join([p for p in parts if p])


def build_input_text(sample: Dict[str, Any]) -> str:
    keys = sample.get("keys", {})
    now = sample.get("input", {}).get("now_inputs", {})
    guide = sample.get("input", {}).get("guidance_inputs", {})

    state_a = now.get("current_state", {}).get("from_noaa_forecast_advisory", {})
    center = state_a.get("center", {})
    motion = state_a.get("motion", {})
    intensity = state_a.get("intensity", {})

    gt = now.get("current_state", {}).get("from_groundtruth_alignment", {})
    env = now.get("now_env_ec", {})
    feats = env.get("features", {})

    shear = feats.get("vertical_wind_shear", {})
    div = feats.get("upper_level_divergence", {})
    ohc = feats.get("ocean_heat_content_or_sst", {})
    sth = feats.get("subtropical_high", {})
    wt = feats.get("westerly_trough", {})
    mt = feats.get("monsoon_trough", {})

    discussion_excerpt = now.get("evidence_text", {}).get("discussion_excerpt", "")
    public_lines = now.get("evidence_text", {}).get("public_summary_lines", [])

    hres_track = guide.get("ec_track_guidance_hres", {})
    hres_points = hres_track.get("points", [])
    hres_env = guide.get("ec_environment_guidance_hres", {}).get("features_by_tau", [])

    planned = guide.get("planned_but_not_integrated_yet", {})

    p1 = (
        f"样本对应 {keys.get('basin')} 盆地风暴 {keys.get('storm_name')}（storm_id={keys.get('storm_id')}），"
        f"预报时次为 {keys.get('issue_time_utc')}，advisory 编号 {keys.get('advisory_no')}。"
    )

    p2 = (
        f"当前状态（来自 NOAA advisory）显示中心位于 {format_num(center.get('lat'))}N, "
        f"{format_num(center.get('lon'))}W，最大持续风 {intensity.get('max_wind_kt', '未知')} kt，"
        f"最低中心气压 {intensity.get('min_pressure_mb', '未知')} mb，"
        f"移动方向/速度为 {motion.get('direction_deg', '未知')}°、{motion.get('speed_kt', '未知')} kt。"
        f" 与 GroundTruth 对齐点时间为 {gt.get('matched_datetime_utc')}。"
    )

    p3 = (
        f"EC 实况环境（CDS_real）在同一时次给出：垂直风切变约 {format_num(shear.get('value'))} {shear.get('unit', '')}，"
        f"高层散度约 {format_num(div.get('value'))} {div.get('unit', '')}，"
        f"海温/海洋热含量指标约 {format_num(ohc.get('value'))} {ohc.get('unit', '')}；"
        f"同时识别到副热带高压（{format_num(sth.get('value'))} {sth.get('unit', '')}）、"
        f"西风槽（{format_num(wt.get('value'))} {wt.get('unit', '')}）和季风槽（{format_num(mt.get('value'))} {mt.get('unit', '')}）信号。"
    )

    track_sent = []
    for x in hres_points:
        track_sent.append(
            f"tau{int(x.get('tau_h', 0))}h: ({format_num(x.get('lat'))}, {format_num(x.get('lon'))}), "
            f"{format_num(x.get('mslp_hpa'))} hPa, {format_num(x.get('wind_kt'))} kt"
        )
    p4 = (
        f"EC guidance（HRES）使用 {hres_track.get('init_time_utc')} 初始化。"
        f"轨迹-强度关键点为：" + "；".join(track_sent) + "。"
    ) if track_sent else "EC guidance（HRES）轨迹点缺失。"

    env_sent = []
    for x in hres_env:
        systems = x.get("systems", {})
        vs = systems.get("VerticalWindShear", {})
        ud = systems.get("UpperLevelDivergence", {})
        oc = systems.get("OceanHeatContent", {})
        env_sent.append(
            f"tau{int(x.get('tau_h', 0))}h: shear={format_num(vs.get('value'))} {vs.get('unit', '')}, "
            f"div={format_num(ud.get('value'))} {ud.get('unit', '')}, "
            f"ohc/sst={format_num(oc.get('value'))} {oc.get('unit', '')}"
        )
    p5 = ("EC 环境指导关键点为：" + "；".join(env_sent) + "。") if env_sent else "EC 环境指导缺失。"

    missing_blocks = []
    for k, v in planned.items():
        if isinstance(v, dict):
            miss = v.get("missing_reason")
            if miss:
                missing_blocks.append(f"{k}: {miss}")

    p6 = (
        "证据文本（discussion 摘要）显示：" + discussion_excerpt[:600].replace("\n", " ") + "..."
        if discussion_excerpt
        else "discussion 摘要缺失。"
    )

    p7 = (
        "当前未接入但已预留的数据块包括：" + "；".join(missing_blocks) + "。"
        if missing_blocks
        else "当前无未接入预留块。"
    )

    p8 = (
        "public advisory 摘要信息：" + "；".join(public_lines) + "。"
        if public_lines
        else "public advisory 摘要缺失。"
    )

    return "\n\n".join([p1, p2, p3, p4, p5, p6, p7, p8])


def build_output_text(sample: Dict[str, Any]) -> str:
    out = sample.get("output", {}).get("official_outputs_noaa", {})

    adv_table = out.get("forecast_advisory_table", [])
    dis_table = out.get("forecast_discussion_table", [])
    text = out.get("reasoning_target_text", "")
    pub = out.get("public_advisory_summary", {})
    wind = out.get("wind_speed_probabilities", {})

    adv_lines = []
    for x in adv_table:
        adv_lines.append(
            f"{x.get('valid_day')}/{x.get('valid_hhmmz')}Z: ({format_num(x.get('lat'))}, {format_num(x.get('lon'))}), {x.get('vmax_kt')} kt"
        )

    dis_lines = []
    for x in dis_table:
        dis_lines.append(
            f"tau{int(x.get('tau_h', 0))}h -> ({format_num(x.get('lat'))}, {format_num(x.get('lon'))}), {x.get('vmax_kt')} kt"
        )

    p1 = "官方预报（forecast advisory）给出的路径与强度为：" + "；".join(adv_lines) + "。" if adv_lines else "官方预报表为空。"
    p2 = "官方讨论（forecast discussion）的结构化路径为：" + "；".join(dis_lines) + "。" if dis_lines else "官方讨论路径表为空。"
    p3 = "官方推理文本目标为：" + text[:900].replace("\n", " ") + "..." if text else "官方推理文本缺失。"

    summary_lines = pub.get("summary_lines", [])
    p4 = (
        "官方公众通告摘要：" + "；".join(summary_lines) + "。"
        if summary_lines
        else "官方公众通告摘要缺失。"
    )

    p5 = (
        f"风速概率产品编号 {wind.get('product_number', '未知')}，"
        f"当前是否含地点行：{wind.get('has_location_rows')}，地点行数：{wind.get('location_rows')}。"
    )

    return "\n\n".join([p1, p2, p3, p4, p5])


def main() -> int:
    in_file = Path('data/interim/schema/dataset_v0_sample_preview_ec_single_source.json')
    out_json = Path('data/interim/schema/dataset_v0_sample_preview_ec_single_source_text_io.json')
    out_jsonl = Path('data/interim/schema/dataset_v0_sample_preview_ec_single_source_text_io.jsonl')
    out_md = Path('data/interim/schema/dataset_v0_sample_preview_ec_single_source_text_io.md')

    sample = json.loads(in_file.read_text(encoding='utf-8'))

    input_text = build_input_text(sample)
    output_text = build_output_text(sample)

    item = {
        'sample_id': sample.get('sample_meta', {}).get('sample_id'),
        'input': input_text,
        'output': output_text,
    }

    out_json.write_text(json.dumps(item, ensure_ascii=False, indent=2), encoding='utf-8')
    out_jsonl.write_text(json.dumps(item, ensure_ascii=False) + '\n', encoding='utf-8')

    md = (
        f"# Sample {item['sample_id']}\n\n"
        f"## INPUT\n\n{input_text}\n\n"
        f"## OUTPUT\n\n{output_text}\n"
    )
    out_md.write_text(md, encoding='utf-8')

    print(out_json)
    print(out_jsonl)
    print(out_md)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
