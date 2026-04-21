#!/usr/bin/env python3
"""Analyze sample-level differences between two forecast evaluation reports."""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


STORM_ID_PATTERN = re.compile(r"^\d{7}[NS]\d{5}$")
INTENSITY_PATTERN = re.compile(r"Intensity\s+(\d+)\s+kt")
SPREAD_PATTERN = re.compile(r"spread\s+([0-9.]+)\s*km/([0-9.]+)\s*kt")
SHEAR_PATTERN = re.compile(r"Vertical wind shear:\s*([0-9.]+)\s*m/s\s*\(([^)]+)\)")
COVERAGE_PATTERN = re.compile(r"Coverage:\s*([^\n]+)")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _parse_sample_id(sample_id: str) -> dict[str, str]:
    parts = sample_id.split("_")
    storm_index = next(index for index, part in enumerate(parts) if STORM_ID_PATTERN.match(part))
    basin = "_".join(parts[:storm_index])
    storm_id = parts[storm_index]
    issue_time = parts[storm_index + 1]
    sample_index = parts[storm_index + 2]
    return {
        "basin": basin,
        "storm_id": storm_id,
        "issue_time": issue_time,
        "sample_index": sample_index,
    }


def _mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def _median_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return float(statistics.median(values))


def _feature_row(row: dict[str, Any]) -> dict[str, Any]:
    sample_meta = _parse_sample_id(str(row["sample_id"]))
    prompt = str(row["messages"][1]["content"])

    init_intensity_match = INTENSITY_PATTERN.search(prompt)
    init_intensity = int(init_intensity_match.group(1)) if init_intensity_match else None
    if init_intensity is None:
        intensity_bin = "unknown"
    elif init_intensity <= 34:
        intensity_bin = "<=34kt"
    elif init_intensity <= 63:
        intensity_bin = "35-63kt"
    else:
        intensity_bin = ">=64kt"

    spreads = [float(spatial) for spatial, _ in SPREAD_PATTERN.findall(prompt)]
    mean_spread_km = _mean_or_none(spreads)
    if mean_spread_km is None:
        spread_bin = "unknown"
    elif mean_spread_km < 1000:
        spread_bin = "<1000km"
    elif mean_spread_km < 1800:
        spread_bin = "1000-1800km"
    else:
        spread_bin = ">=1800km"

    shear_match = SHEAR_PATTERN.search(prompt)
    shear_ms = float(shear_match.group(1)) if shear_match else None
    shear_desc = shear_match.group(2).strip() if shear_match else "unknown"

    coverage_match = COVERAGE_PATTERN.search(prompt)
    coverage = coverage_match.group(1).strip() if coverage_match else "unknown"
    recon = "Recon available" if "Recon available" in coverage else "Recon missing"

    future_track = list(row["verification"].get("future_best_track") or [])
    max_future_intensity = max(
        (float(point["vmax_kt"]) for point in future_track),
        default=float(init_intensity or 0.0),
    )
    final_future_intensity = (
        float(future_track[-1]["vmax_kt"]) if future_track else float(init_intensity or 0.0)
    )
    delta_peak_intensity = max_future_intensity - float(init_intensity or 0.0)
    delta_end_intensity = final_future_intensity - float(init_intensity or 0.0)
    if delta_peak_intensity >= 15:
        trend = "strengthen>=15"
    elif delta_peak_intensity <= -10:
        trend = "weaken<=-10"
    else:
        trend = "steady/mixed"

    return {
        **sample_meta,
        "init_intensity_kt": init_intensity,
        "intensity_bin": intensity_bin,
        "mean_guidance_spread_km": mean_spread_km,
        "spread_bin": spread_bin,
        "shear_ms": shear_ms,
        "shear_desc": shear_desc,
        "coverage": coverage,
        "recon": recon,
        "trend": trend,
        "delta_peak_intensity_kt": delta_peak_intensity,
        "delta_end_intensity_kt": delta_end_intensity,
    }


def _group_rows(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[key])].append(row)

    summaries: list[dict[str, Any]] = []
    for group_value, group_rows in grouped.items():
        left_wins = sum(1 for row in group_rows if row["winner"] == "left")
        right_wins = sum(1 for row in group_rows if row["winner"] == "right")
        ties = sum(1 for row in group_rows if row["winner"] == "tie")
        decisive = left_wins + right_wins
        reward_deltas = [float(row["delta_reward"]) for row in group_rows]
        track_deltas = [
            float(row["delta_track_error_km"])
            for row in group_rows
            if row["delta_track_error_km"] is not None
        ]
        intensity_deltas = [
            float(row["delta_intensity_error_kt"])
            for row in group_rows
            if row["delta_intensity_error_kt"] is not None
        ]
        coverage_deltas = [
            float(row["delta_match_coverage_vs_target"])
            for row in group_rows
            if row["delta_match_coverage_vs_target"] is not None
        ]
        summaries.append(
            {
                "group": group_value,
                "sample_count": len(group_rows),
                "mean_delta_reward": _mean_or_none(reward_deltas),
                "median_delta_reward": _median_or_none(reward_deltas),
                "left_wins": left_wins,
                "right_wins": right_wins,
                "ties": ties,
                "left_win_rate_all_samples": left_wins / len(group_rows) if group_rows else None,
                "left_win_rate_decisive_only": (
                    left_wins / decisive if decisive else None
                ),
                "mean_delta_match_coverage": _mean_or_none(coverage_deltas),
                "mean_delta_track_error_km": _mean_or_none(track_deltas),
                "mean_delta_intensity_error_kt": _mean_or_none(intensity_deltas),
            }
        )
    summaries.sort(key=lambda item: (item["mean_delta_reward"] or 0.0, item["sample_count"]), reverse=True)
    return summaries


def _storm_summaries(rows: list[dict[str, Any]], *, min_samples: int) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["basin"]), str(row["storm_id"]))].append(row)

    summaries: list[dict[str, Any]] = []
    for (basin, storm_id), group_rows in grouped.items():
        if len(group_rows) < min_samples:
            continue
        left_wins = sum(1 for row in group_rows if row["winner"] == "left")
        right_wins = sum(1 for row in group_rows if row["winner"] == "right")
        ties = sum(1 for row in group_rows if row["winner"] == "tie")
        decisive = left_wins + right_wins
        reward_deltas = [float(row["delta_reward"]) for row in group_rows]
        summaries.append(
            {
                "basin": basin,
                "storm_id": storm_id,
                "sample_count": len(group_rows),
                "mean_delta_reward": _mean_or_none(reward_deltas),
                "median_delta_reward": _median_or_none(reward_deltas),
                "left_wins": left_wins,
                "right_wins": right_wins,
                "ties": ties,
                "left_win_rate_all_samples": left_wins / len(group_rows) if group_rows else None,
                "left_win_rate_decisive_only": (
                    left_wins / decisive if decisive else None
                ),
            }
        )
    summaries.sort(key=lambda item: (item["mean_delta_reward"] or 0.0, item["sample_count"]), reverse=True)
    return summaries


def _winner_mechanisms(rows: list[dict[str, Any]], winner: str) -> dict[str, Any]:
    group_rows = [row for row in rows if row["winner"] == winner]
    return {
        "sample_count": len(group_rows),
        "mean_delta_reward": _mean_or_none([float(row["delta_reward"]) for row in group_rows]),
        "mean_delta_match_coverage": _mean_or_none(
            [
                float(row["delta_match_coverage_vs_target"])
                for row in group_rows
                if row["delta_match_coverage_vs_target"] is not None
            ]
        ),
        "median_delta_match_coverage": _median_or_none(
            [
                float(row["delta_match_coverage_vs_target"])
                for row in group_rows
                if row["delta_match_coverage_vs_target"] is not None
            ]
        ),
        "mean_delta_track_error_km": _mean_or_none(
            [
                float(row["delta_track_error_km"])
                for row in group_rows
                if row["delta_track_error_km"] is not None
            ]
        ),
        "median_delta_track_error_km": _median_or_none(
            [
                float(row["delta_track_error_km"])
                for row in group_rows
                if row["delta_track_error_km"] is not None
            ]
        ),
        "mean_delta_intensity_error_kt": _mean_or_none(
            [
                float(row["delta_intensity_error_kt"])
                for row in group_rows
                if row["delta_intensity_error_kt"] is not None
            ]
        ),
        "median_delta_intensity_error_kt": _median_or_none(
            [
                float(row["delta_intensity_error_kt"])
                for row in group_rows
                if row["delta_intensity_error_kt"] is not None
            ]
        ),
    }


def _representative_cases(rows: list[dict[str, Any]], *, top_n: int) -> dict[str, list[dict[str, Any]]]:
    ranked = sorted(rows, key=lambda row: float(row["delta_reward"]), reverse=True)
    keep_keys = [
        "sample_id",
        "basin",
        "storm_id",
        "issue_time",
        "init_intensity_kt",
        "intensity_bin",
        "trend",
        "mean_guidance_spread_km",
        "spread_bin",
        "recon",
        "shear_desc",
        "delta_reward",
        "delta_match_coverage_vs_target",
        "delta_track_error_km",
        "delta_intensity_error_kt",
        "left_reward",
        "right_reward",
    ]

    def trim(case_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [{key: row.get(key) for key in keep_keys} for row in case_rows]

    return {
        "top_left_wins": trim(ranked[:top_n]),
        "top_right_wins": trim(list(reversed(ranked[-top_n:]))),
    }


def _build_markdown(report: dict[str, Any]) -> str:
    left_label = report["left_label"]
    right_label = report["right_label"]
    overall = report["overall"]
    sections: list[str] = []
    sections.append(f"# {left_label} vs {right_label} Case Analysis")
    sections.append("")
    sections.append("## Overall")
    sections.append("")
    sections.append(
        f"- Sample count: `{overall['sample_count']}` | "
        f"`{left_label}` wins: `{overall['left_wins']}` | "
        f"`{right_label}` wins: `{overall['right_wins']}` | ties: `{overall['ties']}`"
    )
    sections.append(
        f"- Mean reward delta (`{left_label} - {right_label}`): "
        f"`{overall['mean_delta_reward']:.6f}`"
    )
    sections.append(
        f"- Mean track-error delta: `{overall['mean_delta_track_error_km']:.2f} km` | "
        f"Mean intensity-error delta: `{overall['mean_delta_intensity_error_kt']:.3f} kt` | "
        f"Mean coverage delta: `{overall['mean_delta_match_coverage']:.4f}`"
    )
    sections.append(
        f"- Identical outputs: `{overall['identical_generation_count']}` | "
        f"Identical rewards: `{overall['identical_reward_count']}` | "
        f"Identical schema: `{overall['identical_schema_count']}`"
    )
    sections.append("")
    sections.append("## Mechanism")
    sections.append("")
    for winner_key, label in [("left_win_cases", left_label), ("right_win_cases", right_label)]:
        winner = report["winner_mechanisms"][winner_key]
        sections.append(
            f"- When `{label}` wins ({winner['sample_count']} cases), mean reward delta is "
            f"`{winner['mean_delta_reward']:.4f}`, coverage delta is "
            f"`{winner['mean_delta_match_coverage']:.4f}`, track-error delta is "
            f"`{winner['mean_delta_track_error_km']:.2f} km`, intensity-error delta is "
            f"`{winner['mean_delta_intensity_error_kt']:.3f} kt`."
        )
    sections.append("")

    def table_section(title: str, rows: list[dict[str, Any]], limit: int = 8) -> None:
        sections.append(f"## {title}")
        sections.append("")
        sections.append(
            "| group | n | mean_delta_reward | left_wins | right_wins | ties |"
        )
        sections.append("| --- | ---: | ---: | ---: | ---: | ---: |")
        for row in rows[:limit]:
            sections.append(
                f"| `{row['group']}` | {row['sample_count']} | {row['mean_delta_reward']:.6f} | "
                f"{row['left_wins']} | {row['right_wins']} | {row['ties']} |"
            )
        sections.append("")

    table_section("By Basin", report["group_summaries"]["basin"])
    table_section("By Intensity Bin", report["group_summaries"]["intensity_bin"])
    table_section("By Trend", report["group_summaries"]["trend"])
    table_section("By Guidance Spread", report["group_summaries"]["spread_bin"])
    table_section("By Recon Availability", report["group_summaries"]["recon"])
    table_section("By Shear", report["group_summaries"]["shear_desc"])

    sections.append("## Storms")
    sections.append("")
    sections.append("| storm | n | mean_delta_reward | left_wins | right_wins | ties |")
    sections.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for row in report["top_storms_left"][:8]:
        sections.append(
            f"| `{row['basin']} {row['storm_id']}` | {row['sample_count']} | "
            f"{row['mean_delta_reward']:.6f} | {row['left_wins']} | {row['right_wins']} | {row['ties']} |"
        )
    sections.append("")
    sections.append("Worst storms for left model:")
    sections.append("")
    sections.append("| storm | n | mean_delta_reward | left_wins | right_wins | ties |")
    sections.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for row in report["top_storms_right"][:8]:
        sections.append(
            f"| `{row['basin']} {row['storm_id']}` | {row['sample_count']} | "
            f"{row['mean_delta_reward']:.6f} | {row['left_wins']} | {row['right_wins']} | {row['ties']} |"
        )
    sections.append("")

    sections.append("## Representative Cases")
    sections.append("")
    sections.append(f"Top `{left_label}` wins:")
    sections.append("")
    sections.append("| sample_id | delta_reward | basin | intensity | trend | spread | recon | shear |")
    sections.append("| --- | ---: | --- | --- | --- | --- | --- | --- |")
    for row in report["representative_cases"]["top_left_wins"][:8]:
        sections.append(
            f"| `{row['sample_id']}` | {row['delta_reward']:.6f} | `{row['basin']}` | "
            f"`{row['intensity_bin']}` | `{row['trend']}` | `{row['spread_bin']}` | "
            f"`{row['recon']}` | `{row['shear_desc']}` |"
        )
    sections.append("")
    sections.append(f"Top `{right_label}` wins:")
    sections.append("")
    sections.append("| sample_id | delta_reward | basin | intensity | trend | spread | recon | shear |")
    sections.append("| --- | ---: | --- | --- | --- | --- | --- | --- |")
    for row in report["representative_cases"]["top_right_wins"][:8]:
        sections.append(
            f"| `{row['sample_id']}` | {row['delta_reward']:.6f} | `{row['basin']}` | "
            f"`{row['intensity_bin']}` | `{row['trend']}` | `{row['spread_bin']}` | "
            f"`{row['recon']}` | `{row['shear_desc']}` |"
        )
    sections.append("")
    return "\n".join(sections)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Analyze sample-level case differences between two forecast evaluation reports."
    )
    parser.add_argument("--left-report", required=True)
    parser.add_argument("--right-report", required=True)
    parser.add_argument("--rl-dataset", required=True)
    parser.add_argument("--left-label")
    parser.add_argument("--right-label")
    parser.add_argument("--storm-min-samples", type=int, default=8)
    parser.add_argument("--top-n", type=int, default=12)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args(argv)

    left_report = _read_json(Path(args.left_report).resolve())
    right_report = _read_json(Path(args.right_report).resolve())
    rl_rows = _read_jsonl(Path(args.rl_dataset).resolve())

    left_label = args.left_label or str(left_report.get("model_label") or "left")
    right_label = args.right_label or str(right_report.get("model_label") or "right")

    left_samples = {str(sample["sample_id"]): sample for sample in left_report["samples"]}
    right_samples = {str(sample["sample_id"]): sample for sample in right_report["samples"]}

    rows: list[dict[str, Any]] = []
    identical_generation_count = 0
    identical_reward_count = 0
    identical_schema_count = 0
    for raw_row in rl_rows:
        sample_id = str(raw_row["sample_id"])
        left_sample = left_samples[sample_id]
        right_sample = right_samples[sample_id]

        if str(left_sample["generated"]) == str(right_sample["generated"]):
            identical_generation_count += 1

        left_reward = float(left_sample["reward_details"]["reward"])
        right_reward = float(right_sample["reward_details"]["reward"])
        if left_reward == right_reward:
            identical_reward_count += 1
        delta_reward = left_reward - right_reward
        winner = "left" if delta_reward > 0 else "right" if delta_reward < 0 else "tie"

        if dict(left_sample["schema"]) == dict(right_sample["schema"]):
            identical_schema_count += 1

        left_track = left_sample["reward_details"]["mean_track_error_km"]
        right_track = right_sample["reward_details"]["mean_track_error_km"]
        left_intensity = left_sample["reward_details"]["mean_intensity_error_kt"]
        right_intensity = right_sample["reward_details"]["mean_intensity_error_kt"]
        left_coverage = left_sample["reward_details"]["match_coverage_vs_target"]
        right_coverage = right_sample["reward_details"]["match_coverage_vs_target"]

        rows.append(
            {
                "sample_id": sample_id,
                **_feature_row(raw_row),
                "winner": winner,
                "left_reward": left_reward,
                "right_reward": right_reward,
                "delta_reward": delta_reward,
                "delta_track_error_km": (
                    None if left_track is None or right_track is None else left_track - right_track
                ),
                "delta_intensity_error_kt": (
                    None
                    if left_intensity is None or right_intensity is None
                    else left_intensity - right_intensity
                ),
                "delta_match_coverage_vs_target": (
                    None if left_coverage is None or right_coverage is None else left_coverage - right_coverage
                ),
            }
        )

    overall = {
        "sample_count": len(rows),
        "left_wins": sum(1 for row in rows if row["winner"] == "left"),
        "right_wins": sum(1 for row in rows if row["winner"] == "right"),
        "ties": sum(1 for row in rows if row["winner"] == "tie"),
        "mean_delta_reward": _mean_or_none([float(row["delta_reward"]) for row in rows]),
        "median_delta_reward": _median_or_none([float(row["delta_reward"]) for row in rows]),
        "mean_delta_track_error_km": _mean_or_none(
            [
                float(row["delta_track_error_km"])
                for row in rows
                if row["delta_track_error_km"] is not None
            ]
        ),
        "mean_delta_intensity_error_kt": _mean_or_none(
            [
                float(row["delta_intensity_error_kt"])
                for row in rows
                if row["delta_intensity_error_kt"] is not None
            ]
        ),
        "mean_delta_match_coverage": _mean_or_none(
            [
                float(row["delta_match_coverage_vs_target"])
                for row in rows
                if row["delta_match_coverage_vs_target"] is not None
            ]
        ),
        "identical_generation_count": identical_generation_count,
        "identical_reward_count": identical_reward_count,
        "identical_schema_count": identical_schema_count,
    }

    storm_summaries = _storm_summaries(rows, min_samples=args.storm_min_samples)
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "left_label": left_label,
        "right_label": right_label,
        "left_report": str(Path(args.left_report).resolve()),
        "right_report": str(Path(args.right_report).resolve()),
        "rl_dataset": str(Path(args.rl_dataset).resolve()),
        "overall": overall,
        "winner_mechanisms": {
            "left_win_cases": _winner_mechanisms(rows, "left"),
            "right_win_cases": _winner_mechanisms(rows, "right"),
        },
        "group_summaries": {
            "basin": _group_rows(rows, "basin"),
            "intensity_bin": _group_rows(rows, "intensity_bin"),
            "trend": _group_rows(rows, "trend"),
            "spread_bin": _group_rows(rows, "spread_bin"),
            "recon": _group_rows(rows, "recon"),
            "shear_desc": _group_rows(rows, "shear_desc"),
        },
        "top_storms_left": storm_summaries[: args.top_n],
        "top_storms_right": list(reversed(storm_summaries[-args.top_n :])),
        "representative_cases": _representative_cases(rows, top_n=args.top_n),
    }

    _write_json(Path(args.output_json).resolve(), report)
    Path(args.output_md).resolve().write_text(_build_markdown(report), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
