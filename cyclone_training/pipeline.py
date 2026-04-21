"""Convenience wrapper for sequential SFT -> GRPO runs."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

from .config import GRPOExperimentConfig, load_grpo_config, load_sft_config
from .grpo import run_grpo
from .sft import run_sft
from .utils import configure_logging, ensure_dir, write_json


def _override_stage_outputs(
    sft_config,
    grpo_config,
    run_root: Path,
) -> tuple:
    run_root = ensure_dir(run_root.resolve())
    sft_config = deepcopy(sft_config)
    grpo_config = deepcopy(grpo_config)

    sft_config.trainer.output_dir = run_root / "sft"
    sft_config.trainer.run_name = f"{sft_config.trainer.run_name}_{run_root.name}"
    grpo_config.trainer.output_dir = run_root / "grpo"
    grpo_config.trainer.run_name = f"{grpo_config.trainer.run_name}_{run_root.name}"
    return sft_config, grpo_config


def _write_pipeline_manifest(
    run_root: Path,
    *,
    sft_config_path: Path,
    grpo_config_path: Path,
    sft_config,
    grpo_config,
    sft_adapter_dir: Path | None = None,
    grpo_adapter_dir: Path | None = None,
) -> None:
    payload = {
        "run_root": str(run_root),
        "source_configs": {
            "sft": str(sft_config_path),
            "grpo": str(grpo_config_path),
        },
        "datasets": {
            "sft": {
                "dataset_root": str(sft_config.data.dataset_root),
                "train_file": str(sft_config.data.sft_train_file),
                "eval_file": str(sft_config.data.sft_eval_file) if sft_config.data.sft_eval_file is not None else None,
            },
            "grpo": {
                "dataset_root": str(grpo_config.data.dataset_root),
                "train_file": str(grpo_config.data.rl_train_file) if grpo_config.data.rl_train_file is not None else None,
                "eval_file": str(grpo_config.data.rl_eval_file) if grpo_config.data.rl_eval_file is not None else None,
            },
        },
        "effective_outputs": {
            "sft": str(sft_config.trainer.output_dir),
            "grpo": str(grpo_config.trainer.output_dir),
        },
        "effective_run_names": {
            "sft": sft_config.trainer.run_name,
            "grpo": grpo_config.trainer.run_name,
        },
        "artifacts": {
            "sft_adapter_dir": str(sft_adapter_dir) if sft_adapter_dir is not None else None,
            "grpo_adapter_dir": str(grpo_adapter_dir) if grpo_adapter_dir is not None else None,
        },
    }
    write_json(run_root / "pipeline_manifest.json", payload)


def run_sft_then_grpo(
    sft_config_path: str | Path,
    grpo_config_path: str | Path,
    run_root: str | Path | None = None,
) -> tuple[Path, Path]:
    """Run SFT first, then seed GRPO from the saved SFT adapter if needed."""
    sft_config = load_sft_config(sft_config_path)
    grpo_config = load_grpo_config(grpo_config_path)
    sft_config_path = Path(sft_config_path).resolve()
    grpo_config_path = Path(grpo_config_path).resolve()

    if run_root is not None:
        run_root = Path(run_root)
        sft_config, grpo_config = _override_stage_outputs(sft_config, grpo_config, run_root)
        _write_pipeline_manifest(
            run_root,
            sft_config_path=sft_config_path,
            grpo_config_path=grpo_config_path,
            sft_config=sft_config,
            grpo_config=grpo_config,
        )

    sft_adapter_dir = run_sft(sft_config)

    if grpo_config.trainer.adapter_init_path is None:
        grpo_config = deepcopy(grpo_config)
        grpo_config.trainer.adapter_init_path = sft_adapter_dir

    grpo_adapter_dir = run_grpo(grpo_config)
    if run_root is not None:
        _write_pipeline_manifest(
            Path(run_root),
            sft_config_path=sft_config_path,
            grpo_config_path=grpo_config_path,
            sft_config=sft_config,
            grpo_config=grpo_config,
            sft_adapter_dir=sft_adapter_dir,
            grpo_adapter_dir=grpo_adapter_dir,
        )
    return sft_adapter_dir, grpo_adapter_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run SFT followed by GRPO")
    parser.add_argument("--sft-config", required=True)
    parser.add_argument("--grpo-config", required=True)
    parser.add_argument("--run-root")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    configure_logging(verbose=args.verbose)
    run_sft_then_grpo(args.sft_config, args.grpo_config, run_root=args.run_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
