"""SFT training entrypoint backed by Unsloth + TRL."""

from __future__ import annotations

import argparse
import logging
from copy import deepcopy
from pathlib import Path
from typing import Any

from .compat import prepare_trl_runtime
from .config import SFTExperimentConfig, load_sft_config
from .datasets import load_sft_datasets
from .modeling import load_adapter_weights, load_model_and_tokenizer, save_adapter
from .utils import (
    configure_logging,
    count_jsonl_records,
    ensure_dir,
    filter_supported_kwargs,
    to_jsonable,
    write_json,
)


LOGGER = logging.getLogger("cyclone_training.sft")


def _write_dataset_manifest(
    config: SFTExperimentConfig,
    train_count: int,
    eval_count: int,
    record_format: str,
) -> None:
    payload = {
        "dataset_root": str(config.data.dataset_root),
        "sft_train_file": str(config.data.sft_train_file),
        "sft_eval_file": str(config.data.sft_eval_file) if config.data.sft_eval_file is not None else None,
        "train_count": train_count,
        "eval_count": eval_count,
        "record_format": record_format,
        "sft_resample_fields": list(config.data.sft_resample_fields),
        "sft_resample_power": config.data.sft_resample_power,
        "sft_resample_max_multiplier": config.data.sft_resample_max_multiplier,
        "sft_resample_label_min_multipliers": dict(config.data.sft_resample_label_min_multipliers),
        "adapter_init_path": (
            str(config.trainer.adapter_init_path)
            if config.trainer.adapter_init_path is not None
            else None
        ),
    }
    write_json(config.trainer.output_dir / "dataset_manifest.json", payload)


def _override_stage_output(
    config: SFTExperimentConfig,
    run_root: Path,
) -> SFTExperimentConfig:
    run_root = ensure_dir(run_root.resolve())
    config = deepcopy(config)
    config.trainer.output_dir = run_root / "sft"
    config.trainer.run_name = f"{config.trainer.run_name}_{run_root.name}"
    return config


def _write_pipeline_manifest(
    run_root: Path,
    *,
    config_path: Path,
    sft_config: SFTExperimentConfig,
    sft_adapter_dir: Path | None = None,
) -> None:
    payload = {
        "run_root": str(run_root),
        "source_config": str(config_path),
        "datasets": {
            "sft": {
                "dataset_root": str(sft_config.data.dataset_root),
                "train_file": str(sft_config.data.sft_train_file),
                "eval_file": (
                    str(sft_config.data.sft_eval_file)
                    if sft_config.data.sft_eval_file is not None
                    else None
                ),
            }
        },
        "effective_output": str(sft_config.trainer.output_dir),
        "effective_run_name": sft_config.trainer.run_name,
        "adapter_init_path": (
            str(sft_config.trainer.adapter_init_path)
            if sft_config.trainer.adapter_init_path is not None
            else None
        ),
        "artifacts": {
            "sft_adapter_dir": str(sft_adapter_dir) if sft_adapter_dir is not None else None,
        },
    }
    write_json(run_root / "pipeline_manifest.json", payload)


def _import_trl_sft():
    prepare_trl_runtime()
    try:
        from trl import SFTTrainer
        from trl.trainer.sft_config import SFTConfig
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise ImportError("TRL is required for SFT training.") from exc
    return SFTTrainer, SFTConfig


def _supports_assistant_only(SFTConfig: Any) -> bool:
    return "assistant_only_loss" in getattr(SFTConfig, "__dataclass_fields__", {})


def _is_processor_like(processing_class: Any) -> bool:
    try:
        from transformers import ProcessorMixin
    except ImportError:  # pragma: no cover - runtime dependency
        return hasattr(processing_class, "tokenizer")
    return isinstance(processing_class, ProcessorMixin)


def _trainer_processing_kwargs(trainer_cls: Any, tokenizer: Any) -> dict[str, Any]:
    kwargs = {"processing_class": tokenizer, "tokenizer": tokenizer}
    return filter_supported_kwargs(trainer_cls, kwargs)


def _build_sft_args(
    config: SFTExperimentConfig,
    has_eval: bool,
    conversational_mode: bool,
    prompt_completion_mode: bool,
):
    SFTTrainer, SFTConfig = _import_trl_sft()
    if not has_eval:
        if config.trainer.load_best_model_at_end:
            raise ValueError("SFT load_best_model_at_end requires an eval dataset.")
        if config.trainer.early_stopping_patience is not None:
            raise ValueError("SFT early stopping requires an eval dataset.")
    args_kwargs: dict[str, Any] = {
        "output_dir": str(config.trainer.output_dir),
        "per_device_train_batch_size": config.trainer.per_device_train_batch_size,
        "per_device_eval_batch_size": config.trainer.per_device_eval_batch_size,
        "gradient_accumulation_steps": config.trainer.gradient_accumulation_steps,
        "learning_rate": config.trainer.learning_rate,
        "num_train_epochs": config.trainer.num_train_epochs,
        "max_steps": config.trainer.max_steps,
        "warmup_ratio": config.trainer.warmup_ratio,
        "weight_decay": config.trainer.weight_decay,
        "max_grad_norm": config.trainer.max_grad_norm,
        "lr_scheduler_type": config.trainer.lr_scheduler_type,
        "logging_steps": config.trainer.logging_steps,
        "save_steps": config.trainer.save_steps,
        "eval_steps": config.trainer.eval_steps,
        "save_total_limit": config.trainer.save_total_limit,
        "bf16": config.trainer.bf16,
        "fp16": config.trainer.fp16,
        "gradient_checkpointing": config.trainer.gradient_checkpointing,
        "report_to": config.trainer.report_to,
        "seed": config.trainer.seed,
        "packing": config.trainer.packing,
        "max_length": config.model.max_seq_length,
        "remove_unused_columns": False,
        "load_best_model_at_end": config.trainer.load_best_model_at_end,
    }
    if config.trainer.metric_for_best_model is not None:
        args_kwargs["metric_for_best_model"] = config.trainer.metric_for_best_model
    if config.trainer.greater_is_better is not None:
        args_kwargs["greater_is_better"] = config.trainer.greater_is_better
    strategy_value = "steps" if has_eval else "no"
    if "eval_strategy" in getattr(SFTConfig, "__dataclass_fields__", {}):
        args_kwargs["eval_strategy"] = strategy_value
    elif "evaluation_strategy" in getattr(SFTConfig, "__dataclass_fields__", {}):
        args_kwargs["evaluation_strategy"] = strategy_value
    if config.trainer.completion_only_loss is not None:
        args_kwargs["completion_only_loss"] = config.trainer.completion_only_loss
    elif prompt_completion_mode:
        args_kwargs["completion_only_loss"] = True

    if prompt_completion_mode:
        args_kwargs["assistant_only_loss"] = False
    elif not conversational_mode:
        args_kwargs["dataset_text_field"] = config.trainer.dataset_text_field
        if config.trainer.dataset_num_proc is not None:
            args_kwargs["dataset_num_proc"] = config.trainer.dataset_num_proc
    elif _supports_assistant_only(SFTConfig):
        args_kwargs["assistant_only_loss"] = config.trainer.assistant_only_loss
    args_kwargs = filter_supported_kwargs(SFTConfig, args_kwargs)
    return SFTTrainer, SFTConfig(**args_kwargs)


def _build_early_stopping_callback(config: SFTExperimentConfig) -> Any | None:
    patience = config.trainer.early_stopping_patience
    if patience is None:
        return None
    try:
        from transformers import EarlyStoppingCallback
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise ImportError("transformers is required for early stopping.") from exc
    return EarlyStoppingCallback(
        early_stopping_patience=patience,
        early_stopping_threshold=config.trainer.early_stopping_threshold,
    )


def run_sft(config: SFTExperimentConfig) -> Path:
    """Run one SFT stage and return the saved adapter directory."""
    ensure_dir(config.trainer.output_dir)
    write_json(config.trainer.output_dir / "runtime_config.json", to_jsonable(config))

    model, tokenizer = load_model_and_tokenizer(config.model, config.lora, stage="sft")
    if config.trainer.adapter_init_path is not None:
        LOGGER.info("Loading initial SFT adapter from %s", config.trainer.adapter_init_path)
        model = load_adapter_weights(model, config.trainer.adapter_init_path, is_trainable=True)
    prompt_completion_mode = _is_processor_like(tokenizer)
    conversational_mode = False
    try:
        _, SFTConfig = _import_trl_sft()
        conversational_mode = _supports_assistant_only(SFTConfig) and not prompt_completion_mode
    except Exception:
        conversational_mode = False

    record_format = "prompt_completion" if prompt_completion_mode else ("messages" if conversational_mode else "text")
    if prompt_completion_mode:
        LOGGER.info(
            "Using prompt/completion SFT records for %s to preserve completion-only loss.",
            tokenizer.__class__.__name__,
        )

    train_dataset, eval_dataset = load_sft_datasets(
        config.data,
        tokenizer=tokenizer,
        record_format=record_format,
    )
    train_count = len(train_dataset)
    eval_count = len(eval_dataset) if eval_dataset is not None else 0
    _write_dataset_manifest(
        config=config,
        train_count=train_count,
        eval_count=eval_count,
        record_format=record_format,
    )
    LOGGER.info(
        "SFT dataset ready | root=%s | train=%s (%s rows on disk) | eval=%s (%s rows on disk) | format=%s",
        config.data.dataset_root,
        train_count,
        count_jsonl_records(config.data.sft_train_file),
        eval_count,
        count_jsonl_records(config.data.sft_eval_file),
        record_format,
    )
    has_eval = eval_dataset is not None and len(eval_dataset) > 0
    trainer_cls, training_args = _build_sft_args(
        config=config,
        has_eval=has_eval,
        conversational_mode=conversational_mode,
        prompt_completion_mode=prompt_completion_mode,
    )

    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset if has_eval else None,
    }
    trainer_kwargs.update(_trainer_processing_kwargs(trainer_cls, tokenizer))
    trainer = trainer_cls(**trainer_kwargs)
    early_stopping_callback = _build_early_stopping_callback(config)
    if early_stopping_callback is not None:
        LOGGER.info(
            "Enabling SFT early stopping | patience=%s | threshold=%s",
            config.trainer.early_stopping_patience,
            config.trainer.early_stopping_threshold,
        )
        trainer.add_callback(early_stopping_callback)

    LOGGER.info("Starting SFT run '%s'", config.trainer.run_name)
    trainer.train(resume_from_checkpoint=config.trainer.resume_from_checkpoint)
    if config.trainer.load_best_model_at_end:
        LOGGER.info(
            "SFT best model selection | checkpoint=%s | metric=%s",
            getattr(trainer.state, "best_model_checkpoint", None),
            getattr(trainer.state, "best_metric", None),
        )
    adapter_dir = save_adapter(model, tokenizer, config.trainer.output_dir)
    LOGGER.info("Saved SFT adapter to %s", adapter_dir)
    return adapter_dir


def run_sft_with_overrides(
    config_path: str | Path,
    *,
    run_root: str | Path | None = None,
) -> Path:
    """Load one SFT config, optionally redirect outputs under run_root, and train."""
    config = load_sft_config(config_path)
    resolved_config_path = Path(config_path).resolve()

    if run_root is not None:
        resolved_run_root = Path(run_root)
        config = _override_stage_output(config, resolved_run_root)
        resolved_run_root = config.trainer.output_dir.parent.resolve()
        _write_pipeline_manifest(
            resolved_run_root,
            config_path=resolved_config_path,
            sft_config=config,
        )
        adapter_dir = run_sft(config)
        _write_pipeline_manifest(
            resolved_run_root,
            config_path=resolved_config_path,
            sft_config=config,
            sft_adapter_dir=adapter_dir,
        )
        return adapter_dir

    return run_sft(config)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run SFT with Unsloth + TRL")
    parser.add_argument("--config", required=True, help="Path to the SFT YAML config")
    parser.add_argument(
        "--run-root",
        help="Optional run root. When set, effective SFT artifacts are written to run_root/sft.",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    configure_logging(verbose=args.verbose)
    run_sft_with_overrides(args.config, run_root=args.run_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
