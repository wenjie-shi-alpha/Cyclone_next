"""GRPO training entrypoint backed by Unsloth + TRL."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

from .compat import prepare_trl_runtime
from .config import GRPOExperimentConfig, load_grpo_config
from .datasets import load_grpo_datasets
from .modeling import load_adapter_weights, load_model_and_tokenizer, save_adapter
from .rewards import CycloneRewardFunction
from .utils import (
    configure_logging,
    count_jsonl_records,
    ensure_dir,
    filter_supported_kwargs,
    to_jsonable,
    write_json,
)


LOGGER = logging.getLogger("cyclone_training.grpo")


def _write_dataset_manifest(
    config: GRPOExperimentConfig,
    train_count: int,
    eval_count: int,
) -> None:
    payload = {
        "dataset_root": str(config.data.dataset_root),
        "rl_train_file": str(config.data.rl_train_file) if config.data.rl_train_file is not None else None,
        "rl_eval_file": str(config.data.rl_eval_file) if config.data.rl_eval_file is not None else None,
        "train_count": train_count,
        "eval_count": eval_count,
    }
    write_json(config.trainer.output_dir / "dataset_manifest.json", payload)


def _import_trl_grpo():
    prepare_trl_runtime()
    try:
        from trl import GRPOTrainer
        from trl.trainer.grpo_config import GRPOConfig
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise ImportError("TRL is required for GRPO training.") from exc
    return GRPOTrainer, GRPOConfig


def _import_trainer_callback():
    try:
        from transformers import TrainerCallback
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise ImportError("transformers is required to register GRPO callbacks.") from exc
    return TrainerCallback


def _ensure_warning_registry(model: Any) -> None:
    seen_ids: set[int] = set()
    stack = [model, getattr(model, "base_model", None)]
    while stack:
        candidate = stack.pop()
        if candidate is None or id(candidate) in seen_ids:
            continue
        seen_ids.add(id(candidate))
        if not hasattr(candidate, "warnings_issued") or getattr(candidate, "warnings_issued") is None:
            try:
                candidate.warnings_issued = {}
            except Exception:  # pragma: no cover - best effort
                pass
        stack.append(getattr(candidate, "model", None))


def _trainer_processing_kwargs(trainer_cls: Any, tokenizer: Any) -> dict[str, Any]:
    kwargs = {"processing_class": tokenizer, "tokenizer": tokenizer}
    return filter_supported_kwargs(trainer_cls, kwargs)


def _ensure_unsloth_grpo_args(training_args: Any, config: GRPOExperimentConfig) -> Any:
    defaults = {
        "unsloth_num_chunks": config.trainer.unsloth_num_chunks,
        "unsloth_logit_chunk_multiplier": None,
        "unsloth_grpo_mini_batch": None,
    }
    for attr_name, default_value in defaults.items():
        if not hasattr(training_args, attr_name):
            setattr(training_args, attr_name, default_value)
    return training_args


def _build_grpo_args(config: GRPOExperimentConfig, has_eval: bool):
    GRPOTrainer, GRPOConfig = _import_trl_grpo()
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
        "remove_unused_columns": False,
        "disable_dropout": config.trainer.disable_dropout,
        "scale_rewards": config.trainer.scale_rewards,
        "mask_truncated_completions": config.trainer.mask_truncated_completions,
        "max_prompt_length": config.trainer.max_prompt_length,
        "max_completion_length": config.trainer.max_completion_length,
        "num_generations": config.trainer.num_generations,
        "beta": config.trainer.beta,
        "loss_type": config.trainer.loss_type,
        "temperature": config.trainer.temperature,
        "top_p": config.trainer.top_p,
        "top_k": config.trainer.top_k,
        "use_vllm": config.trainer.use_vllm,
        "vllm_gpu_memory_utilization": config.trainer.vllm_gpu_memory_utilization,
    }
    if config.trainer.generation_batch_size is not None:
        args_kwargs["generation_batch_size"] = config.trainer.generation_batch_size
    if "unsloth_num_chunks" in getattr(GRPOConfig, "__dataclass_fields__", {}):
        args_kwargs["unsloth_num_chunks"] = config.trainer.unsloth_num_chunks

    strategy_value = "steps" if has_eval else "no"
    if "eval_strategy" in getattr(GRPOConfig, "__dataclass_fields__", {}):
        args_kwargs["eval_strategy"] = strategy_value
    elif "evaluation_strategy" in getattr(GRPOConfig, "__dataclass_fields__", {}):
        args_kwargs["evaluation_strategy"] = strategy_value

    args_kwargs = filter_supported_kwargs(GRPOConfig, args_kwargs)
    training_args = GRPOConfig(**args_kwargs)
    return GRPOTrainer, _ensure_unsloth_grpo_args(training_args, config)


def _resolve_reference_adapter_path(config: GRPOExperimentConfig) -> Path | None:
    if config.trainer.beta == 0.0:
        return None
    if config.trainer.reference_adapter_path is not None:
        return config.trainer.reference_adapter_path
    if config.trainer.adapter_init_path is not None:
        return config.trainer.adapter_init_path
    return None


def _freeze_reference_model(model: Any) -> Any:
    try:
        model.requires_grad_(False)
    except Exception:  # pragma: no cover - best effort
        pass

    parameters = getattr(model, "parameters", None)
    if callable(parameters):
        for parameter in parameters():
            parameter.requires_grad_(False)
    model.train()
    return model


def _load_reference_model(config: GRPOExperimentConfig) -> tuple[Any | None, Path | None]:
    reference_adapter_path = _resolve_reference_adapter_path(config)
    if reference_adapter_path is None:
        return None, None

    LOGGER.info("Loading GRPO reference adapter from %s", reference_adapter_path)
    reference_model, _ = load_model_and_tokenizer(config.model, config.lora, stage="grpo")
    reference_model = load_adapter_weights(
        reference_model,
        reference_adapter_path,
        is_trainable=False,
    )
    return _freeze_reference_model(reference_model), reference_adapter_path


def _write_latest_snapshot_pointer(output_dir: Path, snapshot_dir: Path) -> None:
    latest_path = output_dir / "latest_adapter_snapshot.txt"
    latest_path.write_text(f"{snapshot_dir}\n", encoding="utf-8")


def _write_best_reward_pointer(output_dir: Path, snapshot_dir: Path, reward_value: float) -> None:
    best_path = output_dir / "best_reward_checkpoint.txt"
    best_path.write_text(f"reward={reward_value:.6f}\npath={snapshot_dir}\n", encoding="utf-8")


def _extract_reward_metric(logs: dict[str, Any] | None) -> float | None:
    if not logs:
        return None
    for key in ("reward", "rewards/CycloneRewardFunction/mean"):
        value = logs.get(key)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _build_periodic_snapshot_callback(
    model: Any,
    tokenizer: Any,
    output_dir: Path,
    save_steps: int,
    reward_save_threshold: float | None,
    stop_after_reward_checkpoint: bool,
):
    TrainerCallback = _import_trainer_callback()

    class PeriodicAdapterSnapshotCallback(TrainerCallback):
        """Persist a reusable adapter snapshot before slow eval/save hooks run."""

        def __init__(self) -> None:
            self._last_saved_step = -1
            self._best_reward = float("-inf")

        def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any):
            if save_steps <= 0 or state.global_step <= 0:
                return control
            if state.global_step % save_steps != 0 or state.global_step == self._last_saved_step:
                return control

            snapshot_dir = save_adapter(
                model,
                tokenizer,
                output_dir,
                adapter_subdir=f"adapter_step-{state.global_step:06d}",
            )
            _write_latest_snapshot_pointer(output_dir, snapshot_dir)
            LOGGER.info("Saved intermediate GRPO adapter snapshot to %s", snapshot_dir)
            self._last_saved_step = state.global_step
            return control

        def on_log(self, args: Any, state: Any, control: Any, logs: dict[str, Any] | None = None, **kwargs: Any):
            reward_value = _extract_reward_metric(logs)
            if reward_save_threshold is None or reward_value is None:
                return control
            if reward_value < reward_save_threshold or reward_value <= self._best_reward:
                return control

            snapshot_dir = save_adapter(
                model,
                tokenizer,
                output_dir,
                adapter_subdir=f"adapter_reward-{reward_value:.4f}_step-{state.global_step:06d}",
            )
            _write_latest_snapshot_pointer(output_dir, snapshot_dir)
            _write_best_reward_pointer(output_dir, snapshot_dir, reward_value)
            LOGGER.info(
                "Saved reward-threshold GRPO adapter snapshot to %s | reward=%.4f | threshold=%.4f",
                snapshot_dir,
                reward_value,
                reward_save_threshold,
            )
            self._best_reward = reward_value
            if stop_after_reward_checkpoint:
                LOGGER.info(
                    "Stopping GRPO after reward checkpoint because reward %.4f reached threshold %.4f",
                    reward_value,
                    reward_save_threshold,
                )
                control.should_training_stop = True
            return control

    return PeriodicAdapterSnapshotCallback()


def run_grpo(config: GRPOExperimentConfig) -> Path:
    """Run one GRPO stage and return the saved adapter directory."""
    ensure_dir(config.trainer.output_dir)
    write_json(config.trainer.output_dir / "runtime_config.json", to_jsonable(config))

    model, tokenizer = load_model_and_tokenizer(config.model, config.lora, stage="grpo")
    if config.trainer.adapter_init_path is not None:
        LOGGER.info("Loading SFT adapter from %s", config.trainer.adapter_init_path)
        model = load_adapter_weights(model, config.trainer.adapter_init_path)
    reference_model, reference_adapter_path = _load_reference_model(config)

    train_dataset, eval_dataset = load_grpo_datasets(config.data, tokenizer=tokenizer)
    train_count = len(train_dataset)
    eval_count = len(eval_dataset) if eval_dataset is not None else 0
    _write_dataset_manifest(
        config=config,
        train_count=train_count,
        eval_count=eval_count,
    )
    LOGGER.info(
        "GRPO dataset ready | root=%s | train=%s (%s rows on disk) | eval=%s (%s rows on disk)",
        config.data.dataset_root,
        train_count,
        count_jsonl_records(config.data.rl_train_file),
        eval_count,
        count_jsonl_records(config.data.rl_eval_file),
    )
    has_eval = eval_dataset is not None and len(eval_dataset) > 0
    trainer_cls, training_args = _build_grpo_args(config, has_eval=has_eval)

    reward_function = CycloneRewardFunction(config.reward)
    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "args": training_args,
        "reward_funcs": reward_function,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset if has_eval else None,
    }
    trainer_kwargs.update(_trainer_processing_kwargs(trainer_cls, tokenizer))
    _ensure_warning_registry(model)
    trainer = trainer_cls(**trainer_kwargs)
    trainer.add_callback(
        _build_periodic_snapshot_callback(
            model=model,
            tokenizer=tokenizer,
            output_dir=config.trainer.output_dir,
            save_steps=config.trainer.save_steps,
            reward_save_threshold=config.trainer.reward_save_threshold,
            stop_after_reward_checkpoint=config.trainer.stop_after_reward_checkpoint,
        )
    )
    if reference_model is not None:
        trainer.ref_model = trainer.accelerator.prepare_model(reference_model)
        trainer.ref_model.train()
        LOGGER.info(
            "GRPO reference model ready | strategy=adapter_copy_train_path | adapter=%s",
            reference_adapter_path,
        )

    LOGGER.info("Starting GRPO run '%s'", config.trainer.run_name)
    trainer.train(resume_from_checkpoint=config.trainer.resume_from_checkpoint)
    adapter_dir = save_adapter(model, tokenizer, config.trainer.output_dir)
    LOGGER.info("Saved GRPO adapter to %s", adapter_dir)
    return adapter_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run GRPO with Unsloth + TRL")
    parser.add_argument("--config", required=True, help="Path to the GRPO YAML config")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    configure_logging(verbose=args.verbose)
    config = load_grpo_config(args.config)
    run_grpo(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
