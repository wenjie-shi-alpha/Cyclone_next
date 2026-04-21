"""Typed config loading for SFT and GRPO runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from .utils import repo_root


DEFAULT_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


@dataclass(slots=True)
class ModelConfig:
    name_or_path: str
    max_seq_length: int = 4096
    load_in_4bit: bool = True
    dtype: Optional[str] = None
    trust_remote_code: bool = False


@dataclass(slots=True)
class LoRAConfig:
    r: int = 32
    alpha: int = 64
    dropout: float = 0.05
    bias: str = "none"
    target_modules: list[str] = field(default_factory=lambda: list(DEFAULT_TARGET_MODULES))
    random_state: int = 3407


@dataclass(slots=True)
class DataConfig:
    dataset_root: Path
    sft_train_file: Optional[Path] = None
    sft_eval_file: Optional[Path] = None
    rl_train_file: Optional[Path] = None
    rl_eval_file: Optional[Path] = None
    max_train_samples: Optional[int] = None
    max_eval_samples: Optional[int] = None
    sft_resample_fields: list[str] = field(default_factory=list)
    sft_resample_power: float = 0.0
    sft_resample_max_multiplier: int = 1
    sft_resample_label_min_multipliers: dict[str, dict[str, int]] = field(default_factory=dict)


@dataclass(slots=True)
class BaseStageConfig:
    output_dir: Path
    run_name: str
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    num_train_epochs: float
    max_steps: int
    warmup_ratio: float
    weight_decay: float
    max_grad_norm: float
    lr_scheduler_type: str
    logging_steps: int
    save_steps: int
    eval_steps: int
    save_total_limit: int
    bf16: bool
    fp16: bool
    gradient_checkpointing: bool
    report_to: list[str]
    resume_from_checkpoint: Optional[str] = None
    seed: int = 3407


@dataclass(slots=True)
class SFTStageConfig(BaseStageConfig):
    adapter_init_path: Optional[Path] = None
    load_best_model_at_end: bool = False
    metric_for_best_model: Optional[str] = None
    greater_is_better: Optional[bool] = None
    early_stopping_patience: Optional[int] = None
    early_stopping_threshold: float = 0.0
    packing: bool = False
    completion_only_loss: Optional[bool] = None
    assistant_only_loss: bool = True
    dataset_text_field: str = "text"
    dataset_num_proc: Optional[int] = None


@dataclass(slots=True)
class RewardRuntimeConfig:
    truth_slot_tolerance_hours: int = 6
    forecast_slot_tolerance_hours: int = 9
    forecast_slot_time_scale_hours: float = 6.0
    track_error_scale_km: float = 250.0
    intensity_error_scale_kt: float = 20.0
    track_error_weight: float = 0.5
    intensity_error_weight: float = 0.5
    format_bonus: float = 0.02
    soft_slot_max_hours: Optional[float] = None
    soft_slot_reward_weight: float = 0.0


@dataclass(slots=True)
class GRPOStageConfig(BaseStageConfig):
    adapter_init_path: Optional[Path] = None
    reference_adapter_path: Optional[Path] = None
    disable_dropout: bool = True
    reward_save_threshold: Optional[float] = None
    stop_after_reward_checkpoint: bool = False
    scale_rewards: str | bool | None = None
    mask_truncated_completions: bool = False
    max_prompt_length: int = 3072
    max_completion_length: int = 512
    num_generations: int = 4
    beta: float = 0.04
    loss_type: str = "grpo"
    temperature: float = 0.8
    top_p: float = 0.95
    top_k: int = 50
    generation_batch_size: Optional[int] = None
    use_vllm: bool = False
    vllm_gpu_memory_utilization: float = 0.85
    unsloth_num_chunks: int = 1


@dataclass(slots=True)
class SFTExperimentConfig:
    config_path: Path
    model: ModelConfig
    lora: LoRAConfig
    data: DataConfig
    trainer: SFTStageConfig


@dataclass(slots=True)
class GRPOExperimentConfig:
    config_path: Path
    model: ModelConfig
    lora: LoRAConfig
    data: DataConfig
    trainer: GRPOStageConfig
    reward: RewardRuntimeConfig


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise ImportError("PyYAML is required to load training configs.") from exc

    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Config file must contain a mapping: {path}")
    return payload


def _resolve_path(value: Any, base_dir: Path) -> Optional[Path]:
    if value in (None, "", False):
        return None
    path = Path(str(value))
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def _load_model(payload: dict[str, Any]) -> ModelConfig:
    return ModelConfig(
        name_or_path=payload["name_or_path"],
        max_seq_length=int(payload.get("max_seq_length", 4096)),
        load_in_4bit=bool(payload.get("load_in_4bit", True)),
        dtype=payload.get("dtype"),
        trust_remote_code=bool(payload.get("trust_remote_code", False)),
    )


def _load_lora(payload: dict[str, Any]) -> LoRAConfig:
    return LoRAConfig(
        r=int(payload.get("r", 32)),
        alpha=int(payload.get("alpha", 64)),
        dropout=float(payload.get("dropout", 0.05)),
        bias=str(payload.get("bias", "none")),
        target_modules=list(payload.get("target_modules", DEFAULT_TARGET_MODULES)),
        random_state=int(payload.get("random_state", 3407)),
    )


def _load_data(
    payload: dict[str, Any],
    base_dir: Path,
    *,
    default_sft_files: bool,
    default_rl_files: bool,
) -> DataConfig:
    dataset_root = _resolve_path(payload["dataset_root"], base_dir)
    assert dataset_root is not None
    raw_label_multipliers = payload.get("sft_resample_label_min_multipliers", {}) or {}
    label_min_multipliers: dict[str, dict[str, int]] = {}
    for field_name, value_map in raw_label_multipliers.items():
        if not isinstance(value_map, dict):
            continue
        normalized_map = {
            str(label): max(1, int(multiplier))
            for label, multiplier in value_map.items()
            if str(label).strip()
        }
        if normalized_map:
            label_min_multipliers[str(field_name)] = normalized_map
    return DataConfig(
        dataset_root=dataset_root,
        sft_train_file=(
            _resolve_path(payload.get("sft_train_file"), base_dir)
            or (dataset_root / "sft_train.jsonl" if default_sft_files else None)
        ),
        sft_eval_file=_resolve_path(payload.get("sft_eval_file"), base_dir),
        rl_train_file=(
            _resolve_path(payload.get("rl_train_file"), base_dir)
            or (dataset_root / "rl_train.jsonl" if default_rl_files else None)
        ),
        rl_eval_file=_resolve_path(payload.get("rl_eval_file"), base_dir),
        max_train_samples=payload.get("max_train_samples"),
        max_eval_samples=payload.get("max_eval_samples"),
        sft_resample_fields=[
            str(field_name)
            for field_name in payload.get("sft_resample_fields", []) or []
            if str(field_name).strip()
        ],
        sft_resample_power=float(payload.get("sft_resample_power", 0.0)),
        sft_resample_max_multiplier=int(payload.get("sft_resample_max_multiplier", 1)),
        sft_resample_label_min_multipliers=label_min_multipliers,
    )


def _load_base_stage(payload: dict[str, Any], base_dir: Path) -> dict[str, Any]:
    report_to = payload.get("report_to", ["none"])
    if isinstance(report_to, str):
        report_to = [report_to]

    output_dir = _resolve_path(payload["output_dir"], base_dir)
    assert output_dir is not None
    return {
        "output_dir": output_dir,
        "run_name": str(payload["run_name"]),
        "per_device_train_batch_size": int(payload.get("per_device_train_batch_size", 1)),
        "per_device_eval_batch_size": int(payload.get("per_device_eval_batch_size", 1)),
        "gradient_accumulation_steps": int(payload.get("gradient_accumulation_steps", 1)),
        "learning_rate": float(payload.get("learning_rate", 2e-4)),
        "num_train_epochs": float(payload.get("num_train_epochs", 1.0)),
        "max_steps": int(payload.get("max_steps", -1)),
        "warmup_ratio": float(payload.get("warmup_ratio", 0.03)),
        "weight_decay": float(payload.get("weight_decay", 0.0)),
        "max_grad_norm": float(payload.get("max_grad_norm", 1.0)),
        "lr_scheduler_type": str(payload.get("lr_scheduler_type", "cosine")),
        "logging_steps": int(payload.get("logging_steps", 1)),
        "save_steps": int(payload.get("save_steps", 50)),
        "eval_steps": int(payload.get("eval_steps", 50)),
        "save_total_limit": int(payload.get("save_total_limit", 2)),
        "bf16": bool(payload.get("bf16", True)),
        "fp16": bool(payload.get("fp16", False)),
        "gradient_checkpointing": bool(payload.get("gradient_checkpointing", True)),
        "report_to": list(report_to),
        "resume_from_checkpoint": payload.get("resume_from_checkpoint"),
        "seed": int(payload.get("seed", 3407)),
    }


def _load_sft_stage(payload: dict[str, Any], base_dir: Path) -> SFTStageConfig:
    kwargs = _load_base_stage(payload, base_dir)
    kwargs.update(
        {
            "adapter_init_path": _resolve_path(payload.get("adapter_init_path"), base_dir),
            "load_best_model_at_end": bool(payload.get("load_best_model_at_end", False)),
            "metric_for_best_model": (
                str(payload["metric_for_best_model"])
                if payload.get("metric_for_best_model") is not None
                else None
            ),
            "greater_is_better": (
                bool(payload["greater_is_better"])
                if payload.get("greater_is_better") is not None
                else None
            ),
            "early_stopping_patience": (
                int(payload["early_stopping_patience"])
                if payload.get("early_stopping_patience") is not None
                else None
            ),
            "early_stopping_threshold": float(payload.get("early_stopping_threshold", 0.0)),
            "packing": bool(payload.get("packing", False)),
            "completion_only_loss": payload.get("completion_only_loss"),
            "assistant_only_loss": bool(payload.get("assistant_only_loss", True)),
            "dataset_text_field": str(payload.get("dataset_text_field", "text")),
            "dataset_num_proc": payload.get("dataset_num_proc"),
        }
    )
    return SFTStageConfig(**kwargs)


def _load_reward(payload: dict[str, Any]) -> RewardRuntimeConfig:
    return RewardRuntimeConfig(
        truth_slot_tolerance_hours=int(payload.get("truth_slot_tolerance_hours", 6)),
        forecast_slot_tolerance_hours=int(payload.get("forecast_slot_tolerance_hours", 9)),
        forecast_slot_time_scale_hours=float(payload.get("forecast_slot_time_scale_hours", 6.0)),
        track_error_scale_km=float(payload.get("track_error_scale_km", 250.0)),
        intensity_error_scale_kt=float(payload.get("intensity_error_scale_kt", 20.0)),
        track_error_weight=float(payload.get("track_error_weight", 0.5)),
        intensity_error_weight=float(payload.get("intensity_error_weight", 0.5)),
        format_bonus=float(payload.get("format_bonus", 0.02)),
        soft_slot_max_hours=(
            float(payload["soft_slot_max_hours"])
            if payload.get("soft_slot_max_hours") is not None
            else None
        ),
        soft_slot_reward_weight=float(payload.get("soft_slot_reward_weight", 0.0)),
    )


def _load_grpo_stage(payload: dict[str, Any], base_dir: Path) -> GRPOStageConfig:
    kwargs = _load_base_stage(payload, base_dir)
    kwargs.update(
        {
            "adapter_init_path": _resolve_path(payload.get("adapter_init_path"), base_dir),
            "reference_adapter_path": _resolve_path(
                payload.get("reference_adapter_path"), base_dir
            ),
            "disable_dropout": bool(payload.get("disable_dropout", True)),
            "reward_save_threshold": (
                float(payload["reward_save_threshold"])
                if payload.get("reward_save_threshold") is not None
                else None
            ),
            "stop_after_reward_checkpoint": bool(
                payload.get("stop_after_reward_checkpoint", False)
            ),
            "scale_rewards": payload.get("scale_rewards"),
            "mask_truncated_completions": bool(
                payload.get("mask_truncated_completions", False)
            ),
            "max_prompt_length": int(payload.get("max_prompt_length", 3072)),
            "max_completion_length": int(payload.get("max_completion_length", 512)),
            "num_generations": int(payload.get("num_generations", 4)),
            "beta": float(payload.get("beta", 0.04)),
            "loss_type": str(payload.get("loss_type", "grpo")),
            "temperature": float(payload.get("temperature", 0.8)),
            "top_p": float(payload.get("top_p", 0.95)),
            "top_k": int(payload.get("top_k", 50)),
            "generation_batch_size": payload.get("generation_batch_size"),
            "use_vllm": bool(payload.get("use_vllm", False)),
            "vllm_gpu_memory_utilization": float(
                payload.get("vllm_gpu_memory_utilization", 0.85)
            ),
            "unsloth_num_chunks": int(payload.get("unsloth_num_chunks", 1)),
        }
    )
    return GRPOStageConfig(**kwargs)


def load_sft_config(path: str | Path) -> SFTExperimentConfig:
    """Load one SFT config file."""
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = (repo_root() / config_path).resolve()
    payload = _load_yaml(config_path)
    base_dir = repo_root()
    return SFTExperimentConfig(
        config_path=config_path,
        model=_load_model(payload["model"]),
        lora=_load_lora(payload.get("lora", {})),
        data=_load_data(
            payload["data"],
            base_dir,
            default_sft_files=True,
            default_rl_files=False,
        ),
        trainer=_load_sft_stage(payload["trainer"], base_dir),
    )


def load_grpo_config(path: str | Path) -> GRPOExperimentConfig:
    """Load one GRPO config file."""
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = (repo_root() / config_path).resolve()
    payload = _load_yaml(config_path)
    base_dir = repo_root()
    return GRPOExperimentConfig(
        config_path=config_path,
        model=_load_model(payload["model"]),
        lora=_load_lora(payload.get("lora", {})),
        data=_load_data(
            payload["data"],
            base_dir,
            default_sft_files=False,
            default_rl_files=True,
        ),
        trainer=_load_grpo_stage(payload["trainer"], base_dir),
        reward=_load_reward(payload.get("reward", {})),
    )
