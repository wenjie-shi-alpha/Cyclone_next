"""Model loading and adapter utilities built around Unsloth."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from .compat import prepare_unsloth_runtime
from .config import LoRAConfig, ModelConfig
from .utils import ensure_dir, repo_root


LOGGER = logging.getLogger("cyclone_training.modeling")


def _resolve_dtype(dtype_name: str | None):
    if dtype_name in (None, "", "auto"):
        return None
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise ImportError("torch is required to resolve explicit dtype values.") from exc

    dtype_name = dtype_name.lower()
    mapping = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    if dtype_name not in mapping:
        raise ValueError(f"Unsupported dtype value: {dtype_name}")
    return mapping[dtype_name]


def _maybe_patch_fast_rl() -> None:
    prepare_unsloth_runtime()
    try:
        from unsloth import FastLanguageModel, PatchFastRL
    except Exception:
        return

    try:
        PatchFastRL("GRPO", FastLanguageModel)
    except Exception as exc:  # pragma: no cover - best effort
        LOGGER.warning("Unsloth GRPO patch skipped: %s", exc)


def _resolve_model_source(name_or_path: str) -> tuple[str, bool]:
    candidate = Path(name_or_path).expanduser()
    search_paths = [candidate]
    if not candidate.is_absolute():
        search_paths.append(repo_root() / candidate)

    for path in search_paths:
        if path.exists():
            return str(path.resolve()), True
    return name_or_path, False


def load_model_and_tokenizer(
    model_config: ModelConfig,
    lora_config: LoRAConfig,
    stage: str,
    *,
    attach_lora: bool = True,
):
    """Load one Unsloth model and optionally attach a trainable LoRA adapter."""
    if stage.lower() == "grpo":
        _maybe_patch_fast_rl()

    prepare_unsloth_runtime()
    try:
        from unsloth import FastLanguageModel
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise ImportError(
            "Unsloth is required for training. Install it using the official install path."
        ) from exc

    model_name_or_path, local_files_only = _resolve_model_source(model_config.name_or_path)
    if local_files_only:
        LOGGER.info("Loading local base model from %s", model_name_or_path)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name_or_path,
        max_seq_length=model_config.max_seq_length,
        dtype=_resolve_dtype(model_config.dtype),
        load_in_4bit=model_config.load_in_4bit,
        trust_remote_code=model_config.trust_remote_code,
        local_files_only=local_files_only,
    )

    if attach_lora:
        model = FastLanguageModel.get_peft_model(
            model,
            r=lora_config.r,
            target_modules=lora_config.target_modules,
            lora_alpha=lora_config.alpha,
            lora_dropout=lora_config.dropout,
            bias=lora_config.bias,
            use_gradient_checkpointing="unsloth",
            random_state=lora_config.random_state,
        )

    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    if stage.lower() == "grpo":
        tokenizer.padding_side = "left"
        if hasattr(tokenizer, "truncation_side"):
            tokenizer.truncation_side = "left"
    else:
        tokenizer.padding_side = "right"
        if hasattr(tokenizer, "truncation_side"):
            tokenizer.truncation_side = "right"

    return model, tokenizer


def load_adapter_weights(model: Any, adapter_path: Path, *, is_trainable: bool = True) -> Any:
    """Load an existing LoRA adapter into the already prepared model."""
    if not adapter_path.exists():
        raise FileNotFoundError(f"Adapter path does not exist: {adapter_path}")

    if hasattr(model, "load_adapter"):
        model.load_adapter(str(adapter_path), adapter_name="default", is_trainable=is_trainable)
        if hasattr(model, "set_adapter"):
            model.set_adapter("default")
        return model

    try:
        from peft import PeftModel
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise ImportError(
            "peft is required to load a saved adapter when model.load_adapter is unavailable."
        ) from exc

    return PeftModel.from_pretrained(model, str(adapter_path), is_trainable=is_trainable)


def save_adapter(
    model: Any,
    tokenizer: Any,
    output_dir: Path,
    adapter_subdir: str = "final_adapter",
) -> Path:
    """Save one trainable adapter and tokenizer for reuse by the next stage."""
    target_dir = ensure_dir(output_dir / adapter_subdir)
    model.save_pretrained(str(target_dir))
    tokenizer.save_pretrained(str(target_dir))
    return target_dir
