"""Runtime compatibility helpers for Unsloth + xFormers training."""

from __future__ import annotations

import contextlib
import io
import json
import logging
import os
from pathlib import Path
from typing import Any


LOGGER = logging.getLogger("cyclone_training.compat")

_XFORMERS_PATCH_APPLIED = False


def _import_xformers_quietly():
    xformers_logger = logging.getLogger("xformers")
    previous_level = xformers_logger.level
    sink = io.StringIO()
    xformers_logger.setLevel(logging.ERROR)
    try:
        with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            import xformers
            from xformers.ops import fmha
    finally:
        xformers_logger.setLevel(previous_level)
    return xformers, fmha


def _xformers_runtime_is_usable(fmha: Any) -> bool:
    if not hasattr(fmha, "memory_efficient_attention"):
        return False

    usable_backends = []
    for backend_name in ("cutlass", "flash"):
        backend = getattr(fmha, backend_name, None)
        if backend is None:
            continue
        usable_backends.append(
            getattr(backend, "FwOp", None) is not None
            and getattr(backend, "BwOp", None) is not None
        )
    return any(usable_backends)


def _load_xformers_build_metadata(cpp_lib: Any):
    build_metadata = getattr(cpp_lib, "_build_metadata", None)
    if build_metadata is not None:
        return build_metadata

    metadata_path = Path(cpp_lib.__file__).with_name("cpp_lib.json")
    if not metadata_path.exists():
        return None

    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    return cpp_lib._BuildInfo(payload)


def patch_xformers_for_unsloth() -> bool:
    """Patch Unsloth's xFormers registration check when kernels are still usable."""
    global _XFORMERS_PATCH_APPLIED
    if _XFORMERS_PATCH_APPLIED:
        return True

    try:
        _, fmha = _import_xformers_quietly()
    except Exception as exc:  # pragma: no cover - runtime dependency
        LOGGER.debug("Skipping xFormers patch; import probe failed: %s", exc)
        return False

    if not _xformers_runtime_is_usable(fmha):
        return False

    try:
        import xformers._cpp_lib as cpp_lib
    except Exception as exc:  # pragma: no cover - runtime dependency
        LOGGER.debug("Skipping xFormers patch; cpp lib probe failed: %s", exc)
        return False

    original_register = getattr(cpp_lib, "_register_extensions", None)
    if original_register is None:
        return False

    if getattr(original_register, "__name__", "") == "_compat_register_extensions":
        _XFORMERS_PATCH_APPLIED = True
        return True

    def _compat_register_extensions():
        try:
            metadata = original_register()
        except Exception as exc:  # pragma: no cover - runtime dependency
            metadata = _load_xformers_build_metadata(cpp_lib)
            cpp_lib._build_metadata = metadata
            cpp_lib._cpp_library_load_exception = None
            LOGGER.info(
                "Using xFormers compatibility shim after metadata mismatch: %s",
                exc,
            )
            return metadata
        cpp_lib._build_metadata = metadata
        cpp_lib._cpp_library_load_exception = None
        return metadata

    cpp_lib._register_extensions = _compat_register_extensions
    cpp_lib._build_metadata = _load_xformers_build_metadata(cpp_lib)
    cpp_lib._cpp_library_load_exception = None
    _XFORMERS_PATCH_APPLIED = True
    return True


def prepare_trl_runtime() -> None:
    """Patch optional TRL import paths that are irrelevant to this project."""
    try:
        from huggingface_hub import constants as hub_constants
        from transformers.utils import hub as transformers_hub
    except Exception:  # pragma: no cover - runtime dependency
        transformers_hub = None
        hub_constants = None

    if transformers_hub is not None and not hasattr(transformers_hub, "TRANSFORMERS_CACHE"):
        cache_root = None
        if hub_constants is not None:
            cache_root = getattr(hub_constants, "HF_HUB_CACHE", None)
            if cache_root is None:
                hf_home = getattr(hub_constants, "HF_HOME", None)
                if hf_home is not None:
                    cache_root = os.path.join(hf_home, "hub")
        transformers_hub.TRANSFORMERS_CACHE = os.environ.get(
            "TRANSFORMERS_CACHE",
            cache_root or os.path.expanduser("~/.cache/huggingface/hub"),
        )

    try:
        import trl.import_utils as trl_import_utils
    except Exception:  # pragma: no cover - runtime dependency
        return

    # These features are not used by our SFT/GRPO pipeline and currently pull
    # in brittle optional dependencies at import time.
    for attr_name in (
        "_mergekit_available",
        "_llm_blender_available",
        "_weave_available",
        "_vllm_available",
    ):
        if hasattr(trl_import_utils, attr_name):
            setattr(trl_import_utils, attr_name, False)


def prepare_unsloth_runtime() -> None:
    """Apply best-effort runtime patches before importing Unsloth."""
    os.environ.setdefault("UNSLOTH_DISABLE_STATISTICS", "1")

    try:
        import torch
    except ImportError:  # pragma: no cover - runtime dependency
        return

    if not torch.cuda.is_available():
        return

    patch_xformers_for_unsloth()
