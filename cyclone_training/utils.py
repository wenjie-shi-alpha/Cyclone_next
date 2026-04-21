"""Shared helpers for training scripts."""

from __future__ import annotations

import inspect
import json
import logging
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping


LOGGER = logging.getLogger("cyclone_training")


def repo_root() -> Path:
    """Return the repository root."""
    return Path(__file__).resolve().parents[1]


def configure_logging(verbose: bool = False) -> None:
    """Configure one process-wide logging formatter."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def ensure_dir(path: Path) -> Path:
    """Create a directory if it does not exist."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_jsonl(path: Path) -> Iterator[Mapping[str, Any]]:
    """Yield JSONL rows from one file."""
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def count_jsonl_records(path: Path | None) -> int:
    """Count non-empty JSONL rows in one file."""
    if path is None or not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def filter_supported_kwargs(factory: Any, kwargs: Mapping[str, Any]) -> dict[str, Any]:
    """Keep only kwargs accepted by one callable/class."""
    signature = inspect.signature(factory)
    return {
        key: value
        for key, value in kwargs.items()
        if key in signature.parameters and value is not None
    }


def to_jsonable(value: Any) -> Any:
    """Convert dataclasses and paths into JSON-safe values."""
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return to_jsonable(asdict(value))
    if isinstance(value, dict):
        return {key: to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value


def write_json(path: Path, payload: Any) -> None:
    """Write JSON with stable indentation."""
    path.write_text(
        json.dumps(to_jsonable(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def limit_records(records: Iterable[Mapping[str, Any]], limit: int | None) -> list[Mapping[str, Any]]:
    """Materialize at most ``limit`` records."""
    if limit is None or limit < 0:
        return list(records)

    limited: list[Mapping[str, Any]] = []
    for idx, record in enumerate(records):
        if idx >= limit:
            break
        limited.append(record)
    return limited
