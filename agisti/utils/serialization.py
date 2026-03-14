"""
Serialization Utilities — JSON/JSONL reading, writing, and
data conversion for AGISTI.

Handles:
- Safe JSON serialization (tensors, dataclasses, enums, Paths)
- JSONL streaming (read/write/append)
- Checkpoint metadata serialization
- Delta serialization helpers
- Configuration import/export

All I/O functions use UTF-8 encoding and create parent dirs.
"""

from __future__ import annotations

import dataclasses
import json
import logging
from enum import Enum
from pathlib import Path
from typing import Any, Generator, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


# ─── Custom JSON Encoder ─────────────────────────────

class AgistiEncoder(json.JSONEncoder):
    """
    JSON encoder that handles AGISTI-specific types.

    Supports:
    - Dataclasses → dicts
    - Enums → values
    - Paths → strings
    - Tensors → lists (with size guard)
    - Sets → lists
    - bytes → hex strings
    """

    MAX_TENSOR_ELEMENTS = 10000

    def default(self, obj: Any) -> Any:
        # Dataclass
        if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
            return dataclasses.asdict(obj)

        # Enum
        if isinstance(obj, Enum):
            return obj.value

        # Path
        if isinstance(obj, Path):
            return str(obj)

        # PyTorch tensor
        try:
            import torch
            if isinstance(obj, torch.Tensor):
                if obj.numel() > self.MAX_TENSOR_ELEMENTS:
                    return {
                        "__type__": "tensor",
                        "shape": list(obj.shape),
                        "dtype": str(obj.dtype),
                        "numel": obj.numel(),
                        "_truncated": True,
                    }
                return obj.cpu().tolist()
        except ImportError:
            pass

        # Set
        if isinstance(obj, (set, frozenset)):
            return sorted(list(obj))

        # Bytes
        if isinstance(obj, bytes):
            return obj.hex()

        return super().default(obj)


def to_json(
    obj: Any,
    indent: int | None = 2,
    ensure_ascii: bool = False,
) -> str:
    """
    Serialize to JSON string.

    Args:
        obj: Object to serialize.
        indent: Indentation level. None for compact.
        ensure_ascii: If False, allow non-ASCII chars.

    Returns:
        JSON string.
    """
    return json.dumps(
        obj,
        cls=AgistiEncoder,
        indent=indent,
        ensure_ascii=ensure_ascii,
    )


def from_json(text: str) -> Any:
    """Parse JSON string."""
    return json.loads(text)


# ─── File I/O ─────────────────────────────────────────

def save_json(
    data: Any,
    path: Path | str,
    indent: int = 2,
) -> None:
    """
    Save data to JSON file.

    Creates parent directories if needed.

    Args:
        data: Data to serialize.
        path: Output file path.
        indent: JSON indentation.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            cls=AgistiEncoder,
            indent=indent,
            ensure_ascii=False,
        )

    logger.debug("Saved JSON to %s", path)


def load_json(path: Path | str) -> Any:
    """
    Load data from JSON file.

    Args:
        path: Input file path.

    Returns:
        Parsed data.

    Raises:
        FileNotFoundError: If file doesn't exist.
        json.JSONDecodeError: If file is not valid JSON.
    """
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ─── JSONL I/O (Newline-Delimited JSON) ──────────────

def save_jsonl(
    records: list[dict[str, Any]],
    path: Path | str,
) -> None:
    """
    Save records to JSONL file (one JSON object per line).

    Args:
        records: List of dicts to write.
        path: Output file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            line = json.dumps(
                record,
                cls=AgistiEncoder,
                ensure_ascii=False,
            )
            f.write(line + "\n")

    logger.debug("Saved %d records to %s", len(records), path)


def append_jsonl(
    record: dict[str, Any],
    path: Path | str,
) -> None:
    """
    Append a single record to JSONL file.

    Args:
        record: Dict to append.
        path: File path (created if doesn't exist).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    line = json.dumps(
        record,
        cls=AgistiEncoder,
        ensure_ascii=False,
    )
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def load_jsonl(path: Path | str) -> list[dict[str, Any]]:
    """
    Load all records from JSONL file.

    Args:
        path: Input file path.

    Returns:
        List of parsed dicts.
    """
    path = Path(path)
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                logger.warning(
                    "Skipping invalid JSON at %s:%d: %s",
                    path, line_num, e,
                )
    return records


def stream_jsonl(
    path: Path | str,
) -> Generator[dict[str, Any], None, None]:
    """
    Stream records from JSONL file (lazy loading).

    Yields one record at a time, suitable for large files.

    Args:
        path: Input file path.

    Yields:
        Parsed dict for each line.
    """
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def count_jsonl(path: Path | str) -> int:
    """Count records in JSONL file."""
    path = Path(path)
    if not path.exists():
        return 0
    count = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


# ─── Dataclass Serialization ─────────────────────────

def dataclass_to_dict(
    obj: Any,
    exclude: set[str] | None = None,
) -> dict[str, Any]:
    """
    Convert a dataclass to dict with filtering.

    Args:
        obj: Dataclass instance.
        exclude: Set of field names to exclude.

    Returns:
        Dict representation.
    """
    if not dataclasses.is_dataclass(obj):
        raise TypeError(f"Expected dataclass, got {type(obj)}")

    result = {}
    for f in dataclasses.fields(obj):
        if exclude and f.name in exclude:
            continue
        value = getattr(obj, f.name)
        result[f.name] = _serialize_value(value)
    return result


def _serialize_value(value: Any) -> Any:
    """Recursively serialize a value for JSON."""
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_serialize_value(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _serialize_value(v) for k, v in value.items()}
    if isinstance(value, (set, frozenset)):
        return [_serialize_value(v) for v in sorted(value)]
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return dataclass_to_dict(value)

    # PyTorch tensor
    try:
        import torch
        if isinstance(value, torch.Tensor):
            if value.numel() <= 1000:
                return value.cpu().tolist()
            return {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
            }
    except ImportError:
        pass

    return str(value)


# ─── Merge Utilities ──────────────────────────────────

def deep_merge(
    base: dict[str, Any],
    override: dict[str, Any],
) -> dict[str, Any]:
    """
    Deep merge two dicts. Override values take precedence.

    Args:
        base: Base dict.
        override: Dict with values to override.

    Returns:
        New merged dict.
    """
    result = dict(base)
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def flatten_dict(
    d: dict[str, Any],
    sep: str = ".",
    prefix: str = "",
) -> dict[str, Any]:
    """
    Flatten a nested dict using dot-separated keys.

    Args:
        d: Nested dict.
        sep: Separator for keys.
        prefix: Prefix for all keys.

    Returns:
        Flat dict.
    """
    items: dict[str, Any] = {}
    for key, value in d.items():
        full_key = f"{prefix}{sep}{key}" if prefix else key
        if isinstance(value, dict):
            items.update(flatten_dict(value, sep, full_key))
        else:
            items[full_key] = value
    return items
