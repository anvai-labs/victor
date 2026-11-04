"""Pinned native physical-file accounting, separate from detector unit counts."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

_MULTI_FORMATS = {"md", "markdown", "mkd", "vue", "svelte", "astro"}


@dataclass(frozen=True)
class FileAccounting:
    format: str
    tokens: int
    bytes: int

    def __post_init__(self) -> None:
        if not isinstance(self.format, str) or not re.fullmatch(r"[\w.+-]{1,100}", self.format):
            raise ValueError("Invalid native accounted format")
        _integer(self.tokens)
        _integer(self.bytes)


def _integer(value: Any) -> int:
    if type(value) is not int or not 0 <= value <= 128 * 1024 * 1024:
        raise ValueError("Invalid native file-accounting count")
    return value


def physical_file_summary(
    raw: Any, manifest: dict[str, dict[str, Any]], *, metric: str
) -> dict[str, FileAccounting]:
    """Read an untruncated summary of actual recognized physical inputs."""
    try:
        summary = raw["summary"]
        files = summary["files"]
        if (
            summary["by"] != metric
            or not isinstance(files, list)
            or _integer(summary["totalFiles"]) != len(files)
            or len(files) > len(manifest)
        ):
            raise ValueError("Invalid or truncated native physical-file accounting")
        accounted = {}
        for item in files:
            name, format_ = item["path"], item["format"]
            if not isinstance(name, str) or name not in manifest or name in accounted:
                raise ValueError("Foreign or duplicate native physical-file accounting")
            if not isinstance(format_, str) or not re.fullmatch(r"[\w.+-]{1,100}", format_):
                raise ValueError("Invalid native accounted format")
            size = _integer(item["bytes"])
            if size != manifest[name]["bytes"]:
                raise ValueError("Native physical input size differs from its acquired snapshot")
            _integer(item["lines"])
            accounted[name] = FileAccounting(format_, _integer(item["tokens"]), size)
        return accounted
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError("Malformed native physical-file accounting") from exc


def proven_below_threshold(
    entry: FileAccounting, info: dict[str, Any], *, min_tokens: int, min_lines: int
) -> bool:
    """Accept only evidence that an omitted file cannot qualify for detection.

    Embedded streams can exceed the host's display-token count (for example
    Astro frontmatter), so their host-token count alone cannot prove omission.
    """
    return (
        _integer(info["native_lines"]) < min_lines
        or entry.bytes < min_tokens
        or (entry.format not in _MULTI_FORMATS and entry.tokens < min_tokens)
    )
