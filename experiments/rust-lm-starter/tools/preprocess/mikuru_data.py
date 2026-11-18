#!/usr/bin/env python3
"""Shared, read-only helpers for Mikuru JSONL preprocessing and audit tools.

This module mirrors the *contract* enforced by
``crates/mikuru-core/src/dataset.rs``. It does not replace the Rust loader:
the trainer re-validates every record on load. Python lives here to catch
problems before a build, and to inspect datasets without compiling anything.

Standard library only. Never writes to a dataset path.
"""
from __future__ import annotations

import hashlib
import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

#: Split labels accepted by the Rust loader. Absent split is also valid.
SPLIT_LABELS = ("train", "validation", "demo")

#: Fields that must exist and must not be blank (whitespace-only is blank).
REQUIRED_TEXT_FIELDS = (
    "id",
    "language",
    "category",
    "source_dataset",
    "source_record_id",
    "source_url",
    "license",
    "provenance_note",
)

#: Optional array-of-string fields understood by the schema.
OPTIONAL_STRING_ARRAYS = ("references", "source_message_ids")

#: Every field the Rust schema knows about.
KNOWN_FIELDS = (
    "id",
    "messages",
    "language",
    "category",
    "source_dataset",
    "source_record_id",
    "source_url",
    "license",
    "synthetic",
    "split",
    "references",
    "source_message_ids",
    "provenance_note",
)

FNV_OFFSET_BASIS = 0xCBF29CE484222325
FNV_PRIME = 0x100000001B3
FNV_MASK = 0xFFFFFFFFFFFFFFFF


class JsonlError(ValueError):
    """A line that is not a valid JSON object."""

    def __init__(self, path: Path, line: int, message: str) -> None:
        super().__init__(f"{path}:{line}: {message}")
        self.path = path
        self.line = line
        self.message = message


@dataclass(frozen=True)
class Issue:
    """One contract violation, located at a line where possible."""

    path: str
    line: int
    field: str
    message: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "line": self.line,
            "field": self.field,
            "message": self.message,
        }

    def render(self) -> str:
        location = f"{self.path}:{self.line}" if self.line else self.path
        field = f" [{self.field}]" if self.field else ""
        return f"{location}{field}: {self.message}"


def iter_jsonl(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    """Yield ``(line_number, record)`` for every JSON object in ``path``.

    Blank lines are rejected: the Rust loader treats them as malformed
    records, and silently skipping them here would hide that mismatch.
    """
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            text = raw.strip()
            if not text:
                raise JsonlError(path, line_number, "blank line is not a valid record")
            try:
                value = json.loads(text)
            except json.JSONDecodeError as exc:
                raise JsonlError(path, line_number, f"invalid JSON: {exc}") from exc
            if not isinstance(value, dict):
                raise JsonlError(path, line_number, "record must be a JSON object")
            yield line_number, value


def validate_record(record: dict[str, Any]) -> list[tuple[str, str]]:
    """Return ``(field, message)`` pairs for one parsed record.

    Mirrors the Rust rules but collects every problem instead of stopping at
    the first one. Callers must already have checked that *record* is a dict.
    """
    problems: list[tuple[str, str]] = []

    for field in REQUIRED_TEXT_FIELDS:
        if field not in record:
            problems.append((field, f"missing required field `{field}`"))
            continue
        value = record[field]
        if not isinstance(value, str):
            problems.append((field, f"`{field}` must be a string"))
        elif not value.strip():
            problems.append((field, f"`{field}` must not be empty"))

    if "synthetic" not in record:
        problems.append(("synthetic", "missing required field `synthetic`"))
    elif not isinstance(record["synthetic"], bool):
        problems.append(("synthetic", "`synthetic` must be a boolean"))

    messages = record.get("messages")
    if messages is None:
        problems.append(("messages", "missing required field `messages`"))
    elif not isinstance(messages, list):
        problems.append(("messages", "`messages` must be an array"))
    elif not messages:
        problems.append(("messages", "conversation must contain at least one message"))
    else:
        for index, message in enumerate(messages, start=1):
            if not isinstance(message, dict):
                problems.append(("messages", f"message {index} must be an object"))
                continue
            role = message.get("role")
            content = message.get("content")
            if not isinstance(role, str):
                problems.append(("messages", f"message {index} `role` must be a string"))
            elif not role.strip():
                problems.append(("messages", f"message {index} has an empty role"))
            if not isinstance(content, str):
                problems.append(("messages", f"message {index} `content` must be a string"))
            elif not content.strip():
                problems.append(("messages", f"message {index} has empty content"))
            unknown = set(message) - {"role", "content", "source_id"}
            if unknown:
                problems.append(
                    ("messages", f"message {index} has unknown field(s): {', '.join(sorted(unknown))}")
                )

    split = record.get("split")
    if split is not None:
        if not isinstance(split, str):
            problems.append(("split", "`split` must be a string when present"))
        elif split.strip() not in SPLIT_LABELS:
            problems.append(
                (
                    "split",
                    f"unknown split `{split.strip()}`; expected one of {', '.join(SPLIT_LABELS)}",
                )
            )

    for field in OPTIONAL_STRING_ARRAYS:
        if field not in record:
            continue
        value = record[field]
        if not isinstance(value, list):
            problems.append((field, f"`{field}` must be an array when present"))
            continue
        for index, entry in enumerate(value, start=1):
            if not isinstance(entry, str):
                problems.append((field, f"`{field}` item {index} must be a string"))
            elif field == "references" and not entry.strip():
                problems.append((field, f"`{field}` item {index} must not be empty"))

    unknown_fields = set(record) - set(KNOWN_FIELDS)
    if unknown_fields:
        problems.append(
            ("", f"unknown field(s): {', '.join(sorted(unknown_fields))} (ignored by Rust serde)")
        )

    return problems


def find_duplicate_ids(
    located_records: list[tuple[str, int, dict[str, Any]]],
) -> list[tuple[str, int, str, str]]:
    """Return duplicates as ``(path, line, id, first_location)``.

    *located_records* is ``(path, line, record)`` in load order.
    """
    seen: dict[str, str] = {}
    duplicates: list[tuple[str, int, str, str]] = []
    for path, line, record in located_records:
        record_id = record.get("id")
        if not isinstance(record_id, str):
            continue
        location = f"{path}:{line}"
        first = seen.get(record_id)
        if first is None:
            seen[record_id] = location
        else:
            duplicates.append((path, line, record_id, first))
    return duplicates


def conversation_chars(record: dict[str, Any]) -> int:
    """Total characters of message content, matching the Rust inspector."""
    total = 0
    for message in record.get("messages", []):
        if isinstance(message, dict) and isinstance(message.get("content"), str):
            total += len(message["content"])
    return total


def message_count(record: dict[str, Any]) -> int:
    messages = record.get("messages", [])
    return len(messages) if isinstance(messages, list) else 0


def role_counts(record: dict[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for message in record.get("messages", []):
        if isinstance(message, dict) and isinstance(message.get("role"), str):
            role = message["role"].strip()
            counts[role] = counts.get(role, 0) + 1
    return counts


def summarise(lengths: list[int]) -> dict[str, Any]:
    """Compact min/mean/max summary; empty input yields nulls."""
    if not lengths:
        return {"min": None, "mean": None, "max": None}
    return {
        "min": min(lengths),
        "mean": round(statistics.fmean(lengths), 2),
        "max": max(lengths),
    }


def fnv1a_64(text: str) -> int:
    """FNV-1a 64-bit, matching Rust's ``u64`` split rule."""
    value = FNV_OFFSET_BASIS
    for byte in text.encode("utf-8"):
        value ^= byte
        value = (value * FNV_PRIME) & FNV_MASK
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
