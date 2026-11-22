#!/usr/bin/env python3
"""Validate Mikuru JSONL datasets against the Rust loader contract.

Read-only. Exits 0 when every supplied file is clean, 1 when the contract is
violated, 2 on usage or I/O errors. The Rust loader remains the authority;
this tool exists so bad data is caught before a build or a training run.

Usage:
    python3 tools/preprocess/validate_jsonl.py data/prepared/v2/*.jsonl
    python3 tools/preprocess/validate_jsonl.py --stats --json data/prepared/v2
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mikuru_data import (  # noqa: E402
    Issue,
    JsonlError,
    conversation_chars,
    find_duplicate_ids,
    iter_jsonl,
    message_count,
    role_counts,
    sha256_file,
    summarise,
    validate_record,
)


def expand_paths(arguments: list[str]) -> list[Path]:
    """Expand files, and dataset directories to train/validation JSONL."""
    paths: list[Path] = []
    for argument in arguments:
        path = Path(argument).expanduser()
        if path.is_dir():
            found = sorted(path.glob("*.jsonl"))
            if not found:
                raise FileNotFoundError(f"no .jsonl files in directory: {path}")
            paths.extend(found)
        elif path.is_file():
            paths.append(path)
        else:
            raise FileNotFoundError(f"not found: {path}")
    return paths


def validate_paths(paths: list[Path]) -> tuple[list[Issue], list[tuple[str, int, dict[str, Any]]]]:
    issues: list[Issue] = []
    located: list[tuple[str, int, dict[str, Any]]] = []

    for path in paths:
        display = str(path)
        try:
            rows = list(iter_jsonl(path))
        except JsonlError as exc:
            issues.append(Issue(display, exc.line, "", exc.message))
            continue
        except OSError as exc:
            issues.append(Issue(display, 0, "", f"cannot read file: {exc}"))
            continue

        if not rows:
            issues.append(Issue(display, 0, "", "dataset file contains no records"))
            continue

        for line, record in rows:
            located.append((display, line, record))
            for field, message in validate_record(record):
                issues.append(Issue(display, line, field, message))

    for path, line, record_id, first in find_duplicate_ids(located):
        issues.append(
            Issue(path, line, "id", f"duplicate id `{record_id}`; first seen at {first}")
        )

    return issues, located


def build_stats(paths: list[Path], located: list[tuple[str, int, dict[str, Any]]]) -> dict[str, Any]:
    per_file: dict[str, Any] = {}
    for path in paths:
        per_file[str(path)] = {
            "records": 0,
            "messages": 0,
            "characters": 0,
            "sha256": sha256_file(path) if path.is_file() else None,
        }

    char_lengths: list[int] = []
    message_lengths: list[int] = []
    roles: Counter[str] = Counter()
    splits: Counter[str] = Counter()
    licenses: Counter[str] = Counter()
    sources: Counter[str] = Counter()
    synthetic = 0

    for display, _line, record in located:
        messages = message_count(record)
        characters = conversation_chars(record)
        char_lengths.append(characters)
        message_lengths.append(messages)
        roles.update(role_counts(record))
        splits[str(record.get("split") or "(unassigned)")] += 1
        license_value = record.get("license")
        licenses[str(license_value) if license_value is not None else "(missing)"] += 1
        source_value = record.get("source_dataset")
        sources[str(source_value) if source_value is not None else "(missing)"] += 1
        if record.get("synthetic") is True:
            synthetic += 1

        entry = per_file.get(display)
        if entry is not None:
            entry["records"] += 1
            entry["messages"] += messages
            entry["characters"] += characters

    return {
        "files": len(paths),
        "records": len(located),
        "synthetic_records": synthetic,
        "messages": sum(message_lengths),
        "characters": sum(char_lengths),
        "chars_per_conversation": summarise(char_lengths),
        "messages_per_conversation": summarise(message_lengths),
        "roles": dict(sorted(roles.items())),
        "splits": dict(sorted(splits.items())),
        "licenses": dict(sorted(licenses.items())),
        "sources": dict(sorted(sources.items())),
        "per_file": per_file,
    }


def print_stats(stats: dict[str, Any], paths: list[Path]) -> None:
    print("\nDATASET SUMMARY")
    print("=" * 72)
    print(f"Files:            {stats['files']}")
    print(f"Records:          {stats['records']} ({stats['synthetic_records']} synthetic)")
    print(f"Messages:         {stats['messages']}")
    print(f"Characters:       {stats['characters']}")
    chars = stats["chars_per_conversation"]
    print(f"Chars/conversation:   min {chars['min']}, mean {chars['mean']}, max {chars['max']}")
    messages = stats["messages_per_conversation"]
    print(
        f"Messages/conversation: min {messages['min']}, mean {messages['mean']}, max {messages['max']}"
    )
    print(f"Splits:           {stats['splits']}")
    print(f"Roles:            {stats['roles']}")
    print(f"Licenses:         {stats['licenses']}")
    print(f"Sources:          {stats['sources']}")
    print("\nPER FILE")
    print("-" * 72)
    for path in paths:
        entry = stats["per_file"][str(path)]
        print(
            f"{path}\n  records {entry['records']}, messages {entry['messages']}, "
            f"chars {entry['characters']}, sha256 {entry['sha256'][:16]}…"
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate Mikuru JSONL datasets against the Rust loader contract (read-only)."
    )
    parser.add_argument(
        "paths",
        nargs="+",
        metavar="PATH",
        help="JSONL file(s) or prepared dataset directory containing *.jsonl",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="print per-split length, role, license, and source summaries",
    )
    parser.add_argument(
        "--json",
        type=Path,
        metavar="FILE",
        help="write the full report as JSON",
    )
    args = parser.parse_args()

    try:
        paths = expand_paths(args.paths)
    except (OSError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    issues, located = validate_paths(paths)
    stats = build_stats(paths, located)

    if issues:
        print(f"{len(issues)} contract violation(s):", file=sys.stderr)
        for issue in issues:
            print(f"  {issue.render()}", file=sys.stderr)
    else:
        print(f"OK: {len(located)} records across {len(paths)} file(s) satisfy the contract.")

    if args.stats:
        print_stats(stats, paths)

    if args.json:
        report = {
            "ok": not issues,
            "paths": [str(path) for path in paths],
            "issues": [issue.to_dict() for issue in issues],
            "stats": stats,
        }
        try:
            args.json.parent.mkdir(parents=True, exist_ok=True)
            args.json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        except OSError as exc:
            print(f"error writing JSON report: {exc}", file=sys.stderr)
            return 2
        print(f"JSON report written: {args.json}")

    return 1 if issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
