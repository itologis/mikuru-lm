#!/usr/bin/env python3
"""Fixed-prompt evaluation harness for Mikuru checkpoints.

Read-only with respect to the repository: runs ``mikuru-lm generate`` for each
checkpoint/prompt pair and writes a report under the chosen output directory.

This exists because nothing in the Rust CLI compares checkpoints on identical
prompts. Without it, "did mini-batch training help?" would be answered by
memory instead of samples.

Usage:
    python3 tools/preprocess/eval_suite.py \
        --checkpoint artifacts/oasst1-v2-50 \
        --checkpoint artifacts/oasst1-v2-100 \
        --out reports/eval-minibatch
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BINARY = REPO_ROOT / "target" / "debug" / "mikuru-lm"
DEFAULT_TOKENIZER = REPO_ROOT / "artifacts" / "mikuru-bpe-v2.json"
DEFAULT_PROMPTS = Path(__file__).resolve().parent / "prompts.json"

GENERATED_MARKER = "Generated continuation:"


def load_prompts(path: Path) -> list[dict[str, str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    prompts = data.get("prompts") if isinstance(data, dict) else None
    if not isinstance(prompts, list) or not prompts:
        raise ValueError(f"{path} must contain a non-empty `prompts` array")

    seen: set[str] = set()
    cleaned: list[dict[str, str]] = []
    for index, entry in enumerate(prompts, start=1):
        if not isinstance(entry, dict):
            raise ValueError(f"prompt {index} must be an object")
        prompt_id = entry.get("id")
        text = entry.get("text")
        if not isinstance(prompt_id, str) or not prompt_id.strip():
            raise ValueError(f"prompt {index} needs a non-empty string `id`")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"prompt `{prompt_id}` needs a non-empty string `text`")
        if prompt_id in seen:
            raise ValueError(f"duplicate prompt id `{prompt_id}`")
        seen.add(prompt_id)
        cleaned.append({"id": prompt_id, "text": text})
    return cleaned


def load_metadata(checkpoint: Path) -> dict[str, Any]:
    metadata_path = checkpoint.with_suffix(".json")
    if not metadata_path.is_file():
        return {}
    try:
        value = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def extract_continuation(stdout: str) -> str | None:
    marker = stdout.find(GENERATED_MARKER)
    if marker == -1:
        return None
    return stdout[marker + len(GENERATED_MARKER) :].strip("\n")


def run_generate(
    binary: Path,
    tokenizer: Path,
    checkpoint: Path,
    tokens: int,
    prompt: str,
    timeout: float,
) -> dict[str, Any]:
    command = [
        str(binary),
        "generate",
        str(tokenizer),
        str(checkpoint),
        str(tokens),
        prompt,
    ]
    try:
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"ok": False, "error": str(exc), "text": None}

    if completed.returncode != 0:
        return {
            "ok": False,
            "error": f"exit {completed.returncode}: {completed.stderr.strip()}",
            "text": None,
        }

    text = extract_continuation(completed.stdout)
    if text is None:
        return {
            "ok": False,
            "error": f"missing `{GENERATED_MARKER}` in output",
            "text": None,
        }
    return {"ok": True, "error": None, "text": text}


def format_metric(value: Any, digits: int = 5) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Mikuru evaluation report",
        "",
        f"- Generated: {report['generated_at']}",
        f"- Binary: `{report['binary']}`",
        f"- Tokenizer: `{report['tokenizer']}`",
        f"- New tokens per prompt: {report['tokens']}",
        f"- Decoding: greedy (CLI default)",
        "",
        "## Checkpoints",
        "",
        "| Checkpoint | Epochs | Batch policy | Train loss | Validation |",
        "| --- | ---: | --- | ---: | ---: |",
    ]
    for entry in report["checkpoints"]:
        metadata = entry["metadata"]
        lines.append(
            "| `{name}` | {epochs} | {policy} | {train} | {val} |".format(
                name=entry["name"],
                epochs=format_metric(metadata.get("epochs")),
                policy=metadata.get("batching_policy") or "legacy full batch",
                train=format_metric(metadata.get("final_loss")),
                val=format_metric(metadata.get("validation_loss")),
            )
        )

    lines.extend(["", "## Samples", ""])
    for entry in report["checkpoints"]:
        lines.append(f"### {entry['name']}")
        lines.append("")
        for sample in entry["samples"]:
            lines.append(f"**{sample['prompt_id']}** — `{sample['prompt']}`")
            lines.append("")
            if sample["ok"]:
                lines.append("```text")
                lines.append(sample["text"] if sample["text"] else "(empty)")
                lines.append("```")
            else:
                lines.append(f"> ERROR: {sample['error']}")
            lines.append("")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run fixed prompts across Mikuru checkpoints and write a comparison report."
    )
    parser.add_argument(
        "--checkpoint",
        action="append",
        required=True,
        metavar="BASE",
        help="checkpoint base path (without .bpk); repeat for multiple checkpoints",
    )
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--tokens", type=int, default=24, metavar="N")
    parser.add_argument("--timeout", type=float, default=300.0, metavar="SECONDS")
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        metavar="DIR",
        help="output directory for eval.md and eval.json",
    )
    args = parser.parse_args()

    try:
        prompts = load_prompts(args.prompts)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.tokens <= 0:
        print("error: --tokens must be greater than zero", file=sys.stderr)
        return 2

    binary = args.binary.expanduser().resolve()
    tokenizer = args.tokenizer.expanduser().resolve()
    if not binary.is_file():
        print(f"error: binary not found: {binary}", file=sys.stderr)
        return 2
    if not tokenizer.is_file():
        print(f"error: tokenizer not found: {tokenizer}", file=sys.stderr)
        return 2

    checkpoints: list[Path] = []
    for value in args.checkpoint:
        base = Path(value).expanduser()
        if base.suffix in {".bpk", ".json"}:
            base = base.with_suffix("")
        base = base.resolve()
        if not base.with_suffix(".bpk").is_file():
            print(f"error: checkpoint not found: {base.with_suffix('.bpk')}", file=sys.stderr)
            return 2
        checkpoints.append(base)

    report: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "binary": str(binary),
        "tokenizer": str(tokenizer),
        "tokens": args.tokens,
        "prompts_file": str(args.prompts.resolve()),
        "checkpoints": [],
    }

    failures = 0
    for checkpoint in checkpoints:
        entry: dict[str, Any] = {
            "name": checkpoint.name,
            "path": str(checkpoint),
            "metadata": load_metadata(checkpoint),
            "samples": [],
        }
        for prompt in prompts:
            result = run_generate(
                binary, tokenizer, checkpoint, args.tokens, prompt["text"], args.timeout
            )
            result.update({"prompt_id": prompt["id"], "prompt": prompt["text"]})
            entry["samples"].append(result)
            if not result["ok"]:
                failures += 1
                print(
                    f"{checkpoint.name} / {prompt['id']}: ERROR {result['error']}",
                    file=sys.stderr,
                )
            else:
                preview = (result["text"] or "").replace("\n", " ")[:60]
                print(f"{checkpoint.name} / {prompt['id']}: {preview}")
        report["checkpoints"].append(entry)

    out_dir = args.out.expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "eval.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    write_markdown(out_dir / "eval.md", report)

    print(f"\nReport: {out_dir / 'eval.md'}")
    print(f"JSON:   {out_dir / 'eval.json'}")
    print(f"Runs:   {len(checkpoints) * len(prompts)} ({failures} failed)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
