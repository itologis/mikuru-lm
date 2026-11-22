#!/usr/bin/env python3
"""Mission 3A fixed-prompt evaluation suite for Mikuru checkpoints.

Runs the forty prompts in ``m3a_prompts.json`` against one or more
checkpoints with deterministic greedy and/or seeded nucleus decoding, then
writes:

- ``eval.json``  full machine-readable results and metrics
- ``eval.md``    human-readable report with samples and metric tables
- ``scores.csv`` blank human-scoring sheet (1-5 rubric)
- ``README.md``  run configuration and scoring rubric

The harness is read-only with respect to the repository. It never trains and
never edits a checkpoint.

Usage:
    python3 tools/preprocess/m3a_eval.py \
        --checkpoint ~/mikuru-experiments/phase4-2026-11-19/batch32-5ep/model \
        --checkpoint ~/mikuru-experiments/phase4-2026-11-19/m2-batch32-50ep-nostop/model \
        --checkpoint ~/mikuru-experiments/phase4-2026-11-19/fullbatch-100ep-control/model \
        --out ~/mikuru-experiments/phase4-2026-11-19/m3a-eval
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import statistics
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BINARY = REPO_ROOT / "target" / "release" / "mikuru-lm"
DEFAULT_TOKENIZER = REPO_ROOT / "artifacts" / "mikuru-bpe-v2.json"
DEFAULT_PROMPTS = Path(__file__).resolve().parent / "m3a_prompts.json"

WORD_RE = re.compile(r"[A-Za-z0-9']+")
TERMINAL_RE = re.compile(r"[.!?。！？…][\"')\]]?\s*$")
ROLE_MARKER_RE = re.compile(r"(?i)\b(user|assistant)\s*:")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_prompts(path: Path) -> tuple[dict[str, Any], list[dict[str, str]]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    prompts = data.get("prompts")
    categories = data.get("categories")

    if not isinstance(prompts, list) or not prompts:
        raise ValueError(f"{path} must contain a non-empty `prompts` array")
    if not isinstance(categories, dict) or not categories:
        raise ValueError(f"{path} must contain a non-empty `categories` object")

    seen: set[str] = set()
    cleaned: list[dict[str, str]] = []
    per_category: Counter[str] = Counter()

    for index, entry in enumerate(prompts, start=1):
        if not isinstance(entry, dict):
            raise ValueError(f"prompt {index} must be an object")

        prompt_id = entry.get("id")
        text = entry.get("text")
        category = entry.get("category")

        if not isinstance(prompt_id, str) or not prompt_id.strip():
            raise ValueError(f"prompt {index} needs a non-empty string `id`")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"prompt `{prompt_id}` needs a non-empty string `text`")
        if not isinstance(category, str) or category not in categories:
            raise ValueError(f"prompt `{prompt_id}` has unknown category `{category}`")
        if prompt_id in seen:
            raise ValueError(f"duplicate prompt id `{prompt_id}`")

        seen.add(prompt_id)
        per_category[category] += 1
        cleaned.append({"id": prompt_id, "text": text, "category": category})

    if len(cleaned) != 40:
        raise ValueError(f"Mission 3A requires exactly 40 prompts, found {len(cleaned)}")
    if len(per_category) != 8:
        raise ValueError(f"Mission 3A requires 8 categories, found {len(per_category)}")

    uneven = {name: count for name, count in per_category.items() if count != 5}
    if uneven:
        raise ValueError(f"each category must have 5 prompts, found {uneven}")

    order = list(categories)
    for prompt in cleaned:
        prompt["category_index"] = str(order.index(prompt["category"]))

    return data, cleaned


def checkpoint_name(base: Path) -> str:
    if base.stem == "model" and base.parent.name:
        return base.parent.name
    return base.stem


def load_metadata(base: Path) -> dict[str, Any]:
    path = base.with_suffix(".json")
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def build_command(
    binary: Path,
    tokenizer: Path,
    base: Path,
    tokens: int,
    mode: dict[str, Any],
    prompt: str,
) -> list[str]:
    command = [
        str(binary),
        "generate",
        str(tokenizer),
        str(base),
        str(tokens),
    ]

    if mode["name"] == "greedy":
        command.append("--greedy")
    else:
        command.extend(
            [
                "--temperature",
                f"{mode['temperature']:.6f}",
                "--top-p",
                f"{mode['top_p']:.6f}",
                "--seed",
                str(mode["seed"]),
            ]
        )

    command.extend(["--json", prompt])
    return command


def parse_stdout_json(stdout: str) -> dict[str, Any] | None:
    for line in reversed(stdout.splitlines()):
        candidate = line.strip()
        if not candidate.startswith("{"):
            continue
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def run_generate(
    binary: Path,
    tokenizer: Path,
    base: Path,
    tokens: int,
    mode: dict[str, Any],
    prompt: str,
    timeout: float,
) -> dict[str, Any]:
    command = build_command(binary, tokenizer, base, tokens, mode, prompt)
    started = time.perf_counter()

    try:
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return {
            "ok": False,
            "error": str(error),
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            "text": None,
        }

    latency_ms = round((time.perf_counter() - started) * 1000, 2)

    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        return {
            "ok": False,
            "error": f"exit {completed.returncode}: {detail[-500:]}",
            "latency_ms": latency_ms,
            "text": None,
        }

    payload = parse_stdout_json(completed.stdout)
    if payload is None:
        return {
            "ok": False,
            "error": "no JSON payload in generate output",
            "latency_ms": latency_ms,
            "text": None,
        }

    text = payload.get("text")
    if not isinstance(text, str):
        return {
            "ok": False,
            "error": "JSON payload has no string `text`",
            "latency_ms": latency_ms,
            "text": None,
        }

    return {
        "ok": True,
        "error": None,
        "latency_ms": latency_ms,
        "text": text,
        "prompt_tokens": payload.get("prompt_tokens"),
        "generated_tokens": payload.get("generated_tokens"),
        "stop_reason": payload.get("stop_reason"),
    }


def words_of(text: str) -> list[str]:
    return WORD_RE.findall(text.lower())


def distinct_n(words: list[str], n: int) -> float:
    if len(words) < n:
        return 0.0
    grams = [tuple(words[index : index + n]) for index in range(len(words) - n + 1)]
    return len(set(grams)) / len(grams)


def repeated_ngram_fraction(words: list[str], n: int) -> float:
    if len(words) < n:
        return 0.0
    grams = [tuple(words[index : index + n]) for index in range(len(words) - n + 1)]
    return 1.0 - len(set(grams)) / len(grams)


def max_word_run(words: list[str]) -> int:
    if not words:
        return 0
    best = current = 1
    for previous, word in zip(words, words[1:]):
        current = current + 1 if word == previous else 1
        best = max(best, current)
    return best


def repeated_line_fraction(text: str) -> float:
    lines = [line.strip().lower() for line in text.splitlines() if line.strip()]
    if len(lines) < 2:
        return 0.0
    counts = Counter(lines)
    return 1.0 - len(counts) / len(lines)


def measure(text: str) -> dict[str, Any]:
    words = words_of(text)
    role_markers = ROLE_MARKER_RE.findall(text)
    return {
        "chars": len(text),
        "words": len(words),
        "lines": len([line for line in text.splitlines() if line.strip()]),
        "empty": not text.strip(),
        "distinct_1": round(distinct_n(words, 1), 4),
        "distinct_2": round(distinct_n(words, 2), 4),
        "repeated_4gram_fraction": round(repeated_ngram_fraction(words, 4), 4),
        "max_word_run": max_word_run(words),
        "repeated_line_fraction": round(repeated_line_fraction(text), 4),
        "ends_with_terminal_punctuation": bool(TERMINAL_RE.search(text)),
        "role_marker_leak": bool(role_markers),
        "role_marker_count": len(role_markers),
        "starts_with_assistant_marker": text.lstrip().lower().startswith("assistant:"),
        "non_ascii_ratio": round(
            sum(1 for character in text if ord(character) > 127) / max(len(text), 1), 4
        ),
        "newline_count": text.count("\n"),
    }


def mean(values: list[float]) -> float | None:
    return round(statistics.fmean(values), 4) if values else None


def summarize(samples: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(samples)
    ok = [sample for sample in samples if sample["ok"]]
    ok_count = len(ok)

    def ratio(predicate) -> float:
        return round(sum(1 for sample in ok if predicate(sample)) / ok_count, 4) if ok_count else 0.0

    return {
        "samples": total,
        "ok": ok_count,
        "failures": total - ok_count,
        "empty": sum(1 for sample in ok if sample["empty"]),
        "empty_rate": ratio(lambda sample: sample["empty"]),
        "eos_rate": ratio(lambda sample: sample["stop_reason"] == "eos"),
        "token_limit_rate": ratio(lambda sample: sample["stop_reason"] == "token_limit"),
        "structural_special_rate": ratio(
            lambda sample: sample["stop_reason"] == "structural_special"
        ),
        "mean_chars": mean([sample["chars"] for sample in ok]),
        "mean_words": mean([sample["words"] for sample in ok]),
        "mean_distinct_1": mean([sample["distinct_1"] for sample in ok]),
        "mean_distinct_2": mean([sample["distinct_2"] for sample in ok]),
        "mean_repeated_4gram_fraction": mean(
            [sample["repeated_4gram_fraction"] for sample in ok]
        ),
        "mean_max_word_run": mean([float(sample["max_word_run"]) for sample in ok]),
        "mean_repeated_line_fraction": mean(
            [sample["repeated_line_fraction"] for sample in ok]
        ),
        "role_marker_leak_rate": ratio(lambda sample: sample["role_marker_leak"]),
        "mean_role_marker_count": mean(
            [float(sample["role_marker_count"]) for sample in ok]
        ),
        "starts_with_assistant_marker_rate": ratio(
            lambda sample: sample["starts_with_assistant_marker"]
        ),
        "terminal_punctuation_rate": ratio(
            lambda sample: sample["ends_with_terminal_punctuation"]
        ),
        "mean_latency_ms": mean([sample["latency_ms"] for sample in ok]),
    }


def metric(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def preview(text: str | None, limit: int = 180) -> str:
    if text is None:
        return "(generation failed)"
    flat = " ".join(text.split())
    if not flat:
        return "(empty)"
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


def write_markdown(
    path: Path,
    report: dict[str, Any],
    prompts: list[dict[str, str]],
    category_order: list[str],
) -> None:
    run = report["run"]
    lines: list[str] = [
        "# Mission 3A — fixed-prompt evaluation",
        "",
        f"- Generated: {run['generated_at']}",
        f"- Binary: `{run['binary']}` (`{run['binary_sha256'][:12]}…`)",
        f"- Tokenizer: `{run['tokenizer']}` (`{run['tokenizer_sha256'][:12]}…`)",
        f"- Prompts: `{run['prompts_file']}` (`{run['prompts_sha256'][:12]}…`)",
        f"- New tokens per prompt: {run['tokens']}",
        f"- Decoding: {' / '.join(mode_label(mode) for mode in run['modes'])}",
        "",
        "> Generation is deterministic for each `(prompt, mode, seed)` pair on this",
        "> binary and backend. Greedy is the legacy v2 behaviour.",
        "",
        "## Checkpoints",
        "",
        "| Checkpoint | Epochs | Best epoch | Val loss | Final loss | Batch | Checkpoint SHA-256 |",
        "| --- | ---: | ---: | ---: | ---: | --- | --- |",
    ]

    for entry in report["checkpoints"]:
        metadata = entry["metadata"]
        lines.append(
            "| `{name}` | {epochs} | {best} | {val} | {final} | {batch} | `{sha}…` |".format(
                name=entry["name"],
                epochs=metric(metadata.get("epochs")),
                best=metric(metadata.get("best_epoch")),
                val=metric(metadata.get("validation_loss"), 4),
                final=metric(metadata.get("final_loss"), 4),
                batch=metadata.get("batch_size") or "full",
                sha=entry["checkpoint_sha256"][:12],
            )
        )

    lines.extend(
        [
            "",
            "## Generation health",
            "",
            "| Checkpoint | Mode | OK | Empty | EOS | Limit | Role leak | Mean chars | Distinct-1 | Distinct-2 | Rep. 4-gram | Max run | Terminal punct. |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )

    for entry in report["checkpoints"]:
        for mode in report["run"]["modes"]:
            summary = entry["modes"][mode["name"]]["overall"]
            lines.append(
                "| `{name}` | {mode} | {ok}/{total} | {empty} | {eos} | {limit} | {leak} | {chars} | {d1} | {d2} | {rep} | {run} | {terminal} |".format(
                    name=entry["name"],
                    mode=mode["name"],
                    ok=summary["ok"],
                    total=summary["samples"],
                    empty=summary["empty"],
                    eos=f"{summary['eos_rate']:.0%}",
                    limit=f"{summary['token_limit_rate']:.0%}",
                    leak=f"{summary['role_marker_leak_rate']:.0%}",
                    chars=metric(summary["mean_chars"], 1),
                    d1=metric(summary["mean_distinct_1"]),
                    d2=metric(summary["mean_distinct_2"]),
                    rep=metric(summary["mean_repeated_4gram_fraction"]),
                    run=metric(summary["mean_max_word_run"], 1),
                    terminal=f"{summary['terminal_punctuation_rate']:.0%}",
                )
            )

    lines.extend(["", "## Metrics by category", ""])

    for entry in report["checkpoints"]:
        for mode in report["run"]["modes"]:
            lines.append(f"### {entry['name']} — {mode['name']}")
            lines.append("")
            lines.append("| Category | Words | Distinct-1 | Rep. 4-gram | Max run | EOS | Terminal punct. |")
            lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")

            by_category = entry["modes"][mode["name"]]["by_category"]
            for category in category_order:
                summary = by_category[category]
                lines.append(
                    "| {category} | {words} | {d1} | {rep} | {run} | {eos} | {terminal} |".format(
                        category=category,
                        words=metric(summary["mean_words"], 1),
                        d1=metric(summary["mean_distinct_1"]),
                        rep=metric(summary["mean_repeated_4gram_fraction"]),
                        run=metric(summary["mean_max_word_run"], 1),
                        eos=f"{summary['eos_rate']:.0%}",
                        terminal=f"{summary['terminal_punctuation_rate']:.0%}",
                    )
                )
            lines.append("")

    lines.extend(["## Samples", ""])
    lines.append(
        "Preview only; full text is in `eval.json`. Grouped by category so a "
        "reader can compare the same question across checkpoints."
    )
    lines.append("")

    for entry in report["checkpoints"]:
        lines.append(f"### {entry['name']}")
        lines.append("")

        samples_by_key = {
            mode["name"]: {
                sample["prompt_id"]: sample
                for sample in entry["modes"][mode["name"]]["samples"]
            }
            for mode in report["run"]["modes"]
        }

        for category in category_order:
            category_prompts = [prompt for prompt in prompts if prompt["category"] == category]
            if not category_prompts:
                continue

            lines.append(f"#### {category}")
            lines.append("")
            header = "| Prompt | " + " | ".join(
                mode["name"] for mode in report["run"]["modes"]
            ) + " |"
            separator = "| --- | " + " | ".join(
                "---" for _ in report["run"]["modes"]
            ) + " |"
            lines.append(header)
            lines.append(separator)

            for prompt in category_prompts:
                cells = []
                for mode in report["run"]["modes"]:
                    sample = samples_by_key[mode["name"]][prompt["id"]]
                    cells.append(preview(sample.get("text")).replace("|", "\\|"))
                lines.append(f"| `{prompt['id']}`: {prompt['text'].replace(chr(10), ' / ')} | " + " | ".join(cells) + " |")

            lines.append("")

    lines.extend(
        [
            "## Human scoring",
            "",
            "`scores.csv` has one row per checkpoint × mode × prompt with blank",
            "1–5 columns. Score each response on:",
            "",
            "| Dimension | 1 | 3 | 5 |",
            "| --- | --- | --- | --- |",
            "| Relevance | ignores the prompt | partly answers | directly answers |",
            "| Coherence | word salad | understandable, flawed | clear and consistent |",
            "| Naturalness | robotic or broken | stiff | human-like |",
            "| Persona fit | generic or contradictory | flashes of voice | stable warm Mikuru voice |",
            "",
            "Acceptance gate for a data or recipe change: mean overall ≥ 3.0 and",
            "< 5% unusable (score 1) responses. Do not promote a checkpoint on",
            "validation loss alone.",
            "",
        ]
    )

    path.write_text("\n".join(lines), encoding="utf-8")


def mode_label(mode: dict[str, Any]) -> str:
    if mode["name"] == "greedy":
        return "greedy"
    return (
        f"nucleus (T={mode['temperature']}, top_p={mode['top_p']}, seed={mode['seed']})"
    )


def write_scores_csv(
    path: Path,
    report: dict[str, Any],
    prompts_by_id: dict[str, dict[str, str]],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "checkpoint",
                "mode",
                "category",
                "prompt_id",
                "prompt",
                "response_preview",
                "relevance_1_5",
                "coherence_1_5",
                "naturalness_1_5",
                "persona_1_5",
                "overall_1_5",
                "notes",
            ]
        )

        for entry in report["checkpoints"]:
            for mode in report["run"]["modes"]:
                for sample in entry["modes"][mode["name"]]["samples"]:
                    prompt = prompts_by_id[sample["prompt_id"]]
                    writer.writerow(
                        [
                            entry["name"],
                            mode["name"],
                            sample["category"],
                            sample["prompt_id"],
                            prompt["text"],
                            preview(sample.get("text"), 240),
                            "",
                            "",
                            "",
                            "",
                            "",
                            "",
                        ]
                    )


def write_readme(path: Path, report: dict[str, Any]) -> None:
    run = report["run"]
    lines = [
        "# Mission 3A evaluation bundle",
        "",
        "Fixed-prompt conversational evaluation for the Mikuru LM checkpoints.",
        "",
        "## Configuration",
        "",
        f"- Prompts: `{run['prompts_file']}` (40 prompts, 8 categories, 5 each)",
        f"- Tokenizer: `{run['tokenizer']}`",
        f"- New tokens per prompt: {run['tokens']}",
        f"- Sampling: {', '.join(mode_label(mode) for mode in run['modes'])}",
        f"- Binary SHA-256: `{run['binary_sha256']}`",
        f"- Prompts SHA-256: `{run['prompts_sha256']}`",
        f"- Script SHA-256: `{run['script_sha256']}`",
        "",
        "## Files",
        "",
        "| File | Purpose |",
        "| --- | --- |",
        "| `eval.json` | full responses, per-sample metrics, aggregates, hashes |",
        "| `eval.md` | human-readable report: health tables, category tables, samples |",
        "| `scores.csv` | blank 1–5 human scoring sheet |",
        "| `README.md` | this file |",
        "",
        "## Reproduce",
        "",
        "```bash",
        "python3 tools/preprocess/m3a_eval.py \\",
        "  --checkpoint <base> [--checkpoint <base> ...] \\",
        "  --out <dir>",
        "```",
        "",
        "The runner calls `mikuru-lm generate ... --json` once per prompt/mode.",
        "Greedy is deterministic by construction. Nucleus sampling uses a",
        "SplitMix64 sampler seeded from `--seed`, so the same binary, prompt,",
        "and settings reproduce the same text exactly.",
        "",
        "## Scoring",
        "",
        "Use `scores.csv`. Score relevance, coherence, naturalness, and persona",
        "fit from 1 to 5, then record an overall score. The acceptance gate is",
        "mean overall ≥ 3.0 with fewer than 5% unusable (score 1) responses.",
        "Validation loss alone is never sufficient to promote a checkpoint.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the Mission 3A fixed prompts across Mikuru checkpoints."
    )
    parser.add_argument(
        "--checkpoint",
        action="append",
        required=True,
        metavar="BASE",
        help="checkpoint base path (without .bpk); repeat for comparison",
    )
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--tokens", type=int, default=48, metavar="N")
    parser.add_argument("--timeout", type=float, default=300.0, metavar="SECONDS")
    parser.add_argument(
        "--mode",
        action="append",
        choices=["greedy", "nucleus"],
        default=None,
        help="decoding mode; repeat to select both (default: both)",
    )
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, required=True, metavar="DIR")
    args = parser.parse_args()

    try:
        prompt_document, prompts = load_prompts(args.prompts)
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    if args.tokens <= 0:
        print("error: --tokens must be greater than zero", file=sys.stderr)
        return 2
    if not (0.0 < args.top_p <= 1.0):
        print("error: --top-p must be in (0, 1]", file=sys.stderr)
        return 2
    if args.temperature <= 0.0:
        print("error: --temperature must be greater than zero", file=sys.stderr)
        return 2

    binary = args.binary.expanduser().resolve()
    tokenizer = args.tokenizer.expanduser().resolve()
    prompts_path = args.prompts.expanduser().resolve()

    if not binary.is_file():
        print(f"error: binary not found: {binary}", file=sys.stderr)
        return 2
    if not tokenizer.is_file():
        print(f"error: tokenizer not found: {tokenizer}", file=sys.stderr)
        return 2

    modes: list[dict[str, Any]] = []
    selected = args.mode or ["greedy", "nucleus"]
    for name in dict.fromkeys(selected):
        if name == "greedy":
            modes.append({"name": "greedy"})
        else:
            modes.append(
                {
                    "name": "nucleus",
                    "temperature": round(args.temperature, 3),
                    "top_p": round(args.top_p, 3),
                    "seed": args.seed,
                }
            )

    checkpoints: list[Path] = []
    for value in args.checkpoint:
        base = Path(value).expanduser()
        if base.suffix in {".bpk", ".json"}:
            base = base.with_suffix("")
        base = base.resolve()
        if not base.with_suffix(".bpk").is_file() or not base.with_suffix(".json").is_file():
            print(f"error: checkpoint not found: {base}.bpk or .json", file=sys.stderr)
            return 2
        checkpoints.append(base)

    category_order = list(prompt_document["categories"])
    report: dict[str, Any] = {
        "suite": {
            "version": prompt_document.get("version"),
            "prompts": len(prompts),
            "categories": {
                category: sum(1 for prompt in prompts if prompt["category"] == category)
                for category in category_order
            },
        },
        "run": {
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "binary": str(binary),
            "binary_sha256": sha256_file(binary),
            "tokenizer": str(tokenizer),
            "tokenizer_sha256": sha256_file(tokenizer),
            "prompts_file": str(prompts_path),
            "prompts_sha256": sha256_file(prompts_path),
            "script_sha256": sha256_file(Path(__file__).resolve()),
            "tokens": args.tokens,
            "modes": modes,
        },
        "checkpoints": [],
    }

    failures = 0

    for base in checkpoints:
        name = checkpoint_name(base)
        entry: dict[str, Any] = {
            "name": name,
            "base": str(base),
            "checkpoint_sha256": sha256_file(base.with_suffix(".bpk")),
            "metadata": load_metadata(base),
            "modes": {},
        }

        for mode in modes:
            samples: list[dict[str, Any]] = []

            for prompt in prompts:
                result = run_generate(
                    binary,
                    tokenizer,
                    base,
                    args.tokens,
                    mode,
                    prompt["text"],
                    args.timeout,
                )
                sample: dict[str, Any] = {
                    "prompt_id": prompt["id"],
                    "category": prompt["category"],
                    "prompt": prompt["text"],
                    "mode": mode["name"],
                }

                if result["ok"]:
                    sample.update(measure(result["text"]))
                    sample.update(
                        {
                            "text": result["text"],
                            "prompt_tokens": result.get("prompt_tokens"),
                            "generated_tokens": result.get("generated_tokens"),
                            "stop_reason": result.get("stop_reason"),
                            "latency_ms": result["latency_ms"],
                        }
                    )
                else:
                    failures += 1
                    sample.update(
                        {
                            "text": result.get("text"),
                            "prompt_tokens": None,
                            "generated_tokens": None,
                            "stop_reason": None,
                            "latency_ms": result["latency_ms"],
                            "chars": 0,
                            "words": 0,
                            "lines": 0,
                            "empty": True,
                            "distinct_1": 0.0,
                            "distinct_2": 0.0,
                            "repeated_4gram_fraction": 0.0,
                            "max_word_run": 0,
                            "repeated_line_fraction": 0.0,
                            "ends_with_terminal_punctuation": False,
                            "role_marker_leak": False,
                            "role_marker_count": 0,
                            "starts_with_assistant_marker": False,
                            "non_ascii_ratio": 0.0,
                            "newline_count": 0,
                        }
                    )
                    print(f"{name} / {mode['name']} / {prompt['id']}: {result['error']}", file=sys.stderr)

                if "error" in result and result["error"]:
                    sample["error"] = result["error"]
                else:
                    sample["error"] = None

                sample["ok"] = bool(result["ok"])
                samples.append(sample)

            by_category = {
                category: summarize(
                    [sample for sample in samples if sample["category"] == category]
                )
                for category in category_order
            }

            entry["modes"][mode["name"]] = {
                "overall": summarize(samples),
                "by_category": by_category,
                "samples": samples,
            }

            preview_text = preview(
                next(
                    (
                        sample["text"]
                        for sample in samples
                        if sample["prompt_id"] == "persona-01"
                    ),
                    None,
                ),
                60,
            )
            print(
                f"{name} / {mode['name']}: {entry['modes'][mode['name']]['overall']['ok']}/{len(samples)} ok | persona-01: {preview_text}"
            )

        report["checkpoints"].append(entry)

    report["run"]["failures"] = failures

    out_dir = args.out.expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "eval.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    write_markdown(out_dir / "eval.md", report, prompts, category_order)
    write_scores_csv(
        out_dir / "scores.csv",
        report,
        {prompt["id"]: prompt for prompt in prompts},
    )
    write_readme(out_dir / "README.md", report)

    print()
    print(f"Report: {out_dir / 'eval.md'}")
    print(f"JSON:   {out_dir / 'eval.json'}")
    print(f"Scores: {out_dir / 'scores.csv'}")
    print(
        f"Runs:   {len(checkpoints) * len(modes) * len(prompts)} "
        f"({failures} failed)"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
