#!/usr/bin/env python3
"""Compare Mikuru Rust trainer checkpoint metadata without touching weights."""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any


CONFIG_FIELDS = (
    "dataset_path",
    "validation_path",
    "tokenizer_path",
    "seed",
    "epochs",
    "vocab_size",
    "model_dim",
    "num_layers",
    "num_heads",
    "max_seq_len",
    "learning_rate",
)


def metadata_path_from_argument(value: str) -> Path:
    path = Path(value).expanduser()
    if path.suffix.lower() == ".json":
        return path
    if path.suffix.lower() == ".bpk":
        return path.with_suffix(".json")
    if path.suffix:
        return path.with_suffix(".json")
    return Path(f"{path}.json")


def first_value(data: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = data.get(key)
        if value is not None:
            return value
    return None


def number(data: dict[str, Any], *keys: str) -> float | None:
    value = first_value(data, *keys)
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{keys[0]} must be a number, not a boolean")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{keys[0]} must be numeric; got {value!r}") from exc
    if not math.isfinite(result):
        raise ValueError(f"{keys[0]} must be finite; got {value!r}")
    return result


def integer(data: dict[str, Any], key: str) -> int | None:
    value = data.get(key)
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{key} must be an integer, not a boolean")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be an integer; got {value!r}") from exc
    if result < 0:
        raise ValueError(f"{key} must not be negative; got {result}")
    return result


def load_experiment(argument: str) -> dict[str, Any]:
    metadata_file = metadata_path_from_argument(argument)
    if not metadata_file.is_file():
        raise FileNotFoundError(f"metadata file not found: {metadata_file}")

    try:
        metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {metadata_file}: {exc}") from exc

    if not isinstance(metadata, dict):
        raise ValueError(f"expected a JSON object in {metadata_file}")

    # Ignore unrelated JSON files, but reject a likely checkpoint sidecar
    # whose metadata schema is not one this tool understands.
    required = ("epochs", "model_dim", "num_layers", "num_heads", "max_seq_len")
    missing = [key for key in required if key not in metadata]
    if missing:
        raise ValueError(
            f"{metadata_file} does not look like Mikuru checkpoint metadata; "
            f"missing: {', '.join(missing)}"
        )

    train_loss = number(metadata, "final_loss", "final_train_loss")
    validation_loss = number(metadata, "validation_loss", "final_validation_loss")
    initial_loss = number(metadata, "initial_loss")

    if train_loss is None:
        raise ValueError(f"{metadata_file} has no final training loss")

    base = metadata_file.with_suffix("")
    improvement = None
    improvement_percent = None
    if initial_loss is not None:
        improvement = initial_loss - train_loss
        if initial_loss != 0:
            improvement_percent = (improvement / initial_loss) * 100.0

    gap = None
    if validation_loss is not None:
        gap = validation_loss - train_loss

    record: dict[str, Any] = {
        "name": base.name,
        "metadata_file": str(metadata_file),
        "checkpoint_file": str(base.with_suffix(".bpk")),
        "format_version": metadata.get("format_version"),
        "epochs": integer(metadata, "epochs"),
        "learning_rate": number(metadata, "learning_rate"),
        "initial_train_loss": initial_loss,
        "final_train_loss": train_loss,
        "final_validation_loss": validation_loss,
        "train_improvement": improvement,
        "train_improvement_percent": improvement_percent,
        "generalization_gap": gap,
    }

    for key in (
        "dataset_path",
        "tokenizer_path",
        "split_description",
    ):
        record[key] = metadata.get(key)

    for key in (
        "vocab_size",
        "model_dim",
        "num_layers",
        "num_heads",
        "max_seq_len",
        "seed",
        "train_records",
        "validation_records",
    ):
        record[key] = metadata.get(key)

    record["_config"] = tuple(metadata.get(key) for key in CONFIG_FIELDS)
    return record


def show_value(value: Any, digits: int = 5) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def print_report(records: list[dict[str, Any]]) -> None:
    ranked = sorted(
        records,
        key=lambda item: (
            item["final_validation_loss"] is None,
            item["final_validation_loss"]
            if item["final_validation_loss"] is not None
            else item["final_train_loss"],
        ),
    )

    print("\nMIKURU EXPERIMENT COMPARISON")
    print("=" * 78)
    header = (
        f"{'Rank':>4}  {'Checkpoint':<25} {'Epochs':>6} "
        f"{'Train':>10} {'Validation':>10} {'Val-Train':>10} {'Train gain':>11}"
    )
    print(header)
    print("-" * len(header))

    for rank, item in enumerate(ranked, start=1):
        gain = item["train_improvement"]
        gain_text = show_value(gain, 4)
        print(
            f"{rank:>4}  {item['name']:<25.25} "
            f"{show_value(item['epochs'], 0):>6} "
            f"{show_value(item['final_train_loss']):>10} "
            f"{show_value(item['final_validation_loss']):>10} "
            f"{show_value(item['generalization_gap']):>10} "
            f"{gain_text:>11}"
        )

    print("\nDETAILS")
    for item in records:
        print(f"\n[{item['name']}]")
        print(f"  Metadata:          {item['metadata_file']}")
        print(f"  Checkpoint:        {item['checkpoint_file']}")
        print(f"  Architecture:      vocab={show_value(item['vocab_size'], 0)}, "
              f"dim={show_value(item['model_dim'], 0)}, "
              f"layers={show_value(item['num_layers'], 0)}, "
              f"heads={show_value(item['num_heads'], 0)}, "
              f"max_seq_len={show_value(item['max_seq_len'], 0)}")
        print(f"  Learning rate:     {show_value(item['learning_rate'])}")
        print(f"  Initial train:     {show_value(item['initial_train_loss'])}")
        print(f"  Final train:       {show_value(item['final_train_loss'])}")
        print(f"  Final validation:  {show_value(item['final_validation_loss'])}")
        if item["train_improvement"] is not None:
            print(
                f"  Train improvement:{item['train_improvement']:>10.5f} "
                f"({item['train_improvement_percent']:.2f}%)"
            )
        if item["generalization_gap"] is not None:
            print(f"  Val-train gap:     {item['generalization_gap']:.5f}")
        print(f"  Training records:  {show_value(item['train_records'], 0)}")
        print(f"  Validation records:{show_value(item['validation_records'], 0):>10}")
        print(f"  Dataset:           {item['dataset_path'] or 'n/a'}")
        print(f"  Split:             {item['split_description'] or 'n/a'}")

    configs = {item["_config"] for item in records}
    print("\nINTERPRETATION")
    if len(configs) == 1:
        print("  Recorded configuration fields match across these runs.")
    else:
        print("  Runs have different recorded configurations; compare them cautiously.")
    print("  Lower validation loss is better when the validation sets are comparable.")
    print("  A positive Val-Train gap is a generalization gap, not proof of overfitting.")
    print("  The current metadata has final losses only; epoch-by-epoch trends are unavailable.")


def write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    columns = [
        "name",
        "epochs",
        "learning_rate",
        "initial_train_loss",
        "final_train_loss",
        "final_validation_loss",
        "train_improvement",
        "train_improvement_percent",
        "generalization_gap",
        "vocab_size",
        "model_dim",
        "num_layers",
        "num_heads",
        "max_seq_len",
        "train_records",
        "validation_records",
        "dataset_path",
        "metadata_file",
        "checkpoint_file",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare Mikuru Rust trainer checkpoint metadata (read-only)."
    )
    parser.add_argument(
        "checkpoints",
        nargs="+",
        metavar="CHECKPOINT",
        help="checkpoint base path, .bpk path, or metadata .json path",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        metavar="FILE",
        help="also write a CSV comparison report",
    )
    args = parser.parse_args()

    records: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for argument in args.checkpoints:
        try:
            record = load_experiment(argument)
            metadata_file = Path(record["metadata_file"]).resolve()
            if metadata_file in seen:
                raise ValueError(f"checkpoint supplied more than once: {metadata_file}")
            seen.add(metadata_file)
            records.append(record)
        except (OSError, ValueError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2

    print_report(records)
    if args.csv:
        try:
            write_csv(args.csv, records)
        except OSError as exc:
            print(f"error writing CSV: {exc}", file=sys.stderr)
            return 2
        print(f"\nCSV report written: {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
