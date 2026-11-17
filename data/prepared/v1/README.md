# Dataset v1 — pipeline validation set

**Prepared:** 2026-11-16
**Status:** internal pipeline validation only — **not** a model-quality corpus.

## Contents

| Split | Records | Tokenizer use |
| --- | --- | --- |
| `train.jsonl` | 1 | Included |
| `validation.jsonl` | 1 | Excluded |

Both records are two-message English conversations. The split is deterministic:
seed `42`, `validation_fraction` `0.2`, one record per split (the fraction is
clamped so neither split can be empty).

## Provenance

| Field | Value |
| --- | --- |
| Source file | `data/samples/tiny.jsonl` |
| `source_dataset` | `Mikuru Demo` |
| `license` | `Project Internal` |
| `synthetic` | `true` |
| `source_url` | `N/A` (source-provided placeholder) |

The records are the original synthetic demonstrations authored for this
project. Their provenance is traceable but **`Project Internal` is not an open
license**. They are approved for internal experimentation only; redistribution
and public release are not approved.

Preparation preserved every metadata field and only assigned the `split` labels
that were already implied by the file. No text was generated, edited, or
reconstructed.

## Why the set is this small

An audit of `data/research/` found no other approved records on disk:

- **OASST1** — Apache-2.0 candidate, but raw text, label filtering, and privacy
  review are still pending. No records collected.
- **Taskmaster-1** — CC BY 4.0 candidate, official JSON never fetched. No
  records collected.
- **SOC-2508** — excluded pending a defensible role mapping.
- **Original (Culture) / Original (Style)** — listed in the research report as
  collected, but **no JSONL file contains them**. Their ids appear only in
  abbreviated unit-test fixtures in `crates/mikuru-core/src/dataset.rs`, whose
  text is not the collected record. They were deliberately not reconstructed:
  doing so would fabricate data under real record ids.
- **DailyDialog, EmpatheticDialogues, Marugoto/Irodori, Taskmaster-4** — excluded
  for license or format reasons in the research report.

Training on one record and evaluating on another produces a number, but not a
meaningful one. v1 exists to exercise the split, tokenizer-exclusion, validation
evaluation, and checkpoint-metadata paths end to end. Real training is blocked
on approved data, not on code.

## Reproducing

From the workspace root:

```bash
cargo run -p mikuru-cli -- prepare-dataset \
  data/samples/tiny.jsonl \
  data/prepared/v1 \
  --seed 42 \
  --validation-fraction 0.2
```

The command refuses to overwrite the existing files. Remove the directory
deliberately first if you intend to regenerate it.

## Manifest

`manifest.json` records:

- `split_source: "deterministic"`
- `split_seed: 42`
- `validation_fraction: 0.2`
- `train_records: 1`, `validation_records: 1`
- One source: `Mikuru Demo`, license `Project Internal`, 2 records, 2 synthetic

## Using it

```bash
# Tokenizer: train on the training split only.
cargo run -p mikuru-tokenizer -- train \
  data/prepared/v1/train.jsonl \
  artifacts/mikuru-bpe-v1.json 320

# Train: directory input picks up both splits automatically.
cargo run -p mikuru-cli -- train \
  data/prepared/v1 \
  artifacts/mikuru-bpe-v1.json \
  artifacts/expanded-v1 50
```

Expected training output now includes `Training records`, `Validation records`,
and both train and validation loss. With one record per split those numbers are
a plumbing check, not evidence of learning.
