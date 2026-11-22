# Changelog

## Unreleased - 2026-11-18

- Added opt-in deterministic mini-batch training (`train --batch-size N`).
  One optimizer step per batch, token-weighted loss, final partial batch
  included, and a per-epoch Fisher-Yates shuffle driven by SplitMix64
  (`seed ^ epoch`). The legacy full-batch path remains the default and is
  unchanged. See `docs/decisions/0005-deterministic-mini-batch-training.md`.
- Checkpoint metadata now records optional batching and performance fields:
  `batch_size`, `batching_policy`, `steps_per_epoch`, `total_steps`,
  step/epoch time, `peak_rss_kb`, `swap_kb`, and `cpu_utilization_percent`.
  `format_version` stays 1.
- Continuation runs inherit `batch_size` from the source checkpoint metadata.
- Added read-only Python audit tools under `tools/preprocess/`:
  `validate_jsonl.py` (schema mirror with `--stats`/`--json`) and
  `eval_suite.py` (fixed-prompt checkpoint comparison). Stdlib only; the Rust
  loader remains the authority.
- Extended `tools/analyze_experiments.py` to display the new batching and
  performance fields.

## Unreleased - 2026-11-16

- Added dataset loading with required-metadata validation, duplicate-id
  rejection, and strict split-label parsing (`train`, `validation`, legacy
  `demo`).
- Added explicit split partitioning, deterministic seeded splitting, JSONL
  writing, and manifest generation in `mikuru-core`.
- Added `prepare-dataset` CLI command for reproducible train/validation
  preparation. Outputs are never overwritten.
- Added `train --validation <path>` and prepared-directory input. Training now
  reports training records, validation records, and validation loss. Validation
  evaluation performs no optimizer updates.
- Added backward-compatible checkpoint metadata fields: split description,
  record counts, validation path, and validation loss.
- Tokenizer training now excludes records labelled `validation`.
- `inspect-jsonl` now uses the shared loader, so duplicate ids and unknown
  split labels are reported consistently.
- Added `data/prepared/v1`: 1 train / 1 validation record prepared from the two
  internal demo records. Pipeline validation only; see its README for the
  unapproved/missing-source audit.
- Documented that Burn Flex 0.22.1 `Backend::seed` is a no-op, so model weight
  initialisation is not reproducible on this backend.

### Dataset Expansion v2 — OASST1

- Verified OASST1 licensing against the dataset card **and** the repository
  `LICENSE` file (Apache-2.0, 11,357 bytes, SHA-256 `c71d239d…`) and pinned
  revision `fdf72ae0827c1cda404aff25b6603abec9e3399b`.
- Added `mikuru-core` `oasst1` module: message filtering (deleted, unreviewed,
  failed review, non-English, invalid role, synthetic, empty, toxic, PII,
  unsafe labels), deterministic best-path tree reconstruction with strict role
  alternation, and a JSONL importer. 16 new tests cover reconstruction,
  branching, filtering, tie-breaks, malformed trees, and provenance.
- Added `ConversationExample::source_message_ids`, validated against message
  count, so original OASST1 message ids survive import and preparation.
- Added `mikuru-lm import-oasst1 <messages.jsonl> <output.jsonl>
  [--max-chars N] [--max-messages N] [--revision SHA]`. The importer refuses to
  overwrite an existing output.
- Imported the canonical `2023-04-12_oasst_ready.messages.jsonl.gz` export
  (88,838 messages, 10,364 trees) into 335 English conversations (1,122
  messages, 110,007 characters). Rejection counts are documented in
  `data/prepared/v2/README.md`.
- Added `data/imports/oasst1-2023-04-12-en-v1.jsonl`, a verbatim OASST1
  `LICENSE` copy under `data/third-party/oasst1/`, and an attribution/modification
  notice.
- Added `data/prepared/v2`: 301 train / 34 validation, deterministic seed 42.
  v1 is untouched.
- Added `train --max-seq-len N` (default unchanged at 128). v2's longest record
  is 272 tokens, so its first run needs `--max-seq-len 288`.
- Trained `artifacts/mikuru-bpe-v2.json` (vocab 1024) on the v2 training split
  only. A 2-epoch smoke run produced `artifacts/oasst1-v2` (train loss
  7.12907 → 6.80933, validation 6.83059; reload difference 0.0000000), and a
  full 50-epoch run produced `artifacts/oasst1-v2-50` (train loss
  7.12907 → 4.89625, validation 5.32537).
- Marked OASST1 approved for internal training and redistribution of derived
  records, with PII and scale caveats documented. Added ADR 0003 for the
  reconstruction policy and updated the research report, attribution document,
  source manifest, prepared-data index, and root README.

### Taskmaster-1 — inspection and CONDITIONAL qualification

- Added `mikuru-core` `taskmaster1` module: a streaming JSON-array reader,
  per-record analysis, `USER`/`ASSISTANT` role mapping, duplicate-id and
  ordered speaker/text payload detection, contact-pattern heuristics,
  length/budget statistics, and official split coverage. No conversation text
  is written to any artifact.
- Added `mikuru-lm inspect-taskmaster1 <dataset_dir> <output_dir>
  [--revision SHA]`. It writes `taskmaster1-report.md`,
  `taskmaster1-report.json`, and a candidate `taskmaster1-manifest.jsonl`,
  refuses archives and existing outputs, and leaves source files read-only.
- Added 10 synthetic-fixture tests covering valid records, role mapping,
  unknown/missing speakers, empty and malformed records, duplicates, deleted
  placeholders, contact patterns, split coverage, nested directories,
  deterministic output, and overwrite refusal.
- Verified the official `google-research-datasets/Taskmaster` release at
  revision `d92cb6af3005f1dc09c39e75e7daf4a04905e00b`. The repository has no
  separate LICENSE file; `TM-1-2019/README.md` carries an explicit copyright
  notice applying CC BY 4.0 to the dialog data. A legal-code copy and the
  upstream README are stored under `data/third-party/taskmaster1/`.
- Inspected all official files: 13,215 records, 303,066 utterances,
  12,602,704 characters. Self-dialogs: 7,708 records, 6,755 valid, 953 flagged
  only for contact-like patterns. WOZ: 5,507 records, 440 valid, 5,063 flagged,
  4 invalid, including 1,038 deleted placeholders, 55 blank speakers, 6
  duplicate-id records, 11 payload-duplicate records, and 23,628 same-role
  adjacencies.
- Verified the official written splits cover all 7,708 self-dialog ids exactly
  (6,168 train / 770 dev / 770 test, no overlap). The WOZ subset has no
  official split list.
- Both subsets are **CONDITIONAL** and `record_count_used` remains 0. No
  Taskmaster-1 record was imported, no tokenizer or model artifact changed, and
  `artifacts/oasst1-v2-50.*` is untouched. Privacy policy, WOZ repair/dedup
  policy, the sequence-budget decision, and final training approval remain
  open. Added ADR 0004 and qualification artifacts under
  `data/third-party/taskmaster1/`.

## 0.1.0 - 2026-11-15

- Created initial Rust 2024 workspace.
- Added Burn Flex FP32/autodiff smoke-test command.
- Added JSONL record schema and sample inspector.
- Reserved separate Transformer and NCM crates.
- Documented locked stack decisions and open architecture questions.
