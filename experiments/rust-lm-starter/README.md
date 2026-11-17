# Mikuru LM

A Rust workspace for experimenting with two language-model approaches:

1. **Autoregressive Transformer** — the first baseline.
2. **Next-Concept Modeling (NCM)** — a separate experimental path with its own multiscale residual VQ tokenizer.

**Status: Stage 2 — first external corpus imported.** A 96,230-parameter model
has memorized two demonstration conversations on CPU. Dataset preparation,
train/validation splits, and validation-loss evaluation are implemented.
Dataset Expansion v2 adds **335 English OASST1 conversations** (Apache-2.0,
license verified): 301 train / 34 validation, with a tokenizer trained on the
training split only. A full 50-epoch run produced `artifacts/oasst1-v2-50`
(train loss 7.12907 → 4.89625, validation 5.32537; checkpoint reload difference
0.0000014) — a working tiny baseline, not a quality model. v1 is preserved
unchanged at [`data/prepared/v1/`](data/prepared/v1/README.md).

## Locked initial choices

| Component | Decision |
|---|---|
| Language | Rust 2024 |
| ML framework | Burn 0.22.x |
| Initial CPU backend | Burn Flex |
| Transformer tokenizer | BPE |
| Initial training precision | FP32 |
| Dataset interchange | JSONL |
| Model weight format | SafeTensors |

> Implementation note: the current trainer writes Burnpack (`.bpk`) checkpoints
> through Burn's `save_file`. The SafeTensors decision in
> [ADR 0001](docs/decisions/0001-initial-stack.md) has not been reconciled yet.

The NCM tokenizer is not BPE. It will be implemented as a distinct multiscale
residual vector-quantization component, following the selected NCM paper and its
reference implementation.

## Requirements

- Rust stable toolchain with Cargo (Rust 1.85 or newer for edition 2024).
- Linux, macOS, or Windows.

No GPU backend is enabled in this starter workspace. Burn Flex is the first CPU
backend so the initial test stays portable and avoids adding GPU runtime
requirements.

## Checks

From this directory, run:

```bash
rustc --version
cargo --version
cargo check --workspace --all-targets
cargo test --workspace
cargo run -p mikuru-cli -- burn-smoke
cargo run -p mikuru-cli -- inspect-jsonl data/prepared/v1/train.jsonl
cargo run -p mikuru-cli -- inspect-jsonl data/prepared/v2/train.jsonl
```

`burn-smoke` checks that Burn can create FP32 tensors and calculate gradients on
the Flex backend. `inspect-jsonl` validates the JSONL schema, duplicate ids, and
split labels, then reports a summary without printing the full text.
`inspect-taskmaster1` audits an extracted official Taskmaster-1 directory and
writes deterministic qualification reports without importing anything:

```bash
cargo run -p mikuru-cli -- inspect-taskmaster1 \
  /tmp/opencode/taskmaster1-inspect /tmp/opencode/taskmaster1-qualification \
  --revision d92cb6af3005f1dc09c39e75e7daf4a04905e00b
```

## Dataset workflow

1. **Collect only approved records.** A documented source is not an approved
   source. See [`data/research/ATTRIBUTION_AND_LICENSES.md`](data/research/ATTRIBUTION_AND_LICENSES.md).
2. **Convert an external source into conversation records.** Importers live in
   `mikuru-core` and keep source ids, license, revision, and message ids:

   ```bash
   cargo run -p mikuru-cli -- import-oasst1 \
     /tmp/opencode/oasst1-ready.messages.jsonl \
     data/imports/oasst1-2023-04-12-en-v1.jsonl \
     --max-chars 500 --max-messages 8 \
     --revision fdf72ae0827c1cda404aff25b6603abec9e3399b
   ```

   The exact download and decompression commands are in
   [`data/prepared/v2/README.md`](data/prepared/v2/README.md) and
   [`data/third-party/oasst1/README.md`](data/third-party/oasst1/README.md).
   Importers never overwrite an existing output file.
3. **Prepare a split** — explicit labels are respected; otherwise a
   deterministic seeded split is generated:

   ```bash
   cargo run -p mikuru-cli -- prepare-dataset \
     data/samples/tiny.jsonl data/prepared/v1 --seed 42 --validation-fraction 0.2
   ```

   Outputs: `train.jsonl`, `validation.jsonl`, `manifest.json`. Existing files
   are never overwritten.
4. **Inspect** both splits. Duplicate ids, missing provenance, and unknown split
   labels are rejected.
5. **Train the tokenizer on the training split only.** The tokenizer CLI also
   filters records labelled `validation` from combined files:

   ```bash
   # v2 (vocab 1024, longest record 272 tokens)
   cargo run -p mikuru-tokenizer -- train \
     data/prepared/v2/train.jsonl artifacts/mikuru-bpe-v2.json 1024

   # v1 smoke tokenizer, used by artifacts/first-model
   cargo run -p mikuru-tokenizer -- train \
     data/prepared/v1/train.jsonl artifacts/mikuru-bpe.json 294
   ```

6. **Train** with a prepared directory, or with a separate validation file:

   ```bash
   # Prepared directory (uses train.jsonl and validation.jsonl)
   cargo run -p mikuru-cli -- train \
     data/prepared/v2 artifacts/mikuru-bpe-v2.json artifacts/oasst1-v2 50 \
     --max-seq-len 288

   # Single training file plus explicit validation file
   cargo run -p mikuru-cli -- train \
     data/prepared/v2/train.jsonl artifacts/mikuru-bpe-v2.json artifacts/oasst1-v2 50 \
     --validation data/prepared/v2/validation.jsonl --max-seq-len 288
   ```

   Training reports training records, validation records, and both losses.
   Validation performs a forward pass only — no gradients, no optimizer step.
   `--max-seq-len N` overrides the 128-token default; records longer than the
   context are rejected, never truncated. The v1 smoke run is the same command
   without `--max-seq-len`, using `artifacts/mikuru-bpe.json` and
   `data/prepared/v1`.

7. **Verify and generate** as before:

   ```bash
   cargo run -p mikuru-cli -- verify-checkpoint \
     data/samples/tiny.jsonl artifacts/mikuru-bpe.json artifacts/first-model

   cargo run -p mikuru-cli -- generate \
     artifacts/mikuru-bpe.json artifacts/first-model 64 "Hello, Mikuru!"
   ```

## Dataset status

| Version | Train | Validation | Source | Status |
| --- | ---: | ---: | --- | --- |
| [v1](data/prepared/v1/README.md) | 1 | 1 | `Mikuru Demo`, `Project Internal` | Pipeline validation only |
| [v2](data/prepared/v2/README.md) | 301 | 34 | OASST1, Apache-2.0 (verified) | First external corpus |

**v1** contains the two original synthetic `Mikuru Demo` conversations in
`data/samples/tiny.jsonl`, licensed `Project Internal` (not an open license).
It is a pipeline validation set, not a quality corpus.

**v2** is derived from the OpenAssistant OASST1 `2023-04-12` ready export,
pinned at revision `fdf72ae0827c1cda404aff25b6603abec9e3399b`. The dataset
card declares `apache-2.0` and the repository `LICENSE` file was downloaded and
verified (Apache License 2.0, 11,357 bytes, SHA-256 `c71d239d…`). Only one
source file was downloaded (34.2 MB gzip, 88,838 messages, 10,364 trees);
converted records and the license copy are committed, the raw archive is not.
335 English conversations passed the documented filters — one
highest-quality root-to-leaf path per tree, no merged branches, no generated
replies. Provenance fields and `source_message_ids` travel with every record.

**Taskmaster-1** was inspected at revision
`d92cb6af3005f1dc09c39e75e7daf4a04905e00b` on 2026-11-17. All 13,215 official
records parsed (7,708 written / 5,507 spoken; 303,066 utterances). The license
notice in `TM-1-2019/README.md` is CC BY 4.0 and was verified against the
official files. Both subsets are **CONDITIONAL, not imported, not approved for
training**: 953 self-dialog and 948 WOZ conversations contain contact-like
patterns with no upstream PII policy, the WOZ subset has deleted placeholders,
blank speakers, duplicate ids/payloads, and same-role runs, and almost nothing
fits the current 8-message / 500-character budget. Inspection artifacts and
rerun commands: [`data/third-party/taskmaster1/README.md`](data/third-party/taskmaster1/README.md).
The official written split lists exactly cover the 7,708 self-dialogs
(6,168 / 770 / 770) with no overlap.

SOC-2508 is excluded pending role mapping. The two "Original" culture/style
records listed in the research report are **absent from the repository as data**
and were not reconstructed. Details:
[`data/research/RESEARCH_REPORT.md`](data/research/RESEARCH_REPORT.md) and
[`data/third-party/oasst1/README.md`](data/third-party/oasst1/README.md).

## Workspace map

```text
crates/
  mikuru-core/         Shared data types, dataset loading/splitting
  mikuru-backend/      Burn backend smoke test and backend boundary
  mikuru-cli/          Command-line entry point
  mikuru-transformer/  Autoregressive baseline
  mikuru-ncm/          Reserved for the NCM experiment
  mikuru-tokenizer/    Byte-level BPE tokenizer + CLI
  mikuru-trainer/      Training, validation, checkpoint, generation
data/
  samples/             Tiny synthetic format example only
  imports/             Source-native converted records (v2: OASST1 English)
  third-party/         Third-party license copies and attribution notices
  prepared/            Generated train/validation datasets (v1 and v2)
  research/            Source audit, licenses, manifest
configs/               Future model/training configurations
docs/
  architecture/        Architecture boundaries and tokenizer notes
  decisions/           Locked project decisions (ADR 0001–0004)
  plan/                Milestones and acceptance checks
```

## Scope rules

- Do not mix the existing Mikuru CLI application with this Mikuru LM workspace.
- Add corpora only through the documented audit → import → prepare path. A new
  source needs a verified license, a pinned revision, an attribution note, and
  a deterministic importer before it may be used.
- Do not build both model architectures at once. First verify the toolchain and
  backend, then implement a tiny Transformer baseline.
- Keep exact dependency versions in `Cargo.lock` after a successful local build.
  The version requirements in this starter intentionally track the Burn 0.22
  release line.
- Benchmark Burn Flex against Burn CubeCL CPU later on the target machine. Do
  not assume one is faster without measuring.

## Known limitations

- **Burn Flex 0.22.1 does not implement `Backend::seed`.** Model weight
  initialisation is not reproducible on this backend; checkpoint metadata still
  records the requested seed for future backends. Split generation *is*
  reproducible.
- Training processes one padded batch per split (no shuffling or mini-batches).
  This is deliberate for the first experiments, but it caps the practical corpus
  size: v2 is 301 training records and maxes out at 272 tokens per record.
- v2 requires `--max-seq-len 288` with `artifacts/mikuru-bpe-v2.json`; the
  default 128 will reject every v2 conversation by design. Raising the corpus
  budget further needs larger contexts and real batching, not a bigger download.
- Generation is greedy and re-runs the full context each token (no KV cache).
- Checkpoints are Burnpack `.bpk` files; see the implementation note above.

## Current status

See [`docs/plan/stage-0.md`](docs/plan/stage-0.md),
[ADR 0001](docs/decisions/0001-initial-stack.md),
[ADR 0002](docs/decisions/0002-dataset-splits.md), and
[ADR 0003](docs/decisions/0003-oasst1-import.md).
