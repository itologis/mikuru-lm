# ADR 0002: Dataset splits and validation

- **Status:** Accepted
- **Date:** 2026-11-16

## Context

The first training experiments used every record in a single file as one padded
batch. There was no held-out data, so loss could only show memorization. Dataset
Expansion v1 needs reproducible train/validation separation without silently
leaking validation records into training or into the BPE vocabulary.

## Decision

1. **Two assigned labels.** A record's `split` field must be `train` or
   `validation` when it is assigned. The legacy `demo` label from the starter
   sample means "no assignment" and is accepted only so existing files load.
   Any other label is rejected.
2. **Unique ids everywhere.** Loading rejects duplicate record ids, including
   duplicates across multiple input files. A record cannot exist in both splits.
3. **Preparation, not the trainer, creates automatic splits.** `prepare-dataset`
   respects explicit labels when every record has one. If no record has an
   assignment, it generates a deterministic split. Partially labelled input is
   rejected.
4. **Deterministic split algorithm.** Records are ranked by an FNV-1a hash of
   `seed` and record id (id as tie-breaker). The validation count is
   `clamp(ceil(n * fraction), 1, n - 1)`, so both splits are non-empty for any
   input of at least two records. Source order is preserved inside each split.
5. **Prepared outputs are self-describing.** `train.jsonl` and
   `validation.jsonl` carry their assigned labels; `manifest.json` records the
   split source, seed, fraction, counts, and a per-source license summary.
   Existing output files are never overwritten.
6. **The trainer never auto-splits a legacy file.** An unlabelled (or
   `demo`-labelled) file trains as a whole with no validation, preserving the
   original CLI behaviour. Labelled files are partitioned; a prepared directory
   or `--validation <path>` is the explicit way to supply held-out data.
7. **Validation is read-only.** Evaluation uses a forward pass and cross-entropy
   only. No gradients are calculated and no optimizer is involved.
8. **Tokenizer training excludes `validation` records.** The tokenizer CLI skips
   them even when given a combined file; `train.jsonl` is the preferred input.

## Consequences

- Tiny corpora can split 1/1. The numbers are a pipeline check, not evidence of
  generalization.
- Model initialisation is **not** reproducible on Burn Flex 0.22.1:
  `Backend::seed` is a no-op. Split generation is reproducible; weight init is
  not. A future backend or an explicit weight-init scheme is required.
- The manifest does not yet hash record contents, so manual edits inside a
  prepared version directory are not detected. Do not hand-edit prepared data.
- License approval remains a human/documentation decision. The tooling enforces
  structure, not legal policy.

## Alternatives rejected

- **Random shuffle at training time** — needs an RNG dependency, changes
  meaning between runs, and splits after tokenization.
- **One hash bucket per record** — can produce an empty split on small corpora
  and makes the fraction unreliable.
- **Auto-splitting inside the trainer** — ambiguous when explicit labels, a
  directory, and `--validation` are all possible.
