# ADR 0004: Taskmaster-1 inspection and CONDITIONAL qualification

- **Status:** Accepted
- **Date:** 2026-11-17

## Context

Taskmaster-1 is the next planned external source after OASST1 v2. It ships two
JSON arrays collected by different procedures: written self-dialogs and spoken
Wizard-of-Oz dialogs. Before any import, the project needs to know whether the
data is structurally usable, what license applies, and what privacy or repair
problems would block training.

Loading both arrays into memory or writing a one-off script would be hard to
reproduce and easy to leave unaudited. The project also must not disturb the
OASST1 v2 baseline or silently create a training-ready corpus.

## Decision

1. **Inspect before importing.** A dedicated read-only command,
   `mikuru-lm inspect-taskmaster1 <dataset_dir> <output_dir> [--revision SHA]`,
   streams both official JSON arrays, accounts for every record, and writes
   Markdown, JSON, and JSONL manifest artifacts.
2. **Pin the revision and record hashes.** The first pass uses
   `d92cb6af3005f1dc09c39e75e7daf4a04905e00b` and the official
   `google-research-datasets/Taskmaster` files only. File sizes and SHA-256
   values are recorded in the report and in
   `data/third-party/taskmaster1/README.md`.
3. **Separate the subsets.** Self-dialogs and WOZ dialogs keep independent
   statistics and qualification decisions. Duplicate detection runs across both
   so a reused id or payload is visible.
4. **Preserve every record.** Malformed, empty, duplicate, unmapped-role, and
   deleted-placeholder records are counted and flagged, never dropped from the
   manifest. A malformed record is recorded with its line number and status.
5. **Do not write conversation text.** Reports and manifests contain counts,
   ids, and hashes only. Contact-like patterns are counted, not echoed.
6. **Define duplicates precisely.** An identifier duplicate is a repeated
   `conversation_id`. A payload duplicate is an identical ordered
   `(speaker, text)` sequence; annotations, indices, and instruction ids are
   ignored.
7. **Qualify conservatively.** Both subsets are **CONDITIONAL**. The license is
   verified (CC BY 4.0 via the README copyright notice), but contact-like
   patterns, WOZ structural defects, and length-budget incompatibility lack an
   approved policy. `ACCEPTED` is not used until those policies and final
   approval exist.
8. **No import, no training, no baseline change.** This ADR authorizes
   inspection artifacts only. It does not authorize merging Taskmaster-1 into
   OASST1, retraining, changing splits, or altering
   `artifacts/oasst1-v2-50.*`.
9. **Refuse unsafe output behavior.** The command rejects archives with
   extraction guidance, requires both official JSON files, and refuses to
   overwrite existing report files. Writes go through temporary files and
   renames.

## Consequences

- The project has a reproducible audit trail for Taskmaster-1 without committing
  the 103 MB raw arrays.
- The written subset looks usable after a privacy decision; the spoken subset
  needs a repair/dedup policy and probably a larger sequence budget.
- Only 81 WOZ conversations fit the current 8-message / 500-character budget,
  and no self-dialog does. This is a trainer constraint, not a source defect.
- The inspector will need a conversion/importer step before any records can
  enter a prepared dataset, and that step must encode the missing policies.
- CC BY 4.0 attribution and change notices will be required if derived records
  or model artifacts are redistributed.

## Alternatives rejected

- **Import first and clean later** — would create an unauthorized corpus and
  make defects harder to attribute.
- **Use only the self-dialogs because they are clean** — still blocked by the
  unresolved contact-pattern question, and would quietly discard half the
  source.
- **Treat the README notice as a full license file** — the notice is explicit
  about CC BY 4.0, but the repository has no separate LICENSE file; the
  distinction is recorded rather than hidden.
- **Load both arrays into memory** — simple, but unnecessary for 103 MB and
  harder to run on constrained machines.
- **Count only unique conversations for report totals** — hides the duplicate
  and malformed records the qualification decision depends on.
