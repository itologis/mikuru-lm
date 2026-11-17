# Mikuru LM Pilot Corpus Research Report

**Research date:** 2026-11-15  
**Corpus target:** 5,000 conversations

## Executive Summary

The pilot collection contains two original synthetic English conversations:
one Japanese culture knowledge example and one Japanese-inspired style example.

No external dataset records have been collected into the pilot JSONL.

The original examples have traceable project provenance, but their license
remains Pending / Project Internal. They are not yet cleared for redistribution
or unrestricted training use.

External dataset candidates remain pending local collection and verification
or have been excluded for documented reasons.

## Verification Status Matrix

_Initial status as of 2026-11-15; see the dated updates below for current
status._

| Dataset | Source ID | Raw Text | Reconstruction | License | Privacy | Decision |
|---|---|---|---|---|---|---|
| OASST1 | Yes | Unverified: viewer truncation | Pending local validation | Verified candidate | Filtering required | Pending |
| Taskmaster-1 | Yes | Pending local fetch | Pending | Verified candidate | Review required | Pending |
| SOC-2508 | Yes | Available | Role mapping unresolved | Verified candidate | Review required | Excluded pending review |
| Original Culture | Yes | Yes | Yes | Project Internal, pending | Reviewed | Internal sample only |
| Original Style | Yes | Yes | Yes | Project Internal, pending | Reviewed | Internal sample only |

## Unresolved Items

1. **OASST1 raw text verification**
   - Retrieve the complete, untruncated text for message
     `c8e83833-ecbc-44fe-b6db-735228c25a1c`.
   - Verify `review_result` and `rank` selection logic.
   - Inspect nested labels, including PII and quality indicators.
   - Exclude records that fail the project's privacy and quality thresholds.

2. **Taskmaster-1 raw fetch**
   - Download `self-dialogs.json` and `woz-dialogs.json` from the official
     repository.
   - Verify genuine `conversationId` values and utterance indexes.
   - Preserve source attribution.

3. **SOC-2508 role mapping**
   - Inspect persona identifiers and sender relationships.
   - Do not arbitrarily assign user and assistant roles.
   - Keep excluded until a defensible mapping is established.

4. **Original record licensing**
   - Decide and document the project's internal-use and redistribution policy.
   - Do not treat Project Internal as an open license.

## Category Counts

| Category | Actual Collected | Pilot Target |
|---|---:|---:|
| General Conversation | 0 | 3,500 |
| Japanese Culture Knowledge | 1 | 750 |
| Japanese-Inspired Style | 1 | 750 |
| **Total** | **2** | **5,000** |

Targets are planning allocations, not completed collection counts.

## Cultural Fact-Checking References

The synthetic culture record `culture-genkan-nuance-001` discusses
variation in restroom slipper practices in Japan.

References:
- Web Japan / Kids Web Japan: https://web-japan.org/kidsweb/
- CLAIR Multicultural Life-in-Japan: https://www.clair.or.jp/e/multiculture/index.html

These references were used to avoid presenting regional and household
customs as universal rules.

## Compatibility

The JSONL schema has not yet been fully validated against every Rust
dataset parser field. Run schema validation before treating the pilot
records as approved training input.

## Final Summary (as of 2026-11-15, before the updates below)

- Actual collected records: 2 original synthetic examples.
- External records collected: 0.
- External candidates: OASST1 and Taskmaster-1 pending local verification.
- SOC-2508: excluded pending role-mapping review.
- Restricted or proprietary sources: excluded.
- Training approval: pending.

## Update — 2026-11-16: Dataset Expansion v1 audit

The corpus on disk was audited before preparing the first split dataset.

1. **Only two records exist as data.** `data/samples/tiny.jsonl` contains the
   two `Mikuru Demo` conversations (`split: "demo"`). These are the only
   approved records available for internal pipeline use.
2. **The two "Original" records are missing.** `culture-genkan-nuance-001` and
   `style-tactful-refactor-001` are listed in the verification matrix above as
   collected, but no JSONL file in the repository contains them. Their ids
   appear only in abbreviated unit-test fixtures inside
   `crates/mikuru-core/src/dataset.rs`; that fixture text is not the collected
   record. They were **not reconstructed**, and must not be treated as
   available data until the real records are recovered and their approval
   status is decided.
3. **No external candidate completed verification.** OASST1 and Taskmaster-1
   remain pending local verification/fetch; SOC-2508 remains excluded;
   restricted sources remain excluded.
4. **Dataset v1 was prepared from the two demo records only.** It is a pipeline
   validation set (1 train / 1 validation), not a corpus suitable for quality
   claims. See `data/prepared/v1/README.md`.
5. **Schema note:** the legacy `demo` split label is recognised as "no
   train/validation assignment". Preparation relabels records to `train` or
   `validation`; unknown split labels are rejected.

## Update — 2026-11-16: OASST1 verified and imported (Dataset Expansion v2)

The audit gap for OASST1 is closed.

- **License verified, not assumed.** The dataset card declares `apache-2.0`, and
  the repository `LICENSE` file was downloaded and checked: Apache License 2.0,
  11,357 bytes, SHA-256 `c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4`.
  A copy lives at `data/third-party/oasst1/LICENSE`.
- **Revision pinned:** `fdf72ae0827c1cda404aff25b6603abec9e3399b`
  (card last modified 2023-05-02).
- **One file downloaded:** `2023-04-12_oasst_ready.messages.jsonl.gz`
  (34,196,309 bytes, 88,838 messages, 10,364 trees). The parquet files and the
  `all` / `spam` / `trees` exports were not used.
- **Structure confirmed empirically.** The flat table is a message forest:
  `message_id`, `parent_id` (null at roots), `message_tree_id`, `role`
  (`prompter` / `assistant`), `lang`, `deleted`, `review_result`, `synthetic`,
  `labels` (including `quality` and `pii`), `detoxify`, `tree_state`. All
  10,364 trees are `ready_for_export` and contiguous in the file.
- **Data-quality findings:** 1,553 messages flagged `deleted`, 1,334 with
  `review_result: false`, 732 without `review_result` (700 of them also
  deleted), 12,745 without `detoxify` scores, and 41,305 English messages of
  88,838 total. The "ready" export still needs filtering; it is not
  pre-cleaned.
- **Conversion:** one highest-quality root-to-leaf path per tree, strict role
  alternation, English-only, 500-character / 8-message budget. 335
  conversations retained; 6,757 trees had no usable path and 3,270 exceeded
  the budget. Details in `data/prepared/v2/README.md`.
- **Dataset v2:** 301 train / 34 validation, deterministic seed 42. The
  tokenizer was trained on `train.jsonl` only (vocab 1024). Longest record is
  272 tokens, so the first v2 run needs `--max-seq-len 288`.
- **Approval:** OASST1 is approved for internal training and for redistribution
  of derived records under Apache-2.0 with the included license and attribution.
  Taskmaster-1, SOC-2508, and the missing original records remain unchanged.

## Update — 2026-11-17: Taskmaster-1 inspected (CONDITIONAL)

The Taskmaster-1 audit gap is closed for the official release; the source is
**not** approved for training.

- **Official provenance verified.** Repository
  `google-research-datasets/Taskmaster`, subset directory `TM-1-2019`, pinned
  at `d92cb6af3005f1dc09c39e75e7daf4a04905e00b`. Only official files were used;
  no mirror or Kaggle copy.
- **License verified.** The repository has no separate `LICENSE` file, but
  `TM-1-2019/README.md` opens with a COPYRIGHT NOTICE applying CC BY 4.0 to the
  dialog data. The notice and the CC BY 4.0 legal code are stored under
  `data/third-party/taskmaster1/`. Attribution requirements are recorded there.
- **All records accounted for.** 13,215 records parsed with no malformed
  top-level records: 7,708 written self-dialogs (169,469 utterances,
  7,390,510 characters) and 5,507 spoken WOZ dialogs (133,597 utterances,
  5,212,194 characters).
- **Written subset is structurally clean.** 6,755 valid, 953 flagged only for
  contact-like patterns, 0 invalid. Strict `USER`/`ASSISTANT` alternation and
  sequential indices throughout. Zero identifier duplicates and zero payload
  duplicates.
- **WOZ subset is structurally messy.** 440 valid, 5,063 flagged, 4 invalid;
  1,038 `(deleted)` placeholders, 55 blank-speaker utterances, 6 records with
  a repeated conversation id, 11 records in 3 duplicate payload groups,
  23,628 same-role adjacencies, and 1,410 adjacent duplicate utterances.
- **Privacy is unresolved.** 953 self-dialog and 948 WOZ conversations contain
  email-like, URL-like, or phone-like patterns. These are heuristics, not proof
  of personal data, and the README and paper contain no consent, anonymization,
  or PII-scrubbing statement.
- **Length compatibility is poor.** 0 self-dialogs and 81 WOZ conversations fit
  the current 8-message / 500-character budget. Using the source needs a larger
  sequence budget or a documented selection rule.
- **Split coverage is exact for written dialogs.** The official CSVs contain
  6,168 train / 770 dev / 770 test ids, covering all 7,708 self-dialogs with no
  missing ids, extras, or overlap. WOZ has no official split list.
- **Decision:** both subsets are **CONDITIONAL**, `record_count_used = 0`, and
  no records were imported. No model or tokenizer artifact changed.
- **Tooling:** `mikuru-lm inspect-taskmaster1` streams both arrays, writes
  deterministic Markdown/JSON reports plus a per-record candidate manifest, and
  refuses to write conversation text. Ten synthetic-fixture tests cover
  malformed records, roles, duplicates, privacy flags, splits, determinism, and
  overwrite refusal. See ADR 0004.

Open questions: contact-data policy, WOZ repair/dedup policy, sequence-budget
decision, and final training approval.
