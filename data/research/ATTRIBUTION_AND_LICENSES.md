# Attribution and Licenses

## Included Sources: Original Generation

The pilot corpus currently contains two original synthetic generations.
External sources are documented in the manifest for future local collection
but are not included in the JSONL.

Records categorized as `japanese_culture_knowledge` and
`japanese_inspired_style` are original English dialogues generated
specifically for the Mikuru LM pilot.

- **License:** Pending / Project Internal.
- **Redistribution:** Not approved until a formal license decision is made.
- **Factual basis:** Cultural facts were checked against Web Japan and
  CLAIR resources to account for regional and household variations.

## Documented External Sources

### OASST1
- Publisher: OpenAssistant
- License: Apache-2.0
- Source: https://huggingface.co/datasets/OpenAssistant/oasst1
- Status: Pending local raw-text verification.
- Requirements: Fetch Parquet files, verify complete text, inspect review
  labels, and apply privacy and quality filters.

### Taskmaster-1
- Publisher: Google LLC
- License: CC BY 4.0, stated in the `TM-1-2019/README.md` copyright notice
  (the repository has no separate `LICENSE` file)
- Source: https://github.com/google-research-datasets/Taskmaster/tree/d92cb6af3005f1dc09c39e75e7daf4a04905e00b/TM-1-2019
- Revision inspected: `d92cb6af3005f1dc09c39e75e7daf4a04905e00b`
- Status: **CONDITIONAL — inspected, not imported, not approved for training.**
- Findings: both official subsets parse; 953 self-dialog and 948 WOZ
  conversations contain contact-like patterns with no upstream PII statement;
  WOZ has 1,038 deleted placeholders, 55 blank speakers, repeated ids, and
  payload duplicates; 0 self-dialogs and only 81 WOZ conversations fit the
  current 8-message / 500-character budget.
- Evidence and rerun commands:
  [`../third-party/taskmaster1/README.md`](../third-party/taskmaster1/README.md).

### SOC-2508
- Publisher: marcodsn
- License: CC BY 4.0
- Source: https://huggingface.co/datasets/marcodsn/SOC-2508
- Status: Excluded pending role-mapping review.
- Reason: Peer-to-peer persona structure requires reliable role mapping.

## Excluded Sources

- **DailyDialog:** CC BY-NC-SA 4.0. NonCommercial and ShareAlike restrictions.
- **EmpatheticDialogues:** CC BY-NC 4.0. NonCommercial restriction.
- **Taskmaster-4:** Self-dialogue format, not genuine two-party dialogue.
- **Marugoto / Irodori:** Proprietary Japan Foundation educational materials.

## Approval Policy

A source must pass provenance, raw-text, reconstruction, license, privacy,
and quality checks before its records can enter an approved training corpus.

A documented source is not automatically an approved source.
A generated example is not automatically cleared for redistribution.

## Update — 2026-11-16: Dataset Expansion v1 approval audit

| Source | Records on disk | License | Approval for v1 |
| --- | ---: | --- | --- |
| `Mikuru Demo` (`data/samples/tiny.jsonl`) | 2 | `Project Internal` | Internal pipeline use only |
| OASST1 | 0 | Apache-2.0 (documented) | Not approved at v1 time — superseded by the v2 update below |
| Taskmaster-1 | 0 | CC BY 4.0 (documented) | Not approved — not fetched |
| SOC-2508 | 0 | CC BY 4.0 (documented) | Excluded — role mapping unresolved |
| Original (Culture) / (Style) | 0 | `Pending / Project Internal` | No data file exists; not approved |

- No license, URL, or provenance value was invented during preparation. All
  fields are copied from the source records.
- `Project Internal` is not an open license. Internal training does not imply
  redistribution rights.
- Training approval for any external corpus was **pending** at v1 time; see
  the OASST1 v2 approval below. `data/prepared/v1/README.md` documents the
  exact scope of the v1 dataset.

## Update — 2026-11-16: OASST1 verified and imported (Dataset Expansion v2)

OASST1 was verified against the official repository, not just the dataset card.

| Check | Result |
| --- | --- |
| Dataset card license | `apache-2.0` |
| `LICENSE` file | Apache License 2.0, 11,357 bytes, SHA-256 `c71d239d…` |
| Revision pinned | `fdf72ae0827c1cda404aff25b6603abec9e3399b` |
| Source file | `2023-04-12_oasst_ready.messages.jsonl.gz`, 34,196,309 bytes, SHA-256 `286a6e9a…` |
| Attribution | Publisher, page, paper, revision, and license recorded in `data/third-party/oasst1/README.md` |
| License copy included | Yes — `data/third-party/oasst1/LICENSE` |
| Modifications declared | Yes — filtering, path reconstruction, role rename, schema conversion |

**Decision:** OASST1 is **approved for internal training and redistribution of
derived records under Apache-2.0**, with the attribution and license copy above.
It is not approved for claims about model quality at this scale.

The 335 exported conversations are derived works. Any redistribution must keep
the Apache-2.0 license copy and the provenance fields
(`source_dataset`, `source_url`, `license`, `source_record_id`,
`source_message_ids`).

Remaining concerns: crowd-sourced PII (filtered via review labels, not
guaranteed), dropped sibling branches, and the 500-character budget imposed by
the current single-batch trainer. See `data/prepared/v2/README.md`.

## Update — 2026-11-17: Taskmaster-1 verified and inspected

Taskmaster-1 was verified against the official
`google-research-datasets/Taskmaster` repository at revision
`d92cb6af3005f1dc09c39e75e7daf4a04905e00b`, not a mirror or Kaggle copy.

| Check | Result |
| --- | --- |
| Repository-level license metadata | None; no separate `LICENSE` file at the pinned revision |
| Data notice | `TM-1-2019/README.md` COPYRIGHT NOTICE applies CC BY 4.0 to the dialog data |
| License legal code | Stored as `data/third-party/taskmaster1/cc-by-4.0-legalcode.txt`, SHA-256 `9ba9550a…` |
| Upstream README | Preserved as `data/third-party/taskmaster1/upstream-README.md`, SHA-256 `05e1bc05…` |
| Attribution | Authors, publisher, repository, revision, paper, and license recorded in `data/third-party/taskmaster1/README.md` |
| Modifications declared | No conversion performed; inspection only |
| Raw files committed | No — 103 MB of official JSON stays outside the repository |
| Records imported | 0 |

**Inspection totals:** 13,215 records, 303,066 utterances, 12,602,704
characters. Self-dialogs: 7,708 records, 6,755 valid, 953 flagged for
contact-like patterns only. WOZ: 5,507 records, 440 valid, 5,063 flagged,
4 invalid; 1,038 deleted placeholders, 55 blank speakers, 6 duplicate-id
records, 11 payload-duplicate records, 23,628 same-role adjacencies. The
official split CSVs cover all 7,708 self-dialog ids exactly.

**Decision:** both subsets are **CONDITIONAL**. The license permits training
and adaptation with attribution, but the project has not approved a policy for
contact-like content, WOZ repair/deduplication, or the larger sequence budget
that Taskmaster-1 needs. `record_count_used` remains 0. See ADR 0004 and the
Taskmaster-1 third-party README for exact findings and rerun commands.

Remaining concerns: heuristic contact-pattern hits are not confirmed PII, no
upstream consent or anonymization statement was found, and CC BY 4.0
attribution/change notices must accompany any future derived records or model
artifacts.
