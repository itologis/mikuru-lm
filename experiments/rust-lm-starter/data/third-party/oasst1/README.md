# OASST1 — attribution and license

**Source:** OpenAssistant Conversations Dataset (OASST1)
**Publisher:** OpenAssistant / LAION-AI
**Dataset page:** https://huggingface.co/datasets/OpenAssistant/oasst1
**Paper:** https://arxiv.org/abs/2304.07327
**Revision used:** `fdf72ae0827c1cda404aff25b6603abec9e3399b`
**Version label:** `2023-04-12` export (card last modified 2023-05-02)
**License:** Apache-2.0 (declared on the dataset card **and** confirmed in the
repository's `LICENSE` file, copied here as [`LICENSE`](LICENSE))

## What was downloaded

Only one data file:

| File | Compressed size | SHA-256 of the downloaded file |
| --- | ---: | --- |
| `2023-04-12_oasst_ready.messages.jsonl.gz` | 34,196,309 bytes | `286a6e9a5a413b3272ae9c0b5a20d327983dea1c24342ae28cb244a6da65185c` |
| `LICENSE` | 11,357 bytes | `c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4` |

The gzip file was decompressed with `gzip -dc` (165 MB) and processed by
`mikuru-lm import-oasst1`. Neither the archive nor the decompressed file is
committed; only the derived conversation records and this license copy are.

The card's `ready.messages` export contains 88,838 messages in 10,364
`ready_for_export` trees. The Hugging Face parquet files were **not** used:
the card documents the JSONL exports as the canonical flat message data, and
the flat JSONL is readable without adding Arrow/Parquet dependencies.

## What was created

`data/imports/oasst1-2023-04-12-en-v1.jsonl` — 335 English conversations
derived from the messages above. SHA-256:
`2066628a0064572a8e3b237dde18d27e88017bdd757d9d1c4470e8dbfcab7619`.

**Changes made to the source data** (Apache-2.0 §4(b) notice):

1. Flat message rows were grouped by `message_tree_id` and reconstructed into
   single root-to-leaf conversation paths.
2. Only English messages (`lang == "en"`) were retained.
3. Deleted, unreviewed, failed-review, synthetic, toxic, PII-labelled, and
   unsafe-labelled messages were excluded.
4. Role `prompter` was renamed to `user`; `assistant` was kept.
5. Messages were serialized into Mikuru's JSONL conversation schema with
   provenance fields, original message ids, and the dataset revision.
6. No message text was rewritten, summarized, translated, or generated.

## Attribution requirement

Apache-2.0 requires retaining copyright, patent, trademark, and attribution
notices and including a copy of the license with any redistribution. This
directory provides the license copy and the source identification above. The
derived records keep `source_dataset`, `source_url`, `license`,
`source_record_id`, and `source_message_ids` so attribution travels with the
data.

OASST1 is a crowd-sourced corpus. Individual messages were written by
volunteers; the dataset does not attach per-message copyright statements. The
project does not claim authorship of the message text.

## Privacy

The importer rejects messages with a positive `pii` review label, but automated
and human labelling is not perfect. OASST1 may still contain personal
information. Do not publish model outputs that could reproduce such content
without review.

## Not included

- No Spanish, Russian, German, or other non-English messages.
- No `spam` supplemental export.
- No `all.trees` / `all.messages` files.
- No OASST2 data.
