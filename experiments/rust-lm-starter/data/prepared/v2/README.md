# Dataset v2 — OASST1 English conversations

**Prepared:** 2026-11-16
**Status:** first external corpus. Apache-2.0 verified. Pipeline-validation
scale, not a benchmark corpus.

## Contents

| Split | Records | Messages | Characters |
| --- | ---: | ---: | ---: |
| `train.jsonl` | 301 | 1,010 | 100,295 |
| `validation.jsonl` | 34 | 112 | 9,712 |
| **Total** | **335** | **1,122** | **110,007** |

Split: deterministic, seed `42`, `validation_fraction` `0.1`
(`ceil(335 × 0.1) = 34`). No record id is in both splits; the ids are
`oasst1-<message_tree_id>`.

## Source

| Field | Value |
| --- | --- |
| Dataset | OpenAssistant Conversations (OASST1) |
| Publisher | OpenAssistant / LAION-AI |
| Page | https://huggingface.co/datasets/OpenAssistant/oasst1 |
| Paper | https://arxiv.org/abs/2304.07327 |
| Revision | `fdf72ae0827c1cda404aff25b6603abec9e3399b` |
| Source file | `2023-04-12_oasst_ready.messages.jsonl.gz` (88,838 messages) |
| License | Apache-2.0 (verified; see `data/third-party/oasst1/`) |
| Synthetic | false — every exported message is human-written |

## Import report

| Measure | Count |
| --- | ---: |
| Messages read | 88,838 |
| Trees read | 10,364 |
| Messages passing all filters | 39,164 |
| Conversations written | **335** |
| Messages inside conversations | 1,122 |
| Longest conversation | 500 chars, 6 messages |

### Message rejections (first failing rule; mutually exclusive)

| Reason | Messages |
| --- | ---: |
| Not English (`lang != "en"`) | 46,327 |
| Deleted | 1,553 |
| `review_result: false` | 1,207 |
| Toxicity ≥ 0.5 | 244 |
| Unsafe review label > 0.5 (spam, hate speech, etc.) | 225 |
| PII review label > 0 | 86 |
| `review_result` missing | 32 |
| **Total rejected** | **49,674** |

700 of the 732 messages without `review_result` are also `deleted`, so
`deleted` is reported first. The counts above sum to 88,838.

### Tree rejections

| Reason | Trees |
| --- | ---: |
| No accessible accepted path (root or all replies filtered) | 6,757 |
| Best path longer than 500 chars or 8 messages | 3,270 |
| Best path shorter than 2 messages | 2 |
| **Total rejected** | **10,029** |

335 accepted + 10,029 rejected = 10,364 trees.

## Conversion rules

1. **One tree → one conversation.** The exported path is the highest-scoring
   root-to-leaf chain in the tree. Sibling branches are never combined, and no
   reply is ever generated or paraphrased.
2. **Scoring.** Sum of `labels.quality.value` along the path, then path length,
   then lexicographically smallest message ids. Deterministic for a given
   input.
3. **Strict alternation.** A child is only eligible when its role differs from
   its parent's (`prompter` ↔ `assistant`).
4. **Message filters** (in order): deleted, `review_result` not `true`,
   non-English, unknown role, synthetic, empty text, toxicity ≥ 0.5, positive
   `pii` label, any of spam / not_appropriate / hate_speech / sexual_content /
   violence > 0.5.
5. **Budget.** Total conversation text ≤ 500 characters and ≤ 8 messages. The
   tree is dropped if its best path exceeds the budget; the importer does not
   fall back to a shorter branch.
6. **Provenance.** Each record keeps `source_dataset`, `source_record_id`
   (`message_tree_id`), `source_url`, `license`, `references`, the dataset
   revision in `provenance_note`, and `source_message_ids` in path order.
7. **Role mapping.** `prompter` → `user`, `assistant` → `assistant`.

## Reproducing

```bash
# 1. Fetch only the canonical flat messages export (34 MB) and its license.
hf download OpenAssistant/oasst1 --repo-type dataset \
  --revision fdf72ae0827c1cda404aff25b6603abec9e3399b \
  --include "2023-04-12_oasst_ready.messages.jsonl.gz" --include "LICENSE" \
  --local-dir /tmp/opencode/oasst1-raw

# 2. Decompress with standard tools (no extra dependency).
gzip -dc /tmp/opencode/oasst1-raw/2023-04-12_oasst_ready.messages.jsonl.gz \
  > /tmp/opencode/oasst1-ready.messages.jsonl

# 3. Reconstruct conversations.
cargo run -p mikuru-cli -- import-oasst1 \
  /tmp/opencode/oasst1-ready.messages.jsonl \
  data/imports/oasst1-2023-04-12-en-v1.jsonl \
  --max-chars 500 --max-messages 8 \
  --revision fdf72ae0827c1cda404aff25b6603abec9e3399b

# 4. Split deterministically.
cargo run -p mikuru-cli -- prepare-dataset \
  data/imports/oasst1-2023-04-12-en-v1.jsonl data/prepared/v2 \
  --seed 42 --validation-fraction 0.1
```

The importer and `prepare-dataset` both refuse to overwrite existing outputs.

## Training configuration implied by this corpus

Tokenizing all 335 records with `artifacts/mikuru-bpe-v2.json` (vocab 1024)
gives **272 tokens** for the longest training record and 232 for the longest
validation record (mean 143 / 130). The first run therefore needs
`--max-seq-len 288`; the trainer's default of 128 will reject these records by
design rather than truncate them.

## Observed training runs

Both runs use 1 layer / 64 dim / 4 heads, lr 1e-3, seed 42, batch
`[301, 271]` (272 tokens shifted into 271 next-token inputs).

| Run | Checkpoint | Epochs | Train loss | Validation loss | Reload diff |
| --- | --- | ---: | ---: | ---: | ---: |
| Smoke | `artifacts/oasst1-v2` | 2 | 7.12907 → 6.80933 | 6.83059 | 0.0000000 |
| Full | `artifacts/oasst1-v2-50` | 50 | 7.12907 → 4.89625 | 5.32537 | 0.0000014 |

```bash
cargo run -p mikuru-cli -- train data/prepared/v2 \
  artifacts/mikuru-bpe-v2.json artifacts/oasst1-v2-50 50 --max-seq-len 288
```

Loss fell monotonically on both splits and validation was still improving at
epoch 50. That is a healthy tiny baseline, not evidence of quality — with 301
records and ~200 k parameters the model remains far from fluent.

A 24-token greedy continuation from "Hello, Mikuru!" on the 50-epoch checkpoint
degenerates into repeated `I I I I am ...`. The checkpoint proves the pipeline
trains, reloads, and generates; it does not yet produce coherent replies.
More epochs, a larger model, and real batching are the next levers.

## Caveats

- **Scale.** 335 conversations is enough to exercise real data and produce a
  meaningful loss curve, not enough to claim general model quality.
- **Dropped branches.** Only one branch per tree is exported. OASST1 contains
  many alternative replies per prompt; they are deliberately not used to avoid
  near-duplicate conversations.
- **Short conversations only.** The 500-character budget is dictated by the
  single-batch CPU trainer and a 288-token context. Longer OASST1 threads are
  excluded, so the corpus skews toward short exchanges.
- **Topic mix is unfiltered.** Conversations were selected by quality, not by
  topic. The corpus includes role-play, coding, and open-domain chat.
- **PII risk.** Filtering uses OASST1's own review labels, which are not
  perfect. Treat generated text accordingly.
- **No content hash.** Manual edits inside this directory are not detected.
  Never hand-edit prepared data.
