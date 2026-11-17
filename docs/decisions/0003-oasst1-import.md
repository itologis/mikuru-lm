# ADR 0003: OASST1 conversation reconstruction policy

- **Status:** Accepted
- **Date:** 2026-11-16

## Context

OASST1 stores conversation **trees**, not conversations: every message is one
row, linked by `message_id` and `parent_id`. A prompt can have many sibling
replies, and those replies can branch again. The dataset also contains
deleted messages, failed reviews, unreviewed messages, non-English text,
synthetic replies, toxicity scores, and per-message quality labels.

The project needs a deterministic, auditable way to turn that forest into
Mikuru's linear `ConversationExample` records without inventing content or
mixing unrelated branches.

## Decision

1. **One conversation per tree.** The exporter selects a single root-to-leaf
   path. It never merges sibling replies into one conversation and never emits
   a synthetic reply.
2. **Quality-first path selection.** Paths are ranked by the sum of
   `labels.quality.value`, then by length, then by lexicographically smallest
   message ids. The tie-break makes the choice independent of file ordering.
3. **Strict alternation.** A child is eligible only when its role differs from
   its parent's (`prompter` ↔ `assistant`). This mirrors the dataset card's
   guarantee and rejects malformed trees.
4. **Strict message filter.** A message is used only when it is:
   not `deleted`; `review_result: true`; `lang == "en"`; a known role;
   `synthetic: false`; non-empty; `detoxify.toxicity < 0.5`; `pii` label `0`;
   and every safety label (`spam`, `not_appropriate`, `hate_speech`,
   `sexual_content`, `violence`) `<= 0.5`.
5. **Length budget with rejection, not truncation.** The path must be
   2–8 messages and at most 500 characters. A tree whose best path exceeds the
   budget is dropped. Truncating a conversation would change what the assistant
   replied to.
6. **Provenance is mandatory.** Records keep `source_dataset`, `license`,
   `source_url`, `source_record_id` (`message_tree_id`), the dataset revision,
   and `source_message_ids` in path order. `prompter` maps to `user`.
7. **The importer is strict about input shape.** Trees must be contiguous and
   tree ids may not reappear. Duplicate message ids inside a tree, missing
   parents, multiple roots, and cycles are rejected rather than repaired.
8. **No new parsing dependency.** The canonical gzip JSONL export is
   decompressed with `gzip -dc`; the workspace reads plain JSONL and does not
   add Arrow/Parquet or compression crates.

## Consequences

- Many trees are lost: 6,757 had no usable path and 3,270 exceeded the budget.
  The result is 335 conversations, which is honest but small.
- Only one branch per tree survives, so alternative good replies are discarded.
  This avoids near-duplicate conversations that would inflate the corpus.
- The 500-character budget is a trainer constraint, not a dataset property.
  Raising it requires a larger `max_seq_len` and a different batching strategy.
- Validation stays conversation-level: a tree is entirely in train or entirely
  in validation because the split operates on exported records.
- `review_result` is required. Messages that were never reviewed are excluded
  even though some may be fine. This is deliberate: unreviewed text cannot be
  quality-screened.

## Alternatives rejected

- **All root-to-leaf paths** — produces thousands of near-duplicate
  conversations sharing prefixes, and multiplies the same prompt across splits
  if split naively.
- **Best reply at each depth** — mixes branches: a reply is chosen for its own
  quality, then a different branch is followed below it, so the conversation no
  longer represents one thread.
- **Truncate long paths** — silently changes the dialogue and can end on a user
  turn, training the model to answer nothing.
- **Keep synthetic OASST1 replies** — the corpus would no longer be
  human-written, and synthetic provenance would need separate handling.
- **Use the Hugging Face parquet splits directly** — their 95/5 split is a
  message-level research split, not a conversation-level split for training,
  and it would require Arrow/Parquet dependencies.
