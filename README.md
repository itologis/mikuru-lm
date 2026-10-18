# Mikuru

Mikuru is a from-scratch C++ learning project exploring small neural networks and language generation.

## Current milestone: GRU-backed hybrid chat experiment

The default chat path uses the experimental, CPU-only GRU language model and
its matching vocabulary. Mikuru checks system-inspector questions first,
searches the local retrieval database next, and uses GRU generation only when
neither has a suitable answer. The earlier feed-forward MLP remains available
as a legacy option.

Expect incoherent replies sometimes. This project is for learning how model
training and generation work; it is not a general-purpose assistant and does
not understand language like a person.

## Quick start

Requires a C++17 compiler such as GCC and GNU Make.

    make
    make chat

`make chat` uses the existing GRU snapshot under
`experiments/mikuru-v3/gru-v1/build/`. To train a new GRU from the preserved
GRU corpus snapshot, `make train-gru` writes a separate, non-overwriting pair
to `build/gru/`; a later `make chat` will prefer that pair. The training target
refuses to overwrite files already at that output path.

To use the legacy MLP instead:

    make train
    make chat-mlp

The MLP writes `build/dialogue-vocab.txt` and
`build/mikuru-dialogue.weights`. Both models require the vocabulary produced
by their own trainer; the GRU loader checks its model format, dimensions, and
vocabulary fingerprint before inference.

## Chat commands

- /help: show commands.
- /reset: clear in-memory conversation context.
- /quit: exit.

Chat messages are kept in memory only and are not logged or saved to disk.

## Project layout

- src/main.cpp: original single-neuron calculation.
- src/dialogue_common.hpp: shared tokenizer, vocabulary, network, serialization, and text utilities.
- src/train_dialogue.cpp: deterministic vocabulary, dialogue samples, training, validation loss, and model serialization.
- src/chat_gru.cpp: default hybrid chat executable; loads GRU V1 and preserves system/retrieval priority.
- src/chat.cpp: legacy MLP chat executable.
- src/response_quality.hpp: shared heuristic response filter and contextual fallback.
- src/test_response_quality.cpp: focused filter and fallback tests.
- data/dialogue.tsv: original user/assistant examples, one tab-separated pair per line.
- data/personality_corpus_proposal.tsv: separate proposal; not used for training.
- PERSONALITY.md: intended style and reliability goals.
- Makefile: build, train, chat, test, and clean commands.

## Model outline

- C++17 and the standard library only; no external dependencies or pretrained weights.
- The default GRU has 32-dimensional embeddings and 64 hidden units. Its GRU implementation, trainer, model serializer, and gradient tests are shared with the isolated V1 experiment.
- `make chat-mlp` retains the earlier fixed-context feed-forward model for comparison.
- Simple lowercase word/punctuation tokenizer.
- GRU training uses Adam, gradient clipping, output dropout, early stopping, and a deterministic pair-level validation split.
- Vocabulary IDs are deterministic for the training split; the versioned GRU model records the vocabulary fingerprint to reject incompatible files.
- The GRU training targets are assistant response tokens and the end-of-turn marker, with recurrent state carried across the prompt.

## Tests

Run `make test-response-quality` for the focused response-filter checks.
`make test` also runs the existing smoke suite, including MLP training; that
target rewrites the generated MLP vocabulary and weights in `build/`.

## Limitations

The corpora are small and hand-authored. Training and validation loss are
diagnostic, and a validation split from a small dataset is noisy. The GRU
snapshot's existing evaluation includes incoherent unseen-prompt outputs. The
response filter catches only obvious patterns such as repeated words,
reserved-token leakage, and filler-dominated fragments; it is not a language
understanding or correctness check.

Mikuru does not learn from conversations at runtime, does not persist chat history after exit, does not browse the web, and does not have tool access. The personality file is a specification; the model does not independently understand those traits. The separate personality corpus proposal is not automatically trained. The existing single-neuron demo is retained as a learning checkpoint.

## Generated files

Model weights and the generated vocabulary are rebuildable artifacts and should not normally be committed. The original dialogue corpus, source code, and documentation are tracked.
