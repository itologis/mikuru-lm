# Mikuru

Mikuru is a from-scratch C++ learning project exploring small neural networks and language generation.

## Current milestone: v0.1 experimental dialogue prototype

This branch adds a deliberately tiny CPU-only conversational language-model prototype. It trains a feed-forward next-token predictor from original user/assistant dialogue examples, saves a matching vocabulary and model, and loads both for terminal chat.

Expect nonsense sometimes. This model is not a general-purpose assistant, does not understand language like a person, and cannot compete with modern pretrained language models. Its purpose is to make the training-to-generation pipeline understandable and reproducible.

## Quick start

Requires a C++17 compiler such as GCC and GNU Make.

    make
    make train
    make chat

Or explicitly:

    g++ -std=c++17 -O2 -Wall -Wextra src/train_dialogue.cpp -o build/mikuru-train
    ./build/mikuru-train
    g++ -std=c++17 -O2 -Wall -Wextra src/chat.cpp -o build/mikuru-chat
    ./build/mikuru-chat

Run training before chatting. Training writes build/dialogue-vocab.txt and build/mikuru-dialogue.weights; generated artifacts are intentionally ignored by Git and can be recreated from the tracked corpus.

## Chat commands

- /help: show commands.
- /reset: clear in-memory conversation context.
- /quit: exit.

Chat messages are kept in memory only and are not logged or saved to disk.

## Project layout

- src/main.cpp: original single-neuron calculation.
- src/dialogue_common.hpp: shared tokenizer, vocabulary, network, serialization, and text utilities.
- src/train_dialogue.cpp: deterministic vocabulary, dialogue samples, training, validation loss, and model serialization.
- src/chat.cpp: loads the matching vocabulary/model and generates assistant replies.
- data/dialogue.tsv: original user/assistant examples, one tab-separated pair per line.
- PERSONALITY.md: intended style and reliability goals.
- Makefile: build, train, chat, test, and clean commands.

## Model outline

- C++17 and the standard library only; no external dependencies or pretrained weights.
- Simple lowercase word/punctuation tokenizer.
- Fixed context window, tanh hidden layer, softmax token scores, cross-entropy, and manually implemented backpropagation.
- A deterministic held-out validation split at dialogue-pair level.
- Vocabulary IDs are deterministic for the training split; a fingerprint is stored in the versioned model file to reject incompatible model/vocabulary combinations.
- Dialogue examples include user/assistant boundaries. The training targets are assistant response tokens and the end-of-turn marker.
- This is a small feed-forward model, not a Transformer.

## Tests

Run make test. The smoke suite builds the binaries, trains the small model, loads it into the chat program, checks basic commands, and verifies that special tokens are not printed as normal replies.

## Limitations

The corpus is small and hand-authored. The network has a short context window and modest capacity. Training and validation loss are diagnostic, and a validation split from a small dataset is noisy. The generator may repeat itself or produce incoherent text.

Mikuru does not learn from conversations at runtime, does not persist chat history after exit, does not browse the web, and does not have tool access. The personality file is a specification; the model does not independently understand those traits. The existing single-neuron demo is retained as a learning checkpoint.

## Generated files

Model weights and the generated vocabulary are rebuildable artifacts and should not normally be committed. The original dialogue corpus, source code, and documentation are tracked.
