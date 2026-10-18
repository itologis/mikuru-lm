# Mikuru GRU V1 experiment

An isolated C++17, from-scratch GRU language-model experiment.

- Embedding: 32 dimensions
- Single GRU layer: 64 hidden units
- Adam optimizer, learning rate 0.001
- Global gradient clipping at norm 1.0
- Output-projection dropout 0.2 during training
- Loss only for assistant response tokens and EOS
- Pair-level deterministic validation split
- Early stopping on validation loss
- Includes finite-difference gradient checks

This experiment uses a snapshot of the dialogue corpus. It does not modify
the main chatbot or its model files.

## Commands

Run from this directory:

    make test
    make train
    make chat

The GRU is an experiment, not a guaranteed improvement. Compare validation
loss and unseen-prompt outputs against the MLP before considering integration.
