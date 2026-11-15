# Model boundaries

## Autoregressive Transformer

First implementation target: a tiny decoder-only model that can overfit a tiny synthetic batch. It should establish embeddings, positional information, causal masking, attention, feed-forward blocks, normalization, logits, loss, and one optimizer step.

## Next-Concept Modeling

Separate experiment. Implement only after the NCM paper and official/reference implementation have been checked carefully. The generator is Transformer-based; the multiscale residual VQ representation is a separate tokenizer/codec subsystem.

## Shared infrastructure

The model experiments may share Burn backend configuration, JSONL ingestion, checkpoint metadata conventions, evaluation harnesses, and benchmark utilities. They must not be forced into one tokenizer or one output-generation interface before their needs are understood.
