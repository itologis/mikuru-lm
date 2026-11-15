# Tokenizer boundary

## Transformer path

Use BPE. The vocabulary size, normalization policy, pre-tokenization, byte handling, and special-token set remain undecided until we inspect the intended English/Japanese training text and run tokenizer coverage tests.

## NCM path

Use a multiscale residual VQ tokenizer as required by the selected Next-Concept Modeling approach. This is not a BPE tokenizer. Encoder, decoder, codebook layout, number of scales, and codebook sizes remain research/configuration decisions.

## Shared rule

Keep tokenizer artifacts versioned with the model checkpoint. A model must not silently load a different tokenizer vocabulary or codebook configuration.
