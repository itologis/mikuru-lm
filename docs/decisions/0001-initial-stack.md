# ADR 0001: Initial LM stack

- **Status:** Accepted
- **Date:** 2026-11-15

## Decisions

| Area | Choice |
|---|---|
| Language | Rust 2024 |
| ML framework | Burn 0.22.x |
| Initial CPU backend | Burn Flex |
| Transformer tokenizer | BPE |
| Initial precision | FP32 |
| Dataset interchange | JSONL |
| Model weights | SafeTensors |

## Rationale

Start with the Rust-native Burn framework and a CPU backend that does not require a GPU runtime. FP32 makes early numerical debugging easier. JSONL is inspectable and easy to stream. SafeTensors avoids inventing a model-weight file format.

## Important boundaries

- BPE is for the conventional autoregressive Transformer path.
- NCM uses a distinct multiscale residual vector-quantization tokenizer.
- Burn Flex is only the first backend choice. Benchmark it against Burn CubeCL CPU before making performance claims.
- SafeTensors stores tensor data; model configuration, tokenizer artifacts, vocabulary metadata, and training metadata must be stored alongside the weights.

## Revisit when

- The first local build passes.
- We have baseline throughput and peak-memory measurements.
- We have enough tokenizer experiments to pick a vocabulary size.
