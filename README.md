# Mikuru LM

A Rust workspace for experimenting with two language-model approaches:

1. **Autoregressive Transformer** — the first baseline.
2. **Next-Concept Modeling (NCM)** — a separate experimental path with its own multiscale residual VQ tokenizer.

This repository is at **Stage 0: foundation and toolchain verification**. It is not yet a trained language model.

## Locked initial choices

| Component | Decision |
|---|---|
| Language | Rust 2024 |
| ML framework | Burn 0.22.x |
| Initial CPU backend | Burn Flex |
| Transformer tokenizer | BPE |
| Initial training precision | FP32 |
| Training data | JSONL |
| Model weight format | SafeTensors |

The NCM tokenizer is not BPE. It will be implemented as a distinct multiscale residual vector-quantization component, following the selected NCM paper and its reference implementation.

## Requirements

- Rust stable toolchain with Cargo (Rust 1.85 or newer for edition 2024).
- Linux, macOS, or Windows.

No GPU backend is enabled in this starter workspace. Burn Flex is the first CPU backend so the initial test stays portable and avoids adding GPU runtime requirements.

## First checks

From this directory, run:

```bash
rustc --version
cargo --version
cargo check --workspace
cargo run -p mikuru-cli -- burn-smoke
cargo run -p mikuru-cli -- inspect-jsonl data/samples/tiny.jsonl
```

`burn-smoke` checks that Burn can create FP32 tensors and calculate gradients on the Flex backend. It does **not** train a language model yet.

`inspect-jsonl` validates the starter JSONL record format and reports a small summary without printing the full text.

## Workspace map

```text
crates/
  mikuru-core/         Shared data structures
  mikuru-backend/      Burn backend smoke test and backend boundary
  mikuru-cli/          Small command-line entry point
  mikuru-transformer/  Reserved for the autoregressive baseline
  mikuru-ncm/          Reserved for the NCM experiment
data/
  samples/             Tiny synthetic format example only
configs/               Future model/training configurations
docs/
  architecture/        Architecture boundaries and tokenizer notes
  decisions/           Locked project decisions
  plan/                Milestones and acceptance checks
```

## Scope rules

- Do not mix the existing Mikuru CLI application with this Mikuru LM workspace.
- Do not add a large corpus yet. The sample in `data/samples/` only demonstrates the JSONL schema.
- Do not build both model architectures at once. First verify the toolchain and backend, then implement a tiny Transformer baseline.
- Keep exact dependency versions in `Cargo.lock` after a successful local build. The version requirements in this starter intentionally track the Burn 0.22 release line.
- Benchmark Burn Flex against Burn CubeCL CPU later on the target machine. Do not assume one is faster without measuring.

## Current status

See [`docs/plan/stage-0.md`](docs/plan/stage-0.md) and [`docs/decisions/0001-initial-stack.md`](docs/decisions/0001-initial-stack.md).
