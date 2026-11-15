# Stage 0: Foundation and toolchain verification

**Status:** Starter scaffold created. Local compilation has not yet been verified in this packaging environment.

## Run these checks on the development laptop

```bash
rustc --version
cargo --version
cargo check --workspace
cargo run -p mikuru-cli -- burn-smoke
cargo run -p mikuru-cli -- inspect-jsonl data/samples/tiny.jsonl
```

## Acceptance criteria

- [ ] Rust stable is installed and edition 2024 is supported.
- [ ] `cargo check --workspace` completes.
- [ ] Burn Flex creates an FP32 tensor and calculates gradients.
- [ ] JSONL inspector accepts the sample file.
- [ ] Cargo.lock is generated and committed.

## Next stage, after Stage 0 passes

Implement a tiny autoregressive Transformer with a deliberately small configuration. First target is a working forward pass and one training step, not useful conversation quality.

## Out of scope for Stage 0

- Training a 500K or 100M model.
- Downloading a large dataset.
- Implementing the full tokenizer trainer.
- Implementing NCM.
- Building the desktop pet or math engine.
