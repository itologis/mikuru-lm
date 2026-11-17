#!/usr/bin/env python3
"""Add checkpoint continuation support to Mikuru's Rust trainer and CLI.

This script is intentionally conservative: it validates expected source anchors,
backs up the two edited Rust files, and refuses to overwrite an existing
resume.rs file. It uses only the Python standard library.
"""

from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path

ROOT = Path.cwd()
TRAINER_DIR = ROOT / "crates/mikuru-trainer/src"
TRAINER_LIB = TRAINER_DIR / "lib.rs"
RESUME_RS = TRAINER_DIR / "resume.rs"
CLI_MAIN = ROOT / "crates/mikuru-cli/src/main.rs"
BACKUP_DIR = ROOT / "patches/resume-support-backup"

RESUME_SOURCE = r'''use super::*;

/// Continue training from an existing checkpoint into a new checkpoint base.
///
/// Older checkpoints contain model weights only, so they warm-start with a new
/// AdamW state. Checkpoints produced by this function also contain an optimizer
/// sidecar and can therefore restore the AdamW moments on later continuations.
pub fn continue_training_from_checkpoint(
    source_base: impl AsRef<Path>,
    output_base: impl AsRef<Path>,
    additional_epochs: usize,
) -> TrainerResult<TrainingReport> {
    let source_base = source_base.as_ref();
    let output_base = output_base.as_ref();

    if additional_epochs == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "Additional epochs must be greater than zero",
        )
        .into());
    }

    if source_base.extension().is_some() || output_base.extension().is_some() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "Provide checkpoint base paths without file extensions",
        )
        .into());
    }

    let source_model_path = source_base.with_extension("bpk");
    let source_metadata_path = source_base.with_extension("json");
    let source_optimizer_base = source_base.with_extension("optim");
    let source_optimizer_path = source_base.with_extension("optim.bpk");

    if !source_model_path.is_file() {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            format!("Source checkpoint not found: {}", source_model_path.display()),
        )
        .into());
    }

    if !source_metadata_path.is_file() {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            format!("Source checkpoint metadata not found: {}", source_metadata_path.display()),
        )
        .into());
    }

    let metadata: CheckpointMetadata =
        serde_json::from_slice(&fs::read(&source_metadata_path)?)?;

    if metadata.format_version != 1 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Unsupported checkpoint metadata version",
        )
        .into());
    }

    let output_model_path = output_base.with_extension("bpk");
    let output_metadata_path = output_base.with_extension("json");
    let output_optimizer_base = output_base.with_extension("optim");
    let output_optimizer_path = output_base.with_extension("optim.bpk");

    if output_model_path.exists()
        || output_metadata_path.exists()
        || output_optimizer_path.exists()
    {
        return Err(io::Error::new(
            io::ErrorKind::AlreadyExists,
            format!("Output checkpoint already exists; choose another name: {}", output_base.display()),
        )
        .into());
    }

    if let Some(parent) = output_base.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent)?;
    }

    let config = TrainerConfig {
        epochs: additional_epochs,
        learning_rate: metadata.learning_rate,
        model_dim: metadata.model_dim,
        num_layers: metadata.num_layers,
        num_heads: metadata.num_heads,
        max_seq_len: metadata.max_seq_len,
        seed: metadata.seed,
    };

    let dataset_path = Path::new(&metadata.dataset_path);
    let tokenizer_path = Path::new(&metadata.tokenizer_path);
    let validation_path = metadata.validation_path.as_deref().map(Path::new);
    let resolved = resolve_dataset(dataset_path, validation_path)?;

    let tokenizer = MikuruTokenizer::from_file(tokenizer_path)?;
    let vocab_size = tokenizer.vocab_size();

    if vocab_size != metadata.vocab_size {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "Vocabulary mismatch: checkpoint expects {}, tokenizer has {}",
                metadata.vocab_size, vocab_size
            ),
        )
        .into());
    }

    let pad_id = tokenizer
        .token_id(PAD_TOKEN)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "Tokenizer has no PAD token"))?;

    let train_sequences = load_sequences(&resolved.train, &tokenizer, config.max_seq_len)?;
    let validation_sequences = if resolved.validation.is_empty() {
        Vec::new()
    } else {
        load_sequences(&resolved.validation, &tokenizer, config.max_seq_len)?
    };

    let model_config = TransformerConfig {
        vocab_size: metadata.vocab_size,
        model_dim: config.model_dim,
        num_layers: config.num_layers,
        num_heads: config.num_heads,
        max_seq_len: config.max_seq_len,
    };
    model_config.validate().map_err(io::Error::other)?;

    let device = Device::flex().autodiff();
    device.seed(config.seed);

    let (inputs, targets, batch_size, sequence_length) =
        build_batch(&train_sequences, pad_id, &device)?;

    let model = TransformerModel::new(&model_config, &device).map_err(io::Error::other)?;
    let mut model = model
        .try_load_file(&source_model_path)
        .map_err(io::Error::other)?;

    let mut optimizer = AdamWConfig::new().with_weight_decay(0.01).init();
    let optimizer_state_restored = source_optimizer_path.is_file();

    if optimizer_state_restored {
        optimizer = optimizer
            .load(&source_optimizer_base)
            .map_err(io::Error::other)?;
        println!("Optimizer state: restored from {}", source_optimizer_path.display());
    } else {
        println!("Optimizer state: not present; continuing as a weights-only warm start");
    }

    let loss_function = CrossEntropyLoss::new(Some(pad_id as usize), &device);
    let initial_logits = model
        .forward(inputs.clone())
        .map_err(io::Error::other)?
        .reshape([batch_size * sequence_length, vocab_size]);
    let initial_loss = loss_function
        .forward(initial_logits, targets.clone())
        .into_scalar::<f32>();

    if !initial_loss.is_finite() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Source checkpoint produced non-finite training loss",
        )
        .into());
    }

    let completed_epochs = metadata.epochs;
    let total_epochs = completed_epochs.saturating_add(additional_epochs);

    println!("Source checkpoint: {}", source_model_path.display());
    println!("Output checkpoint: {}", output_model_path.display());
    println!("Starting at completed epoch {completed_epochs}; adding {additional_epochs} epochs");
    println!("Initial continuation train loss: {initial_loss:.5}");

    for local_epoch in 1..=additional_epochs {
        let logits = model
            .forward(inputs.clone())
            .map_err(io::Error::other)?
            .reshape([batch_size * sequence_length, vocab_size]);

        let loss = loss_function.forward(logits, targets.clone());
        let epoch_loss = loss.clone().into_scalar::<f32>();

        if !epoch_loss.is_finite() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Non-finite training loss at continuation epoch {local_epoch}"),
            )
            .into());
        }

        let gradients = loss.backward();
        let gradients = GradientsParams::from_grads(gradients, &model);
        model = optimizer.step(config.learning_rate, model, gradients);

        let global_epoch = completed_epochs + local_epoch;
        if local_epoch == 1 || local_epoch % 5 == 0 || local_epoch == additional_epochs {
            println!("Epoch {global_epoch}/{total_epochs} - train loss: {epoch_loss:.5}");
        }
    }

    let final_logits = model
        .forward(inputs.clone())
        .map_err(io::Error::other)?
        .reshape([batch_size * sequence_length, vocab_size]);
    let final_train_loss = loss_function
        .forward(final_logits, targets)
        .into_scalar::<f32>();

    let final_validation_loss = if validation_sequences.is_empty() {
        None
    } else {
        let (validation_inputs, validation_targets, validation_batch_size, validation_sequence_length) =
            build_batch(&validation_sequences, pad_id, &device)?;
        let validation_logits = model
            .forward(validation_inputs)
            .map_err(io::Error::other)?
            .reshape([
                validation_batch_size * validation_sequence_length,
                vocab_size,
            ]);
        Some(
            loss_function
                .forward(validation_logits, validation_targets)
                .into_scalar::<f32>(),
        )
    };

    if !final_train_loss.is_finite()
        || final_validation_loss.is_some_and(|loss| !loss.is_finite())
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Continued model produced non-finite final loss",
        )
        .into());
    }

    // Save model and optimizer separately. Burn writes the .bpk extension.
    model
        .save_file(output_base)
        .map_err(io::Error::other)?;
    optimizer
        .save(&output_optimizer_base)
        .map_err(io::Error::other)?;

    // Preserve all source metadata fields and update only the continuation data.
    let mut output_metadata: serde_json::Value =
        serde_json::from_slice(&fs::read(&source_metadata_path)?)?;
    output_metadata["epochs"] = serde_json::json!(total_epochs);
    output_metadata["continuation_initial_loss"] = serde_json::json!(initial_loss);
    output_metadata["final_loss"] = serde_json::json!(final_train_loss);
    output_metadata["validation_loss"] = serde_json::to_value(final_validation_loss)?;
    output_metadata["optimizer_state"] = serde_json::json!("saved");
    output_metadata["optimizer_state_restored_from_source"] =
        serde_json::json!(optimizer_state_restored);
    output_metadata["optimizer_checkpoint"] =
        serde_json::json!(output_optimizer_path.to_string_lossy().to_string());
    output_metadata["resumed_from"] =
        serde_json::json!(source_base.to_string_lossy().to_string());
    output_metadata["last_run_epochs"] = serde_json::json!(additional_epochs);
    fs::write(
        &output_metadata_path,
        serde_json::to_vec_pretty(&output_metadata)?,
    )?;

    match final_validation_loss {
        Some(loss) => println!("Final validation loss: {loss:.5}"),
        None => println!("Final validation loss: not evaluated (no validation records)"),
    }

    println!("Saved checkpoint: {}", output_model_path.display());
    println!("Saved optimizer: {}", output_optimizer_path.display());
    println!("Saved metadata: {}", output_metadata_path.display());

    Ok(TrainingReport {
        train_examples: resolved.train.len(),
        validation_examples: resolved.validation.len(),
        epochs: total_epochs,
        initial_loss,
        final_train_loss,
        final_validation_loss,
        checkpoint_path: output_model_path,
        metadata_path: output_metadata_path,
    })
}
'''


def fail(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


def main() -> None:
    if not TRAINER_LIB.is_file() or not CLI_MAIN.is_file():
        fail(
            "Run this script from the rust-lm-starter directory; expected "
            "crates/mikuru-trainer/src/lib.rs and crates/mikuru-cli/src/main.rs."
        )

    if RESUME_RS.exists():
        fail(f"Refusing to overwrite existing file: {RESUME_RS}")

    lib_text = TRAINER_LIB.read_text(encoding="utf-8")
    cli_text = CLI_MAIN.read_text(encoding="utf-8")

    if "pub fn train_jsonl_with_validation" not in lib_text:
        fail("Could not find train_jsonl_with_validation in trainer lib.rs")
    if "struct CheckpointMetadata" not in lib_text:
        fail("Could not find CheckpointMetadata in trainer lib.rs")
    if 'Some("train") => {' not in cli_text:
        fail('Could not find the CLI Some("train") command arm')
    if "use mikuru_trainer::" not in cli_text:
        fail("Could not find the mikuru_trainer import in CLI main.rs")
    if 'Some("resume") => {' in cli_text:
        fail('The CLI already has a "resume" command; refusing to duplicate it')
    if "mod resume;" in lib_text:
        fail("Trainer lib.rs already declares mod resume; inspect before applying")

    # Validate the current metadata shape before creating the child module.
    required_fields = (
        "format_version",
        "dataset_path",
        "tokenizer_path",
        "epochs",
        "learning_rate",
        "vocab_size",
        "model_dim",
        "num_layers",
        "num_heads",
        "max_seq_len",
        "seed",
        "validation_path",
    )
    metadata_start = lib_text.index("struct CheckpointMetadata")
    metadata_block = lib_text[metadata_start : metadata_start + 2500]
    missing = [name for name in required_fields if not re.search(rf"\b{name}\s*:", metadata_block)]
    if missing:
        fail("CheckpointMetadata is missing expected fields: " + ", ".join(missing))

    # Make all edits only after all source checks have passed.
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(TRAINER_LIB, BACKUP_DIR / "lib.rs")
    shutil.copy2(CLI_MAIN, BACKUP_DIR / "main.rs")

    lib_text = lib_text.rstrip() + "\n\nmod resume;\npub use resume::continue_training_from_checkpoint;\n"
    TRAINER_LIB.write_text(lib_text, encoding="utf-8")
    RESUME_RS.write_text(RESUME_SOURCE, encoding="utf-8")

    cli_text = cli_text.replace(
        "use mikuru_trainer::",
        "use mikuru_trainer::continue_training_from_checkpoint;\nuse mikuru_trainer::",
        1,
    )

    resume_arm = '''        Some("resume") => {
            let source = args.next().ok_or("missing source checkpoint base path")?;
            let output = args.next().ok_or("missing output checkpoint base path")?;
            let additional_epochs = args
                .next()
                .ok_or("missing number of additional epochs")?
                .parse::<usize>()?;

            if args.next().is_some() {
                return Err("unexpected extra argument for resume".into());
            }

            continue_training_from_checkpoint(source, output, additional_epochs)?;
        }
'''
    cli_text = cli_text.replace('        Some("train") => {', resume_arm + '        Some("train") => {', 1)
    cli_text = cli_text.replace(
        "fn print_help() {",
        'fn print_help() {\n    println!("  resume <source-checkpoint> <output-checkpoint> <additional-epochs>");',
        1,
    )
    CLI_MAIN.write_text(cli_text, encoding="utf-8")

    print("Checkpoint continuation support added.")
    print(f"Backups: {BACKUP_DIR}")
    print(f"New trainer module: {RESUME_RS}")
    print(f"Updated trainer: {TRAINER_LIB}")
    print(f"Updated CLI: {CLI_MAIN}")
    print("Next: run cargo fmt --all, then cargo check -p mikuru-trainer -p mikuru-cli.")


if __name__ == "__main__":
    main()
