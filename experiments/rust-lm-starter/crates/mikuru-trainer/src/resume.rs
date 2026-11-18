use super::early_stopping::{
    DEFAULT_MIN_DELTA, DEFAULT_PATIENCE, EarlyStopping, EarlyStoppingDecision,
};
use super::*;

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
    let source_optimizer_legacy_path = source_base.with_extension("optim.bpk");
    let source_optimizer_path = if source_optimizer_base.is_file() {
        source_optimizer_base.clone()
    } else {
        source_optimizer_legacy_path
    };

    if !source_model_path.is_file() {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            format!(
                "Source checkpoint not found: {}",
                source_model_path.display()
            ),
        )
        .into());
    }

    if !source_metadata_path.is_file() {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            format!(
                "Source checkpoint metadata not found: {}",
                source_metadata_path.display()
            ),
        )
        .into());
    }

    let metadata: CheckpointMetadata = serde_json::from_slice(&fs::read(&source_metadata_path)?)?;

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
    let output_optimizer_path = output_optimizer_base.clone();
    let output_optimizer_legacy_path = output_base.with_extension("optim.bpk");

    if output_model_path.exists()
        || output_metadata_path.exists()
        || output_optimizer_path.exists()
        || output_optimizer_legacy_path.exists()
    {
        return Err(io::Error::new(
            io::ErrorKind::AlreadyExists,
            format!(
                "Output checkpoint already exists; choose another name: {}",
                output_base.display()
            ),
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
        batch_size: metadata.batch_size,
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
            .load(&source_optimizer_path)
            .map_err(io::Error::other)?;
        println!(
            "Optimizer state: restored from {}",
            source_optimizer_path.display()
        );
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

    let mut early_stopping = EarlyStopping::default();

    // Treat the source checkpoint as the initial candidate.
    let initial_validation_loss = if validation_sequences.is_empty() {
        None
    } else {
        Some(evaluate_loss(
            &model,
            &validation_sequences,
            pad_id,
            &device,
            vocab_size,
            &loss_function,
        )?)
    };

    let mut best_validation_loss = initial_validation_loss;
    let mut best_epoch = completed_epochs;
    let mut epochs_attempted = 0usize;
    let mut stopped_early = false;

    if let Some(validation_loss) = initial_validation_loss {
        let _ = early_stopping.observe(validation_loss);

        // Save a matching baseline pair of model weights and optimizer state.
        model
            .clone()
            .save_file(output_base)
            .map_err(io::Error::other)?;
        optimizer
            .save(&output_optimizer_base)
            .map_err(io::Error::other)?;

        println!("Initial continuation validation loss: {validation_loss:.5}");
    }

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

        epochs_attempted = local_epoch;
        let global_epoch = completed_epochs + local_epoch;

        if validation_sequences.is_empty() {
            println!("[Epoch {global_epoch}/{total_epochs}] train loss: {epoch_loss:.5}");
            continue;
        }

        let validation_loss = evaluate_loss(
            &model,
            &validation_sequences,
            pad_id,
            &device,
            vocab_size,
            &loss_function,
        )?;

        match early_stopping.observe(validation_loss) {
            EarlyStoppingDecision::Improved => {
                best_validation_loss = Some(validation_loss);
                best_epoch = global_epoch;

                // Keep model weights and optimizer moments from the same epoch.
                model
                    .clone()
                    .save_file(output_base)
                    .map_err(io::Error::other)?;
                optimizer
                    .save(&output_optimizer_base)
                    .map_err(io::Error::other)?;

                println!(
                    "[Epoch {global_epoch}/{total_epochs}] train loss: {epoch_loss:.5} | validation loss: {validation_loss:.5} | best"
                );
            }
            EarlyStoppingDecision::Continue => {
                println!(
                    "[Epoch {global_epoch}/{total_epochs}] train loss: {epoch_loss:.5} | validation loss: {validation_loss:.5}"
                );
            }
            EarlyStoppingDecision::Stop => {
                println!(
                    "[Epoch {global_epoch}/{total_epochs}] train loss: {epoch_loss:.5} | validation loss: {validation_loss:.5}"
                );
                println!(
                    "Early stopping: validation did not improve for {} consecutive epochs.",
                    DEFAULT_PATIENCE
                );
                stopped_early = true;
                break;
            }
        }
    }

    let final_train_loss;
    let final_validation_loss;

    if validation_sequences.is_empty() {
        // No validation split: complete the requested continuation normally.
        best_epoch = completed_epochs.saturating_add(epochs_attempted);

        model
            .clone()
            .save_file(output_base)
            .map_err(io::Error::other)?;
        optimizer
            .save(&output_optimizer_base)
            .map_err(io::Error::other)?;

        final_train_loss = evaluate_loss(
            &model,
            &train_sequences,
            pad_id,
            &device,
            vocab_size,
            &loss_function,
        )?;

        final_validation_loss = None;
    } else {
        // Evaluate the best saved model, not the last in-memory model.
        let best_model = TransformerModel::new(&model_config, &device)
            .map_err(io::Error::other)?
            .try_load_file(&output_model_path)
            .map_err(io::Error::other)?;

        final_train_loss = evaluate_loss(
            &best_model,
            &train_sequences,
            pad_id,
            &device,
            vocab_size,
            &loss_function,
        )?;

        final_validation_loss = best_validation_loss;
    }

    if !final_train_loss.is_finite() || final_validation_loss.is_some_and(|loss| !loss.is_finite())
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Continued model produced non-finite final loss",
        )
        .into());
    }

    // With validation, keep the best model and matching optimizer state
    // already saved during the baseline or an improving epoch.

    // Preserve all source metadata fields and update only the continuation data.
    let mut output_metadata: serde_json::Value =
        serde_json::from_slice(&fs::read(&source_metadata_path)?)?;
    output_metadata["epochs"] = serde_json::json!(best_epoch);
    output_metadata["continuation_initial_loss"] = serde_json::json!(initial_loss);
    output_metadata["final_loss"] = serde_json::json!(final_train_loss);
    output_metadata["validation_loss"] = serde_json::to_value(final_validation_loss)?;
    output_metadata["optimizer_state"] = serde_json::json!("saved");
    output_metadata["optimizer_state_restored_from_source"] =
        serde_json::json!(optimizer_state_restored);
    output_metadata["optimizer_checkpoint"] =
        serde_json::json!(output_optimizer_path.to_string_lossy().to_string());
    output_metadata["resumed_from"] = serde_json::json!(source_base.to_string_lossy().to_string());
    output_metadata["last_run_epochs"] = serde_json::json!(epochs_attempted);
    output_metadata["epochs_attempted"] = serde_json::json!(epochs_attempted);
    output_metadata["best_epoch"] = serde_json::json!(best_epoch);
    output_metadata["early_stopped"] = serde_json::json!(stopped_early);
    output_metadata["early_stopping_patience"] = serde_json::json!(DEFAULT_PATIENCE);
    output_metadata["early_stopping_min_delta"] = serde_json::json!(DEFAULT_MIN_DELTA);
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
        epochs: completed_epochs.saturating_add(epochs_attempted),
        initial_loss,
        final_train_loss,
        final_validation_loss,
        checkpoint_path: output_model_path,
        metadata_path: output_metadata_path,
    })
}
