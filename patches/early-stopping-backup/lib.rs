//! Autoregressive language-model training pipeline.
//!
//! Version 0 processes the supplied small corpus as one padded batch per
//! split. It is intended for the first learning experiments, not large-scale
//! training.
//!
//! Datasets may be supplied in three ways:
//! - a single JSONL file whose records carry explicit `split` labels,
//! - a prepared directory containing `train.jsonl` and `validation.jsonl`,
//! - a training JSONL file plus a separate `--validation` JSONL file.
//!
//! Validation records are only ever evaluated. They never reach the optimizer.

use std::{
    error::Error,
    fs, io,
    path::{Path, PathBuf},
};

use burn::{
    module::Module,
    nn::loss::CrossEntropyLoss,
    optim::{AdamWConfig, GradientsParams},
    prelude::*,
    tensor::{Int, TensorData},
};

use serde::{Deserialize, Serialize};

use mikuru_core::{
    ConversationExample,
    dataset::{
        DatasetSplit, ensure_unique_ids, is_unassigned_split, load_jsonl, partition_by_split,
    },
};
use mikuru_tokenizer::{
    BOS_TOKEN, EOS_TOKEN, MikuruTokenizer, PAD_TOKEN, UNK_TOKEN, format_conversation,
    format_user_prompt,
};
use mikuru_transformer::{TransformerConfig, TransformerModel};

pub type TrainerResult<T> = Result<T, Box<dyn Error + Send + Sync>>;

#[derive(Clone, Debug)]
pub struct TrainerConfig {
    pub epochs: usize,
    pub learning_rate: f64,
    pub model_dim: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub max_seq_len: usize,
    /// Seed recorded in checkpoint metadata.
    ///
    /// Burn Flex 0.22.1 does not implement backend seeding
    /// (`Backend::seed` is a no-op), so weight initialisation is not yet
    /// reproducible on this backend. The seed is still stored so future
    /// backends can honour it.
    pub seed: u64,
}

impl Default for TrainerConfig {
    fn default() -> Self {
        Self {
            epochs: 50,
            learning_rate: 0.001,
            model_dim: 64,
            num_layers: 1,
            num_heads: 4,
            max_seq_len: 128,
            seed: 42,
        }
    }
}

#[derive(Debug)]
pub struct TrainingReport {
    pub train_examples: usize,
    pub validation_examples: usize,
    pub epochs: usize,
    pub initial_loss: f32,
    pub final_train_loss: f32,
    pub final_validation_loss: Option<f32>,
    pub checkpoint_path: PathBuf,
    pub metadata_path: PathBuf,
}

/// Metadata written beside every checkpoint.
///
/// `format_version` stays at 1: all new fields are optional, so checkpoints
/// written before dataset splits existed still load and generate correctly.
#[derive(Deserialize, Serialize)]
struct CheckpointMetadata {
    format_version: u32,
    dataset_path: String,
    tokenizer_path: String,
    epochs: usize,
    learning_rate: f64,
    vocab_size: usize,
    model_dim: usize,
    num_layers: usize,
    num_heads: usize,
    max_seq_len: usize,
    seed: u64,
    #[serde(default)]
    validation_path: Option<String>,
    #[serde(default)]
    split_description: Option<String>,
    #[serde(default)]
    train_records: Option<usize>,
    #[serde(default)]
    validation_records: Option<usize>,
    #[serde(default)]
    initial_loss: Option<f32>,
    #[serde(default)]
    final_loss: Option<f32>,
    #[serde(default)]
    validation_loss: Option<f32>,
}

/// Resolved dataset contents after split handling.
struct ResolvedDataset {
    train: Vec<ConversationExample>,
    validation: Vec<ConversationExample>,
    train_path: PathBuf,
    validation_path: Option<PathBuf>,
    split_description: String,
}

fn invalid_input(message: impl Into<String>) -> Box<dyn Error + Send + Sync> {
    io::Error::new(io::ErrorKind::InvalidInput, message.into()).into()
}

/// Rejects split labels that contradict the role a file is being used for.
fn enforce_file_split(
    records: &[ConversationExample],
    expected: DatasetSplit,
    path: &Path,
) -> TrainerResult<()> {
    for record in records {
        let Some(raw) = record.split.as_deref() else {
            continue;
        };

        let actual = DatasetSplit::parse(raw).map_err(invalid_input)?;

        if actual != expected {
            return Err(invalid_input(format!(
                "record `{}` in {} declares split `{raw}` but the file is used as `{}`",
                record.id,
                path.display(),
                expected.as_str()
            )));
        }
    }

    Ok(())
}

/// Resolves the dataset path(s) into explicit training and validation records.
fn resolve_dataset(
    dataset_path: &Path,
    validation_path: Option<&Path>,
) -> TrainerResult<ResolvedDataset> {
    if dataset_path.is_dir() {
        if validation_path.is_some() {
            return Err(invalid_input(
                "--validation cannot be combined with a prepared dataset directory",
            ));
        }

        let train_path = dataset_path.join("train.jsonl");
        let validation_path = dataset_path.join("validation.jsonl");

        if !train_path.is_file() || !validation_path.is_file() {
            return Err(invalid_input(format!(
                "prepared dataset directory must contain train.jsonl and validation.jsonl: {}",
                dataset_path.display()
            )));
        }

        let train = load_jsonl(&train_path)?;
        let validation = load_jsonl(&validation_path)?;

        enforce_file_split(&train, DatasetSplit::Train, &train_path)?;
        enforce_file_split(&validation, DatasetSplit::Validation, &validation_path)?;

        ensure_unique_ids(train.iter().chain(validation.iter()))?;

        return Ok(ResolvedDataset {
            train,
            validation,
            train_path,
            validation_path: Some(validation_path),
            split_description: "prepared train/validation directory".to_owned(),
        });
    }

    let train = load_jsonl(dataset_path)?;

    if let Some(validation_path) = validation_path {
        if !validation_path.is_file() {
            return Err(invalid_input(format!(
                "validation dataset not found: {}",
                validation_path.display()
            )));
        }

        let validation = load_jsonl(validation_path)?;

        enforce_file_split(&train, DatasetSplit::Train, dataset_path)?;
        enforce_file_split(&validation, DatasetSplit::Validation, validation_path)?;

        ensure_unique_ids(train.iter().chain(validation.iter()))?;

        return Ok(ResolvedDataset {
            train,
            validation,
            train_path: dataset_path.to_path_buf(),
            validation_path: Some(validation_path.to_path_buf()),
            split_description: "separate train and validation files".to_owned(),
        });
    }

    // Legacy behaviour: an unlabelled (or `demo`-labelled) file is trained on
    // as-is, with no validation split. Assigned files must be fully labelled.
    if train
        .iter()
        .all(|record| is_unassigned_split(record.split.as_deref()))
    {
        return Ok(ResolvedDataset {
            train,
            validation: Vec::new(),
            train_path: dataset_path.to_path_buf(),
            validation_path: None,
            split_description: "none (all records used for training)".to_owned(),
        });
    }

    let (train, validation) = partition_by_split(&train)?;

    Ok(ResolvedDataset {
        train,
        validation,
        train_path: dataset_path.to_path_buf(),
        validation_path: None,
        split_description: "explicit split labels in the dataset file".to_owned(),
    })
}

/// Tokenizes conversation records into training sequences.
fn load_sequences(
    records: &[ConversationExample],
    tokenizer: &MikuruTokenizer,
    max_seq_len: usize,
) -> TrainerResult<Vec<Vec<u32>>> {
    if records.is_empty() {
        return Err(invalid_input("Dataset split contains no conversations"));
    }

    if tokenizer.token_id(BOS_TOKEN).is_none() || tokenizer.token_id(EOS_TOKEN).is_none() {
        return Err(invalid_input("Tokenizer is missing BOS or EOS"));
    }

    let vocab_size = tokenizer.vocab_size();
    let mut sequences = Vec::with_capacity(records.len());

    for record in records {
        let formatted = format_conversation(record);
        let ids = tokenizer.encode(&formatted)?;

        if ids.len() < 2 {
            return Err(invalid_input(format!(
                "Record `{}` has fewer than two tokens",
                record.id
            )));
        }

        if ids.len() - 1 > max_seq_len {
            return Err(invalid_input(format!(
                "Record `{}` requires {} input positions, exceeding max_seq_len={max_seq_len}",
                record.id,
                ids.len() - 1
            )));
        }

        if ids.iter().any(|&id| id as usize >= vocab_size) {
            return Err(invalid_input(format!(
                "Token ID outside vocabulary in record `{}`",
                record.id
            )));
        }

        sequences.push(ids);
    }

    Ok(sequences)
}

/// Constructs shifted input and target tensors.
///
/// For [A, B, C, D]:
/// input  = [A, B, C]
/// target = [B, C, D]
///
/// Shorter sequences are padded. PAD targets are ignored by cross entropy.
fn build_batch(
    sequences: &[Vec<u32>],
    pad_id: u32,
    device: &Device,
) -> TrainerResult<(Tensor<2, Int>, Tensor<1, Int>, usize, usize)> {
    let batch_size = sequences.len();

    if batch_size == 0 {
        return Err(invalid_input("Cannot build an empty batch"));
    }

    if sequences.iter().any(|sequence| sequence.len() < 2) {
        return Err(invalid_input(
            "Every sequence needs at least two tokens for a shifted batch",
        ));
    }

    let sequence_length = sequences
        .iter()
        .map(|sequence| sequence.len() - 1)
        .max()
        .unwrap_or(0);

    let total_positions = batch_size
        .checked_mul(sequence_length)
        .ok_or_else(|| invalid_input("Training batch size overflow"))?;

    let pad_value =
        i32::try_from(pad_id).map_err(|_| invalid_input("PAD token ID exceeds i32 range"))?;

    let mut inputs = vec![pad_value; total_positions];
    let mut targets = vec![pad_value; total_positions];

    for (row, sequence) in sequences.iter().enumerate() {
        for (position, pair) in sequence.windows(2).enumerate() {
            let input_id = i32::try_from(pair[0])
                .map_err(|_| invalid_input("Input token ID exceeds i32 range"))?;

            let target_id = i32::try_from(pair[1])
                .map_err(|_| invalid_input("Target token ID exceeds i32 range"))?;

            let offset = row * sequence_length + position;

            inputs[offset] = input_id;
            targets[offset] = target_id;
        }
    }

    let input_tensor = Tensor::<2, Int>::from_data(
        TensorData::new(inputs, [batch_size, sequence_length]),
        device,
    );

    let target_tensor =
        Tensor::<1, Int>::from_data(TensorData::new(targets, [total_positions]), device);

    Ok((input_tensor, target_tensor, batch_size, sequence_length))
}

/// Evaluates cross-entropy loss without touching the optimizer.
///
/// This performs a forward pass only; no gradients are calculated and no
/// parameter update is possible.
fn evaluate_loss(
    model: &TransformerModel,
    sequences: &[Vec<u32>],
    pad_id: u32,
    device: &Device,
    vocab_size: usize,
    loss_function: &CrossEntropyLoss,
) -> TrainerResult<f32> {
    let (inputs, targets, batch_size, sequence_length) = build_batch(sequences, pad_id, device)?;

    let logits = model
        .forward(inputs)
        .map_err(io::Error::other)?
        .reshape([batch_size * sequence_length, vocab_size]);

    let loss = loss_function.forward(logits, targets).into_scalar::<f32>();

    if !loss.is_finite() {
        return Err(invalid_input(format!(
            "Evaluation produced a non-finite loss: {loss}"
        )));
    }

    Ok(loss)
}

/// Trains on a JSONL dataset without an explicit validation file.
///
/// If the dataset records carry split labels they are respected. If they do
/// not, all records are used for training and no validation loss is reported.
pub fn train_jsonl(
    dataset_path: impl AsRef<Path>,
    tokenizer_path: impl AsRef<Path>,
    checkpoint_base: impl AsRef<Path>,
    config: &TrainerConfig,
) -> TrainerResult<TrainingReport> {
    train_jsonl_with_validation(dataset_path, None, tokenizer_path, checkpoint_base, config)
}

/// Trains on a JSONL dataset with an optional separate validation file.
///
/// Validation records are evaluated only and are never passed to the
/// optimizer.
pub fn train_jsonl_with_validation(
    dataset_path: impl AsRef<Path>,
    validation_path: Option<&Path>,
    tokenizer_path: impl AsRef<Path>,
    checkpoint_base: impl AsRef<Path>,
    config: &TrainerConfig,
) -> TrainerResult<TrainingReport> {
    if config.epochs == 0 {
        return Err(invalid_input("Epoch count must be greater than zero"));
    }

    if !config.learning_rate.is_finite() || config.learning_rate <= 0.0 {
        return Err(invalid_input("Learning rate must be positive and finite"));
    }

    if config.max_seq_len == 0 {
        return Err(invalid_input("max_seq_len must be greater than zero"));
    }

    let dataset_path = dataset_path.as_ref();
    let tokenizer_path = tokenizer_path.as_ref();
    let checkpoint_base = checkpoint_base.as_ref();

    if checkpoint_base.extension().is_some() {
        return Err(invalid_input(
            "Provide a checkpoint base path without a file extension",
        ));
    }

    let checkpoint_path = checkpoint_base.with_extension("bpk");
    let metadata_path = checkpoint_base.with_extension("json");

    if checkpoint_path.exists() || metadata_path.exists() {
        return Err(invalid_input(format!(
            "Checkpoint already exists. Choose another path: {}",
            checkpoint_base.display()
        )));
    }

    if let Some(parent) = checkpoint_base.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent)?;
    }

    let resolved = resolve_dataset(dataset_path, validation_path)?;

    let tokenizer = MikuruTokenizer::from_file(tokenizer_path)?;

    let pad_id = tokenizer
        .token_id(PAD_TOKEN)
        .ok_or_else(|| invalid_input("Tokenizer has no PAD token"))?;

    let train_sequences = load_sequences(&resolved.train, &tokenizer, config.max_seq_len)?;

    let validation_sequences = if resolved.validation.is_empty() {
        Vec::new()
    } else {
        load_sequences(&resolved.validation, &tokenizer, config.max_seq_len)?
    };

    let vocab_size = tokenizer.vocab_size();

    let model_config = TransformerConfig {
        vocab_size,
        model_dim: config.model_dim,
        num_layers: config.num_layers,
        num_heads: config.num_heads,
        max_seq_len: config.max_seq_len,
    };

    model_config.validate().map_err(io::Error::other)?;

    // Burn Flex executes on CPU. Autodiff enables parameter gradients.
    let device = Device::flex().autodiff();
    device.seed(config.seed);

    let (inputs, targets, batch_size, sequence_length) =
        build_batch(&train_sequences, pad_id, &device)?;

    let mut model = TransformerModel::new(&model_config, &device).map_err(io::Error::other)?;

    let mut optimizer = AdamWConfig::new().with_weight_decay(0.01).init();

    let loss_function = CrossEntropyLoss::new(Some(pad_id as usize), &device);

    println!("Backend: Burn Flex CPU");
    println!("Precision: FP32");
    println!("Training records: {}", resolved.train.len());
    println!("Validation records: {}", resolved.validation.len());
    println!("Dataset split: {}", resolved.split_description);
    println!("Batch shape: [{batch_size}, {sequence_length}]");
    println!("Vocabulary size: {vocab_size}");
    println!(
        "Transformer: {} layer(s), {} dimensions, {} heads",
        config.num_layers, config.model_dim, config.num_heads
    );
    println!("Epochs: {}", config.epochs);
    println!("Learning rate: {}", config.learning_rate);
    println!();

    let mut initial_loss = None;
    let log_interval = (config.epochs / 10).max(1);

    for epoch in 1..=config.epochs {
        let logits = model.forward(inputs.clone()).map_err(io::Error::other)?;

        let logits = logits.reshape([batch_size * sequence_length, vocab_size]);

        let loss = loss_function.forward(logits, targets.clone());

        let loss_value = loss.clone().into_scalar::<f32>();

        if !loss_value.is_finite() {
            return Err(invalid_input(format!(
                "Non-finite training loss at epoch {epoch}: {loss_value}"
            )));
        }

        if initial_loss.is_none() {
            initial_loss = Some(loss_value);
        }

        let gradients = loss.backward();
        let gradients = GradientsParams::from_grads(gradients, &model);

        // Burn consumes the model and returns its updated parameters.
        model = optimizer.step(config.learning_rate, model, gradients);

        if epoch == 1 || epoch % log_interval == 0 || epoch == config.epochs {
            if validation_sequences.is_empty() {
                println!(
                    "[Epoch {epoch:>3}/{:>3}] train loss: {loss_value:.5}",
                    config.epochs
                );
            } else {
                let validation_loss = evaluate_loss(
                    &model,
                    &validation_sequences,
                    pad_id,
                    &device,
                    vocab_size,
                    &loss_function,
                )?;

                println!(
                    "[Epoch {epoch:>3}/{:>3}] train loss: {loss_value:.5} | validation loss: {validation_loss:.5}",
                    config.epochs
                );
            }
        }
    }

    // Evaluate once more after the final optimizer update.
    let final_train_loss = evaluate_loss(
        &model,
        &train_sequences,
        pad_id,
        &device,
        vocab_size,
        &loss_function,
    )?;

    let final_validation_loss = if validation_sequences.is_empty() {
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

    let initial_loss =
        initial_loss.ok_or_else(|| invalid_input("No training steps were performed"))?;

    // Save the trained model weights in Burnpack format.
    model.save_file(checkpoint_base)?;

    // Save the architecture and training information beside the weights.
    let metadata = CheckpointMetadata {
        format_version: 1,
        dataset_path: resolved.train_path.display().to_string(),
        validation_path: resolved
            .validation_path
            .as_ref()
            .map(|path| path.display().to_string()),
        split_description: Some(resolved.split_description.clone()),
        tokenizer_path: tokenizer_path.display().to_string(),
        epochs: config.epochs,
        learning_rate: config.learning_rate,
        vocab_size,
        model_dim: config.model_dim,
        num_layers: config.num_layers,
        num_heads: config.num_heads,
        max_seq_len: config.max_seq_len,
        seed: config.seed,
        train_records: Some(resolved.train.len()),
        validation_records: Some(resolved.validation.len()),
        initial_loss: Some(initial_loss),
        final_loss: Some(final_train_loss),
        validation_loss: final_validation_loss,
    };

    fs::write(&metadata_path, serde_json::to_vec_pretty(&metadata)?)?;

    println!();
    println!("Initial train loss: {initial_loss:.5}");
    println!("Final train loss:   {final_train_loss:.5}");

    match final_validation_loss {
        Some(loss) => println!("Final validation loss: {loss:.5}"),
        None => println!("Final validation loss: not evaluated (no validation records)"),
    }

    println!("Checkpoint:   {}", checkpoint_path.display());
    println!("Metadata:     {}", metadata_path.display());

    Ok(TrainingReport {
        train_examples: resolved.train.len(),
        validation_examples: resolved.validation.len(),
        epochs: config.epochs,
        initial_loss,
        final_train_loss,
        final_validation_loss,
        checkpoint_path,
        metadata_path,
    })
}

/// Reload a saved checkpoint into a fresh model and evaluate its loss.
pub fn verify_checkpoint(
    dataset_path: impl AsRef<Path>,
    tokenizer_path: impl AsRef<Path>,
    checkpoint_base: impl AsRef<Path>,
    expected_loss: Option<f32>,
) -> TrainerResult<f32> {
    if expected_loss.is_some_and(|value| !value.is_finite()) {
        return Err(invalid_input("Expected loss must be finite"));
    }

    let dataset_path = dataset_path.as_ref();
    let tokenizer_path = tokenizer_path.as_ref();
    let checkpoint_base = checkpoint_base.as_ref();

    let checkpoint_path = checkpoint_base.with_extension("bpk");
    let metadata_path = checkpoint_base.with_extension("json");

    if !checkpoint_path.is_file() {
        return Err(invalid_input(format!(
            "Checkpoint not found: {}",
            checkpoint_path.display()
        )));
    }

    if !metadata_path.is_file() {
        return Err(invalid_input(format!(
            "Checkpoint metadata not found: {}",
            metadata_path.display()
        )));
    }

    let metadata: CheckpointMetadata = serde_json::from_slice(&fs::read(&metadata_path)?)?;

    if metadata.format_version != 1 {
        return Err(invalid_input("Unsupported checkpoint metadata version"));
    }

    let tokenizer = MikuruTokenizer::from_file(tokenizer_path)?;
    let vocab_size = tokenizer.vocab_size();

    if vocab_size != metadata.vocab_size {
        return Err(invalid_input(format!(
            "Vocabulary mismatch: expected {}, got {}",
            metadata.vocab_size, vocab_size
        )));
    }

    let pad_id = tokenizer
        .token_id(PAD_TOKEN)
        .ok_or_else(|| invalid_input("Tokenizer has no PAD token"))?;

    // Verification evaluates every record in the file, reproducing the
    // single-batch training loss of the original checkpoint.
    let records = load_jsonl(dataset_path)?;
    let sequences = load_sequences(&records, &tokenizer, metadata.max_seq_len)?;

    let model_config = TransformerConfig {
        vocab_size: metadata.vocab_size,
        model_dim: metadata.model_dim,
        num_layers: metadata.num_layers,
        num_heads: metadata.num_heads,
        max_seq_len: metadata.max_seq_len,
    };

    model_config.validate().map_err(io::Error::other)?;

    let device = Device::flex().autodiff();
    device.seed(metadata.seed);

    let (inputs, targets, batch_size, sequence_length) = build_batch(&sequences, pad_id, &device)?;

    // Initialize fresh weights, then replace them with checkpoint weights.
    let model = TransformerModel::new(&model_config, &device).map_err(io::Error::other)?;

    let model = model
        .try_load_file(&checkpoint_path)
        .map_err(io::Error::other)?;

    let parameter_count = model.num_params();

    // Evaluate the restored model.
    let logits = model
        .forward(inputs)
        .map_err(io::Error::other)?
        .reshape([batch_size * sequence_length, vocab_size]);

    let loss = CrossEntropyLoss::new(Some(pad_id as usize), &device)
        .forward(logits, targets)
        .into_scalar::<f32>();

    if !loss.is_finite() {
        return Err(invalid_input("Reloaded model produced non-finite loss"));
    }

    println!("Checkpoint: {}", checkpoint_path.display());
    println!("Model parameters: {parameter_count}");
    println!("Evaluation examples: {}", sequences.len());
    println!("Loss after reload: {loss:.5}");

    // Older checkpoints may not have recorded their final loss.
    if let Some(expected) = expected_loss.or(metadata.final_loss) {
        const TOLERANCE: f32 = 0.0005;

        let difference = (loss - expected).abs();

        println!("Expected loss:      {expected:.5}");
        println!("Absolute difference: {difference:.7}");

        if difference > TOLERANCE {
            return Err(invalid_input(format!(
                "Checkpoint mismatch: difference {difference:.7} exceeds tolerance {TOLERANCE}"
            )));
        }

        println!("Checkpoint round-trip verification passed.");
    } else {
        println!("No recorded baseline loss; compare with the original training log.");
    }

    Ok(loss)
}

/// Generates a continuation using greedy autoregressive decoding.
///
/// Loads a checkpoint and tokenizer, then predicts one token at a time.
/// The context is limited to the model's configured maximum sequence length.
pub fn generate_from_checkpoint(
    tokenizer_path: impl AsRef<Path>,
    checkpoint_base: impl AsRef<Path>,
    prompt: &str,
    max_new_tokens: usize,
    verbose: bool,
) -> TrainerResult<String> {
    if prompt.trim().is_empty() {
        return Err(invalid_input("Prompt must not be empty"));
    }

    if max_new_tokens == 0 {
        return Err(invalid_input("max_new_tokens must be greater than zero"));
    }

    let tokenizer_path = tokenizer_path.as_ref();
    let checkpoint_base = checkpoint_base.as_ref();

    let checkpoint_path = checkpoint_base.with_extension("bpk");
    let metadata_path = checkpoint_base.with_extension("json");

    if !checkpoint_path.is_file() || !metadata_path.is_file() {
        return Err(invalid_input(
            "Checkpoint or checkpoint metadata is missing",
        ));
    }

    let metadata: CheckpointMetadata = serde_json::from_slice(&fs::read(&metadata_path)?)?;

    if metadata.format_version != 1 {
        return Err(invalid_input("Unsupported checkpoint metadata version"));
    }

    let tokenizer = MikuruTokenizer::from_file(tokenizer_path)?;

    let vocab_size = tokenizer.vocab_size();

    if vocab_size != metadata.vocab_size {
        return Err(invalid_input(format!(
            "Vocabulary mismatch: checkpoint expects {}, tokenizer has {}",
            metadata.vocab_size, vocab_size
        )));
    }

    let bos_id = tokenizer
        .token_id(BOS_TOKEN)
        .ok_or_else(|| invalid_input("Tokenizer has no BOS token"))?;

    let eos_id = tokenizer
        .token_id(EOS_TOKEN)
        .ok_or_else(|| invalid_input("Tokenizer has no EOS token"))?;

    let pad_id = tokenizer
        .token_id(PAD_TOKEN)
        .ok_or_else(|| invalid_input("Tokenizer has no PAD token"))?;

    let unk_id = tokenizer
        .token_id(UNK_TOKEN)
        .ok_or_else(|| invalid_input("Tokenizer has no UNK token"))?;

    // Build the same canonical prefix used by the training formatter.
    let formatted_prompt = format_user_prompt(prompt)?;
    let mut context_ids = tokenizer.encode(&formatted_prompt)?;

    if context_ids.is_empty() {
        return Err(invalid_input("Prompt produced no tokens"));
    }

    if context_ids.iter().any(|&id| id as usize >= vocab_size) {
        return Err(invalid_input(
            "Prompt contains a token outside the vocabulary",
        ));
    }

    let model_config = TransformerConfig {
        vocab_size: metadata.vocab_size,
        model_dim: metadata.model_dim,
        num_layers: metadata.num_layers,
        num_heads: metadata.num_heads,
        max_seq_len: metadata.max_seq_len,
    };

    model_config.validate().map_err(io::Error::other)?;

    let device = Device::flex().autodiff();
    device.seed(metadata.seed);

    let model = TransformerModel::new(&model_config, &device).map_err(io::Error::other)?;

    let model = model
        .try_load_file(&checkpoint_path)
        .map_err(io::Error::other)?;

    let mut generated_ids = Vec::with_capacity(max_new_tokens);
    let mut stop_reason: Option<&str> = None;

    println!("Loaded checkpoint: {}", checkpoint_path.display());
    println!("Prompt tokens: {}", context_ids.len());
    println!("Generating up to {max_new_tokens} tokens...");
    println!();

    for step in 0..max_new_tokens {
        // Keep only the most recent context that fits the model.
        let context_start = context_ids.len().saturating_sub(metadata.max_seq_len);

        let context = &context_ids[context_start..];
        let context_length = context.len();

        if context_length == 0 {
            return Err(invalid_input("Generation context is empty"));
        }

        let input_values = context
            .iter()
            .map(|&id| i32::try_from(id).map_err(|_| invalid_input("Token ID exceeds i32 range")))
            .collect::<TrainerResult<Vec<i32>>>()?;

        let input = Tensor::<2, Int>::from_data(
            TensorData::new(input_values, [1, context_length]),
            &device,
        );

        // Obtain vocabulary logits for the complete context.
        let logits = model.forward(input).map_err(io::Error::other)?;

        // Only the final sequence position predicts the next token.
        let next_token = logits
            .slice([0..1, (context_length - 1)..context_length, 0..vocab_size])
            .reshape([vocab_size])
            .argmax(0)
            .into_scalar::<i32>();

        let next_id = u32::try_from(next_token)
            .map_err(|_| invalid_input("Model produced an invalid token ID"))?;

        if next_id as usize >= vocab_size {
            return Err(invalid_input(
                "Model produced a token outside the vocabulary",
            ));
        }

        // EOS ends generation. Structural special tokens are not emitted.
        if next_id == eos_id {
            stop_reason = Some("EOS");
            break;
        }

        if next_id == bos_id || next_id == pad_id || next_id == unk_id {
            stop_reason = Some("a structural special token");
            break;
        }

        generated_ids.push(next_id);
        context_ids.push(next_id);

        if verbose {
            println!("Generated token {}/{}", step + 1, max_new_tokens);
        }
    }

    let generated_text = tokenizer.decode(&generated_ids)?;

    println!();

    match stop_reason {
        Some(reason) => println!("Generation ended on {reason}."),
        None => println!("Generation reached the token limit."),
    }

    Ok(generated_text)
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use mikuru_core::{ConversationMessage, dataset::write_jsonl};
    use mikuru_tokenizer::BpeTrainingConfig;

    use super::*;

    fn conversation(
        id: &str,
        split: Option<&str>,
        user: &str,
        assistant: &str,
    ) -> ConversationExample {
        ConversationExample {
            id: id.to_owned(),
            messages: vec![
                ConversationMessage {
                    role: "user".to_owned(),
                    content: user.to_owned(),
                },
                ConversationMessage {
                    role: "assistant".to_owned(),
                    content: assistant.to_owned(),
                },
            ],
            language: "en".to_owned(),
            category: "general_conversation".to_owned(),
            source_dataset: "Mikuru Trainer Test".to_owned(),
            source_record_id: id.to_owned(),
            source_url: "N/A".to_owned(),
            license: "Project Internal".to_owned(),
            synthetic: true,
            split: split.map(str::to_owned),
            references: Vec::new(),
            source_message_ids: Vec::new(),
            provenance_note: "Synthetic trainer unit-test record.".to_owned(),
        }
    }

    fn temp_dir(name: &str) -> PathBuf {
        static COUNTER: AtomicUsize = AtomicUsize::new(0);

        let index = COUNTER.fetch_add(1, Ordering::Relaxed);

        let dir = std::env::temp_dir().join(format!(
            "mikuru-trainer-test-{}-{index}-{name}",
            std::process::id()
        ));

        fs::create_dir_all(&dir).expect("temporary directory should be created");

        dir
    }

    fn build_tokenizer(dir: &Path, records: &[ConversationExample]) -> PathBuf {
        let texts = records
            .iter()
            .map(format_conversation)
            .collect::<Vec<String>>();

        let tokenizer = MikuruTokenizer::train_from_iterator(
            texts.iter(),
            &BpeTrainingConfig {
                vocab_size: 320,
                min_frequency: 1,
            },
        )
        .expect("test tokenizer should train");

        let path = dir.join("tokenizer.json");

        tokenizer
            .save(&path)
            .expect("test tokenizer should be saved");

        path
    }

    fn test_config(epochs: usize) -> TrainerConfig {
        TrainerConfig {
            epochs,
            learning_rate: 0.01,
            model_dim: 16,
            num_layers: 1,
            num_heads: 2,
            max_seq_len: 128,
            seed: 7,
        }
    }

    #[test]
    fn trains_with_explicit_splits_and_reports_validation_loss() {
        let dir = temp_dir("explicit-splits");

        let records = vec![
            conversation(
                "train-1",
                Some("train"),
                "Hello, Mikuru!",
                "Hello! What shall we explore today?",
            ),
            conversation(
                "train-2",
                Some("train"),
                "What are you working on?",
                "I'm learning how to organize conversations.",
            ),
            conversation(
                "validation-1",
                Some("validation"),
                "How are you today?",
                "I'm doing well, thank you.",
            ),
        ];

        let input = dir.join("dataset.jsonl");
        write_jsonl(&input, &records).expect("dataset should be written");

        let tokenizer = build_tokenizer(&dir, &records);
        let checkpoint = dir.join("model");

        let report = train_jsonl(&input, &tokenizer, &checkpoint, &test_config(2))
            .expect("training should succeed");

        assert_eq!(report.train_examples, 2);
        assert_eq!(report.validation_examples, 1);
        assert!(
            report.final_validation_loss.is_some_and(f32::is_finite),
            "validation loss should be reported"
        );
        assert!(report.checkpoint_path.is_file());
        assert!(report.metadata_path.is_file());

        let metadata: CheckpointMetadata =
            serde_json::from_slice(&fs::read(&report.metadata_path).expect("metadata readable"))
                .expect("metadata should parse");

        assert_eq!(metadata.train_records, Some(2));
        assert_eq!(metadata.validation_records, Some(1));
        assert!(metadata.validation_loss.is_some());

        let error = train_jsonl(&input, &tokenizer, &checkpoint, &test_config(1))
            .expect_err("existing checkpoint must never be overwritten");

        assert!(
            error.to_string().contains("Checkpoint already exists"),
            "error should refuse to overwrite: {error}"
        );

        fs::remove_dir_all(dir).expect("temporary directory should be removed");
    }

    #[test]
    fn validation_evaluation_is_read_only() {
        let dir = temp_dir("validation-read-only");

        let records = vec![conversation(
            "validation-1",
            Some("validation"),
            "How are you today?",
            "I'm doing well, thank you.",
        )];

        let tokenizer_path = build_tokenizer(&dir, &records);
        let tokenizer = MikuruTokenizer::from_file(&tokenizer_path).expect("tokenizer should load");

        let pad_id = tokenizer
            .token_id(PAD_TOKEN)
            .expect("PAD token should exist");
        let sequences =
            load_sequences(&records, &tokenizer, 128).expect("sequences should tokenize");

        let vocab_size = tokenizer.vocab_size();
        let config = test_config(1);

        let model_config = TransformerConfig {
            vocab_size,
            model_dim: config.model_dim,
            num_layers: config.num_layers,
            num_heads: config.num_heads,
            max_seq_len: config.max_seq_len,
        };

        let device = Device::flex().autodiff();
        device.seed(config.seed);

        let model = TransformerModel::new(&model_config, &device).expect("model should build");

        let loss_function = CrossEntropyLoss::new(Some(pad_id as usize), &device);

        let first = evaluate_loss(
            &model,
            &sequences,
            pad_id,
            &device,
            vocab_size,
            &loss_function,
        )
        .expect("first evaluation should succeed");

        let second = evaluate_loss(
            &model,
            &sequences,
            pad_id,
            &device,
            vocab_size,
            &loss_function,
        )
        .expect("second evaluation should succeed");

        assert_eq!(
            first.to_bits(),
            second.to_bits(),
            "evaluating validation records must never mutate model weights"
        );

        fs::remove_dir_all(dir).expect("temporary directory should be removed");
    }

    #[test]
    fn legacy_dataset_without_splits_trains_on_every_record() {
        let dir = temp_dir("legacy");

        let records = vec![
            conversation(
                "legacy-1",
                None,
                "Hello, Mikuru!",
                "Hello! What shall we explore today?",
            ),
            conversation(
                "legacy-2",
                None,
                "What are you working on?",
                "I'm learning how to organize conversations.",
            ),
        ];

        let input = dir.join("dataset.jsonl");
        write_jsonl(&input, &records).expect("dataset should be written");

        let tokenizer = build_tokenizer(&dir, &records);

        let report = train_jsonl(
            &input,
            &tokenizer,
            dir.join("legacy-model"),
            &test_config(1),
        )
        .expect("legacy dataset should train");

        assert_eq!(report.train_examples, 2);
        assert_eq!(report.validation_examples, 0);
        assert_eq!(report.final_validation_loss, None);

        fs::remove_dir_all(dir).expect("temporary directory should be removed");
    }

    #[test]
    fn rejects_validation_file_with_conflicting_split_labels() {
        let dir = temp_dir("conflicting-labels");

        let train_records = vec![conversation(
            "train-1",
            Some("train"),
            "Hello, Mikuru!",
            "Hello there!",
        )];

        let validation_records = vec![conversation(
            "validation-1",
            Some("train"),
            "How are you?",
            "Doing well.",
        )];

        let train_path = dir.join("train.jsonl");
        let validation_path = dir.join("validation.jsonl");

        write_jsonl(&train_path, &train_records).expect("train split should be written");
        write_jsonl(&validation_path, &validation_records)
            .expect("validation split should be written");

        let tokenizer = build_tokenizer(&dir, &train_records);

        let error = train_jsonl_with_validation(
            &train_path,
            Some(&validation_path),
            &tokenizer,
            dir.join("model"),
            &test_config(1),
        )
        .expect_err("conflicting labels must be rejected");

        assert!(
            error.to_string().contains("declares split"),
            "error should explain the conflict: {error}"
        );

        fs::remove_dir_all(dir).expect("temporary directory should be removed");
    }
}

mod resume;
pub use resume::continue_training_from_checkpoint;
