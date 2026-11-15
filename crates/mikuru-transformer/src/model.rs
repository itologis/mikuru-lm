//! Complete autoregressive Transformer language model.
//!
//! Architecture:
//! 1. Token embeddings
//! 2. Learned positional embeddings
//! 3. Configurable stack of decoder blocks
//! 4. Final layer normalization
//! 5. Vocabulary output projection
//!
//! Input:  [batch, sequence]
//! Output: [batch, sequence, vocab_size]

use burn::nn::{LayerNorm, LayerNormConfig, Linear, LinearConfig};
use burn::prelude::*;

use crate::config::TransformerConfig;
use crate::decoder_block::TransformerDecoderBlock;
use crate::embedding::TokenEmbedding;
use crate::position::PositionEmbedding;

/// Complete decoder-only Transformer language model.
#[derive(Module, Debug)]
pub struct TransformerModel {
    token_embedding: TokenEmbedding,
    position_embedding: PositionEmbedding,
    blocks: Vec<TransformerDecoderBlock>,
    final_norm: LayerNorm,
    output_projection: Linear,

    // Configuration is stored separately from the trainable weights.
    #[module(skip)]
    config: TransformerConfig,
}

impl TransformerModel {
    /// Creates a Transformer using the supplied configuration.
    pub fn new(config: &TransformerConfig, device: &Device) -> Result<Self, &'static str> {
        config.validate()?;

        // Create the decoder blocks.
        let mut blocks = Vec::with_capacity(config.num_layers);

        for _ in 0..config.num_layers {
            blocks.push(TransformerDecoderBlock::new(config, device)?);
        }

        Ok(Self {
            token_embedding: TokenEmbedding::new(config, device)?,

            position_embedding: PositionEmbedding::new(config, device)?,

            blocks,

            final_norm: LayerNormConfig::new(config.model_dim).init(device),

            output_projection: LinearConfig::new(config.model_dim, config.vocab_size).init(device),

            config: config.clone(),
        })
    }

    /// Returns the configuration used to construct this model.
    pub fn config(&self) -> &TransformerConfig {
        &self.config
    }

    /// Runs a sequence through the complete Transformer.
    ///
    /// Input:
    ///   token_ids: [batch, sequence]
    ///
    /// Output:
    ///   logits: [batch, sequence, vocab_size]
    ///
    /// Token IDs must be within the configured vocabulary.
    pub fn forward(&self, token_ids: Tensor<2, Int>) -> Result<Tensor<3>, &'static str> {
        let [batch_size, sequence_length] = token_ids.dims();

        if batch_size == 0 {
            return Err("Batch size must be greater than zero");
        }

        if sequence_length == 0 {
            return Err("Sequence length must be greater than zero");
        }

        if sequence_length > self.config.max_seq_len {
            return Err("Sequence exceeds max_seq_len");
        }

        // Generate position IDs: [0, 1, 2, ..., sequence_length - 1].
        let device = token_ids.device();

        let sequence_length_i64 =
            i64::try_from(sequence_length).map_err(|_| "Sequence length is too large")?;

        let position_ids = Tensor::<1, Int>::arange(0..sequence_length_i64, &device)
            .reshape([1, sequence_length])
            .repeat(&[batch_size, 1]);

        // Convert token IDs and position IDs into embeddings.
        let token_embeddings = self.token_embedding.forward(token_ids);

        let position_embeddings = self.position_embedding.forward(position_ids);

        // Combine token identity and token position.
        let mut hidden = token_embeddings + position_embeddings;

        // Process the sequence through every decoder block.
        for block in &self.blocks {
            hidden = block.forward(hidden, &self.config)?;
        }

        // Normalize the final hidden representations.
        let hidden = self.final_norm.forward(hidden);

        // Produce scores across the vocabulary.
        let logits = self.output_projection.forward(hidden);

        Ok(logits)
    }
}
