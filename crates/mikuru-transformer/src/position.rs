//! Learned positional embeddings.
//!
//! Gives each token position a trainable numerical representation.

use burn::nn::{Embedding, EmbeddingConfig, Initializer};
use burn::prelude::*;

use crate::config::TransformerConfig;

#[derive(Module, Debug)]
pub struct PositionEmbedding {
    table: Embedding,
}

impl PositionEmbedding {
    /// Creates a positional embedding table.
    pub fn new(config: &TransformerConfig, device: &Device) -> Result<Self, &'static str> {
        config.validate()?;

        let table = EmbeddingConfig::new(config.max_seq_len, config.model_dim)
            .with_initializer(Initializer::Normal {
                mean: 0.0,
                std: 0.02,
            })
            .init(device);

        Ok(Self { table })
    }

    /// Converts position IDs [batch, sequence]
    /// into positional vectors [batch, sequence, model_dim].
    pub fn forward(&self, position_ids: Tensor<2, Int>) -> Tensor<3> {
        self.table.forward(position_ids)
    }
}
