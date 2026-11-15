//! Token embedding layer.
//!
//! Converts token IDs into learned numerical representations.

use burn::nn::{Embedding, EmbeddingConfig, Initializer};
use burn::prelude::*;

use crate::config::TransformerConfig;

#[derive(Module, Debug)]
pub struct TokenEmbedding {
    table: Embedding,
}

impl TokenEmbedding {
    /// Creates the embedding table.
    pub fn new(config: &TransformerConfig, device: &Device) -> Result<Self, &'static str> {
        config.validate()?;

        let table = EmbeddingConfig::new(config.vocab_size, config.model_dim)
            .with_initializer(Initializer::Normal {
                mean: 0.0,
                std: 0.02,
            })
            .init(device);

        Ok(Self { table })
    }

    /// Converts token IDs [batch, sequence]
    /// into embeddings [batch, sequence, model_dim].
    pub fn forward(&self, token_ids: Tensor<2, Int>) -> Tensor<3> {
        self.table.forward(token_ids)
    }
}
