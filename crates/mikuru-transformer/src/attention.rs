//! Causal multi-head self-attention.
//!
//! Input shape:  [batch, sequence, model_dim]
//! Output shape: [batch, sequence, model_dim]

use burn::{
    nn::{Linear, LinearConfig},
    prelude::*,
    tensor::{module::attention, ops::AttentionModuleOptions},
};

use crate::config::TransformerConfig;

/// Multi-head self-attention with causal masking.
#[derive(Module, Debug)]
pub struct CausalSelfAttention {
    query: Linear,
    key: Linear,
    value: Linear,
    output: Linear,
}

impl CausalSelfAttention {
    /// Creates the four learned projection layers.
    pub fn new(config: &TransformerConfig, device: &Device) -> Result<Self, &'static str> {
        config.validate()?;

        let model_dim = config.model_dim;

        Ok(Self {
            query: LinearConfig::new(model_dim, model_dim).init(device),
            key: LinearConfig::new(model_dim, model_dim).init(device),
            value: LinearConfig::new(model_dim, model_dim).init(device),
            output: LinearConfig::new(model_dim, model_dim).init(device),
        })
    }

    /// Applies causal multi-head self-attention.
    ///
    /// Each token can attend only to itself and earlier tokens.
    pub fn forward(
        &self,
        input: Tensor<3>,
        config: &TransformerConfig,
    ) -> Result<Tensor<3>, &'static str> {
        config.validate()?;

        let [batch_size, sequence_length, model_dim] = input.dims();

        if model_dim != config.model_dim {
            return Err("Input dimension does not match model_dim");
        }

        if sequence_length == 0 {
            return Err("Sequence length must be greater than zero");
        }

        if sequence_length > config.max_seq_len {
            return Err("Sequence exceeds max_seq_len");
        }

        let num_heads = config.num_heads;
        let head_dim = model_dim / num_heads;

        // Project representations into queries, keys and values.
        let query = self
            .query
            .forward(input.clone())
            .reshape([batch_size, sequence_length, num_heads, head_dim])
            .permute([0, 2, 1, 3]);

        let key = self
            .key
            .forward(input.clone())
            .reshape([batch_size, sequence_length, num_heads, head_dim])
            .permute([0, 2, 1, 3]);

        let value = self
            .value
            .forward(input)
            .reshape([batch_size, sequence_length, num_heads, head_dim])
            .permute([0, 2, 1, 3]);

        // Apply scaled dot-product attention with a causal mask.
        let attended = attention(
            query,
            key,
            value,
            None,
            None,
            AttentionModuleOptions {
                scale: None,
                softcap: None,
                is_causal: true,
            },
        );

        // Merge the attention heads back into one representation.
        let merged =
            attended
                .permute([0, 2, 1, 3])
                .reshape([batch_size, sequence_length, model_dim]);

        // Final output projection.
        Ok(self.output.forward(merged))
    }
}
