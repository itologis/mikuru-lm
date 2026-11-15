//! Transformer decoder block.
//!
//! Uses a Pre-LayerNorm architecture:
//! 1. Layer normalization
//! 2. Causal self-attention
//! 3. Residual connection
//! 4. Layer normalization
//! 5. Feed-forward network
//! 6. Residual connection
//!
//! Input and output shape: [batch, sequence, model_dim]

use burn::nn::{LayerNorm, LayerNormConfig};
use burn::prelude::*;

use crate::attention::CausalSelfAttention;
use crate::config::TransformerConfig;
use crate::feed_forward::FeedForward;

/// One complete Transformer decoder block.
#[derive(Module, Debug)]
pub struct TransformerDecoderBlock {
    attention_norm: LayerNorm,
    attention: CausalSelfAttention,
    feed_forward_norm: LayerNorm,
    feed_forward: FeedForward,
}

impl TransformerDecoderBlock {
    /// Initializes all components of the decoder block.
    pub fn new(config: &TransformerConfig, device: &Device) -> Result<Self, &'static str> {
        config.validate()?;

        let model_dim = config.model_dim;

        Ok(Self {
            attention_norm: LayerNormConfig::new(model_dim).init(device),

            attention: CausalSelfAttention::new(config, device)?,

            feed_forward_norm: LayerNormConfig::new(model_dim).init(device),

            feed_forward: FeedForward::new(config, device)?,
        })
    }

    /// Processes a sequence through the complete decoder block.
    ///
    /// Input:  [batch, sequence, model_dim]
    /// Output: [batch, sequence, model_dim]
    pub fn forward(
        &self,
        input: Tensor<3>,
        config: &TransformerConfig,
    ) -> Result<Tensor<3>, &'static str> {
        config.validate()?;

        let [_, sequence_length, model_dim] = input.dims();

        if model_dim != config.model_dim {
            return Err("Input dimension does not match model_dim");
        }

        if sequence_length == 0 {
            return Err("Sequence length must be greater than zero");
        }

        if sequence_length > config.max_seq_len {
            return Err("Sequence exceeds max_seq_len");
        }

        // Normalize before attention.
        let normalized = self.attention_norm.forward(input.clone());

        // Apply causal self-attention.
        let attended = self.attention.forward(normalized, config)?;

        // First residual connection.
        let hidden = input + attended;

        // Normalize before the feed-forward network.
        let normalized = self.feed_forward_norm.forward(hidden.clone());

        // Apply the feed-forward network.
        let feed_forward_output = self.feed_forward.forward(normalized, config)?;

        // Second residual connection.
        Ok(hidden + feed_forward_output)
    }
}
