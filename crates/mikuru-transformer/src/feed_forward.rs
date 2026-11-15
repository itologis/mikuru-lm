//! Position-wise feed-forward network.
//!
//! Input:  [batch, sequence, model_dim]
//! Hidden: [batch, sequence, feed_forward_dim]
//! Output: [batch, sequence, model_dim]

use burn::nn::{Gelu, Linear, LinearConfig};
use burn::prelude::*;

use crate::config::TransformerConfig;

/// Two linear projections with GELU activation between them.
#[derive(Module, Debug)]
pub struct FeedForward {
    expansion: Linear,
    activation: Gelu,
    contraction: Linear,
}

impl FeedForward {
    /// Creates the feed-forward network.
    pub fn new(config: &TransformerConfig, device: &Device) -> Result<Self, &'static str> {
        config.validate()?;

        let model_dim = config.model_dim;
        let hidden_dim = config
            .feed_forward_dim()
            .ok_or("Feed-forward dimension overflow")?;

        Ok(Self {
            expansion: LinearConfig::new(model_dim, hidden_dim).init(device),

            activation: Gelu::new(),

            contraction: LinearConfig::new(hidden_dim, model_dim).init(device),
        })
    }

    /// Processes each token's representation independently.
    pub fn forward(
        &self,
        input: Tensor<3>,
        config: &TransformerConfig,
    ) -> Result<Tensor<3>, &'static str> {
        config.validate()?;

        let [_, _, model_dim] = input.dims();

        if model_dim != config.model_dim {
            return Err("Input dimension does not match model_dim");
        }

        let expanded = self.expansion.forward(input);
        let activated = self.activation.forward(expanded);

        Ok(self.contraction.forward(activated))
    }
}
