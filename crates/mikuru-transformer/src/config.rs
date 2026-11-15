//! Configuration for the autoregressive Transformer.

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TransformerConfig {
    pub vocab_size: usize,
    pub model_dim: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub max_seq_len: usize,
}

impl Default for TransformerConfig {
    fn default() -> Self {
        Self {
            vocab_size: 256,
            model_dim: 128,
            num_layers: 2,
            num_heads: 4,
            max_seq_len: 128,
        }
    }
}

impl TransformerConfig {
    pub const FEED_FORWARD_MULTIPLIER: usize = 4;

    /// Check whether the configuration is usable.
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.vocab_size < 2 {
            return Err("vocab_size must be at least 2");
        }

        if self.model_dim == 0 {
            return Err("model_dim must be greater than 0");
        }

        if self.num_layers == 0 {
            return Err("num_layers must be greater than 0");
        }

        if self.num_heads == 0 {
            return Err("num_heads must be greater than 0");
        }

        if self.max_seq_len == 0 {
            return Err("max_seq_len must be greater than 0");
        }

        if self.model_dim % self.num_heads != 0 {
            return Err("model_dim must be divisible by num_heads");
        }

        Ok(())
    }

    /// Width of one attention head.
    pub fn head_dim(&self) -> Option<usize> {
        if self.num_heads == 0 || self.model_dim % self.num_heads != 0 {
            return None;
        }

        Some(self.model_dim / self.num_heads)
    }

    /// Width of the feed-forward network's hidden layer.
    pub fn feed_forward_dim(&self) -> Option<usize> {
        self.model_dim.checked_mul(Self::FEED_FORWARD_MULTIPLIER)
    }
}

#[cfg(test)]
mod tests {
    use super::TransformerConfig;

    #[test]
    fn default_configuration_is_valid() {
        let config = TransformerConfig::default();

        assert_eq!(config.validate(), Ok(()));
        assert_eq!(config.model_dim, 128);
        assert_eq!(config.head_dim(), Some(32));
        assert_eq!(config.feed_forward_dim(), Some(512));
    }
}
