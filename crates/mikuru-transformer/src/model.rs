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

#[cfg(test)]
mod tests {
    //! Behavioural guards for the autoregressive contract.
    //!
    //! Both tests fail if the causal mask leaks information from the right
    //! (future tokens or padding), which would make every reported loss
    //! optimistically wrong.

    use burn::tensor::TensorData;

    use super::*;

    fn test_config() -> TransformerConfig {
        TransformerConfig {
            vocab_size: 32,
            model_dim: 32,
            num_layers: 2,
            num_heads: 4,
            max_seq_len: 16,
        }
    }

    fn int_tensor(
        values: &[i64],
        batch: usize,
        sequence: usize,
        device: &Device,
    ) -> Tensor<2, Int> {
        Tensor::<2, Int>::from_data(TensorData::new(values.to_vec(), [batch, sequence]), device)
    }

    fn logits_values(model: &TransformerModel, input: Tensor<2, Int>) -> Vec<f32> {
        model
            .forward(input)
            .expect("forward should succeed")
            .into_data()
            .to_vec::<f32>()
            .expect("logits should convert to f32")
    }

    #[test]
    fn causal_attention_ignores_future_tokens() {
        let device = Device::flex();
        let model = TransformerModel::new(&test_config(), &device).expect("model should build");
        let vocab = test_config().vocab_size;

        // Identical four-token prefix, different fifth token.
        let first = logits_values(&model, int_tensor(&[3, 5, 7, 11, 13], 1, 5, &device));
        let repeat = logits_values(&model, int_tensor(&[3, 5, 7, 11, 13], 1, 5, &device));
        let changed = logits_values(&model, int_tensor(&[3, 5, 7, 11, 2], 1, 5, &device));

        assert_eq!(first.len(), 5 * vocab);
        assert_eq!(
            first, repeat,
            "the same input must produce identical logits"
        );

        // Nothing before position 4 may observe token 4.
        assert_eq!(
            &first[..4 * vocab],
            &changed[..4 * vocab],
            "a future token changed an earlier prediction: the causal mask leaks"
        );

        // Sanity check that the comparison is not vacuous.
        assert_ne!(
            &first[4 * vocab..],
            &changed[4 * vocab..],
            "the final token should affect the final prediction"
        );
    }

    #[test]
    fn right_padding_does_not_change_shorter_sequence_logits() {
        let device = Device::flex();
        let model = TransformerModel::new(&test_config(), &device).expect("model should build");
        let vocab = test_config().vocab_size;

        let sequence = [4i64, 6, 8, 2];
        let pad = 0i64;

        let alone = logits_values(&model, int_tensor(&sequence, 1, 4, &device));
        let padded = logits_values(
            &model,
            int_tensor(
                &[sequence[0], sequence[1], sequence[2], sequence[3], pad, pad],
                1,
                6,
                &device,
            ),
        );

        assert_eq!(
            alone,
            padded[..4 * vocab],
            "right padding changed predictions for the real tokens"
        );
    }
}
