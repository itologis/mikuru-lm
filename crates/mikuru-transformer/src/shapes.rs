//! Tensor shape conventions used by the Transformer.
//!
//! B = batch size
//! S = sequence length
//! D = model dimension
//! H = number of attention heads
//! DH = dimension per attention head
//! V = vocabulary size

/// Token IDs: [B, S]
pub type TokenIdsShape = [usize; 2];

/// Hidden representations: [B, S, D]
pub type HiddenStatesShape = [usize; 3];

/// Attention Q/K/V tensors: [B, H, S, DH]
pub type AttentionHeadsShape = [usize; 4];

/// Attention scores: [B, H, S, S]
pub type AttentionScoresShape = [usize; 4];

/// Output vocabulary scores: [B, S, V]
pub type LogitsShape = [usize; 3];

pub const fn token_ids(batch: usize, sequence: usize) -> TokenIdsShape {
    [batch, sequence]
}

pub const fn hidden_states(batch: usize, sequence: usize, model_dim: usize) -> HiddenStatesShape {
    [batch, sequence, model_dim]
}

pub const fn attention_heads(
    batch: usize,
    heads: usize,
    sequence: usize,
    head_dim: usize,
) -> AttentionHeadsShape {
    [batch, heads, sequence, head_dim]
}

pub const fn attention_scores(batch: usize, heads: usize, sequence: usize) -> AttentionScoresShape {
    [batch, heads, sequence, sequence]
}

pub const fn logits(batch: usize, sequence: usize, vocab_size: usize) -> LogitsShape {
    [batch, sequence, vocab_size]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn produces_expected_shapes() {
        assert_eq!(token_ids(1, 128), [1, 128]);
        assert_eq!(hidden_states(1, 128, 128), [1, 128, 128]);
        assert_eq!(attention_heads(1, 4, 128, 32), [1, 4, 128, 32]);
        assert_eq!(attention_scores(1, 4, 128), [1, 4, 128, 128]);
        assert_eq!(logits(1, 128, 256), [1, 128, 256]);
    }
}
