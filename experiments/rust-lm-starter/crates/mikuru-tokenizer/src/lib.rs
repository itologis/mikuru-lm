//! Byte-Level BPE tokenizer for Mikuru LM.

use std::{collections::HashSet, error::Error, io, path::Path};

use mikuru_core::ConversationExample;

use tokenizers::{
    AddedToken, Tokenizer,
    models::{
        TrainerWrapper,
        bpe::{BPE, BpeTrainerBuilder},
    },
    pre_tokenizers::byte_level::ByteLevel,
};

pub type TokenizerResult<T> = Result<T, Box<dyn Error + Send + Sync>>;

pub const UNK_TOKEN: &str = "<|unk|>";
pub const BOS_TOKEN: &str = "<|bos|>";
pub const EOS_TOKEN: &str = "<|eos|>";
pub const PAD_TOKEN: &str = "<|pad|>";

const SPECIAL_TOKENS: [&str; 4] = [UNK_TOKEN, BOS_TOKEN, EOS_TOKEN, PAD_TOKEN];

/// Settings used when training the BPE vocabulary.
#[derive(Clone, Debug)]
pub struct BpeTrainingConfig {
    /// Target vocabulary size, including special tokens and byte alphabet.
    pub vocab_size: usize,

    /// Minimum frequency required for a pair to be merged.
    pub min_frequency: u64,
}

impl Default for BpeTrainingConfig {
    fn default() -> Self {
        Self {
            vocab_size: 8192,
            min_frequency: 2,
        }
    }
}

/// Mikuru's tokenizer wrapper.
pub struct MikuruTokenizer {
    inner: Tokenizer,
}

impl MikuruTokenizer {
    /// Trains a byte-level BPE tokenizer from an iterator of text records.
    ///
    /// The iterator is consumed during training. It can be backed by a
    /// streaming JSONL reader to avoid loading the entire corpus into RAM.
    pub fn train_from_iterator<I, S>(
        sequences: I,
        config: &BpeTrainingConfig,
    ) -> TokenizerResult<Self>
    where
        I: Iterator<Item = S> + Send,
        S: AsRef<str> + Send,
    {
        let byte_alphabet = ByteLevel::alphabet();

        let minimum_vocab_size = byte_alphabet.len() + SPECIAL_TOKENS.len();

        if config.vocab_size < minimum_vocab_size {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("Vocabulary size must be at least {minimum_vocab_size}"),
            )
            .into());
        }

        // Include every possible byte so unseen Unicode text can still
        // be represented without requiring an unknown-character token.
        let initial_alphabet: HashSet<char> = byte_alphabet.into_iter().collect();

        let special_tokens = SPECIAL_TOKENS
            .iter()
            .map(|token| AddedToken::from((*token).to_owned(), true))
            .collect();

        // Do not add an artificial leading space to each record.
        let byte_level = ByteLevel::default().add_prefix_space(false);

        let mut tokenizer = Tokenizer::new(BPE::default());

        tokenizer.with_pre_tokenizer(Some(byte_level));
        tokenizer.with_post_processor(Some(byte_level));
        tokenizer.with_decoder(Some(byte_level));

        let mut trainer: TrainerWrapper = BpeTrainerBuilder::new()
            .vocab_size(config.vocab_size)
            .min_frequency(config.min_frequency)
            .show_progress(false)
            .special_tokens(special_tokens)
            .initial_alphabet(initial_alphabet)
            .build()
            .into();

        tokenizer.train(&mut trainer, sequences)?;

        Ok(Self { inner: tokenizer })
    }

    /// Loads a previously saved tokenizer.json file.
    pub fn from_file(path: impl AsRef<Path>) -> TokenizerResult<Self> {
        let inner = Tokenizer::from_file(path)?;
        Ok(Self { inner })
    }

    /// Saves the complete tokenizer pipeline as tokenizer.json.
    pub fn save(&self, path: impl AsRef<Path>) -> TokenizerResult<()> {
        self.inner.save(path, true)?;
        Ok(())
    }

    /// Converts text into token IDs.
    pub fn encode(&self, text: &str) -> TokenizerResult<Vec<u32>> {
        let encoding = self.inner.encode(text, false)?;
        Ok(encoding.get_ids().to_vec())
    }

    /// Returns the visible token pieces produced by encoding.
    pub fn encode_tokens(&self, text: &str) -> TokenizerResult<Vec<String>> {
        let encoding = self.inner.encode(text, false)?;
        Ok(encoding.get_tokens().to_vec())
    }

    /// Converts token IDs back into text.
    pub fn decode(&self, ids: &[u32]) -> TokenizerResult<String> {
        Ok(self.inner.decode(ids, false)?)
    }

    /// Returns the complete vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.inner.get_vocab_size(true)
    }

    /// Looks up a token's numerical ID.
    pub fn token_id(&self, token: &str) -> Option<u32> {
        self.inner.token_to_id(token)
    }
}

/// Converts a structured conversation into the text format consumed by the tokenizer.
///
/// Speaker roles and conversation boundaries are preserved using explicit markers.
pub fn format_conversation(example: &ConversationExample) -> String {
    let mut formatted = String::from(BOS_TOKEN);

    for message in &example.messages {
        formatted.push('\n');
        formatted.push_str(message.role.trim());
        formatted.push_str(": ");
        formatted.push_str(&message.content);
    }

    formatted.push('\n');
    formatted.push_str(EOS_TOKEN);

    formatted
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn formats_conversation_with_roles_and_boundaries() {
        let example = ConversationExample {
            id: "test-001".to_owned(),
            messages: vec![
                mikuru_core::ConversationMessage {
                    role: "user".to_owned(),
                    content: "Hello, Mikuru!".to_owned(),
                },
                mikuru_core::ConversationMessage {
                    role: "assistant".to_owned(),
                    content: "Hello there!".to_owned(),
                },
            ],
            language: "en".to_owned(),
            category: "general".to_owned(),
            source_dataset: "test".to_owned(),
            source_record_id: "test-001".to_owned(),
            source_url: "N/A".to_owned(),
            license: "Project Internal".to_owned(),
            synthetic: true,
            split: Some("test".to_owned()),
            references: Vec::new(),
            provenance_note: "Unit test.".to_owned(),
        };

        assert_eq!(
            format_conversation(&example),
            "<|bos|>\nuser: Hello, Mikuru!\nassistant: Hello there!\n<|eos|>"
        );
    }

    #[test]
    fn byte_level_bpe_round_trips_multilingual_text() {
        let samples = vec![
            "Mikuru is learning English.".to_owned(),
            "こんにちは、世界！".to_owned(),
            "Transformers process sequences.".to_owned(),
        ];

        let config = BpeTrainingConfig {
            vocab_size: 512,
            min_frequency: 1,
        };

        let tokenizer = MikuruTokenizer::train_from_iterator(samples.iter(), &config)
            .expect("tokenizer should train");

        let original = "Mikuru は learning しています。";

        let ids = tokenizer.encode(original).expect("text should encode");

        let decoded = tokenizer.decode(&ids).expect("IDs should decode");

        assert_eq!(decoded, original);
        assert!(tokenizer.vocab_size() >= 260);

        for special in SPECIAL_TOKENS {
            assert!(tokenizer.token_id(special).is_some());
        }
    }
}
