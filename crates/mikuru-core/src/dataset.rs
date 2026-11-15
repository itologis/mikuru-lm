use serde::{Deserialize, Serialize};

/// One text record in the initial JSONL dataset format.
///
/// `text` is required. `source` and `split` are optional metadata fields.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct TextExample {
    pub text: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub split: Option<String>,
}

/// One message inside a conversation.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct ConversationMessage {
    /// Who sent the message, such as "user" or "assistant".
    pub role: String,

    /// The actual message text.
    pub content: String,
}

/// A complete conversation record with its provenance metadata.
///
/// This preserves the original dataset structure before any conversion
/// into a format suitable for tokenization or model training.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct ConversationExample {
    /// Unique identifier for this record.
    pub id: String,

    /// Ordered messages belonging to this conversation.
    pub messages: Vec<ConversationMessage>,

    /// Language code, such as "en".
    pub language: String,

    /// Dataset category, such as "japanese_culture_knowledge".
    pub category: String,

    /// Name of the dataset or collection this record came from.
    pub source_dataset: String,

    /// Original record identifier from the source.
    pub source_record_id: String,

    /// Original source URL or source-provided placeholder.
    pub source_url: String,

    /// License information exactly as supplied by the source.
    pub license: String,

    /// Whether this record was synthetically generated.
    pub synthetic: bool,

    /// Dataset split, such as "train" or "validation".
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub split: Option<String>,

    /// References used to support or verify this record.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub references: Vec<String>,

    /// Notes explaining the record's origin and preparation.
    pub provenance_note: String,
}

#[cfg(test)]
mod tests {
    use super::{ConversationExample, TextExample};

    #[test]
    fn parses_minimal_text_record() {
        let record: TextExample = serde_json::from_str(r#"{"text":"Hello, Mikuru."}"#)
            .expect("valid minimal JSONL record");

        assert_eq!(record.text, "Hello, Mikuru.");
        assert_eq!(record.source, None);
        assert_eq!(record.split, None);
    }

    #[test]
    fn parses_culture_conversation_record() {
        let json = r#"{
            "id": "culture-genkan-nuance-001",
            "messages": [
                {
                    "role": "user",
                    "content": "I'm staying at a traditional ryokan in Kyoto next month. Are there other specific rules I should watch out for?"
                },
                {
                    "role": "assistant",
                    "content": "Yes, the entryway is the main boundary. Follow the host's instructions about footwear."
                }
            ],
            "language": "en",
            "category": "japanese_culture_knowledge",
            "source_dataset": "Original (Mikuru Pilot)",
            "source_record_id": "synthetic-culture-001",
            "source_url": "N/A",
            "license": "Pending / Project Internal",
            "synthetic": true,
            "split": "train",
            "references": [
                "https://web-japan.org/kidsweb/",
                "https://www.clair.or.jp/e/multiculture/index.html"
            ],
            "provenance_note": "Original dialogue generated for pilot."
        }"#;

        let record: ConversationExample =
            serde_json::from_str(json).expect("valid culture conversation");

        assert_eq!(record.id, "culture-genkan-nuance-001");
        assert_eq!(record.messages.len(), 2);
        assert_eq!(record.messages[0].role, "user");
        assert_eq!(record.messages[1].role, "assistant");
        assert_eq!(record.language, "en");
        assert_eq!(record.references.len(), 2);
        assert!(record.synthetic);
    }

    #[test]
    fn parses_style_record_without_references() {
        let json = r#"{
            "id": "style-tactful-refactor-001",
            "messages": [
                {
                    "role": "user",
                    "content": "I think we should rewrite the entire authentication module."
                },
                {
                    "role": "assistant",
                    "content": "I understand why you'd want to overhaul it. Could we isolate one part first?"
                }
            ],
            "language": "en",
            "category": "japanese_inspired_style",
            "source_dataset": "Original (Mikuru Pilot)",
            "source_record_id": "synthetic-style-002",
            "source_url": "N/A",
            "license": "Pending / Project Internal",
            "synthetic": true,
            "split": "train",
            "provenance_note": "Demonstration record only."
        }"#;

        let record: ConversationExample =
            serde_json::from_str(json).expect("valid style conversation");

        assert_eq!(record.id, "style-tactful-refactor-001");
        assert_eq!(record.messages.len(), 2);
        assert!(record.references.is_empty());
        assert_eq!(record.split.as_deref(), Some("train"));
    }

    #[test]
    fn conversation_record_survives_json_round_trip() {
        let json = r#"{
            "id": "round-trip-001",
            "messages": [
                {"role": "user", "content": "Hello!"},
                {"role": "assistant", "content": "Hello there!"}
            ],
            "language": "en",
            "category": "general",
            "source_dataset": "Mikuru",
            "source_record_id": "original-001",
            "source_url": "N/A",
            "license": "Project Internal",
            "synthetic": true,
            "provenance_note": "Test record."
        }"#;

        let original: ConversationExample = serde_json::from_str(json).expect("valid conversation");

        let serialized = serde_json::to_string(&original).expect("serialize conversation");

        let restored: ConversationExample =
            serde_json::from_str(&serialized).expect("deserialize conversation");

        assert_eq!(original, restored);
    }
}
