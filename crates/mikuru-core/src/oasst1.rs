//! OpenAssistant Conversations (OASST1) import.
//!
//! OASST1 stores every message as one row of a flat table. Conversations are
//! trees: each message has a `parent_id`, and a root prompt can have many
//! sibling replies. This module turns one message tree into **one**
//! conversation path and never merges siblings or invents a reply.
//!
//! Rules (all deliberate, all documented in
//! `data/third-party/oasst1/README.md`):
//!
//! - every message in the chosen path must pass [`check_message`];
//! - exactly one root per tree, and exactly one root-to-leaf path is exported;
//! - children must be direct `parent_id` links with alternating roles;
//! - the path is chosen by maximum summed `labels.quality.value`, then by
//!   length, then lexicographically by message id, so the result is
//!   deterministic;
//! - the tree is rejected when that path exceeds the configured character or
//!   message budget.
//!
//! The importer reads the canonical `2023-04-12_oasst_ready.messages.jsonl`
//! export, which is gzip-compressed upstream. Decompress it with standard
//! tools (for example `gzip -dc`) before running this importer; no compression
//! dependency is added to the workspace.

use std::{
    collections::{BTreeMap, HashMap, HashSet},
    error::Error,
    fmt,
    fs::{self, File},
    io::{self, BufRead, BufReader},
    path::{Path, PathBuf},
};

use serde::{Deserialize, Serialize};

use crate::{
    ConversationExample, ConversationMessage,
    dataset::{DatasetError, write_jsonl},
};

/// Canonical dataset page for OASST1.
pub const OASST1_DATASET_URL: &str = "https://huggingface.co/datasets/OpenAssistant/oasst1";

/// Paper describing OASST1.
pub const OASST1_PAPER_URL: &str = "https://arxiv.org/abs/2304.07327";

/// Value recorded in `source_dataset` for imported records.
pub const OASST1_SOURCE_DATASET: &str = "OpenAssistant OASST1";

/// License declared by the dataset card and LICENSE file.
pub const OASST1_LICENSE: &str = "Apache-2.0";

/// Hugging Face revision used for the first verified import.
pub const OASST1_REVISION: &str = "fdf72ae0827c1cda404aff25b6603abec9e3399b";

/// Message toxicity score at or above this value is rejected.
pub const MAX_TOXICITY: f64 = 0.5;

/// Review label value above this threshold is rejected as unsafe.
pub const MAX_UNSAFE_LABEL: f64 = 0.5;

/// Configuration for one import run.
#[derive(Clone, Debug)]
pub struct OasstImportConfig {
    /// Maximum total characters across the whole conversation.
    pub max_chars: usize,
    /// Maximum messages in a conversation.
    pub max_messages: usize,
    /// Minimum messages in a conversation. Two is the smallest useful turn.
    pub min_messages: usize,
    /// Dataset revision recorded in provenance notes.
    pub revision: String,
}

impl Default for OasstImportConfig {
    fn default() -> Self {
        Self {
            max_chars: 500,
            max_messages: 8,
            min_messages: 2,
            revision: OASST1_REVISION.to_owned(),
        }
    }
}

/// One message row from the OASST1 export.
///
/// Only the fields used by the importer are declared; serde ignores the rest
/// (`created_date`, `user_id`, `emojis`, `rank`, `model_name`, ...).
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct OasstMessage {
    pub message_id: String,
    #[serde(default)]
    pub parent_id: Option<String>,
    pub message_tree_id: String,
    pub text: String,
    pub role: String,
    pub lang: String,
    #[serde(default)]
    pub deleted: bool,
    #[serde(default)]
    pub synthetic: bool,
    /// Missing for messages that were never reviewed.
    #[serde(default)]
    pub review_result: Option<bool>,
    #[serde(default)]
    pub tree_state: Option<String>,
    #[serde(default)]
    pub labels: HashMap<String, OasstLabel>,
    #[serde(default)]
    pub detoxify: Option<OasstToxicity>,
}

/// One human review label with its mean value.
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct OasstLabel {
    #[serde(default)]
    pub value: f64,
    #[serde(default)]
    pub count: i32,
}

/// Automatic toxicity scores supplied by OASST1.
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct OasstToxicity {
    #[serde(default)]
    pub toxicity: f64,
}

/// Why one message was excluded.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum MessageReject {
    Deleted,
    ReviewMissing,
    ReviewFailed,
    NotEnglish,
    InvalidRole,
    Synthetic,
    EmptyText,
    Toxic,
    Pii,
    UnsafeLabel,
}

impl MessageReject {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Deleted => "deleted",
            Self::ReviewMissing => "review_result missing",
            Self::ReviewFailed => "review_result false",
            Self::NotEnglish => "not English",
            Self::InvalidRole => "unrecognised role",
            Self::Synthetic => "synthetic message",
            Self::EmptyText => "empty text",
            Self::Toxic => "toxicity >= 0.5",
            Self::Pii => "PII label",
            Self::UnsafeLabel => "unsafe label",
        }
    }
}

/// Why one tree produced no conversation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum TreeReject {
    DuplicateMessageId,
    MissingParent,
    MissingRoot,
    MultipleRoots,
    Cycle,
    NoValidPath,
    TooShort,
    TooLong,
}

impl TreeReject {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::DuplicateMessageId => "duplicate message id",
            Self::MissingParent => "parent id not in tree",
            Self::MissingRoot => "no root message",
            Self::MultipleRoots => "multiple roots",
            Self::Cycle => "parent cycle",
            Self::NoValidPath => "no accessible accepted path",
            Self::TooShort => "shorter than the minimum",
            Self::TooLong => "longer than the character/message budget",
        }
    }
}

/// Counters describing an import run.
#[derive(Clone, Debug, Default)]
pub struct OasstImportStats {
    pub messages_read: usize,
    pub trees_read: usize,
    pub messages_accepted: usize,
    pub conversations_written: usize,
    pub messages_in_conversations: usize,
    pub message_rejections: BTreeMap<MessageReject, usize>,
    pub tree_rejections: BTreeMap<TreeReject, usize>,
    pub total_characters: usize,
    pub max_conversation_chars: usize,
    pub max_conversation_messages: usize,
}

impl OasstImportStats {
    fn reject_message(&mut self, reason: MessageReject) {
        *self.message_rejections.entry(reason).or_insert(0) += 1;
    }

    fn reject_tree(&mut self, reason: TreeReject) {
        *self.tree_rejections.entry(reason).or_insert(0) += 1;
    }
}

/// Errors that stop the import.
#[derive(Debug)]
pub enum OasstImportError {
    Io {
        path: PathBuf,
        source: io::Error,
    },
    Json {
        path: PathBuf,
        line: usize,
        message: String,
    },
    Dataset(DatasetError),
    /// A tree id appeared again after another tree had started.
    DuplicateTree {
        tree_id: String,
        line: usize,
    },
    /// The export contained no messages.
    Empty {
        path: PathBuf,
    },
}

impl fmt::Display for OasstImportError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io { path, source } => {
                write!(formatter, "failed to access {}: {source}", path.display())
            }
            Self::Json {
                path,
                line,
                message,
            } => write!(
                formatter,
                "invalid OASST1 JSON at {}:{line}: {message}",
                path.display()
            ),
            Self::Dataset(error) => write!(formatter, "{error}"),
            Self::DuplicateTree { tree_id, line } => write!(
                formatter,
                "tree `{tree_id}` reappeared at line {line}; trees must be contiguous"
            ),
            Self::Empty { path } => write!(formatter, "{} contains no messages", path.display()),
        }
    }
}

impl Error for OasstImportError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Io { source, .. } => Some(source),
            Self::Dataset(error) => Some(error),
            _ => None,
        }
    }
}

impl From<DatasetError> for OasstImportError {
    fn from(error: DatasetError) -> Self {
        Self::Dataset(error)
    }
}

fn label_value(message: &OasstMessage, name: &str) -> Option<f64> {
    message
        .labels
        .get(name)
        .map(|label| label.value)
        .filter(|value| value.is_finite())
}

/// Returns the first reason a message is not usable, or `Ok(())`.
///
/// The check order is fixed so rejection counters are mutually exclusive.
pub fn check_message(message: &OasstMessage) -> Result<(), MessageReject> {
    if message.deleted {
        return Err(MessageReject::Deleted);
    }

    match message.review_result {
        Some(true) => {}
        Some(false) => return Err(MessageReject::ReviewFailed),
        None => return Err(MessageReject::ReviewMissing),
    }

    if message.lang != "en" {
        return Err(MessageReject::NotEnglish);
    }

    if message.role != "prompter" && message.role != "assistant" {
        return Err(MessageReject::InvalidRole);
    }

    if message.synthetic {
        return Err(MessageReject::Synthetic);
    }

    if message.text.trim().is_empty() {
        return Err(MessageReject::EmptyText);
    }

    if message
        .detoxify
        .as_ref()
        .is_some_and(|scores| scores.toxicity >= MAX_TOXICITY)
    {
        return Err(MessageReject::Toxic);
    }

    if label_value(message, "pii").is_some_and(|value| value > 0.0) {
        return Err(MessageReject::Pii);
    }

    for name in [
        "spam",
        "not_appropriate",
        "hate_speech",
        "sexual_content",
        "violence",
    ] {
        if label_value(message, name).is_some_and(|value| value > MAX_UNSAFE_LABEL) {
            return Err(MessageReject::UnsafeLabel);
        }
    }

    Ok(())
}

fn quality(message: &OasstMessage) -> f64 {
    label_value(message, "quality").unwrap_or(0.0)
}

/// One candidate path: summed quality and message indices, root first.
struct PathChoice {
    score: f64,
    path: Vec<usize>,
}

/// Deterministic total order: quality, then length, then smallest id sequence.
fn is_better(candidate: &PathChoice, current: &PathChoice, tree: &[OasstMessage]) -> bool {
    match candidate.score.total_cmp(&current.score) {
        std::cmp::Ordering::Greater => return true,
        std::cmp::Ordering::Less => return false,
        std::cmp::Ordering::Equal => {}
    }

    if candidate.path.len() != current.path.len() {
        return candidate.path.len() > current.path.len();
    }

    let candidate_ids = candidate
        .path
        .iter()
        .map(|&index| tree[index].message_id.as_str());

    let current_ids = current
        .path
        .iter()
        .map(|&index| tree[index].message_id.as_str());

    candidate_ids.cmp(current_ids) == std::cmp::Ordering::Less
}

/// Search result: best path within budgets and longest valid path.
#[derive(Default)]
struct PathSearch {
    best: Option<PathChoice>,
    longest_valid_path_messages: usize,
}

/// Outcome of reconstructing one tree.
pub struct Reconstruction {
    pub example: ConversationExample,
    pub message_ids: Vec<String>,
    pub characters: usize,
}

fn conversation_from_path(
    tree_id: &str,
    path: &[usize],
    tree: &[OasstMessage],
    config: &OasstImportConfig,
) -> Reconstruction {
    let mut messages = Vec::with_capacity(path.len());
    let mut message_ids = Vec::with_capacity(path.len());
    let mut characters = 0usize;

    for &index in path {
        let message = &tree[index];

        messages.push(ConversationMessage {
            role: match message.role.as_str() {
                "prompter" => "user".to_owned(),
                _ => "assistant".to_owned(),
            },
            content: message.text.clone(),
        });

        message_ids.push(message.message_id.clone());
        characters += message.text.chars().count();
    }

    let example = ConversationExample {
        id: format!("oasst1-{tree_id}"),
        messages,
        language: "en".to_owned(),
        category: "open_domain_conversation".to_owned(),
        source_dataset: OASST1_SOURCE_DATASET.to_owned(),
        source_record_id: tree_id.to_owned(),
        source_url: OASST1_DATASET_URL.to_owned(),
        license: OASST1_LICENSE.to_owned(),
        synthetic: false,
        split: None,
        references: vec![OASST1_PAPER_URL.to_owned()],
        source_message_ids: message_ids.clone(),
        provenance_note: format!(
            "OASST1 ready export at revision {}; English human-written message path \
             selected by highest summed quality among budget-compatible paths; deleted, failed-review, synthetic, \
             toxic, and PII-labelled messages excluded; prompter mapped to user; \
             original message ids preserved.",
            config.revision
        ),
    };

    Reconstruction {
        example,
        message_ids,
        characters,
    }
}

/// Reconstructs one conversation from one OASST1 message tree.
///
/// The input must contain every message of a single tree. The returned
/// conversation is the best-scoring root-to-leaf path that passes
/// [`check_message`]. Sibling branches are never combined.
pub fn reconstruct_tree(
    tree: &[OasstMessage],
    config: &OasstImportConfig,
) -> Result<Reconstruction, TreeReject> {
    if tree.is_empty() {
        return Err(TreeReject::MissingRoot);
    }

    let mut index: HashMap<&str, usize> = HashMap::with_capacity(tree.len());

    for (position, message) in tree.iter().enumerate() {
        if index
            .insert(message.message_id.as_str(), position)
            .is_some()
        {
            return Err(TreeReject::DuplicateMessageId);
        }
    }

    let mut roots = Vec::new();
    let mut children: HashMap<usize, Vec<usize>> = HashMap::new();

    for (position, message) in tree.iter().enumerate() {
        match message.parent_id.as_deref() {
            None => roots.push(position),
            Some(parent) => {
                let Some(&parent_index) = index.get(parent) else {
                    return Err(TreeReject::MissingParent);
                };

                children.entry(parent_index).or_default().push(position);
            }
        }
    }

    let root = match roots.as_slice() {
        [] => return Err(TreeReject::MissingRoot),
        [root] => *root,
        _ => return Err(TreeReject::MultipleRoots),
    };

    // Children are stored in file order; sorting by id makes the search
    // independent of input ordering for equal scores.
    for siblings in children.values_mut() {
        siblings.sort_by(|&left, &right| {
            tree[left]
                .message_id
                .as_str()
                .cmp(tree[right].message_id.as_str())
        });
    }

    let mut accepted = vec![false; tree.len()];

    for (position, message) in tree.iter().enumerate() {
        accepted[position] = check_message(message).is_ok();
    }

    if !accepted[root] {
        return Err(TreeReject::NoValidPath);
    }

    let mut visiting = HashSet::new();
    let mut cycle = false;
    let search = walk(
        root,
        tree,
        &children,
        &accepted,
        &mut visiting,
        &mut cycle,
        config.max_chars,
        config.max_messages,
        config.min_messages,
    );

    if cycle {
        return Err(TreeReject::Cycle);
    }
    if search.longest_valid_path_messages < config.min_messages {
        return Err(TreeReject::TooShort);
    }
    let Some(best) = search.best else {
        return Err(TreeReject::TooLong);
    };

    let reconstruction = conversation_from_path(&tree[0].message_tree_id, &best.path, tree, config);

    if best.path.len() < config.min_messages {
        return Err(TreeReject::TooShort);
    }
    if best.path.len() > config.max_messages || reconstruction.characters > config.max_chars {
        return Err(TreeReject::TooLong);
    }
    Ok(reconstruction)
}

/// Finds the highest-scoring path that satisfies the configured budgets.
fn walk(
    position: usize,
    tree: &[OasstMessage],
    children: &HashMap<usize, Vec<usize>>,
    accepted: &[bool],
    visiting: &mut HashSet<usize>,
    cycle: &mut bool,
    remaining_chars: usize,
    remaining_messages: usize,
    minimum_messages: usize,
) -> PathSearch {
    if !accepted[position] {
        return PathSearch::default();
    }
    if !visiting.insert(position) {
        *cycle = true;
        return PathSearch::default();
    }

    let chars = tree[position].text.chars().count();
    let fits = remaining_messages > 0 && chars <= remaining_chars;

    let mut best = if fits && minimum_messages <= 1 {
        Some(PathChoice {
            score: quality(&tree[position]),
            path: vec![position],
        })
    } else {
        None
    };

    let mut longest = 1;

    if let Some(siblings) = children.get(&position) {
        for &child in siblings {
            if tree[child].role == tree[position].role {
                continue;
            }

            let result = walk(
                child,
                tree,
                children,
                accepted,
                visiting,
                cycle,
                remaining_chars.saturating_sub(chars),
                remaining_messages.saturating_sub(1),
                minimum_messages.saturating_sub(1),
            );

            if result.longest_valid_path_messages > 0 {
                longest = longest.max(result.longest_valid_path_messages + 1);
            }

            if !fits {
                continue;
            }
            let Some(candidate) = result.best else {
                continue;
            };

            let mut path = Vec::with_capacity(candidate.path.len() + 1);
            path.push(position);
            path.extend(candidate.path);

            let choice = PathChoice {
                score: quality(&tree[position]) + candidate.score,
                path,
            };

            if best
                .as_ref()
                .is_none_or(|current| is_better(&choice, current, tree))
            {
                best = Some(choice);
            }
        }
    }

    visiting.remove(&position);
    PathSearch {
        best,
        longest_valid_path_messages: longest,
    }
}

/// Result of an import run.
#[derive(Debug)]
pub struct OasstImportReport {
    pub output_path: PathBuf,
    pub stats: OasstImportStats,
}

/// Reconstructs and records one completed tree, then clears the buffer.
fn flush_tree(
    tree: &mut Vec<OasstMessage>,
    config: &OasstImportConfig,
    stats: &mut OasstImportStats,
    conversations: &mut Vec<ConversationExample>,
) {
    if tree.is_empty() {
        return;
    }

    stats.trees_read += 1;

    match reconstruct_tree(tree, config) {
        Ok(reconstruction) => {
            stats.conversations_written += 1;
            stats.messages_in_conversations += reconstruction.message_ids.len();
            stats.total_characters += reconstruction.characters;
            stats.max_conversation_chars =
                stats.max_conversation_chars.max(reconstruction.characters);
            stats.max_conversation_messages = stats
                .max_conversation_messages
                .max(reconstruction.message_ids.len());

            conversations.push(reconstruction.example);
        }
        Err(reason) => stats.reject_tree(reason),
    }

    tree.clear();
}

/// Imports an OASST1 messages JSONL file into Mikuru conversation records.
///
/// The output path must not exist. Trees must be contiguous in the input, as
/// they are in the official export.
pub fn import_oasst1_messages(
    input: impl AsRef<Path>,
    output: impl AsRef<Path>,
    config: &OasstImportConfig,
) -> Result<OasstImportReport, OasstImportError> {
    let input = input.as_ref();
    let output = output.as_ref();

    if output.exists() {
        return Err(OasstImportError::Dataset(DatasetError::OutputExists {
            path: output.to_path_buf(),
        }));
    }

    let file = File::open(input).map_err(|source| OasstImportError::Io {
        path: input.to_path_buf(),
        source,
    })?;

    let reader = BufReader::new(file);

    let mut stats = OasstImportStats::default();
    let mut conversations: Vec<ConversationExample> = Vec::new();
    let mut current_tree: Vec<OasstMessage> = Vec::new();
    let mut current_tree_id: Option<String> = None;
    let mut seen_trees: HashSet<String> = HashSet::new();

    for (index, line_result) in reader.lines().enumerate() {
        let line_number = index + 1;

        let line = line_result.map_err(|source| OasstImportError::Io {
            path: input.to_path_buf(),
            source,
        })?;

        if line.trim().is_empty() {
            continue;
        }

        let message: OasstMessage =
            serde_json::from_str(&line).map_err(|error| OasstImportError::Json {
                path: input.to_path_buf(),
                line: line_number,
                message: error.to_string(),
            })?;

        stats.messages_read += 1;

        match check_message(&message) {
            Ok(()) => stats.messages_accepted += 1,
            Err(reason) => stats.reject_message(reason),
        }

        if current_tree_id.as_deref() != Some(message.message_tree_id.as_str()) {
            flush_tree(&mut current_tree, config, &mut stats, &mut conversations);

            if !seen_trees.insert(message.message_tree_id.clone()) {
                return Err(OasstImportError::DuplicateTree {
                    tree_id: message.message_tree_id,
                    line: line_number,
                });
            }

            current_tree_id = Some(message.message_tree_id.clone());
        }

        current_tree.push(message);
    }

    flush_tree(&mut current_tree, config, &mut stats, &mut conversations);

    if stats.messages_read == 0 {
        return Err(OasstImportError::Empty {
            path: input.to_path_buf(),
        });
    }

    if conversations.is_empty() {
        return Err(OasstImportError::Empty {
            path: input.to_path_buf(),
        });
    }

    write_jsonl(output, &conversations)?;

    // Keep the filesystem honest even if a future refactor drops a check.
    let written = fs::metadata(output).map_err(|source| OasstImportError::Io {
        path: output.to_path_buf(),
        source,
    })?;

    if written.len() == 0 {
        return Err(OasstImportError::Empty {
            path: output.to_path_buf(),
        });
    }

    Ok(OasstImportReport {
        output_path: output.to_path_buf(),
        stats,
    })
}

#[cfg(test)]
mod tests {
    use std::{
        collections::HashMap,
        sync::atomic::{AtomicUsize, Ordering},
    };

    use super::*;
    use crate::dataset::load_jsonl;

    fn message(
        id: &str,
        parent: Option<&str>,
        role: &str,
        text: &str,
        quality_score: f64,
    ) -> OasstMessage {
        let mut labels = HashMap::new();
        labels.insert(
            "quality".to_owned(),
            OasstLabel {
                value: quality_score,
                count: 1,
            },
        );

        OasstMessage {
            message_id: id.to_owned(),
            parent_id: parent.map(str::to_owned),
            message_tree_id: "tree-1".to_owned(),
            text: text.to_owned(),
            role: role.to_owned(),
            lang: "en".to_owned(),
            deleted: false,
            synthetic: false,
            review_result: Some(true),
            tree_state: Some("ready_for_export".to_owned()),
            labels,
            detoxify: Some(OasstToxicity::default()),
        }
    }

    fn ids(reconstruction: &Reconstruction) -> Vec<&str> {
        reconstruction
            .example
            .messages
            .iter()
            .map(|message| message.role.as_str())
            .collect()
    }

    #[test]
    fn reconstructs_one_alternating_path_and_preserves_provenance() {
        let tree = vec![
            message("root", None, "prompter", "What is a monopsony?", 0.8),
            message(
                "answer",
                Some("root"),
                "assistant",
                "A buyer with market power.",
                0.9,
            ),
        ];

        let reconstruction = reconstruct_tree(&tree, &OasstImportConfig::default())
            .expect("path should reconstruct");

        assert_eq!(ids(&reconstruction), ["user", "assistant"]);
        assert_eq!(
            reconstruction.message_ids,
            ["root".to_owned(), "answer".to_owned()]
        );
        assert_eq!(reconstruction.example.id, "oasst1-tree-1");
        assert_eq!(reconstruction.example.source_record_id, "tree-1");
        assert_eq!(reconstruction.example.source_url, OASST1_DATASET_URL);
        assert_eq!(reconstruction.example.license, OASST1_LICENSE);
        assert_eq!(reconstruction.example.source_dataset, OASST1_SOURCE_DATASET);
        assert!(!reconstruction.example.synthetic);
        assert_eq!(reconstruction.example.language, "en");
        assert_eq!(
            reconstruction.example.source_message_ids,
            reconstruction.message_ids
        );
        assert!(
            reconstruction
                .example
                .provenance_note
                .contains(OASST1_REVISION)
        );
        assert_eq!(
            reconstruction.example.messages[0].content,
            "What is a monopsony?"
        );
    }

    #[test]
    fn selects_best_branch_without_mixing_siblings() {
        // root ─┬─ good ── follow-up
        //       └─ weak ── weak-follow-up
        let tree = vec![
            message("root", None, "prompter", "Tell me about tides.", 0.5),
            message(
                "good",
                Some("root"),
                "assistant",
                "Tides are driven by gravity.",
                0.95,
            ),
            message(
                "weak",
                Some("root"),
                "assistant",
                "Tides happen sometimes.",
                0.1,
            ),
            message(
                "follow",
                Some("good"),
                "prompter",
                "Which body matters most?",
                0.9,
            ),
            message("weak-follow", Some("weak"), "prompter", "Really?", 0.1),
        ];

        let reconstruction = reconstruct_tree(&tree, &OasstImportConfig::default())
            .expect("path should reconstruct");

        assert_eq!(reconstruction.message_ids, ["root", "good", "follow"]);
        assert!(!reconstruction.message_ids.iter().any(|id| id == "weak"));
        assert!(
            !reconstruction
                .message_ids
                .iter()
                .any(|id| id == "weak-follow")
        );
    }

    #[test]
    fn chosen_messages_are_direct_parent_child_links() {
        let tree = vec![
            message("root", None, "prompter", "Explain rain.", 0.5),
            message(
                "a",
                Some("root"),
                "assistant",
                "Rain comes from clouds.",
                0.6,
            ),
            message("b", Some("root"), "assistant", "Rain is water.", 0.4),
            message("c", Some("a"), "prompter", "Why does it fall?", 0.7),
            message("d", Some("b"), "prompter", "Does it always?", 0.2),
        ];

        let reconstruction = reconstruct_tree(&tree, &OasstImportConfig::default())
            .expect("path should reconstruct");

        let parents: HashMap<&str, Option<&str>> = tree
            .iter()
            .map(|message| (message.message_id.as_str(), message.parent_id.as_deref()))
            .collect();

        let selected: Vec<&str> = reconstruction
            .message_ids
            .iter()
            .map(String::as_str)
            .collect();

        for pair in selected.windows(2) {
            assert_eq!(
                parents[pair[1]],
                Some(pair[0]),
                "conversation must follow one parent chain"
            );
        }
    }

    #[test]
    fn tie_breaks_deterministically_on_message_id() {
        let tree = vec![
            message("root", None, "prompter", "Pick one.", 0.0),
            message("aaa", Some("root"), "assistant", "First.", 0.5),
            message("zzz", Some("root"), "assistant", "Second.", 0.5),
        ];

        let reconstruction = reconstruct_tree(&tree, &OasstImportConfig::default())
            .expect("path should reconstruct");

        assert_eq!(reconstruction.message_ids, ["root", "aaa"]);
    }

    #[test]
    fn rejects_deleted_messages_before_any_other_reason() {
        let mut deleted = message("root", None, "prompter", "Hello", 0.0);
        deleted.deleted = true;
        deleted.lang = "de".to_owned();

        assert_eq!(check_message(&deleted), Err(MessageReject::Deleted));

        let reconstruction = reconstruct_tree(&[deleted], &OasstImportConfig::default());
        assert!(matches!(reconstruction, Err(TreeReject::NoValidPath)));
    }

    #[test]
    fn skips_deleted_branch_and_keeps_valid_sibling() {
        let mut deleted = message("bad", Some("root"), "assistant", "Deleted reply.", 0.99);
        deleted.deleted = true;

        let tree = vec![
            message("root", None, "prompter", "Hello!", 0.0),
            deleted,
            message("good", Some("root"), "assistant", "Hello there!", 0.1),
        ];

        let reconstruction =
            reconstruct_tree(&tree, &OasstImportConfig::default()).expect("valid branch remains");

        assert_eq!(reconstruction.message_ids, ["root", "good"]);
    }

    #[test]
    fn rejects_unreviewed_and_failed_review_messages() {
        let mut unreviewed = message("a", None, "prompter", "Hi", 0.0);
        unreviewed.review_result = None;

        let mut failed = message("b", None, "prompter", "Hi", 0.0);
        failed.review_result = Some(false);

        assert_eq!(
            check_message(&unreviewed),
            Err(MessageReject::ReviewMissing)
        );
        assert_eq!(check_message(&failed), Err(MessageReject::ReviewFailed));
    }

    #[test]
    fn rejects_non_english_synthetic_empty_and_unsafe_messages() {
        let mut german = message("a", None, "prompter", "Hallo", 0.0);
        german.lang = "de".to_owned();
        assert_eq!(check_message(&german), Err(MessageReject::NotEnglish));

        let mut synthetic = message("b", None, "assistant", "Generated.", 0.0);
        synthetic.synthetic = true;
        assert_eq!(check_message(&synthetic), Err(MessageReject::Synthetic));

        let mut empty = message("c", None, "prompter", "   ", 0.0);
        empty.text = "   ".to_owned();
        assert_eq!(check_message(&empty), Err(MessageReject::EmptyText));

        let mut toxic = message("d", None, "assistant", "Rude.", 0.0);
        toxic.detoxify = Some(OasstToxicity { toxicity: 0.72 });
        assert_eq!(check_message(&toxic), Err(MessageReject::Toxic));

        let mut pii = message("e", None, "prompter", "My address is ...", 0.0);
        pii.labels.insert(
            "pii".to_owned(),
            OasstLabel {
                value: 0.5,
                count: 2,
            },
        );
        assert_eq!(check_message(&pii), Err(MessageReject::Pii));

        let mut unsafe_label = message("f", None, "assistant", "Bad.", 0.0);
        unsafe_label.labels.insert(
            "hate_speech".to_owned(),
            OasstLabel {
                value: 0.8,
                count: 3,
            },
        );
        assert_eq!(
            check_message(&unsafe_label),
            Err(MessageReject::UnsafeLabel)
        );
    }

    #[test]
    fn rejects_same_role_child_because_roles_must_alternate() {
        let tree = vec![
            message("root", None, "prompter", "Question?", 0.0),
            message("child", Some("root"), "prompter", "Another question.", 0.0),
        ];

        let result = reconstruct_tree(&tree, &OasstImportConfig::default());

        assert!(matches!(result, Err(TreeReject::TooShort)));
    }

    #[test]
    fn rejects_conversations_over_the_character_budget() {
        let tree = vec![
            message("root", None, "prompter", &"a".repeat(300), 0.0),
            message("answer", Some("root"), "assistant", &"b".repeat(300), 0.0),
        ];

        let config = OasstImportConfig {
            max_chars: 500,
            ..OasstImportConfig::default()
        };

        let result = reconstruct_tree(&tree, &config);

        assert!(matches!(result, Err(TreeReject::TooLong)));
    }

    #[test]
    fn rejects_conversations_over_the_message_budget() {
        let mut tree = vec![message("m0", None, "prompter", "q0", 0.0)];
        for index in 1..10 {
            let role = if index % 2 == 1 {
                "assistant"
            } else {
                "prompter"
            };
            let parent = format!("m{}", index - 1);
            tree.push(message(
                &format!("m{index}"),
                Some(&parent),
                role,
                &format!("t{index}"),
                0.0,
            ));
        }
        let config = OasstImportConfig {
            max_messages: 8,
            min_messages: 9,
            ..OasstImportConfig::default()
        };
        assert!(matches!(
            reconstruct_tree(&tree, &config),
            Err(TreeReject::TooLong)
        ));
    }

    #[test]
    fn rejects_structurally_invalid_trees() {
        let duplicate = vec![
            message("same", None, "prompter", "a", 0.0),
            message("same", Some("same"), "assistant", "b", 0.0),
        ];
        assert!(matches!(
            reconstruct_tree(&duplicate, &OasstImportConfig::default()),
            Err(TreeReject::DuplicateMessageId)
        ));

        let orphan = vec![
            message("root", None, "prompter", "a", 0.0),
            message("orphan", Some("missing"), "assistant", "b", 0.0),
        ];
        assert!(matches!(
            reconstruct_tree(&orphan, &OasstImportConfig::default()),
            Err(TreeReject::MissingParent)
        ));

        let two_roots = vec![
            message("root-a", None, "prompter", "a", 0.0),
            message("root-b", None, "prompter", "b", 0.0),
        ];
        assert!(matches!(
            reconstruct_tree(&two_roots, &OasstImportConfig::default()),
            Err(TreeReject::MultipleRoots)
        ));

        let no_root = vec![message("child", Some("missing"), "assistant", "b", 0.0)];
        assert!(matches!(
            reconstruct_tree(&no_root, &OasstImportConfig::default()),
            Err(TreeReject::MissingParent)
        ));
    }

    #[test]
    fn rejects_parent_cycles() {
        let tree = vec![
            message("a", Some("b"), "prompter", "a", 0.0),
            message("b", Some("a"), "assistant", "b", 0.0),
        ];

        let result = reconstruct_tree(&tree, &OasstImportConfig::default());

        assert!(matches!(result, Err(TreeReject::MissingRoot)));
    }

    #[test]
    fn selects_shorter_valid_branch_when_best_branch_exceeds_budget() {
        let long = "L".repeat(300);
        let tree = vec![
            message("root", None, "prompter", "Question?", 0.1),
            message("long", Some("root"), "assistant", &long, 0.99),
            message("short", Some("root"), "assistant", "Useful answer.", 0.1),
        ];
        let config = OasstImportConfig {
            max_chars: 100,
            ..OasstImportConfig::default()
        };
        let result = reconstruct_tree(&tree, &config).unwrap();
        assert_eq!(result.message_ids, ["root", "short"]);
    }

    #[test]
    fn keeps_prefix_that_fits_message_budget() {
        let mut tree = vec![message("m0", None, "prompter", "q0", 0.0)];
        for index in 1..10 {
            let role = if index % 2 == 1 {
                "assistant"
            } else {
                "prompter"
            };
            let parent = format!("m{}", index - 1);
            tree.push(message(
                &format!("m{index}"),
                Some(&parent),
                role,
                &format!("t{index}"),
                0.0,
            ));
        }
        let config = OasstImportConfig {
            max_messages: 8,
            ..OasstImportConfig::default()
        };
        let result = reconstruct_tree(&tree, &config).unwrap();
        assert_eq!(
            result.message_ids,
            ["m0", "m1", "m2", "m3", "m4", "m5", "m6", "m7"]
        );
    }

    #[test]
    fn minimum_length_is_applied_before_quality_ranking() {
        let tree = vec![
            message("root", None, "prompter", "Question?", 0.0),
            message("high", Some("root"), "assistant", "Short.", 0.99),
            message("lower", Some("root"), "assistant", "Another answer.", 0.1),
            message("follow", Some("lower"), "prompter", "Explain more.", 0.1),
        ];
        let config = OasstImportConfig {
            min_messages: 3,
            ..OasstImportConfig::default()
        };
        let result = reconstruct_tree(&tree, &config).unwrap();
        assert_eq!(result.message_ids, ["root", "lower", "follow"]);
    }

    #[test]
    fn selects_a_character_budget_compatible_prefix() {
        let tree = vec![
            message("root", None, "prompter", &"a".repeat(200), 0.0),
            message("answer", Some("root"), "assistant", &"b".repeat(200), 0.0),
            message("follow", Some("answer"), "prompter", &"c".repeat(200), 0.0),
        ];
        let config = OasstImportConfig {
            max_chars: 500,
            ..OasstImportConfig::default()
        };
        let result = reconstruct_tree(&tree, &config).unwrap();
        assert_eq!(result.message_ids, ["root", "answer"]);
        assert_eq!(result.characters, 400);
    }

    fn temp_dir(name: &str) -> PathBuf {
        static COUNTER: AtomicUsize = AtomicUsize::new(0);

        let index = COUNTER.fetch_add(1, Ordering::Relaxed);

        let dir = std::env::temp_dir().join(format!(
            "mikuru-oasst1-test-{}-{index}-{name}",
            std::process::id()
        ));

        fs::create_dir_all(&dir).expect("temporary directory should be created");

        dir
    }

    fn write_messages(dir: &Path, name: &str, messages: &[OasstMessage]) -> PathBuf {
        let path = dir.join(name);
        let contents = messages
            .iter()
            .map(|message| serde_json::to_string(message).expect("message should serialize"))
            .collect::<Vec<_>>()
            .join("\n");

        fs::write(&path, contents + "\n").expect("fixture should be written");
        path
    }

    #[test]
    fn imports_contiguous_trees_and_reports_rejections() {
        let dir = temp_dir("import");

        let mut bad_root = message("bad-root", None, "prompter", "Deleted prompt.", 0.0);
        bad_root.message_tree_id = "tree-2".to_owned();
        bad_root.deleted = true;

        let messages = vec![
            message("root", None, "prompter", "What is a tide?", 0.9),
            message(
                "answer",
                Some("root"),
                "assistant",
                "A gravitational effect.",
                0.9,
            ),
            bad_root,
        ];

        let input = write_messages(&dir, "messages.jsonl", &messages);
        let output = dir.join("records.jsonl");

        let report = import_oasst1_messages(&input, &output, &OasstImportConfig::default())
            .expect("import should succeed");

        assert_eq!(report.stats.messages_read, 3);
        assert_eq!(report.stats.trees_read, 2);
        assert_eq!(report.stats.conversations_written, 1);
        assert_eq!(report.stats.messages_in_conversations, 2);
        assert_eq!(
            report.stats.message_rejections.get(&MessageReject::Deleted),
            Some(&1)
        );

        let records = load_jsonl(&output).expect("imported records should load");
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].id, "oasst1-tree-1");
        assert_eq!(records[0].license, "Apache-2.0");

        let error = import_oasst1_messages(&input, &output, &OasstImportConfig::default())
            .expect_err("existing output must never be overwritten");
        assert!(error.to_string().contains("already exists"));

        fs::remove_dir_all(dir).expect("temporary directory should be removed");
    }

    #[test]
    fn rejects_a_tree_id_that_reappears_later() {
        let dir = temp_dir("duplicate-tree");

        let mut second = message("other-root", None, "prompter", "Other tree.", 0.0);
        second.message_tree_id = "tree-2".to_owned();

        let messages = vec![
            message("root", None, "prompter", "First tree.", 0.0),
            message("answer", Some("root"), "assistant", "Reply.", 0.0),
            second,
            message("root-again", None, "prompter", "First tree again.", 0.0),
        ];

        let input = write_messages(&dir, "messages.jsonl", &messages);
        let output = dir.join("records.jsonl");

        let error = import_oasst1_messages(&input, &output, &OasstImportConfig::default())
            .expect_err("non-contiguous trees must be rejected");

        assert!(
            error.to_string().contains("reappeared"),
            "error should explain the tree ordering: {error}"
        );

        fs::remove_dir_all(dir).expect("temporary directory should be removed");
    }

    #[test]
    fn rejects_broken_json_with_a_line_number() {
        let dir = temp_dir("broken-json");
        let input = dir.join("messages.jsonl");

        fs::write(&input, "{\"message_id\":\"x\",\"text\":\n").expect("fixture should be written");

        let error = import_oasst1_messages(
            &input,
            dir.join("records.jsonl"),
            &OasstImportConfig::default(),
        )
        .expect_err("malformed JSON must be rejected");

        assert!(
            error.to_string().contains("line 1"),
            "error should identify the line: {error}"
        );

        fs::remove_dir_all(dir).expect("temporary directory should be removed");
    }
}
