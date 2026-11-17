//! Taskmaster-1 inspection and qualification.
//!
//! Taskmaster-1 ships two JSON arrays plus small metadata files. This module
//! reads the official files **without loading the whole dataset into memory**,
//! accounts for every record, and writes three deterministic inspection
//! artifacts:
//!
//! - `taskmaster1-report.md` — human-readable qualification report;
//! - `taskmaster1-report.json` — machine-readable findings;
//! - `taskmaster1-manifest.jsonl` — one inspection entry per source record.
//!
//! The inspector is read-only with respect to the source files. It never
//! writes conversation text into a report, and it never produces a
//! training-ready dataset. The manifest is an inspection artifact, not
//! authorization to train on Taskmaster-1.
//!
//! The two subsets are deliberately kept separate because they were collected
//! by different procedures: written self-dialogs and spoken Wizard-of-Oz
//! dialogs. Duplicate detection is global across both subsets so an identifier
//! reused across files is still visible.

use std::{
    collections::{BTreeMap, BTreeSet},
    error::Error,
    fmt::{self, Write as FmtWrite},
    fs::{self, File, OpenOptions},
    io::{self, BufRead, BufReader, BufWriter, Read, Write as IoWrite},
    path::{Path, PathBuf},
};

use serde::Serialize;
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};

use crate::dataset::DatasetError;

/// Value recorded in `source` for this dataset.
pub const TASKMASTER1_SOURCE_DATASET: &str = "Google Taskmaster-1";

/// Official upstream repository.
pub const TASKMASTER1_REPOSITORY_URL: &str =
    "https://github.com/google-research-datasets/Taskmaster";

/// Directory containing the Taskmaster-1 release inside the repository.
pub const TASKMASTER1_SUBSET_DIRECTORY: &str = "TM-1-2019";

/// Pinned upstream revision used for the first qualification pass.
pub const TASKMASTER1_REVISION: &str = "d92cb6af3005f1dc09c39e75e7daf4a04905e00b";

/// License stated by the Taskmaster-1 README copyright notice.
pub const TASKMASTER1_LICENSE: &str = "CC BY 4.0";

/// Human-readable license URL.
pub const TASKMASTER1_LICENSE_URL: &str = "https://creativecommons.org/licenses/by/4.0/";

/// Taskmaster-1 paper.
pub const TASKMASTER1_PAPER_URL: &str = "https://arxiv.org/abs/1909.05358";

/// Markdown report file name.
pub const REPORT_MARKDOWN_FILE: &str = "taskmaster1-report.md";

/// JSON report file name.
pub const REPORT_JSON_FILE: &str = "taskmaster1-report.json";

/// Candidate manifest file name.
pub const MANIFEST_FILE: &str = "taskmaster1-manifest.jsonl";

const STATUS_VALID: &str = "valid";
const STATUS_FLAGGED: &str = "flagged";
const STATUS_INVALID: &str = "invalid";
const STATUS_MALFORMED: &str = "malformed";

const STATUS_ACCEPTED: &str = "ACCEPTED";
const STATUS_CONDITIONAL: &str = "CONDITIONAL";
const STATUS_REJECTED: &str = "REJECTED";

const MIN_MESSAGES: usize = 2;

/// Conversation-length budgets used only for applicability reporting.
///
/// These are inspection thresholds, not qualification criteria. A conversation
/// that does not fit the current training budget is still a valid source
/// record; it simply cannot be used at the current sequence length.
pub const LENGTH_BUDGETS: [(usize, usize); 4] = [(8, 500), (8, 1_000), (16, 2_000), (32, 4_000)];

/// One of the two Taskmaster-1 collection procedures.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Taskmaster1Subset {
    /// Written two-person dialogs, both roles written by one worker.
    SelfDialogs,
    /// Spoken Wizard-of-Oz dialogs.
    WozDialogs,
}

impl Taskmaster1Subset {
    /// Stable name used in reports and manifests.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::SelfDialogs => "self-dialogs",
            Self::WozDialogs => "woz-dialogs",
        }
    }

    /// Official file name.
    pub fn file_name(self) -> &'static str {
        match self {
            Self::SelfDialogs => "self-dialogs.json",
            Self::WozDialogs => "woz-dialogs.json",
        }
    }

    /// Both official subsets, in a fixed order.
    pub fn all() -> [Self; 2] {
        [Self::SelfDialogs, Self::WozDialogs]
    }
}

/// Configuration for one inspection run.
#[derive(Clone, Debug)]
pub struct Taskmaster1InspectConfig {
    /// Upstream revision recorded in the report and manifest.
    pub revision: String,
}

impl Default for Taskmaster1InspectConfig {
    fn default() -> Self {
        Self {
            revision: TASKMASTER1_REVISION.to_owned(),
        }
    }
}

/// Errors that stop an inspection run.
#[derive(Debug)]
pub enum Taskmaster1Error {
    /// A file could not be opened, read, written, or created.
    Io { path: PathBuf, source: io::Error },
    /// An output file already exists and must not be overwritten.
    OutputExists { path: PathBuf },
    /// The supplied input is not a supported Taskmaster-1 directory.
    UnsupportedInput { path: PathBuf, message: String },
    /// The JSON array framing itself is broken.
    MalformedArray {
        path: PathBuf,
        line: usize,
        message: String,
    },
    /// A report value could not be serialized.
    Serialize { message: String },
    /// A configuration value is invalid.
    InvalidConfig { message: String },
    /// A dataset helper rejected an output path.
    Dataset(DatasetError),
}

impl fmt::Display for Taskmaster1Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io { path, source } => {
                write!(formatter, "failed to access {}: {source}", path.display())
            }
            Self::OutputExists { path } => {
                write!(
                    formatter,
                    "{} already exists; refusing to overwrite",
                    path.display()
                )
            }
            Self::UnsupportedInput { path, message } => {
                write!(
                    formatter,
                    "unsupported Taskmaster-1 input {}: {message}",
                    path.display()
                )
            }
            Self::MalformedArray {
                path,
                line,
                message,
            } => write!(
                formatter,
                "invalid Taskmaster-1 JSON array at {}:{line}: {message}",
                path.display()
            ),
            Self::Serialize { message } => {
                write!(formatter, "failed to serialize report: {message}")
            }
            Self::InvalidConfig { message } => {
                write!(formatter, "invalid configuration: {message}")
            }
            Self::Dataset(error) => write!(formatter, "{error}"),
        }
    }
}

impl Error for Taskmaster1Error {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Io { source, .. } => Some(source),
            Self::Dataset(error) => Some(error),
            _ => None,
        }
    }
}

impl From<DatasetError> for Taskmaster1Error {
    fn from(error: DatasetError) -> Self {
        Self::Dataset(error)
    }
}

/// Fixed provenance and license findings for the inspected release.
#[derive(Clone, Debug, Serialize)]
pub struct DatasetIdentity {
    pub name: String,
    pub publisher: String,
    pub upstream_repository: String,
    pub upstream_revision: String,
    pub subset_directory: String,
    pub paper_url: String,
    pub description: String,
    pub license: String,
    pub license_url: String,
    pub license_evidence_file: String,
    pub license_evidence_present: bool,
    pub license_findings: LicenseFindings,
}

/// Structured license notes. These are summaries, not legal advice.
#[derive(Clone, Debug, Serialize)]
pub struct LicenseFindings {
    pub repository_license_metadata: String,
    pub data_notice: String,
    pub attribution: String,
    pub redistribution: String,
    pub modifications: String,
    pub training_use: String,
    pub caveats: Vec<String>,
}

/// One file found in the supplied dataset directory.
#[derive(Clone, Debug, Serialize)]
pub struct FileInventory {
    pub path: String,
    pub kind: String,
    pub subset: Option<String>,
    pub bytes: u64,
    pub sha256: String,
    pub source_url: Option<String>,
}

/// Min/median/p90/max for a length distribution.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize)]
pub struct LengthStats {
    pub min: usize,
    pub median: usize,
    pub p90: usize,
    pub max: usize,
}

impl LengthStats {
    fn from_values(values: &[usize]) -> Self {
        if values.is_empty() {
            return Self::default();
        }

        let mut sorted = values.to_vec();
        sorted.sort_unstable();

        Self {
            min: sorted[0],
            median: sorted[sorted.len() / 2],
            p90: sorted[(sorted.len() * 9 / 10).min(sorted.len() - 1)],
            max: sorted[sorted.len() - 1],
        }
    }
}

/// Heuristic contact-data pattern counts.
///
/// Counts are per utterance. The inspector records counts only and never
/// writes matched values into a report or manifest.
#[derive(Clone, Debug, Default, Serialize)]
pub struct PiiCounts {
    pub email_like_utterances: usize,
    pub url_like_utterances: usize,
    pub phone_like_utterances: usize,
    pub long_digit_run_utterances: usize,
    pub utterances_affected: usize,
    pub conversations_affected: usize,
}

impl PiiCounts {
    fn add(&mut self, other: &Self) {
        self.email_like_utterances += other.email_like_utterances;
        self.url_like_utterances += other.url_like_utterances;
        self.phone_like_utterances += other.phone_like_utterances;
        self.long_digit_run_utterances += other.long_digit_run_utterances;
        self.utterances_affected += other.utterances_affected;
        self.conversations_affected += other.conversations_affected;
    }
}

/// Qualification decision for one subset.
#[derive(Clone, Debug, Serialize)]
pub struct Qualification {
    pub status: String,
    pub reasons: Vec<String>,
}

/// Statistics and findings for one Taskmaster-1 subset.
#[derive(Clone, Debug, Serialize)]
pub struct SubsetReport {
    pub present: bool,
    pub file: String,
    pub records_read: usize,
    pub conversations_read: usize,
    pub malformed_records: usize,
    pub conversations_valid: usize,
    pub conversations_flagged: usize,
    pub conversations_invalid: usize,
    pub messages: usize,
    pub characters: usize,
    pub annotated_utterances: usize,
    pub non_ascii_characters: usize,
    pub cjk_characters: usize,
    pub non_object_utterances: usize,
    pub missing_text_utterances: usize,
    pub empty_text_utterances: usize,
    pub missing_index_utterances: usize,
    pub non_sequential_index_utterances: usize,
    pub missing_speaker_utterances: usize,
    pub unknown_speaker_utterances: usize,
    pub speakers: BTreeMap<String, usize>,
    pub mapped_roles: BTreeMap<String, usize>,
    pub unmapped_speakers: BTreeMap<String, usize>,
    pub domains: BTreeMap<String, usize>,
    pub instruction_ids: BTreeMap<String, usize>,
    pub conversation_messages: LengthStats,
    pub conversation_characters: LengthStats,
    pub utterance_characters: LengthStats,
    pub issues: BTreeMap<String, usize>,
    pub deleted_placeholders: usize,
    pub same_role_adjacencies: usize,
    pub adjacent_duplicate_utterances: usize,
    pub empty_conversations: usize,
    pub budget_fit: BTreeMap<String, usize>,
    pub privacy: PiiCounts,
    pub qualification: Qualification,
}

impl SubsetReport {
    fn new(subset: Taskmaster1Subset) -> Self {
        let mut budget_fit = BTreeMap::new();

        for (messages, characters) in LENGTH_BUDGETS {
            budget_fit.insert(budget_key(messages, characters), 0);
        }

        Self {
            present: true,
            file: subset.file_name().to_owned(),
            records_read: 0,
            conversations_read: 0,
            malformed_records: 0,
            conversations_valid: 0,
            conversations_flagged: 0,
            conversations_invalid: 0,
            messages: 0,
            characters: 0,
            annotated_utterances: 0,
            non_ascii_characters: 0,
            cjk_characters: 0,
            non_object_utterances: 0,
            missing_text_utterances: 0,
            empty_text_utterances: 0,
            missing_index_utterances: 0,
            non_sequential_index_utterances: 0,
            missing_speaker_utterances: 0,
            unknown_speaker_utterances: 0,
            speakers: BTreeMap::new(),
            mapped_roles: BTreeMap::new(),
            unmapped_speakers: BTreeMap::new(),
            domains: BTreeMap::new(),
            instruction_ids: BTreeMap::new(),
            conversation_messages: LengthStats::default(),
            conversation_characters: LengthStats::default(),
            utterance_characters: LengthStats::default(),
            issues: BTreeMap::new(),
            deleted_placeholders: 0,
            same_role_adjacencies: 0,
            adjacent_duplicate_utterances: 0,
            empty_conversations: 0,
            budget_fit,
            privacy: PiiCounts::default(),
            qualification: Qualification {
                status: STATUS_CONDITIONAL.to_owned(),
                reasons: Vec::new(),
            },
        }
    }
}

/// Where one duplicate occurrence lives.
#[derive(Clone, Debug, Serialize)]
pub struct DuplicateLocation {
    pub subset: String,
    pub file: String,
    pub line: usize,
    pub conversation_id: Option<String>,
}

/// A repeated conversation identifier.
#[derive(Clone, Debug, Serialize)]
pub struct DuplicateIdGroup {
    pub conversation_id: String,
    pub occurrences: usize,
    pub locations: Vec<DuplicateLocation>,
}

/// A group of records with identical utterance payloads.
#[derive(Clone, Debug, Serialize)]
pub struct DuplicatePayloadGroup {
    pub payload_sha256: String,
    pub occurrences: usize,
    pub locations: Vec<DuplicateLocation>,
}

/// Aggregate counts across both subsets.
#[derive(Clone, Debug, Default, Serialize)]
pub struct Totals {
    pub records_read: usize,
    pub conversations_read: usize,
    pub conversations_valid: usize,
    pub conversations_flagged: usize,
    pub conversations_invalid: usize,
    pub malformed_records: usize,
    pub messages: usize,
    pub characters: usize,
}

/// Per-split row counts and unique identifiers.
#[derive(Clone, Debug, Serialize)]
pub struct SplitStats {
    pub present: bool,
    pub rows: usize,
    pub unique_ids: usize,
    pub duplicate_ids: usize,
    pub malformed_rows: usize,
}

/// Coverage of the written self-dialogs by the official split lists.
#[derive(Clone, Debug, Serialize)]
pub struct SplitCoverage {
    pub directory: String,
    pub splits: BTreeMap<String, SplitStats>,
    pub self_dialog_ids: usize,
    pub ids_in_splits: usize,
    pub covered_self_ids: usize,
    pub missing_from_splits: usize,
    pub extra_in_splits: usize,
    pub overlaps: BTreeMap<String, usize>,
}

/// One candidate manifest entry. No conversation text is stored.
#[derive(Clone, Debug, Serialize)]
pub struct Taskmaster1ManifestEntry {
    pub source: String,
    pub source_revision: String,
    pub subset: String,
    pub original_file: String,
    pub original_line: usize,
    pub original_conversation_id: Option<String>,
    pub original_instruction_id: Option<String>,
    pub domain: String,
    pub original_speakers: Vec<String>,
    pub mapped_roles: Vec<Option<String>>,
    pub message_count: usize,
    pub character_count: usize,
    pub status: String,
    pub warnings: Vec<String>,
    pub subset_qualification: String,
}

impl Taskmaster1ManifestEntry {
    fn flag(&mut self, warning: &str) {
        if self.status != STATUS_INVALID && self.status != STATUS_MALFORMED {
            self.status = STATUS_FLAGGED.to_owned();
        }

        if !self.warnings.iter().any(|existing| existing == warning) {
            self.warnings.push(warning.to_owned());
            self.warnings.sort();
        }
    }
}

/// The complete deterministic inspection report.
#[derive(Clone, Debug, Serialize)]
pub struct Taskmaster1Report {
    pub tool: String,
    pub format_version: u32,
    pub dataset: DatasetIdentity,
    pub files: Vec<FileInventory>,
    pub subsets: BTreeMap<String, SubsetReport>,
    pub totals: Totals,
    pub duplicate_conversation_ids: Vec<DuplicateIdGroup>,
    pub duplicate_payloads: Vec<DuplicatePayloadGroup>,
    pub split_coverage: Option<SplitCoverage>,
    pub privacy_notes: Vec<String>,
    pub unresolved_questions: Vec<String>,
}

/// Result of a completed inspection.
#[derive(Debug)]
pub struct Taskmaster1InspectResult {
    pub output_dir: PathBuf,
    pub markdown_path: PathBuf,
    pub json_path: PathBuf,
    pub manifest_path: PathBuf,
    pub report: Taskmaster1Report,
}

/// Reads one JSON array without materializing the array.
struct JsonArrayStream<'a, R: BufRead> {
    reader: &'a mut R,
    path: &'a Path,
    line: usize,
    started: bool,
    finished: bool,
    depth: usize,
    in_string: bool,
    escaped: bool,
    in_value: bool,
    value: Vec<u8>,
    value_line: usize,
}

impl<'a, R: BufRead> JsonArrayStream<'a, R> {
    fn new(reader: &'a mut R, path: &'a Path) -> Self {
        Self {
            reader,
            path,
            line: 1,
            started: false,
            finished: false,
            depth: 0,
            in_string: false,
            escaped: false,
            in_value: false,
            value: Vec::new(),
            value_line: 1,
        }
    }

    /// Returns the next top-level array element, including its starting line.
    ///
    /// A syntactically invalid element is returned as bytes so the caller can
    /// record it as a malformed record and keep going. Only broken array
    /// framing (unbalanced brackets, trailing data) is fatal.
    fn next_element(&mut self) -> Result<Option<(usize, Vec<u8>)>, Taskmaster1Error> {
        loop {
            if self.finished {
                let (consumed, trailing) = {
                    let buffer = match self.reader.fill_buf() {
                        Ok(buffer) => buffer,
                        Err(source) => {
                            return Err(Taskmaster1Error::Io {
                                path: self.path.to_path_buf(),
                                source,
                            });
                        }
                    };

                    if buffer.is_empty() {
                        return Ok(None);
                    }

                    (buffer.len(), !buffer.iter().all(u8::is_ascii_whitespace))
                };

                if trailing {
                    return Err(Taskmaster1Error::MalformedArray {
                        path: self.path.to_path_buf(),
                        line: self.line,
                        message: "unexpected data after the closing `]`".to_owned(),
                    });
                }

                self.reader.consume(consumed);
                continue;
            }

            let mut consumed = 0usize;
            let mut completed = false;
            let mut failure: Option<(usize, &'static str)> = None;

            {
                let buffer = match self.reader.fill_buf() {
                    Ok(buffer) => buffer,
                    Err(source) => {
                        return Err(Taskmaster1Error::Io {
                            path: self.path.to_path_buf(),
                            source,
                        });
                    }
                };

                if buffer.is_empty() {
                    failure = Some((
                        self.line,
                        if self.started {
                            "unexpected end of file inside the JSON array"
                        } else {
                            "expected the file to start with a JSON array"
                        },
                    ));
                } else {
                    for &byte in buffer {
                        consumed += 1;

                        if byte == b'\n' {
                            self.line += 1;
                        }

                        if self.in_string {
                            self.value.push(byte);

                            if self.escaped {
                                self.escaped = false;
                            } else if byte == b'\\' {
                                self.escaped = true;
                            } else if byte == b'"' {
                                self.in_string = false;
                            }

                            continue;
                        }

                        if !self.started {
                            if byte.is_ascii_whitespace() {
                                continue;
                            }

                            if byte == b'[' {
                                self.started = true;
                                continue;
                            }

                            failure =
                                Some((self.line, "expected the file to start with a JSON array"));
                            break;
                        }

                        if self.depth == 0 {
                            match byte {
                                b'"' => {
                                    if !self.in_value {
                                        self.in_value = true;
                                        self.value_line = self.line;
                                        self.value.clear();
                                    }

                                    self.value.push(byte);
                                    self.in_string = true;
                                }
                                b'{' | b'[' => {
                                    if !self.in_value {
                                        self.in_value = true;
                                        self.value_line = self.line;
                                        self.value.clear();
                                    }

                                    self.value.push(byte);
                                    self.depth += 1;
                                }
                                b',' => {
                                    if self.in_value {
                                        completed = true;
                                        break;
                                    }
                                }
                                b']' => {
                                    self.finished = true;

                                    if self.in_value {
                                        completed = true;
                                    }

                                    break;
                                }
                                byte if byte.is_ascii_whitespace() => {
                                    if self.in_value {
                                        self.value.push(byte);
                                    }
                                }
                                _ => {
                                    if !self.in_value {
                                        self.in_value = true;
                                        self.value_line = self.line;
                                        self.value.clear();
                                    }

                                    self.value.push(byte);
                                }
                            }
                        } else {
                            self.value.push(byte);

                            match byte {
                                b'"' => self.in_string = true,
                                b'{' | b'[' => self.depth += 1,
                                b'}' | b']' => {
                                    self.depth -= 1;
                                }
                                _ => {}
                            }
                        }
                    }
                }
            }

            self.reader.consume(consumed);

            if let Some((line, message)) = failure {
                return Err(Taskmaster1Error::MalformedArray {
                    path: self.path.to_path_buf(),
                    line,
                    message: message.to_owned(),
                });
            }

            if completed {
                let line = self.value_line;
                let bytes = std::mem::take(&mut self.value);
                self.in_value = false;
                self.escaped = false;

                let trimmed = bytes.trim_ascii();

                if trimmed.is_empty() {
                    continue;
                }

                return Ok(Some((line, trimmed.to_vec())));
            }
        }
    }
}

fn budget_key(messages: usize, characters: usize) -> String {
    format!("{messages}m/{characters}c")
}

fn string_field(object: &Map<String, Value>, snake: &str, camel: &str) -> Option<String> {
    object
        .get(snake)
        .or_else(|| object.get(camel))
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_owned)
}

fn map_speaker(speaker: &str) -> Option<&'static str> {
    match speaker {
        "USER" => Some("user"),
        "ASSISTANT" => Some("assistant"),
        _ => None,
    }
}

fn domain_for(instruction_id: Option<&str>) -> String {
    let Some(instruction_id) = instruction_id else {
        return "unknown".to_owned();
    };

    for (prefix, domain) in [
        ("pizza", "pizza_ordering"),
        ("auto-repair", "auto_repair"),
        ("coffee", "coffee_ordering"),
        ("restaurant", "restaurant_reservation"),
        ("uber-lyft", "ride_service"),
        ("movie", "movie_tickets"),
    ] {
        if instruction_id.starts_with(prefix) {
            return domain.to_owned();
        }
    }

    "unknown".to_owned()
}

fn is_deleted_placeholder(text: &str) -> bool {
    let trimmed = text.trim();

    trimmed.eq_ignore_ascii_case("(deleted)")
        || trimmed.eq_ignore_ascii_case("[deleted]")
        || trimmed.eq_ignore_ascii_case("<deleted>")
}

#[derive(Clone, Copy, Debug, Default)]
struct UtterancePii {
    email: bool,
    url: bool,
    phone: bool,
    long_digits: bool,
}

impl UtterancePii {
    fn any(self) -> bool {
        self.email || self.url || self.phone || self.long_digits
    }
}

fn contains_email_like(text: &str) -> bool {
    for token in text.split_whitespace() {
        let token = token.trim_matches(|character: char| {
            matches!(
                character,
                ',' | '.' | ';' | ':' | ')' | '(' | '"' | '\'' | '!' | '?'
            )
        });

        if let Some((local, domain)) = token.split_once('@')
            && !local.is_empty()
            && domain.contains('.')
            && !domain.starts_with('.')
            && !domain.ends_with('.')
        {
            return true;
        }
    }

    false
}

fn contains_phone_like(text: &str) -> (bool, bool) {
    let mut digits = 0usize;
    let mut consecutive = 0usize;
    let mut in_run = false;
    let mut long_run = false;

    for character in text.chars() {
        if character.is_ascii_digit() {
            digits += 1;
            consecutive += 1;
            in_run = true;

            if consecutive >= 13 {
                long_run = true;
            }
        } else if in_run && matches!(character, '+' | '-' | '(' | ')' | ' ' | '.') {
            continue;
        } else {
            if digits >= 10 {
                return (true, long_run);
            }

            digits = 0;
            consecutive = 0;
            in_run = false;
        }
    }

    (digits >= 10, long_run)
}

fn scan_utterance_pii(text: &str) -> UtterancePii {
    let lower = text.to_ascii_lowercase();
    let (phone, long_digits) = contains_phone_like(text);

    UtterancePii {
        email: contains_email_like(text),
        url: lower.contains("http://") || lower.contains("https://") || lower.contains("www."),
        phone,
        long_digits,
    }
}

fn is_cjk(character: char) -> bool {
    matches!(
        character,
        '\u{3040}'..='\u{30ff}'
            | '\u{3400}'..='\u{4dbf}'
            | '\u{4e00}'..='\u{9fff}'
            | '\u{ac00}'..='\u{d7af}'
    )
}

struct ConversationScan {
    entry: Taskmaster1ManifestEntry,
    id: Option<String>,
    domain: String,
    instruction_id: Option<String>,
    speakers: BTreeMap<String, usize>,
    role_counts: BTreeMap<String, usize>,
    unmapped_speakers: BTreeMap<String, usize>,
    messages: usize,
    characters: usize,
    utterance_lengths: Vec<usize>,
    annotated: usize,
    non_ascii: usize,
    cjk: usize,
    non_object: usize,
    missing_text: usize,
    empty_text: usize,
    missing_index: usize,
    non_sequential_index: usize,
    missing_speaker: usize,
    unknown_speaker: usize,
    deleted_placeholders: usize,
    same_role_adjacencies: usize,
    adjacent_duplicates: usize,
    empty_conversation: bool,
    privacy: PiiCounts,
    payload_digest: [u8; 32],
}

fn analyze_conversation(
    object: &Map<String, Value>,
    subset: Taskmaster1Subset,
    file: &str,
    line: usize,
    revision: &str,
) -> ConversationScan {
    let id = string_field(object, "conversation_id", "conversationId");
    let instruction_id = string_field(object, "instruction_id", "instructionId");

    let mut warnings: BTreeSet<String> = BTreeSet::new();

    if id.is_none() {
        warnings.insert("missing_conversation_id".to_owned());
    }

    if instruction_id.is_none() {
        warnings.insert("missing_instruction_id".to_owned());
    }

    let utterances_value = object.get("utterances");
    let utterances = utterances_value.and_then(Value::as_array);

    if utterances.is_none() {
        warnings.insert("missing_utterances".to_owned());
    }

    let utterances = utterances.map(Vec::as_slice).unwrap_or(&[]);
    let empty_conversation = utterances_value
        .and_then(Value::as_array)
        .is_some_and(|utterances| utterances.is_empty());

    if empty_conversation {
        warnings.insert("empty_conversation".to_owned());
    }

    let mut original_speakers = Vec::with_capacity(utterances.len());
    let mut mapped_roles = Vec::with_capacity(utterances.len());
    let mut speakers: BTreeMap<String, usize> = BTreeMap::new();
    let mut role_counts: BTreeMap<String, usize> = BTreeMap::new();
    let mut unmapped_speakers: BTreeMap<String, usize> = BTreeMap::new();
    let mut utterance_lengths = Vec::with_capacity(utterances.len());
    let mut characters = 0usize;
    let mut annotated = 0usize;
    let mut non_ascii = 0usize;
    let mut cjk = 0usize;
    let mut non_object = 0usize;
    let mut missing_text = 0usize;
    let mut empty_text = 0usize;
    let mut missing_index = 0usize;
    let mut non_sequential_index = 0usize;
    let mut missing_speaker = 0usize;
    let mut unknown_speaker = 0usize;
    let mut deleted_placeholders = 0usize;
    let mut same_role_adjacencies = 0usize;
    let mut adjacent_duplicates = 0usize;
    let mut privacy = PiiCounts::default();
    let mut previous_speaker: Option<&str> = None;
    let mut previous_text: Option<&str> = None;

    for (position, utterance) in utterances.iter().enumerate() {
        let Some(utterance) = utterance.as_object() else {
            warnings.insert("non_object_utterance".to_owned());
            warnings.insert("missing_speaker".to_owned());
            non_object += 1;
            original_speakers.push("<missing>".to_owned());
            mapped_roles.push(None);
            utterance_lengths.push(0);
            *speakers.entry("<missing>".to_owned()).or_insert(0) += 1;
            *unmapped_speakers.entry("<missing>".to_owned()).or_insert(0) += 1;
            missing_speaker += 1;

            if previous_speaker == Some("<missing>") {
                same_role_adjacencies += 1;
                warnings.insert("same_role_adjacency".to_owned());
            }

            previous_speaker = Some("<missing>");
            previous_text = None;
            continue;
        };

        let raw_speaker = utterance
            .get("speaker")
            .and_then(Value::as_str)
            .unwrap_or("<missing>");

        let speaker = if raw_speaker.trim().is_empty() {
            "<missing>"
        } else {
            raw_speaker
        };

        original_speakers.push(speaker.to_owned());
        *speakers.entry(speaker.to_owned()).or_insert(0) += 1;

        let mapped = map_speaker(speaker);
        mapped_roles.push(mapped.map(str::to_owned));

        if let Some(role) = mapped {
            *role_counts.entry(role.to_owned()).or_insert(0) += 1;
        } else if speaker == "<missing>" {
            warnings.insert("missing_speaker".to_owned());
            missing_speaker += 1;
            *unmapped_speakers.entry("<missing>".to_owned()).or_insert(0) += 1;
        } else {
            warnings.insert("unknown_speaker".to_owned());
            unknown_speaker += 1;
            *unmapped_speakers.entry(speaker.to_owned()).or_insert(0) += 1;
        }

        match utterance.get("index") {
            Some(value) => match value.as_u64() {
                Some(index) if index == position as u64 => {}
                Some(_) => {
                    warnings.insert("non_sequential_index".to_owned());
                    non_sequential_index += 1;
                }
                None => {
                    warnings.insert("missing_index".to_owned());
                    missing_index += 1;
                }
            },
            None => {
                warnings.insert("missing_index".to_owned());
                missing_index += 1;
            }
        }

        let text = utterance.get("text");

        let text = match text {
            Some(value) => match value.as_str() {
                Some(text) => Some(text),
                None => {
                    warnings.insert("missing_text".to_owned());
                    missing_text += 1;
                    None
                }
            },
            None => {
                warnings.insert("missing_text".to_owned());
                missing_text += 1;
                None
            }
        };

        if text.is_some_and(|text| text.trim().is_empty()) {
            warnings.insert("empty_text".to_owned());
            empty_text += 1;
        }

        if utterance
            .get("segments")
            .and_then(Value::as_array)
            .is_some_and(|segments| !segments.is_empty())
        {
            annotated += 1;
        }

        if let Some(text) = text {
            let length = text.chars().count();
            characters += length;
            utterance_lengths.push(length);
            non_ascii += text
                .chars()
                .filter(|character| !character.is_ascii())
                .count();
            cjk += text.chars().filter(|&character| is_cjk(character)).count();

            if is_deleted_placeholder(text) {
                warnings.insert("deleted_placeholder".to_owned());
                deleted_placeholders += 1;
            }

            let pii = scan_utterance_pii(text);

            if pii.email {
                privacy.email_like_utterances += 1;
            }

            if pii.url {
                privacy.url_like_utterances += 1;
            }

            if pii.phone {
                privacy.phone_like_utterances += 1;
            }

            if pii.long_digits {
                privacy.long_digit_run_utterances += 1;
            }

            if pii.any() {
                privacy.utterances_affected += 1;
            }
        } else {
            utterance_lengths.push(0);
        }

        if let Some(previous) = previous_speaker
            && previous == speaker
        {
            warnings.insert("same_role_adjacency".to_owned());
            same_role_adjacencies += 1;

            if text.is_some_and(|text| !text.is_empty()) && previous_text == text {
                warnings.insert("adjacent_duplicate_utterance".to_owned());
                adjacent_duplicates += 1;
            }
        }

        previous_speaker = Some(speaker);
        previous_text = text;
    }

    if privacy.utterances_affected > 0 {
        privacy.conversations_affected = 1;
        warnings.insert("potential_contact_data".to_owned());
    }

    let invalid =
        id.is_none() || utterances_value.and_then(Value::as_array).is_none() || empty_conversation;

    let status = if invalid {
        STATUS_INVALID
    } else if warnings.is_empty() {
        STATUS_VALID
    } else {
        STATUS_FLAGGED
    };

    let domain = domain_for(instruction_id.as_deref());

    let payload = utterances
        .iter()
        .map(|utterance| {
            let utterance = utterance.as_object();
            let speaker = utterance
                .and_then(|utterance| utterance.get("speaker"))
                .and_then(Value::as_str)
                .unwrap_or("");
            let text = utterance
                .and_then(|utterance| utterance.get("text"))
                .and_then(Value::as_str)
                .unwrap_or("");

            (speaker, text)
        })
        .collect::<Vec<_>>();

    let payload_bytes = serde_json::to_vec(&payload).unwrap_or_default();
    let payload_digest: [u8; 32] = Sha256::digest(&payload_bytes).into();

    let entry = Taskmaster1ManifestEntry {
        source: TASKMASTER1_SOURCE_DATASET.to_owned(),
        source_revision: revision.to_owned(),
        subset: subset.as_str().to_owned(),
        original_file: file.to_owned(),
        original_line: line,
        original_conversation_id: id.clone(),
        original_instruction_id: instruction_id.clone(),
        domain: domain.clone(),
        original_speakers,
        mapped_roles,
        message_count: utterances.len(),
        character_count: characters,
        status: status.to_owned(),
        warnings: warnings.into_iter().collect(),
        subset_qualification: String::new(),
    };

    ConversationScan {
        entry,
        id,
        domain,
        instruction_id,
        speakers,
        role_counts,
        unmapped_speakers,
        messages: utterances.len(),
        characters,
        utterance_lengths,
        annotated,
        non_ascii,
        cjk,
        non_object,
        missing_text,
        empty_text,
        missing_index,
        non_sequential_index,
        missing_speaker,
        unknown_speaker,
        deleted_placeholders,
        same_role_adjacencies,
        adjacent_duplicates,
        empty_conversation,
        privacy,
        payload_digest,
    }
}

struct SubsetScan {
    report: SubsetReport,
    ids: BTreeSet<String>,
}

fn inspect_subset(
    path: &Path,
    subset: Taskmaster1Subset,
    config: &Taskmaster1InspectConfig,
    manifest: &mut Vec<Taskmaster1ManifestEntry>,
    id_index: &mut BTreeMap<String, Vec<usize>>,
    payload_index: &mut BTreeMap<[u8; 32], Vec<usize>>,
) -> Result<SubsetScan, Taskmaster1Error> {
    let file = File::open(path).map_err(|source| Taskmaster1Error::Io {
        path: path.to_path_buf(),
        source,
    })?;

    let mut reader = BufReader::with_capacity(256 * 1024, file);
    let mut stream = JsonArrayStream::new(&mut reader, path);

    let mut report = SubsetReport::new(subset);
    let mut ids = BTreeSet::new();
    let mut conversation_messages = Vec::new();
    let mut conversation_characters = Vec::new();
    let mut utterance_characters = Vec::new();

    while let Some((line, bytes)) = stream.next_element()? {
        report.records_read += 1;

        let value: Value = match serde_json::from_slice(&bytes) {
            Ok(value) => value,
            Err(error) => {
                report.malformed_records += 1;
                *report
                    .issues
                    .entry("malformed_record".to_owned())
                    .or_insert(0) += 1;
                manifest.push(malformed_entry(
                    subset,
                    line,
                    &config.revision,
                    format!("record is not valid JSON: {error}"),
                ));
                continue;
            }
        };

        let Value::Object(object) = value else {
            report.malformed_records += 1;
            *report
                .issues
                .entry("malformed_record".to_owned())
                .or_insert(0) += 1;
            manifest.push(malformed_entry(
                subset,
                line,
                &config.revision,
                "record is not a JSON object".to_owned(),
            ));
            continue;
        };

        let scan =
            analyze_conversation(&object, subset, subset.file_name(), line, &config.revision);

        report.conversations_read += 1;

        match scan.entry.status.as_str() {
            STATUS_VALID => report.conversations_valid += 1,
            STATUS_FLAGGED => report.conversations_flagged += 1,
            STATUS_INVALID => report.conversations_invalid += 1,
            _ => {}
        }

        report.messages += scan.messages;
        report.characters += scan.characters;
        report.annotated_utterances += scan.annotated;
        report.non_ascii_characters += scan.non_ascii;
        report.cjk_characters += scan.cjk;
        report.non_object_utterances += scan.non_object;
        report.missing_text_utterances += scan.missing_text;
        report.empty_text_utterances += scan.empty_text;
        report.missing_index_utterances += scan.missing_index;
        report.non_sequential_index_utterances += scan.non_sequential_index;
        report.missing_speaker_utterances += scan.missing_speaker;
        report.unknown_speaker_utterances += scan.unknown_speaker;
        report.deleted_placeholders += scan.deleted_placeholders;
        report.same_role_adjacencies += scan.same_role_adjacencies;
        report.adjacent_duplicate_utterances += scan.adjacent_duplicates;
        report.empty_conversations += usize::from(scan.empty_conversation);
        report.privacy.add(&scan.privacy);

        for (label, count) in &scan.speakers {
            *report.speakers.entry(label.clone()).or_insert(0) += count;
        }

        for (label, count) in &scan.role_counts {
            *report.mapped_roles.entry(label.clone()).or_insert(0) += count;
        }

        for (label, count) in &scan.unmapped_speakers {
            *report.unmapped_speakers.entry(label.clone()).or_insert(0) += count;
        }

        *report.domains.entry(scan.domain.clone()).or_insert(0) += 1;

        if let Some(instruction_id) = &scan.instruction_id {
            *report
                .instruction_ids
                .entry(instruction_id.clone())
                .or_insert(0) += 1;
        }

        for warning in &scan.entry.warnings {
            *report.issues.entry(warning.clone()).or_insert(0) += 1;
        }

        for (messages, characters) in LENGTH_BUDGETS {
            if scan.messages >= MIN_MESSAGES
                && scan.messages <= messages
                && scan.characters <= characters
            {
                *report
                    .budget_fit
                    .entry(budget_key(messages, characters))
                    .or_insert(0) += 1;
            }
        }

        conversation_messages.push(scan.messages);
        conversation_characters.push(scan.characters);
        utterance_characters.extend(scan.utterance_lengths.iter().copied());

        let manifest_index = manifest.len();
        let conversation_id = scan.id.clone();
        manifest.push(scan.entry);

        payload_index
            .entry(scan.payload_digest)
            .or_default()
            .push(manifest_index);

        if let Some(conversation_id) = conversation_id {
            id_index
                .entry(conversation_id.clone())
                .or_default()
                .push(manifest_index);
            ids.insert(conversation_id);
        }
    }

    report.conversation_messages = LengthStats::from_values(&conversation_messages);
    report.conversation_characters = LengthStats::from_values(&conversation_characters);
    report.utterance_characters = LengthStats::from_values(&utterance_characters);

    Ok(SubsetScan { report, ids })
}

fn malformed_entry(
    subset: Taskmaster1Subset,
    line: usize,
    revision: &str,
    warning: String,
) -> Taskmaster1ManifestEntry {
    Taskmaster1ManifestEntry {
        source: TASKMASTER1_SOURCE_DATASET.to_owned(),
        source_revision: revision.to_owned(),
        subset: subset.as_str().to_owned(),
        original_file: subset.file_name().to_owned(),
        original_line: line,
        original_conversation_id: None,
        original_instruction_id: None,
        domain: "unknown".to_owned(),
        original_speakers: Vec::new(),
        mapped_roles: Vec::new(),
        message_count: 0,
        character_count: 0,
        status: STATUS_MALFORMED.to_owned(),
        warnings: vec![warning],
        subset_qualification: String::new(),
    }
}

fn apply_duplicates(
    manifest: &mut [Taskmaster1ManifestEntry],
    id_index: &BTreeMap<String, Vec<usize>>,
    payload_index: &BTreeMap<[u8; 32], Vec<usize>>,
    subsets: &mut BTreeMap<String, SubsetReport>,
) -> (Vec<DuplicateIdGroup>, Vec<DuplicatePayloadGroup>) {
    let mut id_groups = Vec::new();
    let mut payload_groups = Vec::new();

    for (conversation_id, indices) in id_index {
        if indices.len() < 2 {
            continue;
        }

        let mut locations = Vec::with_capacity(indices.len());

        for &index in indices {
            manifest[index].flag("duplicate_conversation_id");
            let entry = &manifest[index];

            *subsets
                .get_mut(&entry.subset)
                .expect("manifest subset must have a report")
                .issues
                .entry("duplicate_conversation_id".to_owned())
                .or_insert(0) += 1;

            locations.push(DuplicateLocation {
                subset: entry.subset.clone(),
                file: entry.original_file.clone(),
                line: entry.original_line,
                conversation_id: entry.original_conversation_id.clone(),
            });
        }

        id_groups.push(DuplicateIdGroup {
            conversation_id: conversation_id.clone(),
            occurrences: indices.len(),
            locations,
        });
    }

    for (digest, indices) in payload_index {
        if indices.len() < 2 {
            continue;
        }

        let mut locations = Vec::with_capacity(indices.len());

        for &index in indices {
            manifest[index].flag("duplicate_payload");
            let entry = &manifest[index];

            *subsets
                .get_mut(&entry.subset)
                .expect("manifest subset must have a report")
                .issues
                .entry("duplicate_payload".to_owned())
                .or_insert(0) += 1;

            locations.push(DuplicateLocation {
                subset: entry.subset.clone(),
                file: entry.original_file.clone(),
                line: entry.original_line,
                conversation_id: entry.original_conversation_id.clone(),
            });
        }

        payload_groups.push(DuplicatePayloadGroup {
            payload_sha256: hex_digest(digest),
            occurrences: indices.len(),
            locations,
        });
    }

    (id_groups, payload_groups)
}

fn hex_digest(digest: &[u8; 32]) -> String {
    let mut output = String::with_capacity(64);

    for byte in digest {
        let _ = write!(output, "{byte:02x}");
    }

    output
}

fn qualify(report: &SubsetReport, license_evidence_present: bool, revision: &str) -> Qualification {
    let mut reasons = vec![
        format!(
            "License: {TASKMASTER1_LICENSE}, stated in the TM-1-2019 README copyright notice; the upstream repository has no separate LICENSE file."
        ),
        format!("Pinned upstream revision: {revision}."),
    ];

    if !license_evidence_present {
        reasons.push(
            "The supplied directory does not contain README.md, so the license notice could not be re-read from the inspected copy."
                .to_owned(),
        );
    }

    if report.conversations_read == 0 {
        reasons.push("No conversation records were read from this file.".to_owned());

        return Qualification {
            status: STATUS_REJECTED.to_owned(),
            reasons,
        };
    }

    let usable = report.conversations_valid + report.conversations_flagged;
    let problem_records =
        report.conversations_flagged + report.conversations_invalid + report.malformed_records;
    let unmapped = report.missing_speaker_utterances + report.unknown_speaker_utterances;

    if usable == 0 {
        reasons.push(
            "Every parsed conversation is structurally invalid; no conversation can be mapped to user/assistant roles."
                .to_owned(),
        );

        return Qualification {
            status: STATUS_REJECTED.to_owned(),
            reasons,
        };
    }

    if problem_records > 0 {
        reasons.push(format!(
            "{problem_records} of {} records are flagged, invalid, or malformed; see the structural findings.",
            report.records_read
        ));
    }

    if report.privacy.conversations_affected > 0 {
        reasons.push(format!(
            "{} conversations contain contact-like patterns (email, URL, phone-like digit runs); no consent, anonymization, or PII-scrubbing statement was found upstream.",
            report.privacy.conversations_affected
        ));
    }

    if let Some(count) = report.issues.get("duplicate_conversation_id") {
        reasons.push(format!(
            "{count} records repeat a conversation identifier; the identifier is not a unique key in this subset."
        ));
    }

    if let Some(count) = report.issues.get("duplicate_payload") {
        reasons.push(format!(
            "{count} records share an identical ordered speaker/text payload, ignoring annotations and task identifiers; a duplicate policy is required."
        ));
    }

    if report.deleted_placeholders > 0 {
        reasons.push(format!(
            "{} `(deleted)` placeholder utterances remain in the source; a documented repair or exclusion rule is required.",
            report.deleted_placeholders
        ));
    }

    if unmapped > 0 {
        reasons.push(format!(
            "{unmapped} utterances use missing or non-USER/ASSISTANT speaker labels."
        ));
    }

    if report
        .budget_fit
        .get(&budget_key(8, 500))
        .copied()
        .unwrap_or(0)
        == 0
    {
        reasons.push(
            "No conversation fits the current 8-message / 500-character training budget; the source is only usable at a larger sequence budget."
                .to_owned(),
        );
    }

    let conditional = problem_records > 0
        || report.privacy.conversations_affected > 0
        || !license_evidence_present;

    if conditional {
        reasons.push(
            "Conditional decision: the subset is potentially useful, but the listed issues need explicit repair/exclusion and privacy decisions before training."
                .to_owned(),
        );

        Qualification {
            status: STATUS_CONDITIONAL.to_owned(),
            reasons,
        }
    } else {
        reasons.push(
            "Automated inspection found no structural, role-mapping, duplicate, or contact-pattern findings; final project approval is still required before training."
                .to_owned(),
        );

        Qualification {
            status: STATUS_ACCEPTED.to_owned(),
            reasons,
        }
    }
}

fn resolve_dataset_dir(input: &Path) -> Result<PathBuf, Taskmaster1Error> {
    if !input.exists() {
        return Err(Taskmaster1Error::UnsupportedInput {
            path: input.to_path_buf(),
            message: "path does not exist".to_owned(),
        });
    }

    if input.is_file() {
        let name = input
            .file_name()
            .map(|name| name.to_string_lossy().to_ascii_lowercase())
            .unwrap_or_default();

        let archive = name.ends_with(".zip")
            || name.ends_with(".tar")
            || name.ends_with(".tar.gz")
            || name.ends_with(".tgz");

        return Err(Taskmaster1Error::UnsupportedInput {
            path: input.to_path_buf(),
            message: if archive {
                "archives are not read directly; extract the official files and pass the extracted directory"
                    .to_owned()
            } else {
                "expected the extracted TM-1-2019 directory".to_owned()
            },
        });
    }

    if input.is_dir() {
        let direct = input.to_path_buf();

        if direct.join("self-dialogs.json").is_file() {
            return Ok(direct);
        }

        let nested = input.join(TASKMASTER1_SUBSET_DIRECTORY);

        if nested.join("self-dialogs.json").is_file() {
            return Ok(nested);
        }
    }

    Err(Taskmaster1Error::UnsupportedInput {
        path: input.to_path_buf(),
        message: format!(
            "expected self-dialogs.json and woz-dialogs.json in the directory or its {TASKMASTER1_SUBSET_DIRECTORY} subdirectory"
        ),
    })
}

fn sha256_file(path: &Path) -> Result<String, Taskmaster1Error> {
    let mut file = File::open(path).map_err(|source| Taskmaster1Error::Io {
        path: path.to_path_buf(),
        source,
    })?;

    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 64 * 1024];

    loop {
        let read = file
            .read(&mut buffer)
            .map_err(|source| Taskmaster1Error::Io {
                path: path.to_path_buf(),
                source,
            })?;

        if read == 0 {
            break;
        }

        hasher.update(&buffer[..read]);
    }

    let digest: [u8; 32] = hasher.finalize().into();
    Ok(hex_digest(&digest))
}

fn upstream_url(revision: &str, relative: &str) -> String {
    format!(
        "https://raw.githubusercontent.com/google-research-datasets/Taskmaster/{revision}/{TASKMASTER1_SUBSET_DIRECTORY}/{relative}"
    )
}

fn inventory_files(
    dataset_dir: &Path,
    revision: &str,
) -> Result<Vec<FileInventory>, Taskmaster1Error> {
    let mut entries = Vec::new();

    let known = [
        ("README.md", "license_evidence", None),
        ("sample.json", "sample", None),
        ("ontology.json", "ontology", None),
        (
            "self-dialogs.json",
            "dialog_subset",
            Some(Taskmaster1Subset::SelfDialogs),
        ),
        (
            "woz-dialogs.json",
            "dialog_subset",
            Some(Taskmaster1Subset::WozDialogs),
        ),
    ];

    for &(relative, kind, subset) in &known {
        let path = dataset_dir.join(relative);

        if path.is_file() {
            entries.push(file_inventory(
                relative,
                kind,
                subset,
                &path,
                Some(upstream_url(revision, relative)),
            )?);
        }
    }

    for split in ["train", "dev", "test"] {
        let relative = format!("train-dev-test/{split}.csv");
        let path = dataset_dir.join(&relative);

        if path.is_file() {
            entries.push(file_inventory(
                &relative,
                "split_list",
                None,
                &path,
                Some(upstream_url(revision, &relative)),
            )?);
        }
    }

    let instructions = dataset_dir.join("instructions");

    if instructions.is_dir() {
        let mut names = fs::read_dir(&instructions)
            .map_err(|source| Taskmaster1Error::Io {
                path: instructions.clone(),
                source,
            })?
            .filter_map(Result::ok)
            .filter(|entry| entry.path().is_file())
            .map(|entry| entry.file_name())
            .collect::<Vec<_>>();

        names.sort();

        for name in names {
            let relative = format!("instructions/{}", name.to_string_lossy());
            let path = instructions.join(&name);
            entries.push(file_inventory(
                &relative,
                "instruction",
                None,
                &path,
                Some(upstream_url(revision, &relative)),
            )?);
        }
    }

    let mut known_names = BTreeSet::new();

    for (name, _, _) in &known {
        known_names.insert((*name).to_owned());
    }

    let root_entries = fs::read_dir(dataset_dir).map_err(|source| Taskmaster1Error::Io {
        path: dataset_dir.to_path_buf(),
        source,
    })?;

    for entry in root_entries.filter_map(Result::ok) {
        let path = entry.path();

        if !path.is_file() {
            continue;
        }

        let name = entry.file_name().to_string_lossy().into_owned();

        if known_names.contains(&name) {
            continue;
        }

        entries.push(file_inventory(&name, "unrecognized", None, &path, None)?);
    }

    entries.sort_by(|left, right| left.path.cmp(&right.path));

    Ok(entries)
}

fn file_inventory(
    relative: &str,
    kind: &str,
    subset: Option<Taskmaster1Subset>,
    path: &Path,
    source_url: Option<String>,
) -> Result<FileInventory, Taskmaster1Error> {
    let metadata = fs::metadata(path).map_err(|source| Taskmaster1Error::Io {
        path: path.to_path_buf(),
        source,
    })?;

    Ok(FileInventory {
        path: relative.to_owned(),
        kind: kind.to_owned(),
        subset: subset.map(|subset| subset.as_str().to_owned()),
        bytes: metadata.len(),
        sha256: sha256_file(path)?,
        source_url,
    })
}

fn inspect_splits(
    dataset_dir: &Path,
    self_ids: &BTreeSet<String>,
) -> Result<Option<SplitCoverage>, Taskmaster1Error> {
    let directory = dataset_dir.join("train-dev-test");

    if !directory.is_dir() {
        return Ok(None);
    }

    let mut splits = BTreeMap::new();
    let mut id_to_splits: BTreeMap<String, BTreeSet<&'static str>> = BTreeMap::new();
    let mut present_any = false;

    for (name, filename) in [
        ("train", "train.csv"),
        ("dev", "dev.csv"),
        ("test", "test.csv"),
    ] {
        let path = directory.join(filename);

        if !path.is_file() {
            splits.insert(
                name.to_owned(),
                SplitStats {
                    present: false,
                    rows: 0,
                    unique_ids: 0,
                    duplicate_ids: 0,
                    malformed_rows: 0,
                },
            );
            continue;
        }

        present_any = true;

        let file = File::open(&path).map_err(|source| Taskmaster1Error::Io {
            path: path.clone(),
            source,
        })?;

        let mut rows = 0usize;
        let mut malformed_rows = 0usize;
        let mut unique_ids = BTreeSet::new();
        let mut first_content_row = true;

        for line in BufReader::new(file).lines() {
            let line = line.map_err(|source| Taskmaster1Error::Io {
                path: path.clone(),
                source,
            })?;

            let first = line.split(',').next().unwrap_or("").trim();

            if first.is_empty() {
                if !line.trim().is_empty() {
                    malformed_rows += 1;
                }

                continue;
            }

            if first_content_row {
                first_content_row = false;

                if first.eq_ignore_ascii_case("conversation_id") {
                    continue;
                }
            }

            rows += 1;
            unique_ids.insert(first.to_owned());
            id_to_splits
                .entry(first.to_owned())
                .or_default()
                .insert(name);
        }

        splits.insert(
            name.to_owned(),
            SplitStats {
                present: true,
                rows,
                unique_ids: unique_ids.len(),
                duplicate_ids: rows.saturating_sub(unique_ids.len()),
                malformed_rows,
            },
        );
    }

    if !present_any {
        return Ok(None);
    }

    let ids_in_splits: BTreeSet<String> = id_to_splits.keys().cloned().collect();
    let covered_self_ids = self_ids.intersection(&ids_in_splits).count();

    let mut overlaps: BTreeMap<String, usize> = BTreeMap::new();

    for split_names in id_to_splits.values() {
        if split_names.len() < 2 {
            continue;
        }

        let key = split_names.iter().copied().collect::<Vec<_>>().join("/");

        *overlaps.entry(key).or_insert(0) += 1;
    }

    Ok(Some(SplitCoverage {
        directory: "train-dev-test".to_owned(),
        splits,
        self_dialog_ids: self_ids.len(),
        ids_in_splits: ids_in_splits.len(),
        covered_self_ids,
        missing_from_splits: self_ids.difference(&ids_in_splits).count(),
        extra_in_splits: ids_in_splits.difference(self_ids).count(),
        overlaps,
    }))
}

fn recount_statuses(
    manifest: &[Taskmaster1ManifestEntry],
    subsets: &mut BTreeMap<String, SubsetReport>,
) {
    for subset in subsets.values_mut() {
        subset.conversations_valid = 0;
        subset.conversations_flagged = 0;
        subset.conversations_invalid = 0;
    }

    for entry in manifest {
        let Some(subset) = subsets.get_mut(&entry.subset) else {
            continue;
        };

        match entry.status.as_str() {
            STATUS_VALID => subset.conversations_valid += 1,
            STATUS_FLAGGED => subset.conversations_flagged += 1,
            STATUS_INVALID => subset.conversations_invalid += 1,
            _ => {}
        }
    }
}

fn totals_for(subsets: &BTreeMap<String, SubsetReport>) -> Totals {
    let mut totals = Totals::default();

    for subset in subsets.values() {
        totals.records_read += subset.records_read;
        totals.conversations_read += subset.conversations_read;
        totals.conversations_valid += subset.conversations_valid;
        totals.conversations_flagged += subset.conversations_flagged;
        totals.conversations_invalid += subset.conversations_invalid;
        totals.malformed_records += subset.malformed_records;
        totals.messages += subset.messages;
        totals.characters += subset.characters;
    }

    totals
}

fn dataset_identity(revision: &str, license_evidence_present: bool) -> DatasetIdentity {
    DatasetIdentity {
        name: "Taskmaster-1".to_owned(),
        publisher: "Google LLC".to_owned(),
        upstream_repository: TASKMASTER1_REPOSITORY_URL.to_owned(),
        upstream_revision: revision.to_owned(),
        subset_directory: TASKMASTER1_SUBSET_DIRECTORY.to_owned(),
        paper_url: TASKMASTER1_PAPER_URL.to_owned(),
        description:
            "13,215 task-oriented dialogs: 7,708 written self-dialogs and 5,507 spoken Wizard-of-Oz dialogs across six domains."
                .to_owned(),
        license: TASKMASTER1_LICENSE.to_owned(),
        license_url: TASKMASTER1_LICENSE_URL.to_owned(),
        license_evidence_file: "README.md".to_owned(),
        license_evidence_present,
        license_findings: LicenseFindings {
            repository_license_metadata:
                "No separate LICENSE file or repository-level license metadata was present at the pinned revision."
                    .to_owned(),
            data_notice:
                "The TM-1-2019 README begins with a COPYRIGHT NOTICE applying the Creative Commons Attribution 4.0 License to the dialog data."
                    .to_owned(),
            attribution:
                "Credit Bill Byrne, Karthik Krishnamoorthi, Chinnadhurai Sankar, Arvind Neelakantan, Amit Dubey, Kyu-Young Kim, and Andy Cedilnik (Google LLC), and cite the Taskmaster-1 EMNLP 2019 paper."
                    .to_owned(),
            redistribution:
                "CC BY 4.0 permits sharing and redistribution with attribution. This is a summary of the published terms, not legal advice."
                    .to_owned(),
            modifications:
                "CC BY 4.0 permits adaptation when the adaptation is indicated and attribution is preserved; adaptations must not imply endorsement by the licensor."
                    .to_owned(),
            training_use:
                "No additional training-use restriction was found in the official README or paper; the notice applies to the dialog data, not only to source code."
                    .to_owned(),
            caveats: vec![
                "The license notice is in the dataset README, not in a separate LICENSE file or repository metadata."
                    .to_owned(),
                "Embedded third-party content cannot be ruled out from the published files."
                    .to_owned(),
                "No consent, anonymization, or PII-scrubbing statement was found in the README or the Taskmaster-1 paper."
                    .to_owned(),
            ],
        },
    }
}

fn privacy_notes() -> Vec<String> {
    vec![
        "Pattern scans are heuristics; they do not prove the presence or absence of personal data."
            .to_owned(),
        "No consent, anonymization, or PII-scrubbing statement was found in the official README or the Taskmaster-1 paper."
            .to_owned(),
        "The WOZ subset contains upstream `(deleted)` placeholders and blank speaker labels, which indicate redaction or transcription gaps rather than a project repair."
            .to_owned(),
        "Reports and manifests contain counts and identifiers only; conversation text is never written."
            .to_owned(),
    ]
}

fn unresolved_questions() -> Vec<String> {
    vec![
        "Are contact-like strings actual personal data, and does the project require redaction before training?"
            .to_owned(),
        "What repair policy should apply to WOZ blank speakers, duplicate identifiers, same-role adjacency, and deleted placeholders?"
            .to_owned(),
        "Should the sequence budget grow to admit Taskmaster-1 conversations, or should only the few short conversations be used?"
            .to_owned(),
        "What attribution text and placement will satisfy CC BY 4.0 in model cards and redistributed artifacts?"
            .to_owned(),
        "Can third-party content embedded in the published dialogs be ruled out?"
            .to_owned(),
        "Who gives final training approval, and on what evidence?"
            .to_owned(),
    ]
}

/// Inspects one extracted Taskmaster-1 directory and writes deterministic
/// reports plus a candidate manifest.
///
/// The output directory may exist, but none of the three output files may
/// already exist. Source files are opened read-only.
pub fn inspect_taskmaster1(
    input: impl AsRef<Path>,
    output_dir: impl AsRef<Path>,
    config: &Taskmaster1InspectConfig,
) -> Result<Taskmaster1InspectResult, Taskmaster1Error> {
    if config.revision.trim().is_empty() {
        return Err(Taskmaster1Error::InvalidConfig {
            message: "--revision must not be empty".to_owned(),
        });
    }

    let input = input.as_ref();
    let output_dir = output_dir.as_ref();
    let dataset_dir = resolve_dataset_dir(input)?;

    for subset in Taskmaster1Subset::all() {
        let path = dataset_dir.join(subset.file_name());

        if !path.is_file() {
            return Err(Taskmaster1Error::UnsupportedInput {
                path: path.clone(),
                message: "required Taskmaster-1 file is missing".to_owned(),
            });
        }
    }

    let markdown_path = output_dir.join(REPORT_MARKDOWN_FILE);
    let json_path = output_dir.join(REPORT_JSON_FILE);
    let manifest_path = output_dir.join(MANIFEST_FILE);

    for path in [&markdown_path, &json_path, &manifest_path] {
        if path.exists() {
            return Err(Taskmaster1Error::OutputExists { path: path.clone() });
        }
    }

    fs::create_dir_all(output_dir).map_err(|source| Taskmaster1Error::Io {
        path: output_dir.to_path_buf(),
        source,
    })?;

    let files = inventory_files(&dataset_dir, &config.revision)?;
    let license_evidence_present = files.iter().any(|file| file.kind == "license_evidence");

    let mut manifest = Vec::new();
    let mut id_index: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    let mut payload_index: BTreeMap<[u8; 32], Vec<usize>> = BTreeMap::new();
    let mut subsets: BTreeMap<String, SubsetReport> = BTreeMap::new();
    let mut self_ids = BTreeSet::new();

    for subset in Taskmaster1Subset::all() {
        let path = dataset_dir.join(subset.file_name());

        let scan = inspect_subset(
            &path,
            subset,
            config,
            &mut manifest,
            &mut id_index,
            &mut payload_index,
        )?;

        if subset == Taskmaster1Subset::SelfDialogs {
            self_ids = scan.ids;
        }

        subsets.insert(subset.as_str().to_owned(), scan.report);
    }

    let split_coverage = inspect_splits(&dataset_dir, &self_ids)?;
    let (duplicate_conversation_ids, duplicate_payloads) =
        apply_duplicates(&mut manifest, &id_index, &payload_index, &mut subsets);

    recount_statuses(&manifest, &mut subsets);

    for subset in subsets.values_mut() {
        subset.qualification = qualify(subset, license_evidence_present, &config.revision);
    }

    for entry in &mut manifest {
        if let Some(subset) = subsets.get(&entry.subset) {
            entry.subset_qualification = subset.qualification.status.clone();
        }
    }

    let report = Taskmaster1Report {
        tool: "mikuru-lm inspect-taskmaster1".to_owned(),
        format_version: 1,
        dataset: dataset_identity(&config.revision, license_evidence_present),
        files,
        totals: totals_for(&subsets),
        subsets,
        duplicate_conversation_ids,
        duplicate_payloads,
        split_coverage,
        privacy_notes: privacy_notes(),
        unresolved_questions: unresolved_questions(),
    };

    write_artifacts(
        output_dir,
        &markdown_path,
        &json_path,
        &manifest_path,
        &report,
        &manifest,
    )?;

    Ok(Taskmaster1InspectResult {
        output_dir: output_dir.to_path_buf(),
        markdown_path,
        json_path,
        manifest_path,
        report,
    })
}

fn write_artifacts(
    output_dir: &Path,
    markdown_path: &Path,
    json_path: &Path,
    manifest_path: &Path,
    report: &Taskmaster1Report,
    manifest: &[Taskmaster1ManifestEntry],
) -> Result<(), Taskmaster1Error> {
    let markdown = render_markdown(report);
    let json =
        serde_json::to_string_pretty(report).map_err(|error| Taskmaster1Error::Serialize {
            message: error.to_string(),
        })? + "\n";

    let temporary_markdown = output_dir.join(format!(".{REPORT_MARKDOWN_FILE}.tmp"));
    let temporary_json = output_dir.join(format!(".{REPORT_JSON_FILE}.tmp"));
    let temporary_manifest = output_dir.join(format!(".{MANIFEST_FILE}.tmp"));

    for path in [&temporary_markdown, &temporary_json, &temporary_manifest] {
        if path.exists() {
            return Err(Taskmaster1Error::OutputExists { path: path.clone() });
        }
    }

    let result = (|| {
        write_new(&temporary_markdown, markdown.as_bytes())?;
        write_new(&temporary_json, json.as_bytes())?;
        write_manifest(&temporary_manifest, manifest)?;

        fs::rename(&temporary_markdown, markdown_path).map_err(|source| Taskmaster1Error::Io {
            path: markdown_path.to_path_buf(),
            source,
        })?;
        fs::rename(&temporary_json, json_path).map_err(|source| Taskmaster1Error::Io {
            path: json_path.to_path_buf(),
            source,
        })?;
        fs::rename(&temporary_manifest, manifest_path).map_err(|source| Taskmaster1Error::Io {
            path: manifest_path.to_path_buf(),
            source,
        })?;

        Ok(())
    })();

    if result.is_err() {
        let _ = fs::remove_file(&temporary_markdown);
        let _ = fs::remove_file(&temporary_json);
        let _ = fs::remove_file(&temporary_manifest);
    }

    result
}

fn write_new(path: &Path, contents: &[u8]) -> Result<(), Taskmaster1Error> {
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|source| Taskmaster1Error::Io {
            path: path.to_path_buf(),
            source,
        })?;

    file.write_all(contents)
        .map_err(|source| Taskmaster1Error::Io {
            path: path.to_path_buf(),
            source,
        })
}

fn write_manifest(
    path: &Path,
    manifest: &[Taskmaster1ManifestEntry],
) -> Result<(), Taskmaster1Error> {
    let file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|source| Taskmaster1Error::Io {
            path: path.to_path_buf(),
            source,
        })?;

    let mut writer = BufWriter::new(file);

    for entry in manifest {
        serde_json::to_writer(&mut writer, entry).map_err(|error| Taskmaster1Error::Serialize {
            message: error.to_string(),
        })?;

        writer
            .write_all(b"\n")
            .map_err(|source| Taskmaster1Error::Io {
                path: path.to_path_buf(),
                source,
            })?;
    }

    writer.flush().map_err(|source| Taskmaster1Error::Io {
        path: path.to_path_buf(),
        source,
    })
}

fn render_markdown(report: &Taskmaster1Report) -> String {
    let mut output = String::new();

    let _ = writeln!(output, "# Taskmaster-1 inspection report\n");
    let _ = writeln!(
        output,
        "- Tool: `{}` (`mikuru-core` 0.1.0)\n- Report format: {}\n- Deterministic: no timestamps, host paths, or conversation text are recorded.\n",
        report.tool, report.format_version
    );

    output.push_str("## Dataset identity\n\n");
    output.push_str("| Field | Value |\n|---|---|\n");
    let _ = writeln!(output, "| Name | {} |", report.dataset.name);
    let _ = writeln!(output, "| Publisher | {} |", report.dataset.publisher);
    let _ = writeln!(
        output,
        "| Upstream repository | {} |",
        report.dataset.upstream_repository
    );
    let _ = writeln!(
        output,
        "| Upstream revision | `{}` |",
        report.dataset.upstream_revision
    );
    let _ = writeln!(
        output,
        "| Subset directory | `{}` |",
        report.dataset.subset_directory
    );
    let _ = writeln!(output, "| Paper | {} |", report.dataset.paper_url);
    let _ = writeln!(output, "| License | {} |", report.dataset.license);
    let _ = writeln!(
        output,
        "| License evidence | `{}` (present: {}) |",
        report.dataset.license_evidence_file,
        if report.dataset.license_evidence_present {
            "yes"
        } else {
            "no"
        }
    );
    let _ = writeln!(output, "\n{}\n", report.dataset.description);

    output.push_str("## License findings\n\n");
    let findings = &report.dataset.license_findings;
    let _ = writeln!(
        output,
        "- **Repository metadata:** {}",
        findings.repository_license_metadata
    );
    let _ = writeln!(output, "- **Data notice:** {}", findings.data_notice);
    let _ = writeln!(output, "- **Attribution:** {}", findings.attribution);
    let _ = writeln!(output, "- **Redistribution:** {}", findings.redistribution);
    let _ = writeln!(output, "- **Modifications:** {}", findings.modifications);
    let _ = writeln!(output, "- **Training use:** {}", findings.training_use);
    output.push_str("\nCaveats:\n\n");

    for caveat in &findings.caveats {
        let _ = writeln!(output, "- {caveat}");
    }

    output.push_str("\n## File inventory\n\n");
    output
        .push_str("| Path | Kind | Bytes | SHA-256 | Upstream source |\n|---|---|---:|---|---|\n");

    for file in &report.files {
        let _ = writeln!(
            output,
            "| `{}` | {} | {} | `{}` | {} |",
            file.path,
            file.kind,
            file.bytes,
            file.sha256,
            file.source_url.as_deref().unwrap_or("-")
        );
    }

    output.push_str("\n## Totals\n\n");
    output.push_str("| Metric | Value |\n|---|---:|\n");
    let _ = writeln!(output, "| Records read | {} |", report.totals.records_read);
    let _ = writeln!(
        output,
        "| Conversations parsed | {} |",
        report.totals.conversations_read
    );
    let _ = writeln!(
        output,
        "| Conversations valid | {} |",
        report.totals.conversations_valid
    );
    let _ = writeln!(
        output,
        "| Conversations flagged | {} |",
        report.totals.conversations_flagged
    );
    let _ = writeln!(
        output,
        "| Conversations invalid | {} |",
        report.totals.conversations_invalid
    );
    let _ = writeln!(
        output,
        "| Malformed records | {} |",
        report.totals.malformed_records
    );
    let _ = writeln!(output, "| Messages | {} |", report.totals.messages);
    let _ = writeln!(output, "| Characters | {} |", report.totals.characters);

    output.push_str("\n## Qualification summary\n\n");
    output.push_str("| Subset | Status |\n|---|---|\n");

    for (name, subset) in &report.subsets {
        let _ = writeln!(output, "| `{name}` | **{}** |", subset.qualification.status);
    }

    for (name, subset) in &report.subsets {
        let _ = writeln!(output, "\n## Subset: `{name}` (`{}`)\n", subset.file);

        let _ = writeln!(
            output,
            "**Qualification: {}**\n",
            subset.qualification.status
        );
        output.push_str("Reasons:\n\n");

        for reason in &subset.qualification.reasons {
            let _ = writeln!(output, "- {reason}");
        }

        output.push_str("\n### Counts\n\n");
        output.push_str("| Metric | Value |\n|---|---:|\n");
        let _ = writeln!(output, "| Records read | {} |", subset.records_read);
        let _ = writeln!(
            output,
            "| Conversations parsed | {} |",
            subset.conversations_read
        );
        let _ = writeln!(
            output,
            "| Valid / flagged / invalid | {} / {} / {} |",
            subset.conversations_valid, subset.conversations_flagged, subset.conversations_invalid
        );
        let _ = writeln!(
            output,
            "| Malformed records | {} |",
            subset.malformed_records
        );
        let _ = writeln!(output, "| Messages | {} |", subset.messages);
        let _ = writeln!(output, "| Characters | {} |", subset.characters);
        let _ = writeln!(
            output,
            "| Annotated utterances | {} |",
            subset.annotated_utterances
        );
        let _ = writeln!(
            output,
            "| Non-ASCII characters | {} |",
            subset.non_ascii_characters
        );
        let _ = writeln!(output, "| CJK characters | {} |", subset.cjk_characters);

        output.push_str("\n### Speaker roles\n\n");
        output.push_str("| Original label | Utterances |\n|---|---:|\n");

        for (label, count) in &subset.speakers {
            let _ = writeln!(output, "| `{label}` | {count} |");
        }

        output.push_str("\nMapped roles:\n\n");

        if subset.mapped_roles.is_empty() {
            output.push_str("- none\n");
        } else {
            for (role, count) in &subset.mapped_roles {
                let _ = writeln!(output, "- `{role}`: {count}");
            }
        }

        output.push_str("\nUnmapped speaker labels:\n\n");

        if subset.unmapped_speakers.is_empty() {
            output.push_str("- none\n");
        } else {
            for (label, count) in &subset.unmapped_speakers {
                let _ = writeln!(output, "- `{label}`: {count}");
            }
        }

        output.push_str("\n### Domains\n\n");
        output.push_str("| Domain | Conversations |\n|---|---:|\n");

        for (domain, count) in &subset.domains {
            let _ = writeln!(output, "| `{domain}` | {count} |");
        }

        output.push_str("\n### Instruction identifiers\n\n");
        output.push_str("| Instruction ID | Conversations |\n|---|---:|\n");

        for (instruction, count) in &subset.instruction_ids {
            let _ = writeln!(output, "| `{instruction}` | {count} |");
        }

        output.push_str("\n### Conversation lengths\n\n");
        output.push_str("| Measure | Min | Median | P90 | Max |\n|---|---:|---:|---:|---:|\n");
        let _ = writeln!(
            output,
            "| Messages | {} | {} | {} | {} |",
            subset.conversation_messages.min,
            subset.conversation_messages.median,
            subset.conversation_messages.p90,
            subset.conversation_messages.max
        );
        let _ = writeln!(
            output,
            "| Characters | {} | {} | {} | {} |",
            subset.conversation_characters.min,
            subset.conversation_characters.median,
            subset.conversation_characters.p90,
            subset.conversation_characters.max
        );
        let _ = writeln!(
            output,
            "| Utterance characters | {} | {} | {} | {} |",
            subset.utterance_characters.min,
            subset.utterance_characters.median,
            subset.utterance_characters.p90,
            subset.utterance_characters.max
        );

        output.push_str("\n### Structural findings\n\n");
        output.push_str("| Finding | Records affected |\n|---|---:|\n");

        if subset.issues.is_empty() {
            output.push_str("| None | 0 |\n");
        } else {
            for (issue, count) in &subset.issues {
                let _ = writeln!(output, "| `{issue}` | {count} |");
            }
        }

        let _ = writeln!(
            output,
            "\nOccurrence counts: deleted placeholders {}, same-role adjacencies {}, adjacent duplicate utterances {}, empty conversations {}.\n",
            subset.deleted_placeholders,
            subset.same_role_adjacencies,
            subset.adjacent_duplicate_utterances,
            subset.empty_conversations
        );

        output.push_str("Utterance-level gaps:\n\n");
        let _ = writeln!(
            output,
            "- non-object utterances: {}\n- missing text: {}\n- empty text: {}\n- missing index: {}\n- non-sequential index: {}\n- missing speaker: {}\n- unknown speaker: {}",
            subset.non_object_utterances,
            subset.missing_text_utterances,
            subset.empty_text_utterances,
            subset.missing_index_utterances,
            subset.non_sequential_index_utterances,
            subset.missing_speaker_utterances,
            subset.unknown_speaker_utterances
        );

        output.push_str("\n### Privacy signals\n\n");
        output.push_str("| Pattern | Utterances |\n|---|---:|\n");
        let _ = writeln!(
            output,
            "| Email-like | {} |",
            subset.privacy.email_like_utterances
        );
        let _ = writeln!(
            output,
            "| URL-like | {} |",
            subset.privacy.url_like_utterances
        );
        let _ = writeln!(
            output,
            "| Phone-like digit runs | {} |",
            subset.privacy.phone_like_utterances
        );
        let _ = writeln!(
            output,
            "| Long digit runs (13+) | {} |",
            subset.privacy.long_digit_run_utterances
        );
        let _ = writeln!(
            output,
            "| Any pattern | {} |",
            subset.privacy.utterances_affected
        );
        let _ = writeln!(
            output,
            "\nConversations affected: {}.\n",
            subset.privacy.conversations_affected
        );

        output.push_str("### Budget fit (applicability, not a source defect)\n\n");
        output.push_str("| Budget | Conversations |\n|---|---:|\n");

        for (messages, characters) in LENGTH_BUDGETS {
            let key = budget_key(messages, characters);
            let _ = writeln!(
                output,
                "| {messages} messages / {characters} characters | {} |",
                subset.budget_fit.get(&key).copied().unwrap_or(0)
            );
        }
    }

    output.push_str("\n## Duplicate records\n\n");
    output.push_str("Payload duplicates compare ordered speaker/text pairs; annotations, utterance indices, and task identifiers are ignored. Identifier duplicates are reported separately.\n\n");

    if report.duplicate_conversation_ids.is_empty() {
        output.push_str("No repeated conversation identifiers were detected.\n\n");
    } else {
        output.push_str("### Repeated conversation identifiers\n\n");
        output.push_str("| Conversation ID | Occurrences | Locations |\n|---|---:|---|\n");

        for (shown, group) in report.duplicate_conversation_ids.iter().enumerate() {
            if shown >= 50 {
                let _ = writeln!(
                    output,
                    "| … | … | {} more groups not shown |",
                    report.duplicate_conversation_ids.len() - shown
                );
                break;
            }

            let locations = group
                .locations
                .iter()
                .map(|location| format!("{}:{}", location.file, location.line))
                .collect::<Vec<_>>()
                .join(", ");

            let _ = writeln!(
                output,
                "| `{}` | {} | {} |",
                group.conversation_id, group.occurrences, locations
            );
        }

        output.push('\n');
    }

    if report.duplicate_payloads.is_empty() {
        output.push_str("No structurally identical conversation payloads were detected.\n\n");
    } else {
        output.push_str("### Repeated utterance payloads\n\n");
        output.push_str("| Payload SHA-256 | Occurrences | Locations |\n|---|---:|---|\n");

        for (shown, group) in report.duplicate_payloads.iter().enumerate() {
            if shown >= 50 {
                let _ = writeln!(
                    output,
                    "| … | … | {} more groups not shown |",
                    report.duplicate_payloads.len() - shown
                );
                break;
            }

            let locations = group
                .locations
                .iter()
                .map(|location| format!("{}:{}", location.file, location.line))
                .collect::<Vec<_>>()
                .join(", ");

            let _ = writeln!(
                output,
                "| `{}` | {} | {} |",
                group.payload_sha256, group.occurrences, locations
            );
        }

        output.push('\n');
    }

    output.push_str("## Split coverage\n\n");

    if let Some(coverage) = &report.split_coverage {
        output.push_str("| Split | Rows | Unique IDs | Duplicate IDs | Malformed rows |\n|---|---:|---:|---:|---:|\n");

        for (name, split) in &coverage.splits {
            let _ = writeln!(
                output,
                "| `{name}` | {} | {} | {} | {} |",
                if split.present { split.rows } else { 0 },
                if split.present { split.unique_ids } else { 0 },
                if split.present {
                    split.duplicate_ids
                } else {
                    0
                },
                if split.present {
                    split.malformed_rows
                } else {
                    0
                }
            );
        }

        let _ = writeln!(
            output,
            "\nSelf-dialog IDs: {}; IDs in split lists: {}; covered: {}; missing from splits: {}; extra in splits: {}.",
            coverage.self_dialog_ids,
            coverage.ids_in_splits,
            coverage.covered_self_ids,
            coverage.missing_from_splits,
            coverage.extra_in_splits
        );

        if !coverage.overlaps.is_empty() {
            output.push_str("\nSplit overlaps:\n\n");

            for (pair, count) in &coverage.overlaps {
                let _ = writeln!(output, "- `{pair}`: {count}");
            }
        }

        output.push('\n');
    } else {
        output.push_str(
            "Split lists were not supplied (`train-dev-test/*.csv`); split coverage was not checked.\n\n",
        );
    }

    output.push_str("## Privacy and content concerns\n\n");

    for note in &report.privacy_notes {
        let _ = writeln!(output, "- {note}");
    }

    output.push_str("\n## Limitations\n\n");
    output.push_str("- The inspector reads extracted directories only; archive extraction is left to standard tools.\n");
    output.push_str("- Contact-data detection is a pattern scan, not a PII classifier.\n");
    output.push_str(
        "- The candidate manifest is not a training dataset and does not authorize training.\n",
    );
    output.push_str("- The license summary is not legal advice.\n");

    output.push_str("\n## Unresolved questions\n\n");

    for question in &report.unresolved_questions {
        let _ = writeln!(output, "- {question}");
    }

    output
}

#[cfg(test)]
mod tests {
    use std::{
        fs,
        sync::atomic::{AtomicUsize, Ordering},
    };

    use super::*;

    const VALID_SELF: &str = r#"[
      {
        "conversation_id": "dlg-self-1",
        "instruction_id": "pizza-ordering-1",
        "utterances": [
          {"index": 0, "speaker": "USER", "text": "Hi, I would like a pizza."},
          {"index": 1, "speaker": "ASSISTANT", "text": "Sure. Which size?"},
          {"index": 2, "speaker": "USER", "text": "A large one, please."},
          {"index": 3, "speaker": "ASSISTANT", "text": "Ordered."}
        ]
      },
      {
        "conversation_id": "dlg-self-2",
        "instruction_id": "movie-tickets-1",
        "utterances": [
          {"index": 0, "speaker": "USER", "text": "Two tickets, for the 7 pm show, please."},
          {"index": 1, "speaker": "ASSISTANT", "text": "Done."}
        ]
      }
    ]"#;

    const VALID_WOZ: &str = r#"[
      {
        "conversation_id": "dlg-woz-1",
        "instruction_id": "coffee-ordering-1",
        "utterances": [
          {"index": 0, "speaker": "USER", "text": "I want a latte."},
          {"index": 1, "speaker": "ASSISTANT", "text": "What size?"}
        ]
      }
    ]"#;

    fn temp_dir(name: &str) -> PathBuf {
        static COUNTER: AtomicUsize = AtomicUsize::new(0);

        let index = COUNTER.fetch_add(1, Ordering::Relaxed);

        let dir = std::env::temp_dir().join(format!(
            "mikuru-taskmaster1-test-{}-{index}-{name}",
            std::process::id()
        ));

        fs::create_dir_all(&dir).expect("temporary directory should be created");
        dir
    }

    fn write_fixture(dir: &Path, self_json: &str, woz_json: &str) {
        fs::write(dir.join("README.md"), "# COPYRIGHT NOTICE\nCC BY 4.0\n")
            .expect("README fixture should be written");
        fs::write(dir.join("self-dialogs.json"), self_json)
            .expect("self-dialogs fixture should be written");
        fs::write(dir.join("woz-dialogs.json"), woz_json)
            .expect("woz-dialogs fixture should be written");
    }

    fn manifest(path: &Path) -> Vec<Value> {
        fs::read_to_string(path)
            .expect("manifest should be readable")
            .lines()
            .map(|line| serde_json::from_str(line).expect("manifest line should be JSON"))
            .collect()
    }

    #[test]
    fn inspects_valid_subsets_and_preserves_source_metadata() {
        let dir = temp_dir("valid");
        write_fixture(&dir, VALID_SELF, VALID_WOZ);

        let output = dir.join("out");
        let config = Taskmaster1InspectConfig {
            revision: "test-revision".to_owned(),
        };

        let result = inspect_taskmaster1(&dir, &output, &config).expect("inspection should run");

        assert_eq!(result.report.totals.records_read, 3);
        assert_eq!(result.report.totals.conversations_read, 3);
        assert_eq!(result.report.totals.malformed_records, 0);
        assert_eq!(result.report.totals.messages, 8);
        assert_eq!(result.report.dataset.upstream_revision, "test-revision");
        assert_eq!(result.report.dataset.license, TASKMASTER1_LICENSE);
        assert!(result.report.dataset.license_evidence_present);

        let self_report = &result.report.subsets["self-dialogs"];
        assert_eq!(self_report.conversations_valid, 2);
        assert_eq!(self_report.conversations_flagged, 0);
        assert_eq!(self_report.messages, 6);
        assert_eq!(self_report.domains["pizza_ordering"], 1);
        assert_eq!(self_report.domains["movie_tickets"], 1);
        assert_eq!(self_report.mapped_roles["user"], 3);
        assert_eq!(self_report.mapped_roles["assistant"], 3);
        assert!(self_report.unmapped_speakers.is_empty());
        assert_eq!(self_report.qualification.status, STATUS_ACCEPTED);

        let woz_report = &result.report.subsets["woz-dialogs"];
        assert_eq!(woz_report.conversations_valid, 1);
        assert_eq!(woz_report.qualification.status, STATUS_ACCEPTED);

        let entries = manifest(&result.manifest_path);
        assert_eq!(entries.len(), 3);
        assert_eq!(entries[0]["source"], TASKMASTER1_SOURCE_DATASET);
        assert_eq!(entries[0]["source_revision"], "test-revision");
        assert_eq!(entries[0]["original_conversation_id"], "dlg-self-1");
        assert_eq!(entries[0]["original_instruction_id"], "pizza-ordering-1");
        assert_eq!(entries[0]["original_speakers"][0], "USER");
        assert_eq!(entries[0]["mapped_roles"][0], "user");
        assert_eq!(entries[0]["mapped_roles"][1], "assistant");
        assert_eq!(entries[0]["status"], STATUS_VALID);
        assert_eq!(entries[0]["subset_qualification"], STATUS_ACCEPTED);

        let markdown = fs::read_to_string(&result.markdown_path).expect("report should exist");
        assert!(markdown.contains("CC BY 4.0"));
        assert!(markdown.contains("## Subset: `self-dialogs`"));
        assert!(markdown.contains("pizza_ordering"));
        assert!(!markdown.contains("Hi, I would like a pizza."));

        let json: Value = serde_json::from_str(
            &fs::read_to_string(&result.json_path).expect("json report should exist"),
        )
        .expect("json report should parse");
        assert_eq!(json["dataset"]["upstream_revision"], "test-revision");
        assert_eq!(json["totals"]["messages"], 8);

        fs::remove_dir_all(dir).expect("temporary directory should be removed");
    }

    #[test]
    fn flags_unknown_and_missing_speaker_labels() {
        let dir = temp_dir("roles");
        let self_json = r#"[
          {
            "conversation_id": "dlg-roles",
            "instruction_id": "restaurant-table-1",
            "utterances": [
              {"index": 0, "speaker": "USER", "text": "A table for two."},
              {"index": 1, "speaker": "AGENT", "text": "Certainly."},
              {"index": 2, "speaker": "", "text": "Friday at seven."},
              {"index": 3, "speaker": "USER", "text": "Thanks."}
            ]
          }
        ]"#;
        write_fixture(&dir, self_json, "[]");

        let result =
            inspect_taskmaster1(&dir, dir.join("out"), &Taskmaster1InspectConfig::default())
                .expect("inspection should run");

        let report = &result.report.subsets["self-dialogs"];
        assert_eq!(report.unknown_speaker_utterances, 1);
        assert_eq!(report.missing_speaker_utterances, 1);
        assert_eq!(report.unmapped_speakers["AGENT"], 1);
        assert_eq!(report.unmapped_speakers["<missing>"], 1);
        assert_eq!(report.conversations_flagged, 1);
        assert_eq!(report.issues["unknown_speaker"], 1);
        assert_eq!(report.issues["missing_speaker"], 1);
        assert_eq!(report.qualification.status, STATUS_CONDITIONAL);

        let entries = manifest(&result.manifest_path);
        assert_eq!(entries[0]["mapped_roles"][1], Value::Null);
        assert_eq!(entries[0]["mapped_roles"][2], Value::Null);
        assert_eq!(entries[0]["original_speakers"][1], "AGENT");
        assert_eq!(entries[0]["status"], STATUS_FLAGGED);

        fs::remove_dir_all(dir).expect("temporary directory should be removed");
    }

    #[test]
    fn records_empty_malformed_and_incomplete_records_without_skipping_them() {
        let dir = temp_dir("invalid");
        let self_json = r#"[
          {
            "conversation_id": "dlg-empty",
            "instruction_id": "pizza-ordering-1",
            "utterances": []
          },
          null,
          {
            "conversation_id": "dlg-missing-utterances",
            "instruction_id": "pizza-ordering-1"
          },
          {
            "instruction_id": "pizza-ordering-1",
            "utterances": [
              {"speaker": "USER", "text": "Hi"},
              {"index": 2, "speaker": "ASSISTANT", "text": "  "}
            ]
          },
          {"conversation_id": "dlg-bad-json", "instruction_id": }
        ]"#;
        write_fixture(&dir, self_json, "[]");

        let result =
            inspect_taskmaster1(&dir, dir.join("out"), &Taskmaster1InspectConfig::default())
                .expect("inspection should continue after record-level failures");

        let report = &result.report.subsets["self-dialogs"];
        assert_eq!(report.records_read, 5);
        assert_eq!(report.conversations_read, 3);
        assert_eq!(report.malformed_records, 2);
        assert_eq!(report.empty_conversations, 1);
        assert_eq!(report.missing_index_utterances, 1);
        assert_eq!(report.non_sequential_index_utterances, 1);
        assert_eq!(report.empty_text_utterances, 1);
        assert_eq!(report.conversations_invalid, 3);
        assert_eq!(report.issues["malformed_record"], 2);
        assert_eq!(report.issues["empty_conversation"], 1);
        assert_eq!(report.issues["missing_utterances"], 1);
        assert_eq!(report.issues["missing_conversation_id"], 1);

        let entries = manifest(&result.manifest_path);
        assert_eq!(entries.len(), 5);
        assert_eq!(entries[1]["status"], STATUS_MALFORMED);
        assert!(entries[1]["original_line"].as_u64().is_some());
        assert_eq!(entries[4]["status"], STATUS_MALFORMED);

        fs::remove_dir_all(dir).expect("temporary directory should be removed");
    }

    #[test]
    fn detects_duplicate_ids_and_identical_payloads() {
        let dir = temp_dir("duplicates");
        let self_json = r#"[
          {
            "conversation_id": "dlg-dup",
            "instruction_id": "pizza-ordering-1",
            "utterances": [
              {"index": 0, "speaker": "USER", "text": "First."},
              {"index": 1, "speaker": "ASSISTANT", "text": "Reply one."}
            ]
          },
          {
            "conversation_id": "dlg-dup",
            "instruction_id": "pizza-ordering-1",
            "utterances": [
              {"index": 0, "speaker": "USER", "text": "Second."},
              {"index": 1, "speaker": "ASSISTANT", "text": "Reply two."}
            ]
          },
          {
            "conversation_id": "dlg-payload-a",
            "instruction_id": "movie-tickets-1",
            "utterances": [
              {"index": 0, "speaker": "USER", "text": "Same payload."},
              {"index": 1, "speaker": "ASSISTANT", "text": "Same reply."}
            ]
          },
          {
            "conversation_id": "dlg-payload-b",
            "instruction_id": "restaurant-table-1",
            "utterances": [
              {"index": 0, "speaker": "USER", "text": "Same payload."},
              {"index": 1, "speaker": "ASSISTANT", "text": "Same reply."}
            ]
          }
        ]"#;
        write_fixture(&dir, self_json, "[]");

        let result =
            inspect_taskmaster1(&dir, dir.join("out"), &Taskmaster1InspectConfig::default())
                .expect("inspection should run");

        assert_eq!(result.report.duplicate_conversation_ids.len(), 1);
        assert_eq!(result.report.duplicate_conversation_ids[0].occurrences, 2);
        assert_eq!(result.report.duplicate_payloads.len(), 1);
        assert_eq!(result.report.duplicate_payloads[0].occurrences, 2);

        let report = &result.report.subsets["self-dialogs"];
        assert_eq!(report.issues["duplicate_conversation_id"], 2);
        assert_eq!(report.issues["duplicate_payload"], 2);
        assert_eq!(report.conversations_flagged, 4);
        assert_eq!(report.qualification.status, STATUS_CONDITIONAL);

        let entries = manifest(&result.manifest_path);
        assert_eq!(entries[0]["status"], STATUS_FLAGGED);
        assert!(
            entries[0]["warnings"]
                .as_array()
                .expect("warnings should be an array")
                .iter()
                .any(|warning| warning == "duplicate_conversation_id")
        );
        assert!(
            entries[2]["warnings"]
                .as_array()
                .expect("warnings should be an array")
                .iter()
                .any(|warning| warning == "duplicate_payload")
        );

        fs::remove_dir_all(dir).expect("temporary directory should be removed");
    }

    #[test]
    fn detects_deleted_placeholders_and_contact_patterns_without_echoing_values() {
        let dir = temp_dir("privacy");
        let self_json = r#"[
          {
            "conversation_id": "dlg-privacy",
            "instruction_id": "auto-repair-appt-1",
            "utterances": [
              {"index": 0, "speaker": "USER", "text": "Email me at person@example.test."},
              {"index": 1, "speaker": "ASSISTANT", "text": "Call 555 123 4567 instead."},
              {"index": 2, "speaker": "USER", "text": "(deleted)"}
            ]
          }
        ]"#;
        write_fixture(&dir, self_json, "[]");

        let result =
            inspect_taskmaster1(&dir, dir.join("out"), &Taskmaster1InspectConfig::default())
                .expect("inspection should run");

        let report = &result.report.subsets["self-dialogs"];
        assert_eq!(report.privacy.email_like_utterances, 1);
        assert_eq!(report.privacy.phone_like_utterances, 1);
        assert_eq!(report.privacy.utterances_affected, 2);
        assert_eq!(report.privacy.conversations_affected, 1);
        assert_eq!(report.deleted_placeholders, 1);
        assert_eq!(report.issues["potential_contact_data"], 1);
        assert_eq!(report.issues["deleted_placeholder"], 1);
        assert_eq!(report.qualification.status, STATUS_CONDITIONAL);

        let markdown = fs::read_to_string(&result.markdown_path).expect("report should exist");
        let json = fs::read_to_string(&result.json_path).expect("json report should exist");
        let manifest_text =
            fs::read_to_string(&result.manifest_path).expect("manifest should exist");

        for artifact in [&markdown, &json, &manifest_text] {
            assert!(!artifact.contains("person@example.test"));
            assert!(!artifact.contains("555 123 4567"));
        }

        fs::remove_dir_all(dir).expect("temporary directory should be removed");
    }

    #[test]
    fn detects_same_role_adjacency_and_adjacent_duplicate_utterances() {
        let dir = temp_dir("adjacency");
        let self_json = r#"[
          {
            "conversation_id": "dlg-adjacent",
            "instruction_id": "coffee-ordering-1",
            "utterances": [
              {"index": 0, "speaker": "USER", "text": "Hi."},
              {"index": 1, "speaker": "USER", "text": "Hi."},
              {"index": 2, "speaker": "ASSISTANT", "text": "Hello."}
            ]
          }
        ]"#;
        write_fixture(&dir, self_json, "[]");

        let result =
            inspect_taskmaster1(&dir, dir.join("out"), &Taskmaster1InspectConfig::default())
                .expect("inspection should run");

        let report = &result.report.subsets["self-dialogs"];
        assert_eq!(report.same_role_adjacencies, 1);
        assert_eq!(report.adjacent_duplicate_utterances, 1);
        assert_eq!(report.issues["same_role_adjacency"], 1);
        assert_eq!(report.issues["adjacent_duplicate_utterance"], 1);

        fs::remove_dir_all(dir).expect("temporary directory should be removed");
    }

    #[test]
    fn verifies_split_coverage_and_camel_case_fallbacks() {
        let dir = temp_dir("splits");
        write_fixture(&dir, VALID_SELF, "[]");

        fs::write(
            dir.join("woz-dialogs.json"),
            r#"[
              {
                "conversationId": "dlg-camel",
                "instructionId": "uber-lyft-1",
                "utterances": [
                  {"index": 0, "speaker": "USER", "text": "I need a ride."},
                  {"index": 1, "speaker": "ASSISTANT", "text": "Where to?"}
                ]
              }
            ]"#,
        )
        .expect("camel-case fixture should be written");

        fs::create_dir_all(dir.join("train-dev-test")).expect("split directory should be created");
        fs::write(
            dir.join("train-dev-test/train.csv"),
            "dlg-self-1,\ndlg-self-2,\n",
        )
        .expect("train split should be written");
        fs::write(dir.join("train-dev-test/dev.csv"), "dlg-self-1,\n")
            .expect("dev split should be written");

        let result =
            inspect_taskmaster1(&dir, dir.join("out"), &Taskmaster1InspectConfig::default())
                .expect("inspection should run");

        let coverage = result
            .report
            .split_coverage
            .as_ref()
            .expect("split coverage should be present");
        assert_eq!(coverage.self_dialog_ids, 2);
        assert_eq!(coverage.ids_in_splits, 2);
        assert_eq!(coverage.covered_self_ids, 2);
        assert_eq!(coverage.missing_from_splits, 0);
        assert_eq!(coverage.extra_in_splits, 0);
        assert_eq!(coverage.overlaps["dev/train"], 1);
        assert!(!coverage.splits["test"].present);

        let woz = &result.report.subsets["woz-dialogs"];
        assert_eq!(woz.conversations_valid, 1);
        assert_eq!(woz.domains["ride_service"], 1);

        fs::remove_dir_all(dir).expect("temporary directory should be removed");
    }

    #[test]
    fn report_generation_is_deterministic_and_refuses_overwrite() {
        let dir = temp_dir("deterministic");
        write_fixture(&dir, VALID_SELF, VALID_WOZ);

        let config = Taskmaster1InspectConfig::default();
        let first = inspect_taskmaster1(&dir, dir.join("out-a"), &config)
            .expect("first inspection should run");
        let second = inspect_taskmaster1(&dir, dir.join("out-b"), &config)
            .expect("second inspection should run");

        let first_markdown = fs::read(&first.markdown_path).expect("first report should exist");
        let second_markdown = fs::read(&second.markdown_path).expect("second report should exist");
        let first_json = fs::read(&first.json_path).expect("first json should exist");
        let second_json = fs::read(&second.json_path).expect("second json should exist");
        let first_manifest = fs::read(&first.manifest_path).expect("first manifest should exist");
        let second_manifest =
            fs::read(&second.manifest_path).expect("second manifest should exist");

        assert_eq!(first_markdown, second_markdown);
        assert_eq!(first_json, second_json);
        assert_eq!(first_manifest, second_manifest);

        let error = inspect_taskmaster1(&dir, dir.join("out-a"), &config)
            .expect_err("existing outputs must not be overwritten");
        assert!(error.to_string().contains("already exists"));

        fs::remove_dir_all(dir).expect("temporary directory should be removed");
    }

    #[test]
    fn rejects_broken_array_framing_and_unsupported_inputs() {
        let dir = temp_dir("framing");
        write_fixture(&dir, "[{\"conversation_id\":\"dlg-1\"}", "[]");

        let error =
            inspect_taskmaster1(&dir, dir.join("out"), &Taskmaster1InspectConfig::default())
                .expect_err("unterminated arrays must be rejected");
        assert!(error.to_string().contains("unexpected end of file"));

        let archive = dir.join("TM-1-2019.zip");
        fs::write(&archive, b"not really a zip").expect("archive fixture should be written");
        let error = inspect_taskmaster1(
            &archive,
            dir.join("out-archive"),
            &Taskmaster1InspectConfig::default(),
        )
        .expect_err("archives should not be read directly");
        assert!(error.to_string().contains("extract"));

        fs::remove_dir_all(dir).expect("temporary directory should be removed");
    }

    #[test]
    fn recognizes_a_nested_tm_1_2019_directory() {
        let dir = temp_dir("nested");
        let dataset = dir.join("TM-1-2019");
        fs::create_dir_all(&dataset).expect("nested directory should be created");
        write_fixture(&dataset, VALID_SELF, VALID_WOZ);

        let result =
            inspect_taskmaster1(&dir, dir.join("out"), &Taskmaster1InspectConfig::default())
                .expect("nested dataset should be found");

        assert_eq!(result.report.totals.conversations_read, 3);

        fs::remove_dir_all(dir).expect("temporary directory should be removed");
    }
}
