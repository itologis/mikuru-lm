use std::{
    collections::{BTreeMap, HashMap, HashSet},
    error::Error,
    fmt,
    fs::{self, File},
    io::{self, BufRead, BufReader},
    path::{Path, PathBuf},
};

use serde::{Deserialize, Serialize};

/// Canonical label for records used during training.
pub const TRAIN_SPLIT: &str = "train";

/// Canonical label for records held out for validation.
pub const VALIDATION_SPLIT: &str = "validation";

/// Legacy label used by the starter sample records. `demo` records are not
/// assigned to training or validation; preparation may relabel them.
pub const DEMO_SPLIT: &str = "demo";

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

    /// Original message identifiers from the source dataset, in conversation
    /// order. Empty for records that were not imported from a message table.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub source_message_ids: Vec<String>,

    /// Notes explaining the record's origin and preparation.
    pub provenance_note: String,
}

/// Recognised split labels.
///
/// `demo` is the legacy label used by the starter sample. It marks a record as
/// a demonstration rather than a train/validation assignment. Unknown labels
/// are rejected so a typo cannot silently move validation records into
/// training.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DatasetSplit {
    Train,
    Validation,
    Demo,
}

impl DatasetSplit {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Train => TRAIN_SPLIT,
            Self::Validation => VALIDATION_SPLIT,
            Self::Demo => DEMO_SPLIT,
        }
    }

    /// Whether the label assigns the record to training or validation.
    pub fn is_assigned(self) -> bool {
        !matches!(self, Self::Demo)
    }

    pub fn parse(value: &str) -> Result<Self, String> {
        match value.trim() {
            TRAIN_SPLIT => Ok(Self::Train),
            VALIDATION_SPLIT => Ok(Self::Validation),
            DEMO_SPLIT => Ok(Self::Demo),
            other => Err(format!(
                "unknown split `{other}`; expected `{TRAIN_SPLIT}`, `{VALIDATION_SPLIT}`, or `{DEMO_SPLIT}`"
            )),
        }
    }
}

/// True when a record has no train/validation assignment.
pub fn is_unassigned_split(value: Option<&str>) -> bool {
    match value.map(str::trim) {
        None | Some(DEMO_SPLIT) => true,
        Some(_) => false,
    }
}

/// Errors produced while loading or preparing a dataset.
#[derive(Debug)]
pub enum DatasetError {
    /// The file could not be opened, read, or written.
    Io { path: PathBuf, source: io::Error },
    /// A line was not valid JSON.
    Json {
        path: PathBuf,
        line: usize,
        message: String,
    },
    /// The record parsed but failed validation.
    Invalid {
        path: PathBuf,
        line: usize,
        message: String,
    },
    /// The same record id appeared more than once.
    DuplicateId { id: String, detail: String },
    /// Split configuration is missing, partial, or contradictory.
    Split { message: String },
    /// A JSONL file contained no records.
    Empty { path: PathBuf },
    /// A preparation output file already exists.
    OutputExists { path: PathBuf },
    /// A record could not be serialized back to JSON.
    Serialize { message: String },
}

impl fmt::Display for DatasetError {
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
                "invalid JSON at {}:{line}: {message}",
                path.display()
            ),
            Self::Invalid {
                path,
                line,
                message,
            } => write!(
                formatter,
                "invalid record at {}:{line}: {message}",
                path.display()
            ),
            Self::DuplicateId { id, detail } => {
                write!(formatter, "duplicate record id `{id}`: {detail}")
            }
            Self::Split { message } => write!(formatter, "dataset split error: {message}"),
            Self::Empty { path } => write!(formatter, "{} contains no records", path.display()),
            Self::OutputExists { path } => write!(
                formatter,
                "output already exists: {}; choose another location",
                path.display()
            ),
            Self::Serialize { message } => {
                write!(formatter, "failed to serialize record: {message}")
            }
        }
    }
}

impl Error for DatasetError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Io { source, .. } => Some(source),
            _ => None,
        }
    }
}

/// Provenance summary for one source dataset inside a prepared split.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct SourceSummary {
    pub source_dataset: String,
    pub license: String,
    pub records: usize,
    pub synthetic_records: usize,
}

/// Machine-readable description of a prepared train/validation dataset.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct DatasetManifest {
    pub format_version: u32,
    /// `explicit` when records already carried split labels, `deterministic`
    /// when the split was generated from ids and a seed.
    pub split_source: String,
    pub split_seed: Option<u64>,
    pub validation_fraction: Option<f32>,
    pub train_records: usize,
    pub validation_records: usize,
    pub sources: Vec<SourceSummary>,
}

/// Options for [`prepare_dataset`].
#[derive(Clone, Copy, Debug)]
pub struct PrepareOptions {
    pub validation_fraction: f32,
    pub seed: u64,
}

impl Default for PrepareOptions {
    fn default() -> Self {
        Self {
            validation_fraction: 0.2,
            seed: 42,
        }
    }
}

/// Where the split labels came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SplitSource {
    Explicit,
    Deterministic,
}

impl SplitSource {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Explicit => "explicit",
            Self::Deterministic => "deterministic",
        }
    }
}

/// Result of a successful dataset preparation run.
#[derive(Clone, Debug)]
pub struct PrepareReport {
    pub output_dir: PathBuf,
    pub train_path: PathBuf,
    pub validation_path: PathBuf,
    pub manifest_path: PathBuf,
    pub train_records: usize,
    pub validation_records: usize,
    pub split_source: SplitSource,
}

fn invalid_record(path: &Path, line: usize, message: impl Into<String>) -> DatasetError {
    DatasetError::Invalid {
        path: path.to_path_buf(),
        line,
        message: message.into(),
    }
}

fn require_non_empty(
    field: &str,
    value: &str,
    path: &Path,
    line: usize,
) -> Result<(), DatasetError> {
    if value.trim().is_empty() {
        return Err(invalid_record(
            path,
            line,
            format!("field `{field}` must not be empty"),
        ));
    }

    Ok(())
}

/// Validates one record without touching the filesystem.
fn validate_record(
    record: &ConversationExample,
    path: &Path,
    line: usize,
) -> Result<(), DatasetError> {
    for (field, value) in [
        ("id", record.id.as_str()),
        ("language", record.language.as_str()),
        ("category", record.category.as_str()),
        ("source_dataset", record.source_dataset.as_str()),
        ("source_record_id", record.source_record_id.as_str()),
        ("source_url", record.source_url.as_str()),
        ("license", record.license.as_str()),
        ("provenance_note", record.provenance_note.as_str()),
    ] {
        require_non_empty(field, value, path, line)?;
    }

    if record.messages.is_empty() {
        return Err(invalid_record(
            path,
            line,
            "conversation must contain at least one message",
        ));
    }

    for (index, message) in record.messages.iter().enumerate() {
        if message.role.trim().is_empty() {
            return Err(invalid_record(
                path,
                line,
                format!("message {} has an empty role", index + 1),
            ));
        }

        if message.content.trim().is_empty() {
            return Err(invalid_record(
                path,
                line,
                format!("message {} has empty content", index + 1),
            ));
        }
    }

    for (index, reference) in record.references.iter().enumerate() {
        if reference.trim().is_empty() {
            return Err(invalid_record(
                path,
                line,
                format!("reference {} is empty", index + 1),
            ));
        }
    }

    for (index, message_id) in record.source_message_ids.iter().enumerate() {
        if message_id.trim().is_empty() {
            return Err(invalid_record(
                path,
                line,
                format!("source_message_ids[{index}] is empty"),
            ));
        }
    }

    if !record.source_message_ids.is_empty()
        && record.source_message_ids.len() != record.messages.len()
    {
        return Err(invalid_record(
            path,
            line,
            format!(
                "source_message_ids has {} entries but the conversation has {} messages",
                record.source_message_ids.len(),
                record.messages.len()
            ),
        ));
    }

    if let Some(split) = record.split.as_deref() {
        DatasetSplit::parse(split).map_err(|message| invalid_record(path, line, message))?;
    }

    Ok(())
}

fn read_jsonl(path: &Path) -> Result<Vec<(usize, ConversationExample)>, DatasetError> {
    let file = File::open(path).map_err(|source| DatasetError::Io {
        path: path.to_path_buf(),
        source,
    })?;

    let reader = BufReader::new(file);
    let mut records = Vec::new();

    for (index, line_result) in reader.lines().enumerate() {
        let line_number = index + 1;

        let line = line_result.map_err(|source| DatasetError::Io {
            path: path.to_path_buf(),
            source,
        })?;

        if line.trim().is_empty() {
            continue;
        }

        let record: ConversationExample =
            serde_json::from_str(&line).map_err(|error| DatasetError::Json {
                path: path.to_path_buf(),
                line: line_number,
                message: error.to_string(),
            })?;

        validate_record(&record, path, line_number)?;

        records.push((line_number, record));
    }

    if records.is_empty() {
        return Err(DatasetError::Empty {
            path: path.to_path_buf(),
        });
    }

    Ok(records)
}

/// Loads one JSONL file and rejects malformed records, missing metadata,
/// unknown split labels, and duplicate record ids.
pub fn load_jsonl(path: impl AsRef<Path>) -> Result<Vec<ConversationExample>, DatasetError> {
    load_jsonl_files(&[path.as_ref().to_path_buf()])
}

/// Loads multiple JSONL files as one dataset.
///
/// Record ids must be unique across every supplied file, which guarantees a
/// record cannot be smuggled into both training and validation.
pub fn load_jsonl_files(paths: &[PathBuf]) -> Result<Vec<ConversationExample>, DatasetError> {
    if paths.is_empty() {
        return Err(DatasetError::Split {
            message: "no dataset files were provided".to_owned(),
        });
    }

    let mut records = Vec::new();
    let mut seen: HashMap<String, String> = HashMap::new();

    for path in paths {
        for (line_number, record) in read_jsonl(path)? {
            let location = format!("{}:{line_number}", path.display());

            if let Some(first) = seen.insert(record.id.clone(), location.clone()) {
                return Err(DatasetError::DuplicateId {
                    id: record.id,
                    detail: format!("first seen at {first}, repeated at {location}"),
                });
            }

            records.push(record);
        }
    }

    Ok(records)
}

/// Verifies that every record id in the iterator is unique.
pub fn ensure_unique_ids<'a, I>(records: I) -> Result<(), DatasetError>
where
    I: IntoIterator<Item = &'a ConversationExample>,
{
    let mut seen: HashSet<&str> = HashSet::new();

    for record in records {
        if !seen.insert(record.id.as_str()) {
            return Err(DatasetError::DuplicateId {
                id: record.id.clone(),
                detail: "appears more than once in the dataset".to_owned(),
            });
        }
    }

    Ok(())
}

/// Partitions records using their explicit `split` labels.
///
/// Every record must carry a label. Records labelled `train` go to the first
/// vector, records labelled `validation` to the second.
pub fn partition_by_split(
    records: &[ConversationExample],
) -> Result<(Vec<ConversationExample>, Vec<ConversationExample>), DatasetError> {
    ensure_unique_ids(records.iter())?;

    let labeled = records
        .iter()
        .filter(|record| record.split.is_some())
        .count();

    if labeled == 0 {
        return Err(DatasetError::Split {
            message: "no record declares a split; label the records or run prepare-dataset"
                .to_owned(),
        });
    }

    if labeled != records.len() {
        return Err(DatasetError::Split {
            message: format!(
                "{labeled} of {} records declare a split; label every record or none",
                records.len()
            ),
        });
    }

    let mut train = Vec::new();
    let mut validation = Vec::new();

    for record in records {
        let split = DatasetSplit::parse(record.split.as_deref().unwrap_or_default()).map_err(
            |message| DatasetError::Split {
                message: format!("record `{}`: {message}", record.id),
            },
        )?;

        match split {
            DatasetSplit::Train => train.push(record.clone()),
            DatasetSplit::Validation => validation.push(record.clone()),
            DatasetSplit::Demo => {
                return Err(DatasetError::Split {
                    message: format!(
                        "record `{}` uses the legacy `{DEMO_SPLIT}` label; assign `{TRAIN_SPLIT}` or `{VALIDATION_SPLIT}`, or run prepare-dataset",
                        record.id
                    ),
                });
            }
        }
    }

    let train_ids: HashSet<&str> = train.iter().map(|record| record.id.as_str()).collect();

    if let Some(duplicate) = validation
        .iter()
        .find(|record| train_ids.contains(record.id.as_str()))
    {
        return Err(DatasetError::DuplicateId {
            id: duplicate.id.clone(),
            detail: "appears in both the training and validation splits".to_owned(),
        });
    }

    if train.is_empty() {
        return Err(DatasetError::Split {
            message: "the training split is empty".to_owned(),
        });
    }

    Ok((train, validation))
}

fn fnv1a_id_hash(seed: u64, id: &str) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64 ^ seed;

    for byte in id.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }

    hash
}

/// Splits records deterministically using a seeded id hash.
///
/// The same records, seed, and fraction always produce the same split. Both
/// splits are guaranteed non-empty for any input of at least two records.
/// Original record order is preserved inside each split.
pub fn deterministic_split(
    records: &[ConversationExample],
    validation_fraction: f32,
    seed: u64,
) -> Result<(Vec<ConversationExample>, Vec<ConversationExample>), DatasetError> {
    ensure_unique_ids(records.iter())?;

    if records.len() < 2 {
        return Err(DatasetError::Split {
            message: "at least two records are required for a train/validation split".to_owned(),
        });
    }

    if !validation_fraction.is_finite() || !(0.0..1.0).contains(&validation_fraction) {
        return Err(DatasetError::Split {
            message: format!(
                "validation_fraction must be greater than 0 and less than 1, got {validation_fraction}"
            ),
        });
    }

    let validation_count =
        ((records.len() as f64) * f64::from(validation_fraction)).ceil() as usize;

    let validation_count = validation_count.clamp(1, records.len() - 1);

    let mut ranked: Vec<usize> = (0..records.len()).collect();

    ranked.sort_by(|&left, &right| {
        let left_hash = fnv1a_id_hash(seed, &records[left].id);
        let right_hash = fnv1a_id_hash(seed, &records[right].id);

        left_hash
            .cmp(&right_hash)
            .then_with(|| records[left].id.cmp(&records[right].id))
    });

    let selected: HashSet<usize> = ranked[..validation_count].iter().copied().collect();

    let mut train = Vec::with_capacity(records.len() - validation_count);
    let mut validation = Vec::with_capacity(validation_count);

    for (index, record) in records.iter().enumerate() {
        if selected.contains(&index) {
            validation.push(record.clone());
        } else {
            train.push(record.clone());
        }
    }

    Ok((train, validation))
}

/// Writes records as JSONL without dropping any provenance field.
///
/// The function refuses to overwrite an existing file.
pub fn write_jsonl(
    path: impl AsRef<Path>,
    records: &[ConversationExample],
) -> Result<(), DatasetError> {
    let path = path.as_ref();

    if path.exists() {
        return Err(DatasetError::OutputExists {
            path: path.to_path_buf(),
        });
    }

    if records.is_empty() {
        return Err(DatasetError::Invalid {
            path: path.to_path_buf(),
            line: 0,
            message: "refusing to write an empty dataset file".to_owned(),
        });
    }

    let mut output = String::new();

    for record in records {
        let line = serde_json::to_string(record).map_err(|error| DatasetError::Serialize {
            message: error.to_string(),
        })?;

        output.push_str(&line);
        output.push('\n');
    }

    fs::write(path, output).map_err(|source| DatasetError::Io {
        path: path.to_path_buf(),
        source,
    })
}

/// Builds a manifest describing a prepared split.
pub fn build_manifest(
    train: &[ConversationExample],
    validation: &[ConversationExample],
    split_source: &str,
    split_seed: Option<u64>,
    validation_fraction: Option<f32>,
) -> DatasetManifest {
    let mut sources: BTreeMap<(String, String), (usize, usize)> = BTreeMap::new();

    for record in train.iter().chain(validation.iter()) {
        let entry = sources
            .entry((record.source_dataset.clone(), record.license.clone()))
            .or_insert((0, 0));

        entry.0 += 1;

        if record.synthetic {
            entry.1 += 1;
        }
    }

    let sources = sources
        .into_iter()
        .map(
            |((source_dataset, license), (records, synthetic_records))| SourceSummary {
                source_dataset,
                license,
                records,
                synthetic_records,
            },
        )
        .collect();

    DatasetManifest {
        format_version: 1,
        split_source: split_source.to_owned(),
        split_seed,
        validation_fraction,
        train_records: train.len(),
        validation_records: validation.len(),
        sources,
    }
}

/// Prepares a reproducible train/validation dataset directory.
///
/// If every input record already carries a split label, those labels are
/// respected. If no record carries a label, a deterministic split is generated
/// from the record ids and the supplied seed. Partially labelled input is
/// rejected. The output directory receives `train.jsonl`, `validation.jsonl`,
/// and `manifest.json`; existing files are never overwritten.
pub fn prepare_dataset(
    input: impl AsRef<Path>,
    output_dir: impl AsRef<Path>,
    options: &PrepareOptions,
) -> Result<PrepareReport, DatasetError> {
    if !options.validation_fraction.is_finite()
        || !(0.0..1.0).contains(&options.validation_fraction)
    {
        return Err(DatasetError::Split {
            message: format!(
                "validation_fraction must be greater than 0 and less than 1, got {}",
                options.validation_fraction
            ),
        });
    }

    let output_dir = output_dir.as_ref();
    let train_path = output_dir.join("train.jsonl");
    let validation_path = output_dir.join("validation.jsonl");
    let manifest_path = output_dir.join("manifest.json");

    for path in [&train_path, &validation_path, &manifest_path] {
        if path.exists() {
            return Err(DatasetError::OutputExists { path: path.clone() });
        }
    }

    let records = load_jsonl(input)?;

    let unassigned = records
        .iter()
        .filter(|record| is_unassigned_split(record.split.as_deref()))
        .count();

    let (mut train, mut validation, split_source) = if unassigned == records.len() {
        let (train, validation) =
            deterministic_split(&records, options.validation_fraction, options.seed)?;

        (train, validation, SplitSource::Deterministic)
    } else if unassigned == 0 {
        let (train, validation) = partition_by_split(&records)?;

        (train, validation, SplitSource::Explicit)
    } else {
        return Err(DatasetError::Split {
            message: format!(
                "{unassigned} of {} records have no train/validation assignment; label every record or none",
                records.len()
            ),
        });
    };

    if train.is_empty() || validation.is_empty() {
        return Err(DatasetError::Split {
            message: "a prepared dataset needs at least one training and one validation record"
                .to_owned(),
        });
    }

    // The prepared files carry explicit labels so downstream tools never have
    // to re-run the split logic.
    for record in &mut train {
        record.split = Some(TRAIN_SPLIT.to_owned());
    }

    for record in &mut validation {
        record.split = Some(VALIDATION_SPLIT.to_owned());
    }

    fs::create_dir_all(output_dir).map_err(|source| DatasetError::Io {
        path: output_dir.to_path_buf(),
        source,
    })?;

    write_jsonl(&train_path, &train)?;
    write_jsonl(&validation_path, &validation)?;

    let deterministic = split_source == SplitSource::Deterministic;

    let manifest = build_manifest(
        &train,
        &validation,
        split_source.as_str(),
        deterministic.then_some(options.seed),
        deterministic.then_some(options.validation_fraction),
    );

    let mut manifest_bytes =
        serde_json::to_vec_pretty(&manifest).map_err(|error| DatasetError::Serialize {
            message: error.to_string(),
        })?;

    manifest_bytes.push(b'\n');

    fs::write(&manifest_path, manifest_bytes).map_err(|source| DatasetError::Io {
        path: manifest_path.clone(),
        source,
    })?;

    Ok(PrepareReport {
        output_dir: output_dir.to_path_buf(),
        train_path,
        validation_path,
        manifest_path,
        train_records: train.len(),
        validation_records: validation.len(),
        split_source,
    })
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;

    fn record(id: &str, split: Option<&str>) -> ConversationExample {
        ConversationExample {
            id: id.to_owned(),
            messages: vec![
                ConversationMessage {
                    role: "user".to_owned(),
                    content: "Hello, Mikuru!".to_owned(),
                },
                ConversationMessage {
                    role: "assistant".to_owned(),
                    content: "Hello! What shall we explore today?".to_owned(),
                },
            ],
            language: "en".to_owned(),
            category: "general_conversation".to_owned(),
            source_dataset: "Mikuru Test".to_owned(),
            source_record_id: id.to_owned(),
            source_url: "N/A".to_owned(),
            license: "Project Internal".to_owned(),
            synthetic: true,
            split: split.map(str::to_owned),
            references: vec!["https://example.invalid/reference".to_owned()],
            source_message_ids: Vec::new(),
            provenance_note: "Synthetic unit-test record.".to_owned(),
        }
    }

    fn temp_dir(name: &str) -> PathBuf {
        static COUNTER: AtomicUsize = AtomicUsize::new(0);

        let index = COUNTER.fetch_add(1, Ordering::Relaxed);

        let dir = std::env::temp_dir().join(format!(
            "mikuru-dataset-test-{}-{index}-{name}",
            std::process::id()
        ));

        fs::create_dir_all(&dir).expect("temporary directory should be created");

        dir
    }

    fn write_records(dir: &Path, name: &str, records: &[ConversationExample]) -> PathBuf {
        let path = dir.join(name);
        write_jsonl(&path, records).expect("fixture records should be written");
        path
    }

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

    #[test]
    fn loads_records_with_explicit_split_labels() {
        let dir = temp_dir("loads-splits");
        let path = write_records(
            &dir,
            "dataset.jsonl",
            &[record("a", Some("train")), record("b", Some("validation"))],
        );

        let records = load_jsonl(&path).expect("dataset should load");

        assert_eq!(records.len(), 2);
        assert_eq!(records[0].split.as_deref(), Some("train"));
        assert_eq!(records[1].split.as_deref(), Some("validation"));

        fs::remove_dir_all(dir).expect("temporary directory should be removed");
    }

    #[test]
    fn rejects_duplicate_ids_across_files() {
        let dir = temp_dir("duplicates");
        let first = write_records(&dir, "first.jsonl", &[record("same-id", Some("train"))]);
        let second = write_records(
            &dir,
            "second.jsonl",
            &[record("same-id", Some("validation"))],
        );

        let error = load_jsonl_files(&[first, second]).expect_err("duplicate ids must be rejected");

        assert!(
            error.to_string().contains("same-id"),
            "error should name the duplicate id: {error}"
        );

        fs::remove_dir_all(dir).expect("temporary directory should be removed");
    }

    #[test]
    fn rejects_malformed_json_line() {
        let dir = temp_dir("malformed");
        let path = dir.join("broken.jsonl");

        fs::write(&path, "{\"id\":\"broken\",\"messages\":[\n").expect("fixture should be written");

        let error = load_jsonl(&path).expect_err("malformed JSON must be rejected");

        assert!(
            error.to_string().contains("line 1"),
            "error should identify the line: {error}"
        );

        fs::remove_dir_all(dir).expect("temporary directory should be removed");
    }

    #[test]
    fn rejects_missing_required_metadata() {
        let dir = temp_dir("missing-license");
        let mut invalid = record("missing-license", Some("train"));
        invalid.license = "   ".to_owned();

        let path = write_records(&dir, "dataset.jsonl", &[invalid]);

        let error = load_jsonl(&path).expect_err("missing license must be rejected");

        assert!(
            error.to_string().contains("license"),
            "error should name the field: {error}"
        );

        fs::remove_dir_all(dir).expect("temporary directory should be removed");
    }

    #[test]
    fn rejects_unknown_split_label() {
        let dir = temp_dir("unknown-split");
        let path = write_records(&dir, "dataset.jsonl", &[record("a", Some("test"))]);

        let error = load_jsonl(&path).expect_err("unknown split labels must be rejected");

        assert!(
            error.to_string().contains("unknown split"),
            "error should explain the label: {error}"
        );

        fs::remove_dir_all(dir).expect("temporary directory should be removed");
    }

    #[test]
    fn accepts_legacy_demo_split_on_load() {
        let dir = temp_dir("demo-split");
        let path = write_records(&dir, "dataset.jsonl", &[record("demo-1", Some("demo"))]);

        let records = load_jsonl(&path).expect("legacy demo labels should load");

        assert_eq!(records.len(), 1);
        assert_eq!(records[0].split.as_deref(), Some("demo"));

        fs::remove_dir_all(dir).expect("temporary directory should be removed");
    }

    #[test]
    fn partition_rejects_demo_records() {
        let records = [
            record("demo-1", Some("demo")),
            record("train-1", Some("train")),
        ];

        let error =
            partition_by_split(&records).expect_err("demo records have no training assignment");

        assert!(
            error.to_string().contains("demo"),
            "error should name the legacy label: {error}"
        );
    }

    #[test]
    fn rejects_mismatched_source_message_ids() {
        let dir = temp_dir("source-message-ids");
        let mut invalid = record("mismatch", Some("train"));
        invalid.source_message_ids = vec!["only-one-id".to_owned()];

        let path = write_records(&dir, "dataset.jsonl", &[invalid]);

        let error = load_jsonl(&path).expect_err("length mismatch must be rejected");

        assert!(
            error.to_string().contains("source_message_ids"),
            "error should name the field: {error}"
        );

        fs::remove_dir_all(dir).expect("temporary directory should be removed");
    }

    #[test]
    fn accepts_matching_source_message_ids() {
        let dir = temp_dir("source-message-ids-ok");
        let mut valid = record("matching", Some("train"));
        valid.source_message_ids = vec!["msg-a".to_owned(), "msg-b".to_owned()];

        let path = write_records(&dir, "dataset.jsonl", &[valid]);

        let records = load_jsonl(&path).expect("matching ids should load");

        assert_eq!(records[0].source_message_ids, ["msg-a", "msg-b"]);

        fs::remove_dir_all(dir).expect("temporary directory should be removed");
    }

    #[test]
    fn prepare_dataset_relabels_demo_records() {
        let dir = temp_dir("prepare-demo");
        let records = [
            record("demo-1", Some("demo")),
            record("demo-2", Some("demo")),
            record("demo-3", Some("demo")),
        ];

        let input = write_records(&dir, "input.jsonl", &records);
        let output = dir.join("prepared");

        let report = prepare_dataset(&input, &output, &PrepareOptions::default())
            .expect("demo records should prepare deterministically");

        assert_eq!(report.split_source, SplitSource::Deterministic);

        let train = load_jsonl(&report.train_path).expect("train split should load");
        let validation = load_jsonl(&report.validation_path).expect("validation split should load");

        assert!(
            train
                .iter()
                .all(|record| record.split.as_deref() == Some("train"))
        );
        assert!(
            validation
                .iter()
                .all(|record| record.split.as_deref() == Some("validation"))
        );

        fs::remove_dir_all(dir).expect("temporary directory should be removed");
    }

    #[test]
    fn partition_separates_explicit_splits() {
        let records = [
            record("train-1", Some("train")),
            record("validation-1", Some("validation")),
            record("train-2", Some("train")),
        ];

        let (train, validation) =
            partition_by_split(&records).expect("explicit splits should partition");

        assert_eq!(
            train
                .iter()
                .map(|record| record.id.as_str())
                .collect::<Vec<_>>(),
            ["train-1", "train-2"]
        );
        assert_eq!(
            validation
                .iter()
                .map(|record| record.id.as_str())
                .collect::<Vec<_>>(),
            ["validation-1"]
        );
    }

    #[test]
    fn partition_rejects_partial_split_metadata() {
        let records = [record("train-1", Some("train")), record("unlabeled", None)];

        let error = partition_by_split(&records).expect_err("partial labels must be rejected");

        assert!(
            error.to_string().contains("label every record"),
            "error should explain the fix: {error}"
        );
    }

    #[test]
    fn deterministic_split_is_reproducible_and_complete() {
        let records = (0..12)
            .map(|index| record(&format!("record-{index:02}"), None))
            .collect::<Vec<_>>();

        let (train_a, validation_a) =
            deterministic_split(&records, 0.25, 7).expect("split should succeed");
        let (train_b, validation_b) =
            deterministic_split(&records, 0.25, 7).expect("split should succeed");

        let ids = |records: &[ConversationExample]| {
            records
                .iter()
                .map(|record| record.id.clone())
                .collect::<Vec<_>>()
        };

        assert_eq!(ids(&train_a), ids(&train_b));
        assert_eq!(ids(&validation_a), ids(&validation_b));
        assert_eq!(train_a.len() + validation_a.len(), records.len());
        assert_eq!(validation_a.len(), 3);

        let mut all_ids = ids(&train_a);
        all_ids.extend(ids(&validation_a));
        all_ids.sort();
        all_ids.dedup();

        assert_eq!(all_ids.len(), records.len(), "no record may be dropped");
    }

    #[test]
    fn deterministic_split_keeps_both_splits_non_empty() {
        let records = [record("a", None), record("b", None)];

        let (train, validation) =
            deterministic_split(&records, 0.01, 42).expect("split should succeed");

        assert_eq!(train.len(), 1);
        assert_eq!(validation.len(), 1);
    }

    #[test]
    fn deterministic_split_rejects_duplicate_ids() {
        let records = [record("same", None), record("same", None)];

        let error =
            deterministic_split(&records, 0.5, 42).expect_err("duplicates must be rejected");

        assert!(
            error.to_string().contains("same"),
            "error should name the duplicate id: {error}"
        );
    }

    #[test]
    fn write_jsonl_preserves_provenance() {
        let dir = temp_dir("round-trip");
        let records = [record("round-trip", Some("train"))];
        let path = write_records(&dir, "dataset.jsonl", &records);

        let reloaded = load_jsonl(&path).expect("written dataset should load");

        assert_eq!(records.to_vec(), reloaded);

        let contents = fs::read_to_string(&path).expect("dataset should be readable");

        assert!(contents.ends_with('\n'));
        assert_eq!(contents.lines().count(), 1);

        fs::remove_dir_all(dir).expect("temporary directory should be removed");
    }

    #[test]
    fn prepare_dataset_writes_splits_and_manifest() {
        let dir = temp_dir("prepare");
        let records = [
            record("a", None),
            record("b", None),
            record("c", None),
            record("d", None),
        ];

        let input = write_records(&dir, "input.jsonl", &records);
        let output = dir.join("prepared");

        let report = prepare_dataset(
            &input,
            &output,
            &PrepareOptions {
                validation_fraction: 0.25,
                seed: 42,
            },
        )
        .expect("dataset should prepare");

        assert_eq!(report.train_records, 3);
        assert_eq!(report.validation_records, 1);
        assert_eq!(report.split_source, SplitSource::Deterministic);

        let train = load_jsonl(&report.train_path).expect("train split should load");
        let validation = load_jsonl(&report.validation_path).expect("validation split should load");

        assert!(
            train
                .iter()
                .all(|record| record.split.as_deref() == Some("train"))
        );
        assert!(
            validation
                .iter()
                .all(|record| record.split.as_deref() == Some("validation"))
        );

        let manifest: DatasetManifest = serde_json::from_slice(
            &fs::read(&report.manifest_path).expect("manifest should be readable"),
        )
        .expect("manifest should parse");

        assert_eq!(manifest.train_records, 3);
        assert_eq!(manifest.validation_records, 1);
        assert_eq!(manifest.split_source, "deterministic");
        assert_eq!(manifest.split_seed, Some(42));
        assert_eq!(manifest.sources.len(), 1);
        assert_eq!(manifest.sources[0].records, 4);
        assert_eq!(manifest.sources[0].synthetic_records, 4);

        fs::remove_dir_all(dir).expect("temporary directory should be removed");
    }

    #[test]
    fn prepare_dataset_respects_explicit_splits() {
        let dir = temp_dir("prepare-explicit");
        let records = [
            record("train-1", Some("train")),
            record("train-2", Some("train")),
            record("validation-1", Some("validation")),
        ];

        let input = write_records(&dir, "input.jsonl", &records);
        let output = dir.join("prepared");

        let report = prepare_dataset(&input, &output, &PrepareOptions::default())
            .expect("explicit splits should prepare");

        assert_eq!(report.split_source, SplitSource::Explicit);
        assert_eq!(report.train_records, 2);
        assert_eq!(report.validation_records, 1);

        fs::remove_dir_all(dir).expect("temporary directory should be removed");
    }

    #[test]
    fn prepare_dataset_refuses_to_overwrite_output() {
        let dir = temp_dir("prepare-overwrite");
        let records = [record("a", None), record("b", None)];
        let input = write_records(&dir, "input.jsonl", &records);
        let output = dir.join("prepared");

        prepare_dataset(&input, &output, &PrepareOptions::default())
            .expect("first preparation should succeed");

        let error = prepare_dataset(&input, &output, &PrepareOptions::default())
            .expect_err("second preparation must refuse to overwrite");

        assert!(
            error.to_string().contains("already exists"),
            "error should explain the refusal: {error}"
        );

        fs::remove_dir_all(dir).expect("temporary directory should be removed");
    }
}
