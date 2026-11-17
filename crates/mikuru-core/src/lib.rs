//! Shared, framework-independent types for Mikuru LM.

pub mod dataset;
pub mod oasst1;
pub mod taskmaster1;

pub use dataset::{
    ConversationExample, ConversationMessage, DEMO_SPLIT, DatasetError, DatasetManifest,
    DatasetSplit, PrepareOptions, PrepareReport, SourceSummary, SplitSource, TRAIN_SPLIT,
    TextExample, VALIDATION_SPLIT, build_manifest, deterministic_split, ensure_unique_ids,
    is_unassigned_split, load_jsonl, load_jsonl_files, partition_by_split, prepare_dataset,
    write_jsonl,
};

pub use oasst1::{
    MessageReject, OASST1_DATASET_URL, OASST1_LICENSE, OASST1_PAPER_URL, OASST1_REVISION,
    OASST1_SOURCE_DATASET, OasstImportConfig, OasstImportError, OasstImportReport,
    OasstImportStats, OasstLabel, OasstMessage, OasstToxicity, Reconstruction, TreeReject,
    check_message, import_oasst1_messages, reconstruct_tree,
};

pub use taskmaster1::{
    LENGTH_BUDGETS, MANIFEST_FILE, REPORT_JSON_FILE, REPORT_MARKDOWN_FILE, TASKMASTER1_LICENSE,
    TASKMASTER1_LICENSE_URL, TASKMASTER1_PAPER_URL, TASKMASTER1_REPOSITORY_URL,
    TASKMASTER1_REVISION, TASKMASTER1_SOURCE_DATASET, TASKMASTER1_SUBSET_DIRECTORY,
    Taskmaster1Error, Taskmaster1InspectConfig, Taskmaster1InspectResult, Taskmaster1ManifestEntry,
    Taskmaster1Report, Taskmaster1Subset, inspect_taskmaster1,
};
