use std::{collections::BTreeMap, env, error::Error, path::Path, process::ExitCode};

use mikuru_core::dataset::{
    DEMO_SPLIT, PrepareOptions, TRAIN_SPLIT, VALIDATION_SPLIT, load_jsonl, prepare_dataset,
};
use mikuru_core::oasst1::OasstImportConfig;
use mikuru_core::taskmaster1::Taskmaster1InspectConfig;
use mikuru_trainer::{TrainerConfig, train_jsonl_with_validation};

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("error: {error}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), Box<dyn Error + Send + Sync>> {
    let mut args = env::args().skip(1);

    match args.next().as_deref() {
        Some("burn-smoke") => {
            mikuru_backend::run_smoke_test().map_err(std::io::Error::other)?;
        }

        Some("inspect-jsonl") => {
            let path = args
                .next()
                .ok_or("missing file path; usage: mikuru-lm inspect-jsonl <path>")?;

            inspect_jsonl(&path)?;
        }

        Some("prepare-dataset") => {
            let input = args.next().ok_or("missing input dataset path")?;
            let output_dir = args.next().ok_or("missing output directory")?;

            let options = parse_prepare_options(args)?;

            let report = prepare_dataset(&input, &output_dir, &options)?;

            println!("Prepared dataset: {}", report.output_dir.display());
            println!("Split source: {}", report.split_source.as_str());
            println!("Training records: {}", report.train_records);
            println!("Validation records: {}", report.validation_records);
            println!("Train file: {}", report.train_path.display());
            println!("Validation file: {}", report.validation_path.display());
            println!("Manifest: {}", report.manifest_path.display());
        }

        Some("import-oasst1") => {
            let input = args.next().ok_or("missing OASST1 messages JSONL path")?;
            let output = args.next().ok_or("missing output JSONL path")?;

            let config = parse_oasst_import_options(args)?;

            let report = mikuru_core::oasst1::import_oasst1_messages(&input, &output, &config)?;
            let stats = &report.stats;

            println!("Imported OASST1 records: {}", report.output_path.display());
            println!("Messages read: {}", stats.messages_read);
            println!("Trees read: {}", stats.trees_read);
            println!("Messages passing filters: {}", stats.messages_accepted);
            println!("Conversations written: {}", stats.conversations_written);
            println!(
                "Messages inside conversations: {}",
                stats.messages_in_conversations
            );
            println!("Total characters: {}", stats.total_characters);
            println!(
                "Longest conversation: {} chars, {} messages",
                stats.max_conversation_chars, stats.max_conversation_messages
            );

            println!("Message rejections:");

            if stats.message_rejections.is_empty() {
                println!("  none");
            }

            for (reason, count) in &stats.message_rejections {
                println!("  {}: {count}", reason.as_str());
            }

            println!("Tree rejections:");

            if stats.tree_rejections.is_empty() {
                println!("  none");
            }

            for (reason, count) in &stats.tree_rejections {
                println!("  {}: {count}", reason.as_str());
            }
        }

        Some("inspect-taskmaster1") => {
            let input = args.next().ok_or("missing Taskmaster-1 directory")?;
            let output_dir = args.next().ok_or("missing output directory")?;
            let config = parse_taskmaster1_options(args)?;

            let result =
                mikuru_core::taskmaster1::inspect_taskmaster1(&input, &output_dir, &config)?;

            println!("Inspected Taskmaster-1: {}", result.output_dir.display());
            println!("Markdown report: {}", result.markdown_path.display());
            println!("JSON report: {}", result.json_path.display());
            println!("Candidate manifest: {}", result.manifest_path.display());
            println!("Records read: {}", result.report.totals.records_read);
            println!(
                "Malformed records: {}",
                result.report.totals.malformed_records
            );

            for (name, subset) in &result.report.subsets {
                println!(
                    "{name}: {} conversations, {} messages, {}",
                    subset.conversations_read, subset.messages, subset.qualification.status
                );
            }
        }

        Some("train") => {
            let dataset = args.next().ok_or("missing dataset path")?;

            let tokenizer = args.next().ok_or("missing tokenizer path")?;

            let checkpoint = args.next().ok_or("missing checkpoint base path")?;

            let options = parse_train_options(args)?;

            let mut config = TrainerConfig::default();

            if let Some(epochs) = options.epochs {
                config.epochs = epochs;
            }

            if let Some(max_seq_len) = options.max_seq_len {
                config.max_seq_len = max_seq_len;
            }

            let validation = options.validation.as_deref().map(Path::new);

            train_jsonl_with_validation(dataset, validation, tokenizer, checkpoint, &config)?;
        }

        Some("verify-checkpoint") => {
            let dataset = args.next().ok_or("missing dataset path")?;
            let tokenizer = args.next().ok_or("missing tokenizer path")?;
            let checkpoint = args.next().ok_or("missing checkpoint base path")?;

            let expected_loss = args.next().map(|value| value.parse::<f32>()).transpose()?;

            if args.next().is_some() {
                return Err("too many arguments for verify-checkpoint".into());
            }

            mikuru_trainer::verify_checkpoint(dataset, tokenizer, checkpoint, expected_loss)?;
        }

        Some("generate") => {
            let tokenizer = args.next().ok_or("missing tokenizer path")?;
            let checkpoint = args.next().ok_or("missing checkpoint base path")?;

            let max_new_tokens = args
                .next()
                .ok_or("missing maximum new token count")?
                .parse::<usize>()?;

            let remaining = args.collect::<Vec<_>>();

            let verbose = remaining
                .first()
                .is_some_and(|argument| argument == "--verbose");

            let prompt_parts = if verbose {
                &remaining[1..]
            } else {
                &remaining[..]
            };

            let prompt = prompt_parts.join(" ");

            if prompt.trim().is_empty() {
                return Err("missing user message".into());
            }

            let generated = mikuru_trainer::generate_from_checkpoint(
                tokenizer,
                checkpoint,
                &prompt,
                max_new_tokens,
                verbose,
            )?;

            println!("Generated continuation:");
            println!("{generated}");
        }

        Some("help") | Some("--help") | Some("-h") | None => print_help(),

        Some(command) => {
            return Err(format!("unknown command `{command}`; run `mikuru-lm help`").into());
        }
    }

    Ok(())
}

#[derive(Debug, Default, PartialEq, Eq)]
struct TrainOptions {
    epochs: Option<usize>,
    validation: Option<String>,
    max_seq_len: Option<usize>,
}

/// Parses the optional arguments that follow the three required train paths.
fn parse_train_options<I>(args: I) -> Result<TrainOptions, Box<dyn Error + Send + Sync>>
where
    I: Iterator<Item = String>,
{
    let mut options = TrainOptions::default();
    let mut args = args;

    while let Some(argument) = args.next() {
        match argument.as_str() {
            "--validation" => {
                let value = args.next().ok_or("missing value for --validation")?;

                if options.validation.is_some() {
                    return Err("--validation may only be supplied once".into());
                }

                options.validation = Some(value);
            }

            "--max-seq-len" => {
                let value = args.next().ok_or("missing value for --max-seq-len")?;

                if options.max_seq_len.is_some() {
                    return Err("--max-seq-len may only be supplied once".into());
                }

                let parsed = value.parse::<usize>()?;

                if parsed == 0 {
                    return Err("--max-seq-len must be greater than zero".into());
                }

                options.max_seq_len = Some(parsed);
            }

            value if value.starts_with("--") => {
                return Err(format!("unknown train option `{value}`").into());
            }

            value => {
                if options.epochs.is_some() {
                    return Err(format!("unexpected extra argument `{value}` for train").into());
                }

                options.epochs = Some(value.parse::<usize>()?);
            }
        }
    }

    Ok(options)
}

/// Parses `import-oasst1` options after the two required paths.
fn parse_oasst_import_options<I>(args: I) -> Result<OasstImportConfig, Box<dyn Error + Send + Sync>>
where
    I: Iterator<Item = String>,
{
    let mut config = OasstImportConfig::default();
    let mut args = args;

    while let Some(argument) = args.next() {
        match argument.as_str() {
            "--max-chars" => {
                let value = args.next().ok_or("missing value for --max-chars")?;

                config.max_chars = value.parse::<usize>()?;

                if config.max_chars == 0 {
                    return Err("--max-chars must be greater than zero".into());
                }
            }

            "--max-messages" => {
                let value = args.next().ok_or("missing value for --max-messages")?;

                config.max_messages = value.parse::<usize>()?;

                if config.max_messages < 2 {
                    return Err("--max-messages must be at least two".into());
                }
            }

            "--revision" => {
                config.revision = args.next().ok_or("missing value for --revision")?;

                if config.revision.trim().is_empty() {
                    return Err("--revision must not be empty".into());
                }
            }

            value => {
                return Err(format!("unknown import-oasst1 option `{value}`").into());
            }
        }
    }

    Ok(config)
}

/// Parses `inspect-taskmaster1` options after the two required paths.
fn parse_taskmaster1_options<I>(
    args: I,
) -> Result<Taskmaster1InspectConfig, Box<dyn Error + Send + Sync>>
where
    I: Iterator<Item = String>,
{
    let mut config = Taskmaster1InspectConfig::default();
    let mut args = args;

    while let Some(argument) = args.next() {
        match argument.as_str() {
            "--revision" => {
                let revision = args.next().ok_or("missing value for --revision")?;

                if revision.trim().is_empty() {
                    return Err("--revision must not be empty".into());
                }

                config.revision = revision;
            }

            value => {
                return Err(format!("unknown inspect-taskmaster1 option `{value}`").into());
            }
        }
    }

    Ok(config)
}

/// Parses `prepare-dataset` options after the two required paths.
fn parse_prepare_options<I>(args: I) -> Result<PrepareOptions, Box<dyn Error + Send + Sync>>
where
    I: Iterator<Item = String>,
{
    let mut options = PrepareOptions::default();
    let mut args = args;

    while let Some(argument) = args.next() {
        match argument.as_str() {
            "--validation-fraction" => {
                let value = args
                    .next()
                    .ok_or("missing value for --validation-fraction")?;

                options.validation_fraction = value.parse::<f32>()?;
            }

            "--seed" => {
                let value = args.next().ok_or("missing value for --seed")?;

                options.seed = value.parse::<u64>()?;
            }

            value => {
                return Err(format!("unknown prepare-dataset option `{value}`").into());
            }
        }
    }

    Ok(options)
}

fn inspect_jsonl(path: &str) -> Result<(), Box<dyn Error + Send + Sync>> {
    let records = load_jsonl(path)?;

    let mut train = 0usize;
    let mut validation = 0usize;
    let mut demo = 0usize;
    let mut unlabeled = 0usize;
    let mut total_messages = 0usize;
    let mut total_characters = 0usize;

    let mut sources: BTreeMap<(String, String), usize> = BTreeMap::new();

    for record in &records {
        match record.split.as_deref() {
            Some(TRAIN_SPLIT) => train += 1,
            Some(VALIDATION_SPLIT) => validation += 1,
            Some(DEMO_SPLIT) => demo += 1,
            Some(_) => {
                return Err(format!("record `{}` has an unexpected split label", record.id).into());
            }
            None => unlabeled += 1,
        }

        total_messages += record.messages.len();

        total_characters += record
            .messages
            .iter()
            .map(|message| message.content.chars().count())
            .sum::<usize>();

        *sources
            .entry((record.source_dataset.clone(), record.license.clone()))
            .or_insert(0) += 1;
    }

    println!("File: {path}");
    println!("Valid conversations: {}", records.len());
    println!("Training records: {train}");
    println!("Validation records: {validation}");
    println!("Demo records: {demo}");
    println!("Unlabeled records: {unlabeled}");
    println!("Total messages: {total_messages}");
    println!("Total message characters: {total_characters}");

    println!("Sources:");

    for ((source, license), count) in sources {
        println!("  {source} | license: {license} | records: {count}");
    }

    Ok(())
}

fn print_help() {
    println!(
        "Mikuru LM\n\n\
         Commands:\n\
           burn-smoke                                  Verify Burn Flex + autodiff\n\
           inspect-jsonl <path>                        Validate conversation JSONL\n\
           prepare-dataset <input.jsonl> <output_dir> [--validation-fraction F] [--seed N]\n\
                                                       Write train/validation splits + manifest\n\
           import-oasst1 <messages.jsonl> <output.jsonl> [--max-chars N] [--max-messages N]\n\
                                                       Convert OASST1 message trees into conversations\n\
           inspect-taskmaster1 <dataset_dir> <output_dir> [--revision SHA]\n\
                                                       Inspect Taskmaster-1 and write qualification reports\n\
           train <dataset|prepared_dir> <tokenizer> <checkpoint> [epochs]\n\
                 [--validation <path>] [--max-seq-len N]\n\
                                                       Train the autoregressive LM\n\
           verify-checkpoint <dataset> <tokenizer> <checkpoint> [expected_loss]\n\
                                                       Reload and verify model weights\n\
           generate <tokenizer> <checkpoint> <max_tokens> [--verbose] <user_message>\n\
                                                       Generate text from a saved model\n\
           help                                        Show this help"
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(values: &[&str]) -> std::vec::IntoIter<String> {
        values
            .iter()
            .map(|value| (*value).to_owned())
            .collect::<Vec<_>>()
            .into_iter()
    }

    #[test]
    fn train_options_accept_epochs_and_validation_path() {
        let parsed = parse_train_options(args(&["25", "--validation", "data/val.jsonl"]))
            .expect("options should parse");

        assert_eq!(
            parsed,
            TrainOptions {
                epochs: Some(25),
                validation: Some("data/val.jsonl".to_owned()),
                max_seq_len: None,
            }
        );
    }

    #[test]
    fn train_options_accept_max_seq_len() {
        let parsed = parse_train_options(args(&["--max-seq-len", "256"]))
            .expect("max sequence length should parse");

        assert_eq!(parsed.max_seq_len, Some(256));
    }

    #[test]
    fn train_options_reject_zero_max_seq_len() {
        let error = parse_train_options(args(&["--max-seq-len", "0"]))
            .expect_err("zero max sequence length must be rejected");

        assert!(error.to_string().contains("greater than zero"));
    }

    #[test]
    fn train_options_reject_unknown_flags() {
        let error =
            parse_train_options(args(&["--bogus"])).expect_err("unknown flags must be rejected");

        assert!(error.to_string().contains("unknown train option"));
    }

    #[test]
    fn train_options_require_validation_path() {
        let error = parse_train_options(args(&["--validation"]))
            .expect_err("a missing validation path must be rejected");

        assert!(error.to_string().contains("missing value"));
    }

    #[test]
    fn prepare_options_use_defaults() {
        let parsed = parse_prepare_options(args(&[])).expect("defaults should parse");

        assert_eq!(parsed.validation_fraction, 0.2);
        assert_eq!(parsed.seed, 42);
    }

    #[test]
    fn prepare_options_accept_overrides() {
        let parsed = parse_prepare_options(args(&["--validation-fraction", "0.25", "--seed", "7"]))
            .expect("overrides should parse");

        assert_eq!(parsed.validation_fraction, 0.25);
        assert_eq!(parsed.seed, 7);
    }

    #[test]
    fn oasst_import_options_use_documented_defaults() {
        let parsed = parse_oasst_import_options(args(&[])).expect("defaults should parse");

        assert_eq!(parsed.max_chars, 500);
        assert_eq!(parsed.max_messages, 8);
        assert_eq!(parsed.min_messages, 2);
        assert_eq!(parsed.revision, mikuru_core::OASST1_REVISION);
    }

    #[test]
    fn oasst_import_options_accept_overrides() {
        let parsed = parse_oasst_import_options(args(&[
            "--max-chars",
            "400",
            "--max-messages",
            "6",
            "--revision",
            "abc123",
        ]))
        .expect("overrides should parse");

        assert_eq!(parsed.max_chars, 400);
        assert_eq!(parsed.max_messages, 6);
        assert_eq!(parsed.revision, "abc123");
    }

    #[test]
    fn oasst_import_options_reject_unknown_flags() {
        let error = parse_oasst_import_options(args(&["--bogus"]))
            .expect_err("unknown flags must be rejected");

        assert!(error.to_string().contains("unknown import-oasst1 option"));
    }

    #[test]
    fn taskmaster1_options_use_pinned_revision_by_default() {
        let parsed = parse_taskmaster1_options(args(&[])).expect("defaults should parse");

        assert_eq!(parsed.revision, mikuru_core::TASKMASTER1_REVISION);
    }

    #[test]
    fn taskmaster1_options_accept_revision_override() {
        let parsed = parse_taskmaster1_options(args(&["--revision", "abc123"]))
            .expect("revision override should parse");

        assert_eq!(parsed.revision, "abc123");
    }

    #[test]
    fn taskmaster1_options_reject_unknown_flags_and_empty_revision() {
        let error = parse_taskmaster1_options(args(&["--bogus"]))
            .expect_err("unknown flags must be rejected");
        assert!(
            error
                .to_string()
                .contains("unknown inspect-taskmaster1 option")
        );

        let error = parse_taskmaster1_options(args(&["--revision", " "]))
            .expect_err("empty revisions must be rejected");
        assert!(error.to_string().contains("must not be empty"));
    }
}
