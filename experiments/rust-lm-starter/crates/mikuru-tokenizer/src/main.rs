use std::{
    env,
    error::Error,
    fs::{File, create_dir_all},
    io::{self, BufRead, BufReader, Lines},
    path::{Path, PathBuf},
    process,
};

use mikuru_core::{ConversationExample, dataset::DatasetSplit};

use mikuru_tokenizer::{
    BOS_TOKEN, BpeTrainingConfig, EOS_TOKEN, MikuruTokenizer, PAD_TOKEN, TokenizerResult,
    UNK_TOKEN, format_conversation,
};

fn main() {
    if let Err(error) = run() {
        eprintln!("Error: {error}");
        process::exit(1);
    }
}

fn run() -> TokenizerResult<()> {
    let mut args = env::args().skip(1);

    let Some(command) = args.next() else {
        print_usage();
        return Ok(());
    };

    match command.as_str() {
        "train" => {
            let dataset_path = PathBuf::from(required_arg(&mut args, "dataset.jsonl")?);

            let output_path = PathBuf::from(required_arg(&mut args, "output.json")?);

            let vocab_size = match args.next() {
                Some(value) => value.parse::<usize>()?,
                None => BpeTrainingConfig::default().vocab_size,
            };

            if args.next().is_some() {
                return Err(input_error("Too many arguments for train"));
            }

            if output_path.exists() {
                return Err(input_error(format!(
                    "Output already exists: {}. Choose another filename.",
                    output_path.display()
                )));
            }

            if let Some(parent) = output_path.parent()
                && !parent.as_os_str().is_empty()
            {
                create_dir_all(parent)?;
            }

            let config = BpeTrainingConfig {
                vocab_size,
                ..BpeTrainingConfig::default()
            };

            let mut corpus = JsonlTextIter::open(&dataset_path)?;

            let tokenizer = MikuruTokenizer::train_from_iterator(corpus.by_ref(), &config)?;

            if let Some(error) = corpus.take_error() {
                return Err(input_error(error));
            }

            if corpus.records_read() == 0 {
                return Err(input_error(
                    "The dataset contains no non-empty text records",
                ));
            }

            tokenizer.save(&output_path)?;

            println!("BPE training completed.");
            println!("Training records: {}", corpus.records_read());
            println!(
                "Validation records excluded: {}",
                corpus.validation_skipped()
            );
            println!("Vocabulary size: {}", tokenizer.vocab_size());
            println!("Tokenizer saved: {}", output_path.display());

            println!("\nSpecial token IDs:");

            for token in [UNK_TOKEN, BOS_TOKEN, EOS_TOKEN, PAD_TOKEN] {
                println!("{token}: {:?}", tokenizer.token_id(token));
            }
        }

        "encode" => {
            let tokenizer_path = required_arg(&mut args, "tokenizer.json")?;

            let text = args.collect::<Vec<_>>().join(" ");

            if text.is_empty() {
                return Err(input_error("Text to encode is required"));
            }

            let tokenizer = MikuruTokenizer::from_file(tokenizer_path)?;

            println!("Tokens: {:?}", tokenizer.encode_tokens(&text)?);
            println!("IDs: {:?}", tokenizer.encode(&text)?);
        }

        "decode" => {
            let tokenizer_path = required_arg(&mut args, "tokenizer.json")?;

            let ids = args
                .map(|value| value.parse::<u32>())
                .collect::<Result<Vec<_>, _>>()?;

            if ids.is_empty() {
                return Err(input_error("At least one token ID is required"));
            }

            let tokenizer = MikuruTokenizer::from_file(tokenizer_path)?;

            println!("{}", tokenizer.decode(&ids)?);
        }

        _ => print_usage(),
    }

    Ok(())
}

fn required_arg<I>(args: &mut I, description: &str) -> TokenizerResult<String>
where
    I: Iterator<Item = String>,
{
    args.next()
        .ok_or_else(|| input_error(format!("Missing argument: {description}")))
}

fn input_error(message: impl Into<String>) -> Box<dyn Error + Send + Sync> {
    io::Error::new(io::ErrorKind::InvalidInput, message.into()).into()
}

/// Reads one JSONL record at a time instead of keeping the whole corpus
/// in memory.
struct JsonlTextIter {
    lines: Lines<BufReader<File>>,
    line_number: usize,
    records_read: usize,
    validation_skipped: usize,
    error: Option<String>,
}

impl JsonlTextIter {
    fn open(path: &Path) -> TokenizerResult<Self> {
        let reader = BufReader::new(File::open(path)?);

        Ok(Self {
            lines: reader.lines(),
            line_number: 0,
            records_read: 0,
            validation_skipped: 0,
            error: None,
        })
    }

    fn records_read(&self) -> usize {
        self.records_read
    }

    fn validation_skipped(&self) -> usize {
        self.validation_skipped
    }

    fn take_error(&mut self) -> Option<String> {
        self.error.take()
    }
}

impl Iterator for JsonlTextIter {
    type Item = String;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let next_line = self.lines.next()?;

            self.line_number += 1;

            let line = match next_line {
                Ok(line) => line,
                Err(error) => {
                    self.error = Some(format!(
                        "Failed reading JSONL line {}: {}",
                        self.line_number, error
                    ));
                    return None;
                }
            };

            if line.trim().is_empty() {
                continue;
            }

            let example: ConversationExample = match serde_json::from_str(&line) {
                Ok(example) => example,
                Err(error) => {
                    self.error = Some(format!(
                        "Invalid JSONL record on line {}: {}",
                        self.line_number, error
                    ));
                    return None;
                }
            };

            if example.messages.is_empty()
                || example.messages.iter().any(|message| {
                    message.role.trim().is_empty() || message.content.trim().is_empty()
                })
            {
                self.error = Some(format!(
                    "Invalid conversation on line {}: empty messages are not allowed",
                    self.line_number
                ));
                return None;
            }

            // Validation records must never influence the BPE vocabulary.
            if let Some(raw_split) = example.split.as_deref() {
                match DatasetSplit::parse(raw_split) {
                    Ok(DatasetSplit::Validation) => {
                        self.validation_skipped += 1;
                        continue;
                    }
                    // Train and legacy demo records are usable training text.
                    Ok(DatasetSplit::Train | DatasetSplit::Demo) => {}
                    Err(message) => {
                        self.error = Some(format!(
                            "Invalid split on line {}: {message}",
                            self.line_number
                        ));
                        return None;
                    }
                }
            }

            self.records_read += 1;

            return Some(format_conversation(&example));
        }
    }
}

fn print_usage() {
    println!(
        "\
Mikuru Tokenizer

Commands:

  train <dataset.jsonl> <output.json> [vocab_size]
  encode <tokenizer.json> <text>
  decode <tokenizer.json> <id> [id...]

Records labelled with split \"validation\" are excluded from tokenizer training.
Train on a prepared train.jsonl, or pass a combined file and let the tool filter.

Default vocabulary target: 8192
Default minimum pair frequency: 2
"
    );
}

#[cfg(test)]
mod cli_tests {
    use super::*;

    #[test]
    fn jsonl_iterator_formats_conversation_records() {
        let path = std::env::temp_dir().join(format!(
            "mikuru-tokenizer-cli-test-{}.jsonl",
            std::process::id()
        ));

        let record = r#"{"id":"cli-test-001","messages":[{"role":"user","content":"Hello!"},{"role":"assistant","content":"Hi there!"}],"language":"en","category":"general_conversation","source_dataset":"Mikuru Test","source_record_id":"cli-test-001","source_url":"N/A","license":"Project Internal","synthetic":true,"provenance_note":"Temporary unit-test fixture."}"#;

        std::fs::write(&path, record).expect("test fixture should be written");

        let mut corpus = JsonlTextIter::open(&path).expect("JSONL reader should open");

        let formatted = corpus.next();

        assert_eq!(
            formatted.as_deref(),
            Some("<|bos|>\nuser: Hello!\nassistant: Hi there!\n<|eos|>")
        );

        assert_eq!(corpus.next(), None);
        assert_eq!(corpus.records_read(), 1);
        assert_eq!(corpus.take_error(), None);

        std::fs::remove_file(&path).expect("test fixture should be removed");
    }
}

#[cfg(test)]
mod cli_error_tests {
    use super::*;

    #[test]
    fn jsonl_iterator_reports_malformed_records() {
        let path = std::env::temp_dir().join(format!(
            "mikuru-tokenizer-invalid-test-{}.jsonl",
            std::process::id()
        ));

        std::fs::write(&path, r#"{"id":"broken","messages":["#)
            .expect("invalid fixture should be written");

        let mut corpus = JsonlTextIter::open(&path).expect("JSONL reader should open");

        assert!(corpus.next().is_none());

        let error = corpus
            .take_error()
            .expect("malformed JSONL should produce an error");

        assert!(
            error.contains("line 1"),
            "Error should identify the malformed line: {error}"
        );

        std::fs::remove_file(&path).expect("temporary fixture should be removed");
    }

    #[test]
    fn jsonl_iterator_excludes_validation_records_from_training() {
        let path = std::env::temp_dir().join(format!(
            "mikuru-tokenizer-split-test-{}.jsonl",
            std::process::id()
        ));

        let train_record = r#"{"id":"tokenizer-train-001","messages":[{"role":"user","content":"Hello!"},{"role":"assistant","content":"Hi there!"}],"language":"en","category":"general_conversation","source_dataset":"Mikuru Test","source_record_id":"tokenizer-train-001","source_url":"N/A","license":"Project Internal","synthetic":true,"split":"train","provenance_note":"Temporary unit-test fixture."}"#;

        let validation_record = r#"{"id":"tokenizer-validation-001","messages":[{"role":"user","content":"Hidden validation text"},{"role":"assistant","content":"Should not appear"}],"language":"en","category":"general_conversation","source_dataset":"Mikuru Test","source_record_id":"tokenizer-validation-001","source_url":"N/A","license":"Project Internal","synthetic":true,"split":"validation","provenance_note":"Temporary unit-test fixture."}"#;

        std::fs::write(&path, format!("{train_record}\n{validation_record}\n"))
            .expect("fixture should be written");

        let mut corpus = JsonlTextIter::open(&path).expect("JSONL reader should open");

        let formatted = corpus.next().expect("training record should be emitted");

        assert!(formatted.contains("Hello!"));
        assert!(!formatted.contains("Hidden validation text"));
        assert!(corpus.next().is_none());

        assert_eq!(corpus.records_read(), 1);
        assert_eq!(corpus.validation_skipped(), 1);
        assert_eq!(corpus.take_error(), None);

        std::fs::remove_file(&path).expect("temporary fixture should be removed");
    }
}
