use std::{
    env,
    error::Error,
    fs::File,
    io::{BufRead, BufReader},
    process::ExitCode,
};

use mikuru_core::ConversationExample;

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("error: {error}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), Box<dyn Error>> {
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
        Some("help") | Some("--help") | Some("-h") | None => print_help(),
        Some(command) => {
            return Err(format!("unknown command `{command}`; run `mikuru-lm help`").into());
        }
    }

    Ok(())
}

fn require_nonempty(
    value: &str,
    field: &str,
    path: &str,
    line_number: usize,
) -> Result<(), String> {
    if value.trim().is_empty() {
        return Err(format!("empty `{field}` field at {path}:{line_number}"));
    }

    Ok(())
}

fn inspect_jsonl(path: &str) -> Result<(), Box<dyn Error>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    let mut records = 0usize;
    let mut total_messages = 0usize;
    let mut total_characters = 0usize;

    for (index, line_result) in reader.lines().enumerate() {
        let line_number = index + 1;
        let line = line_result?;

        if line.trim().is_empty() {
            continue;
        }

        let example: ConversationExample = serde_json::from_str(&line).map_err(|error| {
            format!("invalid conversation record at {path}:{line_number}: {error}")
        })?;

        require_nonempty(&example.id, "id", path, line_number)?;
        require_nonempty(&example.language, "language", path, line_number)?;
        require_nonempty(&example.category, "category", path, line_number)?;
        require_nonempty(&example.source_dataset, "source_dataset", path, line_number)?;
        require_nonempty(
            &example.source_record_id,
            "source_record_id",
            path,
            line_number,
        )?;
        require_nonempty(&example.source_url, "source_url", path, line_number)?;
        require_nonempty(&example.license, "license", path, line_number)?;
        require_nonempty(
            &example.provenance_note,
            "provenance_note",
            path,
            line_number,
        )?;

        if let Some(split) = &example.split {
            require_nonempty(split, "split", path, line_number)?;
        }

        if example.messages.is_empty() {
            return Err(format!("conversation has no messages at {path}:{line_number}").into());
        }

        for (message_index, message) in example.messages.iter().enumerate() {
            if message.role.trim().is_empty() {
                return Err(format!(
                    "empty role in message {} at {path}:{line_number}",
                    message_index + 1
                )
                .into());
            }

            if message.content.trim().is_empty() {
                return Err(format!(
                    "empty content in message {} at {path}:{line_number}",
                    message_index + 1
                )
                .into());
            }

            total_characters += message.content.chars().count();
        }

        for (reference_index, reference) in example.references.iter().enumerate() {
            if reference.trim().is_empty() {
                return Err(format!(
                    "empty reference {} at {path}:{line_number}",
                    reference_index + 1
                )
                .into());
            }
        }

        total_messages += example.messages.len();
        records += 1;
    }

    println!("File: {path}");
    println!("Valid conversations: {records}");
    println!("Total messages: {total_messages}");
    println!("Total message characters: {total_characters}");

    Ok(())
}

fn print_help() {
    println!(
        "Mikuru LM starter\n\n\
         Commands:\n\
           burn-smoke                  Verify Burn Flex FP32 + autodiff\n\
           inspect-jsonl <path>        Validate conversation JSONL records\n\
           help                        Show this help"
    );
}

#[cfg(test)]
mod tests {
    use super::require_nonempty;

    #[test]
    fn rejects_whitespace_only_metadata() {
        let result = require_nonempty("   ", "language", "test.jsonl", 1);

        assert!(result.is_err());
        assert!(result.unwrap_err().contains("language"));
    }

    #[test]
    fn accepts_nonempty_metadata() {
        let result = require_nonempty("en", "language", "test.jsonl", 1);

        assert!(result.is_ok());
    }
}
