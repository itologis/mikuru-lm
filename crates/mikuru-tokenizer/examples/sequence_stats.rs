use mikuru_core::dataset::load_jsonl;
use mikuru_tokenizer::{MikuruTokenizer, format_conversation};
use std::{env, error::Error};

fn percentile(values: &[usize], p: f64) -> usize {
    let index = ((values.len() as f64 * p).ceil() as usize)
        .saturating_sub(1)
        .min(values.len() - 1);
    values[index]
}

fn inspect(path: &str, tokenizer: &MikuruTokenizer) -> Result<(), Box<dyn Error + Send + Sync>> {
    let records = load_jsonl(path)?;
    let mut lengths = Vec::with_capacity(records.len());

    for record in &records {
        let formatted = format_conversation(record);
        let ids = tokenizer.encode(&formatted)?;
        lengths.push(ids.len().saturating_sub(1));
    }

    if lengths.is_empty() {
        println!("{path}: no records");
        return Ok(());
    }

    lengths.sort_unstable();

    println!("\n{path}");
    println!("Records: {}", lengths.len());
    println!("Total input tokens: {}", lengths.iter().sum::<usize>());
    println!("Min: {}", lengths[0]);
    println!("Median: {}", percentile(&lengths, 0.50));
    println!("P90: {}", percentile(&lengths, 0.90));
    println!("P95: {}", percentile(&lengths, 0.95));
    println!("P99: {}", percentile(&lengths, 0.99));
    println!("Max: {}", lengths[lengths.len() - 1]);

    for limit in [128, 192, 256, 288, 320, 384, 512] {
        let count = lengths.iter().filter(|&&n| n > limit).count();
        println!("Exceeding {limit}: {count}");
    }

    Ok(())
}

fn main() -> Result<(), Box<dyn Error + Send + Sync>> {
    let args: Vec<String> = env::args().collect();

    if args.len() != 4 {
        return Err("Usage: sequence_stats TOKENIZER TRAIN_JSONL VALIDATION_JSONL".into());
    }

    let tokenizer = MikuruTokenizer::from_file(&args[1])?;
    println!("Vocabulary size: {}", tokenizer.vocab_size());

    inspect(&args[2], &tokenizer)?;
    inspect(&args[3], &tokenizer)?;

    Ok(())
}
