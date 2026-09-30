//! One explicit evaluation request produces one complete JSON record.
mod backend;
mod completion;
mod loading;
mod measurement;
mod prediction;
mod preparation;
mod request;
mod run;
#[cfg(test)]
mod tests;

use request::Request;
use std::{env, fs, time::Instant};
type HostResult<T> = Result<T, Box<dyn std::error::Error>>;

fn tokenize(tokenizer_path: &str, texts_path: &str) -> HostResult<()> {
    let bytes = fs::read(tokenizer_path)?;
    let tokenizer = minifield_text_tokenizer::Tokenizer::from_json_bytes(
        &bytes,
        minifield_text_tokenizer::TokenizerLimits::default(),
    )?;
    let texts: Vec<String> = serde_json::from_slice(&fs::read(texts_path)?)?;
    let ids = texts
        .iter()
        .map(|text| {
            tokenizer.encode(
                text,
                minifield_text_tokenizer::EncodeOptions {
                    add_special_tokens: false,
                },
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    println!("{}", serde_json::to_string(&ids)?);
    Ok(())
}

fn main() -> HostResult<()> {
    let args = env::args().skip(1).collect::<Vec<_>>();
    if args.first().map(String::as_str) == Some("--tokenize") && args.len() == 3 {
        return tokenize(&args[1], &args[2]);
    }
    if args.len() != 2 || args[0] != "--request" {
        return Err(
            "usage: minifield-eval --request PATH (or --tokenize TOKENIZER TEXTS_JSON)".into(),
        );
    }
    let request: Request = serde_json::from_slice(&fs::read(&args[1])?)?;
    request.validate()?;
    let result = backend::evaluate(&request, Instant::now())?;
    println!("{}", serde_json::to_string(&result)?);
    Ok(())
}
