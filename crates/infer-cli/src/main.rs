#![forbid(unsafe_code)]

use std::io::{self, Write};

use minifield_infer::{parse_args, run_with_io};

fn main() -> std::process::ExitCode {
    match run() {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(error) => {
            let _ = writeln!(io::stderr(), "minifield-infer: {error}");
            std::process::ExitCode::from(2)
        }
    }
}

fn run() -> Result<(), minifield_infer::CliError> {
    let options = parse_args(std::env::args_os().skip(1))?;
    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut output = stdout.lock();
    let _ = run_with_io(&options, stdin.lock(), &mut output)?;
    Ok(())
}
