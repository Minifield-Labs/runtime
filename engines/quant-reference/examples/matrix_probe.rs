//! Dependency-free CLI for comparing exported matrices with the Python oracle.
use minifield_quant_reference::q8::Int8Matrix;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 {
        return Err("usage: matrix_probe MATRIX.mfq8 INPUTS.txt".into());
    }
    let matrix = Int8Matrix::from_bytes(&std::fs::read(&args[1])?, 16_777_216)?;
    let input: Vec<f32> = std::fs::read_to_string(&args[2])?
        .split_whitespace()
        .map(str::parse)
        .collect::<Result<_, _>>()?;
    if input.is_empty() || input.len() % matrix.cols() != 0 {
        return Err("invalid input shape".into());
    }
    let batch = input.len() / matrix.cols();
    let mut output = vec![0.0; batch.checked_mul(matrix.rows()).ok_or("output overflow")?];
    matrix.matmul(&input, batch, &mut output)?;
    println!(
        "[{}]",
        output
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join(",")
    );
    Ok(())
}
