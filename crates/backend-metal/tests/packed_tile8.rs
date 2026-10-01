//! Native packed-tile conformance on a real device, with independent scalar math.
//! Run with --ignored --test-threads=1. Construction and MSL compilation must
//! succeed; reference-device tiled cases fail if dispatch falls back.
//! Draft source: compilation, formatting and hardware validation are pending.
#![cfg(target_os = "macos")]
#![allow(clippy::expect_used, clippy::float_cmp, clippy::cast_precision_loss)]

use minifield_backend_metal::{MetalBackend, MetalBuffer};
use minifield_engine_api::{
    AllocationClass, CompletionPoll, InferenceCompletion, InferenceOps, ResourceLimits, Shape,
};
use std::collections::{BTreeMap, BTreeSet};
use std::time::{Duration, Instant};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Format {
    Ternary,
    Nf4,
    Int8,
}

const FORMATS: [Format; 3] = [Format::Ternary, Format::Nf4, Format::Int8];

// Canonical format values, rather than an imported backend decode/dot helper.
#[allow(clippy::excessive_precision, clippy::unreadable_literal)]
const NF4: [f32; 16] = [
    -1.0,
    -0.6961928009986877,
    -0.5250730514526367,
    -0.39491748809814453,
    -0.28444138169288635,
    -0.18477343022823334,
    -0.09105003625154495,
    0.0,
    0.07958029955625534,
    0.16093020141124725,
    0.24611230194568634,
    0.33791524171829224,
    0.44070982933044434,
    0.5626170039176941,
    0.7229568362236023,
    1.0,
];

impl Format {
    fn per_byte(self) -> usize {
        match self {
            Self::Ternary => 4,
            Self::Nf4 => 2,
            Self::Int8 => 1,
        }
    }

    fn zero(self) -> u8 {
        match self {
            Self::Ternary => 1,
            Self::Nf4 => 7,
            Self::Int8 => 0,
        }
    }

    fn extreme(self, negative: bool) -> u8 {
        match (self, negative) {
            (Self::Ternary | Self::Nf4, true) => 0,
            (Self::Ternary, false) => 2,
            (Self::Nf4, false) => 15,
            (Self::Int8, true) => 129, // -127; the reserved -128 is absent.
            (Self::Int8, false) => 127,
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct Case {
    tokens: usize,
    rows: usize,
    width: usize,
    tiled: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum InputKind {
    Dense,
    Sparse,
    BoundaryImpulses,
}

fn shape(rows: usize, columns: usize) -> Shape {
    Shape::new(&[
        u64::try_from(rows).expect("fixture rows fit u64"),
        u64::try_from(columns).expect("fixture columns fit u64"),
    ])
    .expect("fixture shape")
}

fn metal() -> MetalBackend {
    let backend = MetalBackend::new(
        801,
        ResourceLimits {
            max_allocation_bytes: 64 << 20,
            max_total_bytes: 256 << 20,
            max_pending_operations: 64,
        },
    )
    .expect("real native Metal device and actual MSL compilation required");
    assert_eq!(backend.device_info().api, "metal");
    backend
}

fn wait<C: InferenceCompletion>(mut completion: C) -> C::Output {
    let start = Instant::now();
    loop {
        match completion.poll_step() {
            CompletionPoll::Ready(result) => return result.expect("native Metal completion"),
            CompletionPoll::Pending => {
                assert!(
                    start.elapsed() < Duration::from_secs(30),
                    "native packed-tile completion deadline"
                );
                std::thread::sleep(Duration::from_millis(1));
            }
        }
    }
}

fn read(backend: &MetalBackend, buffer: &MetalBuffer) -> Vec<f32> {
    wait(backend.read_f32_async(buffer).expect("native readback"))
}

fn close(expected: &[f32], actual: &[f32], context: &str) {
    assert_eq!(expected.len(), actual.len(), "{context}: output length");
    for (index, (&reference, &value)) in expected.iter().zip(actual).enumerate() {
        assert!(
            reference.is_finite() && value.is_finite(),
            "{context}: nonfinite output at {index}: scalar={reference}, Metal={value}"
        );
        assert!(
            (reference - value).abs() <= 1.0e-4 + 1.0e-4 * reference.abs(),
            "{context}: output {index}: scalar={reference}, Metal={value}"
        );
    }
}

fn zero_cases(case: Case, kind: InputKind, plain: bool, actual: &[f32], context: &str) {
    assert!(
        actual[..case.rows].iter().all(|&value| value == 0.0),
        "{context}: zero input token"
    );
    for token in 0..case.tokens {
        assert_eq!(actual[token * case.rows], 0.0, "{context}: zero codes");
        assert_eq!(actual[token * case.rows + 4], 0.0, "{context}: zero scales");
    }
    if plain && kind == InputKind::Dense {
        assert_eq!(
            actual[2 * case.rows + 2],
            0.0,
            "{context}: exact paired cancellation"
        );
    }
}

fn dispatched_once(
    backend: &MetalBackend,
    mut before: BTreeMap<&'static str, u64>,
    entry: &'static str,
    context: &str,
) {
    let count = before.entry(entry).or_default();
    *count = count.checked_add(1).expect("test dispatch count");
    assert_eq!(
        backend.dispatch_counts(),
        before,
        "{context}: exactly one actual {entry} dispatch is required"
    );
}

// Decode positive, finite canonical F16 bits directly into F32. Scales are
// uploaded as the same decoded values a canonical model loader supplies.
fn scale_from_f16(bits: u16) -> f32 {
    assert_eq!(bits & 0x8000, 0, "fixture scales are nonnegative");
    let exponent = (bits >> 10) & 31;
    let mantissa = bits & 1023;
    assert!(exponent < 31, "fixture scales are finite");
    if exponent == 0 {
        f32::from(mantissa) * 2.0_f32.powi(-24)
    } else {
        f32::from_bits(((u32::from(exponent) + 112) << 23) | (u32::from(mantissa) << 13))
    }
}

fn code(format: Format, row: usize, column: usize, seed: usize) -> u8 {
    if row == 0 {
        return format.zero();
    }
    if row == 5 || row == 6 {
        return if column == 0 {
            format.extreme(false)
        } else {
            format.zero()
        };
    }
    if row == 2 {
        return format.extreme(false);
    }
    if row == 7 && column % 128 != 0 && column % 128 != 127 {
        return format.zero();
    }
    if row == 1 || row == 7 {
        return format.extreme((column + column / 128 + seed) % 2 == 0);
    }
    let pattern = row * 17 + column * 7 + (column / 128) * 5 + seed * 11;
    match format {
        Format::Ternary => u8::try_from(pattern % 3).expect("ternary symbol"),
        Format::Nf4 => u8::try_from(pattern % 16).expect("NF4 symbol"),
        Format::Int8 => [0, 1, 127, 255, 129, 3, 253, 63, 193][pattern % 9],
    }
}

struct PackedFixture {
    format: Format,
    rows: usize,
    width: usize,
    codes: Vec<u8>,
    scales: Vec<f32>,
}

struct UploadedWeights {
    codes: MetalBuffer,
    scales: MetalBuffer,
}

impl PackedFixture {
    fn new(case: Case, format: Format, seed: usize) -> Self {
        assert_eq!(case.width % 128, 0);
        let per_byte = format.per_byte();
        let code_width = case.width / per_byte;
        let bits_per_code = 8 / per_byte;
        let mut codes = vec![0_u8; case.rows * code_width];
        for row in 0..case.rows {
            for column in 0..case.width {
                let byte = row * code_width + column / per_byte;
                codes[byte] |=
                    code(format, row, column, seed) << ((column % per_byte) * bits_per_code);
            }
        }
        let groups = case.width / 128;
        assert!((1..=64).contains(&groups), "bounded fixture scale groups");
        // Positive finite F16 bits increase by 32 for each canonical K group.
        // A bounded row/seed offset keeps same-format matrices independent.
        // The old six-value palette repeated after two groups, hiding %2 bugs.
        // Rows 4/5/6 separately retain zero, maximum and minimum-subnormal scales.
        let mut scales = Vec::with_capacity(case.rows * groups);
        for row in 0..case.rows {
            for group in 0..groups {
                let bits = match row {
                    4 => 0x0000,
                    5 => 0x7bff,
                    6 => 0x0001,
                    _ => {
                        let offset = u16::try_from((row * 7 + seed * 11) % 128)
                            .expect("bounded row/seed scale offset");
                        let group_offset = u16::try_from(group * 32)
                            .expect("bounded canonical scale group offset");
                        0x2e01_u16 + offset + group_offset
                    }
                };
                scales.push(scale_from_f16(bits));
            }
            let row_scales = &scales[row * groups..(row + 1) * groups];
            if ![4, 5, 6].contains(&row) {
                assert!(
                    row_scales
                        .iter()
                        .all(|value| value.is_finite() && *value > 0.0),
                    "positive finite group scales"
                );
                let unique: BTreeSet<_> = row_scales.iter().map(|value| value.to_bits()).collect();
                assert_eq!(
                    unique.len(),
                    groups,
                    "every canonical K group has a distinct scale"
                );
                if case.width == 384 {
                    assert_ne!(row_scales[0], row_scales[1], "K384 groups 0 and 1");
                    assert_ne!(row_scales[1], row_scales[2], "K384 groups 1 and 2");
                    assert_ne!(
                        row_scales[0], row_scales[2],
                        "K384 groups 0 and 2 detect modulo-2 indexing"
                    );
                }
            }
        }
        Self {
            format,
            rows: case.rows,
            width: case.width,
            codes,
            scales,
        }
    }

    fn upload(&self, backend: &mut MetalBackend) -> UploadedWeights {
        UploadedWeights {
            codes: backend
                .upload_u8_classified(
                    shape(self.rows, self.width / self.format.per_byte()),
                    &self.codes,
                    AllocationClass::Weight,
                )
                .expect("canonical packed codes"),
            scales: backend
                .upload_f32(shape(self.rows, self.width / 128), &self.scales)
                .expect("canonical F16-decoded scales"),
        }
    }

    fn coefficient(&self, row: usize, column: usize) -> f32 {
        // Canonical row-major offsets only. No candidate tile coordinates,
        // producer ownership, padded pitches or staging helpers enter the oracle.
        let decoded = match self.format {
            Format::Ternary => {
                let byte = self.codes[row * (self.width / 4) + column / 4];
                let symbol = (byte >> (2 * (column % 4))) & 3;
                assert!(symbol < 3, "reserved ternary code absent");
                f32::from(i16::from(symbol) - 1)
            }
            Format::Nf4 => {
                let byte = self.codes[row * (self.width / 2) + column / 2];
                NF4[usize::from((byte >> (4 * (column % 2))) & 15)]
            }
            Format::Int8 => {
                let byte = self.codes[row * self.width + column];
                let signed = i8::from_ne_bytes([byte]);
                assert_ne!(signed, i8::MIN, "reserved INT8 code absent");
                f32::from(signed)
            }
        };
        let scale = self.scales[row * (self.width / 128) + column / 128];
        decoded * scale
    }
}

fn input_value(kind: InputKind, token: usize, column: usize, width: usize) -> f32 {
    if token == 0 {
        return 0.0;
    }
    if kind == InputKind::BoundaryImpulses {
        return match column {
            127 => (1 + token % 4) as f32 * 0.125,
            128 => -((1 + token % 3) as f32) * 0.25,
            _ => 0.0,
        };
    }
    if kind == InputKind::Sparse {
        if ![0, 31, 32, 127, 128, 255, 256, width - 1].contains(&column) {
            return 0.0;
        }
        return ((token * 3 + column) % 9) as f32 * 0.25 - 1.0;
    }
    match token {
        1 => [32.0, -32.0, 0.5, -0.5][column % 4],
        2 => {
            if column % 2 == 0 {
                0.25
            } else {
                -0.25
            }
        }
        3 => {
            if [31, 32, 127, 128, 255, 256, width - 1].contains(&column) {
                (column % 5) as f32 * 0.25 - 0.5
            } else {
                0.0
            }
        }
        _ => ((token * 13 + column * 7) % 23) as f32 * 0.0625 - 0.6875,
    }
}

fn inputs(case: Case, kind: InputKind) -> (Vec<f32>, Vec<f32>) {
    let mut gate = Vec::with_capacity(case.tokens * case.width);
    let mut up = Vec::with_capacity(case.tokens * case.width);
    for token in 0..case.tokens {
        for column in 0..case.width {
            gate.push(input_value(kind, token, column, case.width));
            up.push(((token * 5 + column * 11 + 3) % 17) as f32 * 0.125 - 1.0);
        }
    }
    (gate, up)
}

fn silu(value: f32) -> f32 {
    value / (1.0_f32 + (-value).exp())
}

fn scalar_linear(
    case: Case,
    input: &[f32],
    up: Option<&[f32]>,
    weights: &PackedFixture,
) -> Vec<f32> {
    let mut output = vec![0.0_f32; case.tokens * case.rows];
    for token in 0..case.tokens {
        // Retain ascending canonical K order. With finite coefficients and a
        // +0 accumulator, omitted zero products leave each finite sum unchanged.
        // Sparse actual-size fixtures therefore avoid vacuous scalar dot work.
        let mut terms = Vec::new();
        for column in 0..case.width {
            let index = token * case.width + column;
            let activation = up.map_or(input[index], |values| silu(input[index]) * values[index]);
            assert!(activation.is_finite(), "finite fixture activation");
            if activation != 0.0 {
                terms.push((column, activation));
            }
        }
        for row in 0..case.rows {
            let mut sum = 0.0_f32;
            for &(column, activation) in &terms {
                // Decode and scale first, then multiply the activation. Keep
                // ordinary F32 multiply/add; no F64 sum, reassociation or mul_add.
                let weight = weights.coefficient(row, column);
                sum += activation * weight;
            }
            output[token * case.rows + row] = sum;
        }
    }
    output
}

fn output(backend: &mut MetalBackend, case: Case) -> MetalBuffer {
    backend
        .upload_f32(
            shape(case.tokens, case.rows),
            &vec![-999_123.0; case.tokens * case.rows],
        )
        .expect("sentinel output")
}

fn check_single(backend: &mut MetalBackend, case: Case, format: Format, kind: InputKind) {
    let fixture = PackedFixture::new(case, format, 1);
    let weights = fixture.upload(backend);
    let (gate_values, up_values) = inputs(case, kind);
    let gate = backend
        .upload_f32(shape(case.tokens, case.width), &gate_values)
        .expect("single input");
    let up = backend
        .upload_f32(shape(case.tokens, case.width), &up_values)
        .expect("independent input-SiLU up");
    let entry = if case.tiled {
        "packed_linear_tile8"
    } else {
        "packed_linear"
    };
    for fused in [false, true] {
        let mut out = output(backend, case);
        let context = format!("single {case:?} {format:?} {kind:?} input-SiLU={fused}");
        let before = backend.dispatch_counts();
        if fused {
            backend
                .packed_swiglu_linear(&mut out, &gate, &up, &weights.codes, &weights.scales)
                .expect("native input-SiLU packed linear");
        } else {
            backend
                .packed_linear(&mut out, &gate, &weights.codes, &weights.scales)
                .expect("native packed linear");
        }
        dispatched_once(backend, before, entry, &context);
        let expected = scalar_linear(
            case,
            &gate_values,
            fused.then_some(up_values.as_slice()),
            &fixture,
        );
        if !fused && kind == InputKind::Dense {
            assert_eq!(
                expected[2 * case.rows + 2],
                0.0,
                "exact paired cancellation"
            );
        }
        let actual = read(backend, &out);
        close(&expected, &actual, &context);
        zero_cases(case, kind, !fused, &actual, &context);
    }
}

fn check_pair(
    backend: &mut MetalBackend,
    case: Case,
    format_a: Format,
    format_b: Format,
    kind: InputKind,
) {
    let fixture_a = PackedFixture::new(case, format_a, 1);
    let fixture_b = PackedFixture::new(case, format_b, 2);
    assert!(fixture_a.codes != fixture_b.codes || fixture_a.scales != fixture_b.scales);
    let a = fixture_a.upload(backend);
    let b = fixture_b.upload(backend);
    let (input_values, _) = inputs(case, kind);
    let input = backend
        .upload_f32(shape(case.tokens, case.width), &input_values)
        .expect("pair input");
    let expected_a = scalar_linear(case, &input_values, None, &fixture_a);
    let expected_b = scalar_linear(case, &input_values, None, &fixture_b);
    assert_ne!(expected_a, expected_b, "independent pair outputs");
    let entry = if case.tiled {
        "packed_pair_tile8"
    } else {
        "packed_pair"
    };
    let context = format!("pair {case:?} {format_a:?}/{format_b:?} {kind:?}");
    let mut out_a = output(backend, case);
    let mut out_b = output(backend, case);
    let before = backend.dispatch_counts();
    backend
        .packed_linear_pair(
            &mut out_a, &mut out_b, &input, &a.codes, &a.scales, &b.codes, &b.scales,
        )
        .expect("native independently formatted packed pair");
    dispatched_once(backend, before, entry, &context);
    let actual_a = read(backend, &out_a);
    let actual_b = read(backend, &out_b);
    close(&expected_a, &actual_a, &format!("{context} A"));
    close(&expected_b, &actual_b, &format!("{context} B"));
    zero_cases(case, kind, true, &actual_a, &format!("{context} A"));
    zero_cases(case, kind, true, &actual_b, &format!("{context} B"));

    let mut fused = output(backend, case);
    let before = backend.dispatch_counts();
    backend
        .packed_swiglu_pair(&mut fused, &input, &a.codes, &a.scales, &b.codes, &b.scales)
        .expect("native packed pair with F32 SiLU epilogue");
    dispatched_once(backend, before, entry, &format!("{context} epilogue-SiLU"));
    let expected: Vec<_> = expected_a
        .iter()
        .zip(&expected_b)
        .map(|(&gate, &up)| silu(gate) * up)
        .collect();
    let actual = read(backend, &fused);
    close(&expected, &actual, &format!("{context} epilogue-SiLU"));
    zero_cases(
        case,
        kind,
        false,
        &actual,
        &format!("{context} epilogue-SiLU"),
    );
}

#[test]
#[ignore = "requires real native Metal and admitted tile8 pipelines"]
fn tile8_thresholds_all_formats_and_all_independent_pairs_match_scalar() {
    let mut backend = metal();
    assert_eq!(scale_from_f16(0x0000).to_bits(), 0);
    assert_eq!(scale_from_f16(0x0001).to_bits(), 0x3380_0000);
    assert_eq!(scale_from_f16(0x2e66).to_bits(), 0x3dcc_c000);
    assert_eq!(scale_from_f16(0x3555).to_bits(), 0x3eaa_a000);
    assert_eq!(scale_from_f16(0x7bff), 65_504.0);
    for (tokens, tiled) in [(7, false), (8, true), (9, true)] {
        for width in [128, 256, 384] {
            let case = Case {
                tokens,
                rows: 33,
                width,
                tiled,
            };
            for format in FORMATS {
                check_single(&mut backend, case, format, InputKind::Dense);
            }
            for format_a in FORMATS {
                for format_b in FORMATS {
                    check_pair(&mut backend, case, format_a, format_b, InputKind::Dense);
                }
            }
        }
    }
}

#[test]
#[ignore = "requires real native Metal and admitted tile8 pipelines"]
fn tile8_deployment_token_tails_match_sparse_scalar() {
    let mut backend = metal();
    for (index, tokens) in [47, 111, 220, 298, 299, 300, 308, 345, 346, 347]
        .into_iter()
        .enumerate()
    {
        let case = Case {
            tokens,
            rows: 33,
            width: 384,
            tiled: true,
        };
        let format_a = FORMATS[index % FORMATS.len()];
        let format_b = FORMATS[(index + 1) % FORMATS.len()];
        check_single(&mut backend, case, format_a, InputKind::Sparse);
        check_pair(&mut backend, case, format_a, format_b, InputKind::Sparse);
    }
}

#[test]
#[ignore = "requires real native Metal and admitted tile8 pipelines"]
fn tile8_actual_packed_dimensions_match_sparse_scalar() {
    let mut backend = metal();
    // Unique admitted classifier/pointer packed N/K dimensions. The protected
    // dense F16 task head is absent. Fixtures contain synthetic canonical bytes.
    for (rows, width) in [
        (512, 1024),
        (1024, 1024),
        (1024, 2560),
        (2560, 1024),
        (3072, 1024),
        (1024, 4608),
        (4608, 1024),
    ] {
        let case = Case {
            tokens: 9,
            rows,
            width,
            tiled: true,
        };
        for format in FORMATS {
            check_single(&mut backend, case, format, InputKind::Sparse);
        }
    }
    // KV and classifier gate/up pair geometries, plus the pointer FFN shape.
    for rows in [512, 2560, 4608] {
        let case = Case {
            tokens: 9,
            rows,
            width: 1024,
            tiled: true,
        };
        check_pair(
            &mut backend,
            case,
            Format::Nf4,
            Format::Ternary,
            InputKind::Sparse,
        );
    }
}

#[test]
#[ignore = "requires real native Metal and admitted tile8 pipelines"]
fn tile8_production_ffn_shape_matches_ordered_group_boundary_impulses() {
    let mut backend = metal();
    let case = Case {
        tokens: 345,
        rows: 2560,
        width: 1024,
        tiled: true,
    };
    // Each active token has exactly two inputs, at canonical K127 then K128.
    // Expected projections are their ordered F32 products with two separately
    // decoded group scales. This bounds the host oracle at production geometry.
    check_single(&mut backend, case, Format::Nf4, InputKind::BoundaryImpulses);
    check_pair(
        &mut backend,
        case,
        Format::Ternary,
        Format::Int8,
        InputKind::BoundaryImpulses,
    );
}
