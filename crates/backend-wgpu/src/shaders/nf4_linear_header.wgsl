
const PAIR: bool = false;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<storage, read> x: array<f32>;
@group(0) @binding(3) var<storage, read> codes: array<u32>;
@group(0) @binding(4) var<storage, read> scales: array<f32>;
@group(0) @binding(5) var<storage, read> x4: array<vec4<f32>>;
fn input_value(i: u32) -> f32 { return x[i]; }
fn weight_a(row: u32, col: u32, k: u32) -> f32 {
    let word = codes[row * (k / 8u) + col / 8u];
    return NF4[(word >> ((col % 8u) * 4u)) & 15u] * scales[row * (k / 128u) + col / 128u];
}
fn weight_b(row: u32, col: u32, k: u32) -> f32 { return 0.0; }
fn store_output(i: u32, a: f32, b: f32) { dst[i] = a; }
