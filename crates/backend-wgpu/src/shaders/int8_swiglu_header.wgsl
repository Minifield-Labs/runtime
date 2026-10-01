
const PAIR: bool = false;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<storage, read> gate: array<f32>;
@group(0) @binding(3) var<storage, read> up: array<f32>;
@group(0) @binding(4) var<storage, read> codes: array<u32>;
@group(0) @binding(5) var<storage, read> scales: array<f32>;
@group(0) @binding(6) var<storage, read> gate4: array<vec4<f32>>;
@group(0) @binding(7) var<storage, read> up4: array<vec4<f32>>;
fn input_value(i: u32) -> f32 { let g = gate[i]; return (g / (1.0 + exp(-g))) * up[i]; }
fn weight_a(row: u32, col: u32, k: u32) -> f32 {
    let word = codes[row * (k / 4u) + col / 4u];
    return f32(bitcast<i32>(((word >> ((col % 4u) * 8u)) & 255u) << 24u) >> 24u) * scales[row * (k / 128u) + col / 128u];
}
fn weight_b(row: u32, col: u32, k: u32) -> f32 { return 0.0; }
fn store_output(i: u32, a: f32, b: f32) { dst[i] = a; }
