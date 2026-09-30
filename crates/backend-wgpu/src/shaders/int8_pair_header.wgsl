
const PAIR: bool = true;
@group(0) @binding(1) var<storage, read_write> dst_a: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst_b: array<f32>;
@group(0) @binding(3) var<storage, read> x: array<f32>;
@group(0) @binding(4) var<storage, read> codes_a: array<u32>;
@group(0) @binding(5) var<storage, read> scales_a: array<f32>;
@group(0) @binding(6) var<storage, read> codes_b: array<u32>;
@group(0) @binding(7) var<storage, read> scales_b: array<f32>;
@group(0) @binding(8) var<storage, read> x4: array<vec4<f32>>;
fn input_value(i: u32) -> f32 { return x[i]; }
fn weight_a(row: u32, col: u32, k: u32) -> f32 {
    let word = codes_a[row * (k / 4u) + col / 4u];
    return f32(bitcast<i32>(((word >> ((col % 4u) * 8u)) & 255u) << 24u) >> 24u) * scales_a[row * (k / 128u) + col / 128u];
}
fn weight_b(row: u32, col: u32, k: u32) -> f32 {
    let word = codes_b[row * (k / 4u) + col / 4u];
    return f32(bitcast<i32>(((word >> ((col % 4u) * 8u)) & 255u) << 24u) >> 24u) * scales_b[row * (k / 128u) + col / 128u];
}
fn store_output(i: u32, a: f32, b: f32) { dst_a[i] = a; dst_b[i] = b; }
