// Canonical two's-complement bytes, low byte first. Scaling stays F32 and
// occurs before the arithmetic tile multiplies each coefficient by input.
fn int8_decode_word(word: u32, scale: f32) -> vec4<f32> {
    let bytes = (vec4<u32>(word) >> vec4<u32>(0u, 8u, 16u, 24u)) & vec4<u32>(255u);
    let signed = bitcast<vec4<i32>>(bytes << vec4<u32>(24u)) >> vec4<u32>(24u);
    return vec4<f32>(signed) * scale;
}
