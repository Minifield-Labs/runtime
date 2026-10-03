// Independent native Metal baseline. Host validation bounds every allocation,
// shape product and selector before encoding. Cooperative reductions require
// the complete threadgroups and pipeline limits checked by the host.
#include <metal_stdlib>
using namespace metal;
struct Params { uint v[16]; };

constant float NF4[16] = {
    -1.0f, -0.6961928009986877f, -0.5250730514526367f, -0.39491748809814453f,
    -0.28444138169288635f, -0.18477343022823334f, -0.09105003625154495f, 0.0f,
    0.07958029955625534f, 0.16093020141124725f, 0.24611230194568634f, 0.33791524171829224f,
    0.44070982933044434f, 0.5626170039176941f, 0.7229568362236023f, 1.0f
};
inline float silu(float x) { return x / (1.0f + exp(-x)); }

// Formats: 0 = 4 ternary codes/byte (LSB first), 1 = 2 NF4 codes/byte,
// 2 = signed two's-complement INT8. All use one scale per 128 weights.
inline float weight(device const uchar *codes, device const float *scales,
                    uint row, uint column, uint width, uint format) {
    uint code;
    float decoded;
    if (format == 0) {
        code = (codes[row * (width / 4) + column / 4] >> (2 * (column % 4))) & 3;
        decoded = float(int(code) - 1);
    } else if (format == 1) {
        code = (codes[row * (width / 2) + column / 2] >> (4 * (column % 2))) & 15;
        decoded = NF4[code];
    } else {
        code = codes[row * width + column];
        decoded = float(code < 128 ? int(code) : int(code) - 256);
    }
    return decoded * scales[row * (width / 128) + column / 128];
}

// Params: [elements, operation: 0=add, 1=multiply, 2=SwiGLU].
kernel void elementwise(device float *out [[buffer(0)]],
                        device const float *a [[buffer(1)]],
                        device const float *b [[buffer(2)]],
                        constant Params &p [[buffer(8)]], uint index [[thread_position_in_grid]]) {
    if (index >= p.v[0]) return;
    if (p.v[1] == 0) out[index] = a[index] + b[index];
    else if (p.v[1] == 1) out[index] = a[index] * b[index];
    else out[index] = silu(a[index]) * b[index];
}

// Params: [rows,columns,dest_row,dest_stride,dest_col,source_row,source_stride,source_col].
kernel void rect_copy(device float *out [[buffer(0)]], device const float *in [[buffer(1)]],
                      constant Params &p [[buffer(8)]], uint index [[thread_position_in_grid]]) {
    uint columns = p.v[1];
    if (index >= p.v[0] * columns) return;
    uint row = index / columns, column = index % columns;
    out[(row + p.v[2]) * p.v[3] + column + p.v[4]] =
        in[(row + p.v[5]) * p.v[6] + column + p.v[7]];
}

// Params: [selected_rows,width,table_rows]. Check the exact uint conversion
// range first; table_rows stays integer because its F32 rounding can shrink it.
kernel void gather(device float *out [[buffer(0)]], device const float *in [[buffer(1)]],
                   device const float *ids [[buffer(2)]], constant Params &p [[buffer(8)]],
                   uint index [[thread_position_in_grid]]) {
    uint width = p.v[1];
    if (index >= p.v[0] * width) return;
    float raw = ids[index / width];
    if (!isfinite(raw) || raw < 0 || raw >= 4294967296.0f) { out[index] = NAN; return; }
    uint row = uint(raw);
    out[index] = (raw != float(row) || row >= p.v[2]) ? NAN : in[row * width + index % width];
}

// Params: [rows,selected_columns,input_width]. Host checks every column id.
kernel void columns(device float *out [[buffer(0)]], device const float *in [[buffer(1)]],
                    device const uint *ids [[buffer(2)]], constant Params &p [[buffer(8)]],
                    uint index [[thread_position_in_grid]]) {
    uint columns = p.v[1];
    if (index < p.v[0] * columns) out[index] = in[(index / columns) * p.v[2] + ids[index % columns]];
}

// Params: [rows,width,masked]. The 64-bit portable mask is staged as LSB-first uints.
kernel void argmax_rows(device float *out [[buffer(0)]], device const float *in [[buffer(1)]],
                        device const uint *mask [[buffer(2)]], constant Params &p [[buffer(8)]],
                        uint row [[thread_position_in_grid]]) {
    if (row >= p.v[0]) return;
    uint best = 0;
    float maximum = -INFINITY;
    bool found = false;
    for (uint column = 0; column < p.v[1]; column++) {
        if (p.v[2] && !(mask[column / 32] & (1u << (column % 32)))) continue;
        float value = in[row * p.v[1] + column];
        if (!isfinite(value)) { out[row] = NAN; return; }
        if (!found || value > maximum) { found = true; maximum = value; best = column; }
    }
    out[row] = found ? float(best) : NAN;
}

// Params: [tokens,output_rows,input_width]. One thread reduces one output element.
kernel void dense_linear(device float *out [[buffer(0)]], device const float *in [[buffer(1)]],
                         device const float *weights [[buffer(2)]], constant Params &p [[buffer(8)]],
                         uint index [[thread_position_in_grid]]) {
    uint rows = p.v[1], width = p.v[2];
    if (index >= p.v[0] * rows) return;
    uint token = index / rows, row = index % rows;
    float sum = 0;
    for (uint column = 0; column < width; column++) sum += in[token * width + column] * weights[row * width + column];
    out[index] = sum;
}

// Params: [selected_rows,width,table_rows,format].
kernel void packed_gather(device float *out [[buffer(0)]], device const uchar *codes [[buffer(1)]],
                          device const float *scales [[buffer(2)]], device const float *ids [[buffer(3)]],
                          constant Params &p [[buffer(8)]], uint index [[thread_position_in_grid]]) {
    uint width = p.v[1];
    if (index >= p.v[0] * width) return;
    float raw = ids[index / width];
    if (!isfinite(raw) || raw < 0 || raw >= 4294967296.0f) { out[index] = NAN; return; }
    uint row = uint(raw);
    out[index] = (raw != float(row) || row >= p.v[2]) ? NAN : weight(codes, scales, row, index % width, width, p.v[3]);
}

// Params: [tokens,output_rows,input_width,format,fused_activation].
kernel void packed_linear(device float *out [[buffer(0)]], device const float *in [[buffer(1)]],
                          device const uchar *codes [[buffer(2)]], device const float *scales [[buffer(3)]],
                          device const float *up [[buffer(4)]], constant Params &p [[buffer(8)]],
                          uint index [[thread_position_in_grid]]) {
    uint rows = p.v[1], width = p.v[2];
    if (index >= p.v[0] * rows) return;
    uint token = index / rows, row = index % rows;
    float sum = 0;
    for (uint column = 0; column < width; column++) {
        float activation = in[token * width + column];
        if (p.v[4]) activation = silu(activation) * up[token * width + column];
        sum += activation * weight(codes, scales, row, column, width, p.v[3]);
    }
    out[index] = sum;
}

// Params: [tokens,output_rows,input_width,format_a,format_b,fused_epilogue].
// A fused epilogue only writes out_a, so aliasing out_b to out_a is intentional.
kernel void packed_pair(device float *out_a [[buffer(0)]], device float *out_b [[buffer(1)]],
                        device const float *in [[buffer(2)]], device const uchar *codes_a [[buffer(3)]],
                        device const float *scales_a [[buffer(4)]], device const uchar *codes_b [[buffer(5)]],
                        device const float *scales_b [[buffer(6)]], constant Params &p [[buffer(8)]],
                        uint index [[thread_position_in_grid]]) {
    uint rows = p.v[1], width = p.v[2];
    if (index >= p.v[0] * rows) return;
    uint token = index / rows, row = index % rows;
    float sum_a = 0, sum_b = 0;
    for (uint column = 0; column < width; column++) {
        float activation = in[token * width + column];
        sum_a += activation * weight(codes_a, scales_a, row, column, width, p.v[3]);
        sum_b += activation * weight(codes_b, scales_b, row, column, width, p.v[4]);
    }
    if (p.v[5]) out_a[index] = silu(sum_a) * sum_b;
    else { out_a[index] = sum_a; out_b[index] = sum_b; }
}

// Params: [normalization_rows,width,epsilon_bits]. Rows may represent tokens or heads.
kernel void rms_norm(device float *out [[buffer(0)]], device const float *in [[buffer(1)]],
                     device const float *weights [[buffer(2)]], constant Params &p [[buffer(8)]],
                     uint row [[thread_position_in_grid]]) {
    uint width = p.v[1];
    if (row >= p.v[0]) return;
    float sum = 0;
    for (uint column = 0; column < width; column++) { float value = in[row * width + column]; sum += value * value; }
    float inverse_rms = rsqrt(sum / float(width) + as_type<float>(p.v[2]));
    for (uint column = 0; column < width; column++) out[row * width + column] = in[row * width + column] * inverse_rms * weights[column];
}

// Same parameters as rms_norm. Host admits a complete 256-thread group per
// row only for a compiled SIMD width of 32. Small widths retain scalar dispatch.
kernel void rms_norm_simd(device float *out [[buffer(0)]], device const float *in [[buffer(1)]],
                          device const float *weights [[buffer(2)]], constant Params &p [[buffer(8)]],
                          uint row [[threadgroup_position_in_grid]],
                          uint tid [[thread_index_in_threadgroup]],
                          uint group [[simdgroup_index_in_threadgroup]],
                          uint lane [[thread_index_in_simdgroup]]) {
    uint width = p.v[1];
    threadgroup float partials[8];
    threadgroup float inverse_rms;
    float sum = 0;
    for (uint column = tid; column < width; column += 256) {
        float value = in[row * width + column];
        sum += value * value;
    }
    sum = simd_sum(sum);
    if (lane == 0) partials[group] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float total = 0;
        for (uint i = 0; i < 8; i++) total += partials[i];
        inverse_rms = rsqrt(total / float(width) + as_type<float>(p.v[2]));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint column = tid; column < width; column += 256) {
        out[row * width + column] = in[row * width + column] * inverse_rms * weights[column];
    }
}

// Params: [tokens,heads,head_dimension]. Host stages portable F64-to-F32 cosine
// and sine tables, one value per token and half-dimension. Heads reuse each table.
kernel void rotary(device float *out [[buffer(0)]], device const float *in [[buffer(1)]],
                   device const float *table [[buffer(2)]], constant Params &p [[buffer(8)]],
                   uint index [[thread_position_in_grid]]) {
    uint dimension = p.v[2], half_dimension = dimension / 2;
    if (index >= p.v[0] * p.v[1] * half_dimension) return;
    uint row = index / half_dimension, column = index % half_dimension, token = row / p.v[1];
    uint table_index = token * half_dimension + column;
    float cosine = table[table_index], sine = table[p.v[0] * half_dimension + table_index];
    float first = in[row * dimension + column], second = in[row * dimension + column + half_dimension];
    out[row * dimension + column] = first * cosine - second * sine;
    out[row * dimension + column + half_dimension] = second * cosine + first * sine;
}

// Params: [query_tokens,query_heads,head_dimension,kv_heads,key_tokens,cache_offset,bidirectional].
// One thread owns one query/head score row and output row. Scores are disjoint.
// Causal rows see only their populated prefix; encoder rows see their active segment.
kernel void attention(device float *out [[buffer(0)]], device const float *query [[buffer(1)]],
                      device const float *key [[buffer(2)]], device const float *value [[buffer(3)]],
                      device float *scores [[buffer(4)]], device const uint *segments [[buffer(5)]],
                      constant Params &p [[buffer(8)]], uint row [[thread_position_in_grid]]) {
    uint tokens = p.v[0], heads = p.v[1], dimension = p.v[2], kv_heads = p.v[3], key_tokens = p.v[4];
    if (row >= tokens * heads) return;
    uint token = row / heads, head = row % heads, kv_head = head / (heads / kv_heads);
    bool bidirectional = p.v[6] != 0;
    uint visible_tokens = bidirectional ? key_tokens : p.v[5] + token + 1;
    if (bidirectional && segments[token] == 0) {
        for (uint column = 0; column < dimension; column++) out[row * dimension + column] = 0;
        return;
    }
    float maximum = -INFINITY;
    for (uint source = 0; source < visible_tokens; source++) {
        float score = -INFINITY;
        if (!bidirectional || segments[source] == segments[token]) {
            score = 0;
            for (uint column = 0; column < dimension; column++) score += query[row * dimension + column] * key[(source * kv_heads + kv_head) * dimension + column];
            score *= rsqrt(float(dimension));
        }
        scores[row * key_tokens + source] = score;
        maximum = max(maximum, score);
    }
    float denominator = 0;
    for (uint source = 0; source < visible_tokens; source++) {
        float probability = exp(scores[row * key_tokens + source] - maximum);
        scores[row * key_tokens + source] = probability;
        denominator += probability;
    }
    for (uint column = 0; column < dimension; column++) {
        float sum = 0;
        for (uint source = 0; source < visible_tokens; source++) {
            // Skip the value read too. Multiplying a masked infinity by zero
            // would otherwise contaminate a valid query row with NaN.
            if (bidirectional && segments[source] != segments[token]) continue;
            sum += scores[row * key_tokens + source] * value[(source * kv_heads + kv_head) * dimension + column];
        }
        out[row * dimension + column] = sum / denominator;
    }
}

// Params: [tokens,hidden,width]. Kernel taps run oldest-to-current. A snapshot
// preserves the old history until this pass and the following update are encoded.
kernel void causal_conv(device float *out [[buffer(0)]], device const float *projection [[buffer(1)]],
                        device const float *weights [[buffer(2)]], device const float *history [[buffer(3)]],
                        constant Params &p [[buffer(8)]], uint index [[thread_position_in_grid]]) {
    uint tokens = p.v[0], hidden = p.v[1], width = p.v[2];
    if (index >= tokens * hidden) return;
    uint token = index / hidden, column = index % hidden;
    float sum = 0;
    for (uint tap = 0; tap < width; tap++) {
        int source = int(token) + int(tap) - int(width - 1);
        float gated_value;
        if (source < 0) gated_value = history[(source + int(width - 1)) * hidden + column];
        else gated_value = projection[uint(source) * 3 * hidden + column] * projection[uint(source) * 3 * hidden + 2 * hidden + column];
        sum += gated_value * weights[column * width + tap];
    }
    out[index] = sum * projection[token * 3 * hidden + hidden + column];
}

// Params: [tokens,hidden,history_rows]. Every new history element is independently
// selected from the old snapshot or the newly projected sequence.
kernel void conv_history(device float *out [[buffer(0)]], device const float *projection [[buffer(1)]],
                         device const float *old [[buffer(2)]], constant Params &p [[buffer(8)]],
                         uint index [[thread_position_in_grid]]) {
    uint tokens = p.v[0], hidden = p.v[1], history_rows = p.v[2];
    if (index >= history_rows * hidden) return;
    int source = int(tokens) + int(index / hidden) - int(history_rows);
    uint column = index % hidden;
    out[index] = source < 0 ? old[(source + int(history_rows)) * hidden + column] :
        projection[uint(source) * 3 * hidden + column] * projection[uint(source) * 3 * hidden + 2 * hidden + column];
}

// Params: [tokens,hidden,width]. floor(width/2) left padding also defines the
// even-width crop. Padding and other segments never read projected values.
kernel void centered_conv(device float *out [[buffer(0)]], device const float *projection [[buffer(1)]],
                          device const float *weights [[buffer(2)]], device const uint *segments [[buffer(3)]],
                          constant Params &p [[buffer(8)]], uint index [[thread_position_in_grid]]) {
    uint tokens = p.v[0], hidden = p.v[1], width = p.v[2];
    if (index >= tokens * hidden) return;
    uint token = index / hidden, column = index % hidden;
    if (segments[token] == 0) { out[index] = 0; return; }
    float sum = 0;
    for (uint tap = 0; tap < width; tap++) {
        int source = int(token) + int(tap) - int(width / 2);
        if (source >= 0 && source < int(tokens) && segments[uint(source)] == segments[token]) {
            sum += projection[uint(source) * 3 * hidden + column] * projection[uint(source) * 3 * hidden + 2 * hidden + column] * weights[column * width + tap];
        }
    }
    out[index] = sum * projection[token * 3 * hidden + hidden + column];
}
