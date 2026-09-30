// One fixed native family: T8 x N32 x K32, 256 threads, F32 staging.
// This source follows kernels.metal in the same independent native library,
// reusing its canonical NF4 table and exact SiLU expression.

inline uint tile8_weights_per_byte(uint format) {
    if (format == 0) return 4;
    if (format == 1) return 2;
    return 1;
}

inline float tile8_decode(uchar packed, uint component, uint format) {
    // Ternary: 4 LSB-first two-bit symbols. NF4: low nibble first.
    // INT8: one two's-complement byte. Validated artifacts exclude reserved
    // ternary symbol3 and INT8 byte128; native arithmetic keeps the same decode.
    if (format == 0) {
        uint code = (uint(packed) >> (2 * component)) & 3;
        return float(int(code) - 1);
    }
    if (format == 1) {
        uint code = (uint(packed) >> (4 * component)) & 15;
        return NF4[code];
    }
    uint code = uint(packed);
    return float(code < 128 ? int(code) : int(code) - 256);
}

inline void tile8_stage_weights(threadgroup float *staged,
                                device const uchar *codes,
                                device const float *scales,
                                uint output_rows, uint input_width,
                                uint output_base, uint k_base,
                                uint format, uint thread_index) {
    uint weights_per_byte = tile8_weights_per_byte(format);
    uint bytes_per_tile_row = 32 / weights_per_byte;
    uint code_row_stride = input_width / weights_per_byte;
    uint scale_row_stride = input_width / 128;

    // Each canonical byte has one owner. Adjacent loader indices read adjacent
    // K bytes within a row, then transpose decoded coefficients into [k][out].
    // K%128==0 keeps every K32 tile inside one group-128 scale block.
    for (uint byte_index = thread_index;
         byte_index < 32 * bytes_per_tile_row;
         byte_index += 256) {
        uint local_output = byte_index / bytes_per_tile_row;
        uint local_byte = byte_index % bytes_per_tile_row;
        uint output_row = output_base + local_output;
        if (output_row < output_rows) {
            uint code_address = output_row * code_row_stride
                + k_base / weights_per_byte + local_byte;
            uint scale_address = output_row * scale_row_stride + k_base / 128;
            uchar packed = codes[code_address];
            float scale = scales[scale_address];
            for (uint component = 0; component < weights_per_byte; component++) {
                uint local_k = weights_per_byte * local_byte + component;
                // Scale-before-product and F32 storage preserve the scalar
                // source arithmetic. Padding output column32 stays untouched.
                staged[local_k * 33 + local_output]
                    = tile8_decode(packed, component, format) * scale;
            }
        } else {
            // A tail column never reads the code or scale allocation.
            for (uint component = 0; component < weights_per_byte; component++) {
                uint local_k = weights_per_byte * local_byte + component;
                staged[local_k * 33 + local_output] = 0.0f;
            }
        }
    }
}

// Params: [tokens,output_rows,input_width,format,input_fusion], slot8.
// Bindings match packed_linear: out,input-or-gate,codes,scales,up.
// Full groups are exactly (256,1,1); grid is (ceil(N/32),ceil(T/8),1).
kernel void packed_linear_tile8(device float *out [[buffer(0)]],
                                device const float *in [[buffer(1)]],
                                device const uchar *codes [[buffer(2)]],
                                device const float *scales [[buffer(3)]],
                                device const float *up [[buffer(4)]],
                                constant Params &p [[buffer(8)]],
                                uint3 group [[threadgroup_position_in_grid]],
                                uint thread_index [[thread_index_in_threadgroup]]) {
    uint tokens = p.v[0];
    uint output_rows = p.v[1];
    uint input_width = p.v[2];
    uint token_slot = thread_index / 32;
    uint output_slot = thread_index % 32;
    uint token_row = group.y * 8 + token_slot;
    uint output_base = group.x * 32;
    uint output_row = output_base + output_slot;

    // Declared logical storage: 4*(8*32 + 32*33) = 5248 bytes.
    threadgroup float staged_input[8][32];
    threadgroup float staged_weights[32][33];
    float sum = 0.0f;
    for (uint k_base = 0; k_base < input_width; k_base += 32) {
        float activation = 0.0f;
        if (token_row < tokens) {
            uint input_address = token_row * input_width + k_base + output_slot;
            activation = in[input_address];
            if (p.v[4]) activation = silu(activation) * up[input_address];
        }
        staged_input[token_slot][output_slot] = activation;
        tile8_stage_weights(&staged_weights[0][0], codes, scales,
                            output_rows, input_width, output_base, k_base,
                            p.v[3], thread_index);

        // Masked token/output threads participate in both barriers and the
        // same K loop. There is no early return in a cooperative group.
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint column = 0; column < 32; column++) {
            sum += staged_input[token_slot][column] * staged_weights[column][output_slot];
        }
        // Every consumer finishes reading before the next tile overwrites.
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (token_row < tokens && output_row < output_rows) {
        out[token_row * output_rows + output_row] = sum;
    }
}

// Params: [tokens,output_rows,input_width,format_a,format_b,epilogue_fusion].
// Bindings match packed_pair: out_a,out_b,input,codes_a,scales_a,codes_b,scales_b.
// Each matrix has independent packing and group scales, including same-format
// pairs with distinct bindings. Fused epilogue only writes A; B may alias A.
kernel void packed_pair_tile8(device float *out_a [[buffer(0)]],
                              device float *out_b [[buffer(1)]],
                              device const float *in [[buffer(2)]],
                              device const uchar *codes_a [[buffer(3)]],
                              device const float *scales_a [[buffer(4)]],
                              device const uchar *codes_b [[buffer(5)]],
                              device const float *scales_b [[buffer(6)]],
                              constant Params &p [[buffer(8)]],
                              uint3 group [[threadgroup_position_in_grid]],
                              uint thread_index [[thread_index_in_threadgroup]]) {
    uint tokens = p.v[0];
    uint output_rows = p.v[1];
    uint input_width = p.v[2];
    uint token_slot = thread_index / 32;
    uint output_slot = thread_index % 32;
    uint token_row = group.y * 8 + token_slot;
    uint output_base = group.x * 32;
    uint output_row = output_base + output_slot;

    // Declared logical storage: 4*(8*32 + 2*32*33) = 9472 bytes.
    threadgroup float staged_input[8][32];
    threadgroup float staged_a[32][33];
    threadgroup float staged_b[32][33];
    float sum_a = 0.0f;
    float sum_b = 0.0f;
    for (uint k_base = 0; k_base < input_width; k_base += 32) {
        float activation = 0.0f;
        if (token_row < tokens) {
            uint input_address = token_row * input_width + k_base + output_slot;
            activation = in[input_address];
        }
        staged_input[token_slot][output_slot] = activation;
        tile8_stage_weights(&staged_a[0][0], codes_a, scales_a,
                            output_rows, input_width, output_base, k_base,
                            p.v[3], thread_index);
        tile8_stage_weights(&staged_b[0][0], codes_b, scales_b,
                            output_rows, input_width, output_base, k_base,
                            p.v[4], thread_index);

        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint column = 0; column < 32; column++) {
            float input_value = staged_input[token_slot][column];
            sum_a += input_value * staged_a[column][output_slot];
            sum_b += input_value * staged_b[column][output_slot];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (token_row < tokens && output_row < output_rows) {
        uint output_address = token_row * output_rows + output_row;
        if (p.v[5]) {
            out_a[output_address] = silu(sum_a) * sum_b;
        } else {
            out_a[output_address] = sum_a;
            out_b[output_address] = sum_b;
        }
    }
}
