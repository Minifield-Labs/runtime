        // Override the template's scalar ownership only. All 256 invocations
        // have already staged inputs and reach both barriers below this block.
        let linear = x + 16u * y;
        if linear < 128u {
            let c = linear / 4u;
            let q = 4u * (linear % 4u);
            var wa = vec4<f32>(0.0);
            var wb = vec4<f32>(0.0);
            // Admitted K is a multiple of 128. Keep the whole-word guard
            // explicit, and never read codes/scales for an invalid N column.
            if col0 + c < n && base + q + 3u < k {
                wa = weight_a4(col0 + c, base + q, k);
                if PAIR { wb = weight_b4(col0 + c, base + q, k); }
            }
            for (var j = 0u; j < 4u; j += 1u) {
                weights_a[(q + j) * 32u + c] = wa[j];
                if PAIR { weights_b[(q + j) * 32u + c] = wb[j]; }
            }
        }
