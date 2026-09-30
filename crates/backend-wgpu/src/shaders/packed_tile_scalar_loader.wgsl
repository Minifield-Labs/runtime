        for (var lo = 0u; lo < 2u; lo += 1u) {
            let c = x + lo * 16u;
            var wa = 0.0;
            var wb = 0.0;
            if col0 + c < n && base + y < k {
                wa = weight_a(col0 + c, base + y, k);
                if PAIR { wb = weight_b(col0 + c, base + y, k); }
            }
            weights_a[y * 32u + c] = wa;
            if PAIR { weights_b[y * 32u + c] = wb; }
        }
