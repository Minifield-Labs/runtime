//! Content-free work accounting for recorded LFM2 forward passes.

use super::{LayerKind, Lfm2Config};

/// Versioned dense-equivalent estimate: matrix products, causal QK/AV, and convolution.
/// A multiply-add is 2 FLOPs. Elementwise, normalization, dequantization, and data movement
/// are excluded. This estimates model arithmetic, independently of kernel implementation.
pub const FLOPS_ESTIMATOR_VERSION: &str = "lfm2-dense-equivalent/1";

/// Monotone counters for one loaded executor. Snapshot around a serialized inference.
/// Work is counted after a forward pass has been recorded; a later device failure must
/// mark the inference estimate partial. Cache creation and discarded fallback work count.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct InferenceWork {
    pub forward_passes: u64,
    pub token_positions_processed: u64,
    pub estimated_flops: u128,
}

impl InferenceWork {
    /// Difference from an earlier snapshot of this same executor.
    #[must_use]
    pub fn since(self, earlier: Self) -> Self {
        Self {
            forward_passes: self.forward_passes.saturating_sub(earlier.forward_passes),
            token_positions_processed: self
                .token_positions_processed
                .saturating_sub(earlier.token_positions_processed),
            estimated_flops: self.estimated_flops.saturating_sub(earlier.estimated_flops),
        }
    }

    pub(crate) fn record(&mut self, config: &Lfm2Config, rows: u64, base: u64, head: Option<u32>) {
        let rows128 = u128::from(rows);
        let hidden = u128::from(config.hidden_size);
        let kv = u128::from(config.key_value_heads) * u128::from(config.head_dim);
        let intermediate = u128::from(config.effective_intermediate_size);
        let mut flops = 0;
        for (index, kind) in config.layers.iter().enumerate() {
            flops += match kind {
                LayerKind::Conv => {
                    8 * rows128 * hidden * hidden
                        + 2 * rows128 * hidden * u128::from(config.conv_width)
                }
                LayerKind::FullAttention => {
                    let attended = rows128 * u128::from(base) + rows128 * (rows128 + 1) / 2;
                    4 * rows128 * hidden * (hidden + kv) + 4 * hidden * attended
                }
            };
            // The final layer executes its FFN only for the final row, including base prefill.
            let ffn_rows = if index + 1 == config.layers.len() {
                1
            } else {
                rows128
            };
            flops += 6 * ffn_rows * hidden * intermediate;
        }
        if let Some(width) = head {
            flops += 2 * hidden * u128::from(width);
        }
        self.forward_passes = self.forward_passes.saturating_add(1);
        self.token_positions_processed = self.token_positions_processed.saturating_add(rows);
        self.estimated_flops = self.estimated_flops.saturating_add(flops);
    }
}
