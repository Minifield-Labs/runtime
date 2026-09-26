//! Executor scalar probability regressions.

use super::scoring::{candidate_score_transport, token_log_probability};

#[test]
fn token_log_probability_keeps_normalization_when_equal_finite_logits_are_large() {
    let large = f32::from_bits(0x62b5_02e8);
    let logits = [large, large, large, large];
    let expected = -(4.0_f64).ln();
    let actual = match token_log_probability(&logits, 0) {
        Ok(value) => value,
        Err(error) => panic!("finite logits unexpectedly failed: {error:?}"),
    };
    assert!((actual - expected).abs() <= f64::EPSILON);
    let normal = match token_log_probability(&[0.0, 1.0], 1) {
        Ok(value) => value,
        Err(error) => panic!("normal logits unexpectedly failed: {error:?}"),
    };
    assert!((normal - (1.0_f64 - (1.0_f64.exp() + 1.0).ln())).abs() <= f64::EPSILON);
}

#[test]
fn candidate_score_transport_rejects_nonfinite_or_unrepresentable_f64() {
    match candidate_score_transport(-1.25) {
        Ok(value) => assert!((value + 1.25).abs() <= f32::EPSILON),
        Err(error) => panic!("finite transport unexpectedly failed: {error:?}"),
    }
    assert!(candidate_score_transport(f64::INFINITY).is_err());
    assert!(candidate_score_transport(f64::NEG_INFINITY).is_err());
    assert!(candidate_score_transport(f64::MAX).is_err());
    assert!(candidate_score_transport(-f64::MAX).is_err());
}
