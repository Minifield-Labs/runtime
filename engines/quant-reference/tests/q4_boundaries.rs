use minifield_quant_reference::q4::QuantizedMatrix;

#[test]
fn odd_row_boundaries_keep_nibble_and_scale_indices_separate() {
    // Continuous packed codes: [-8, -1, 7, 1, -2, 3]. Row 1 starts mid-byte.
    let matrix =
        QuantizedMatrix::from_parts(2, 3, 2, vec![0xf8, 0x17, 0x3e], vec![1.0, 2.0, 3.0, 4.0])
            .unwrap();
    let mut output = [0.0; 2];
    matrix.matvec(&[1.0; 3], &mut output).unwrap();
    assert_eq!(output, [5.0, 9.0]);
    assert_eq!(matrix.value(0, 2), Some(14.0));
    assert_eq!(matrix.value(1, 0), Some(3.0));
    assert_eq!(matrix.value(2, 0), None);
}

#[test]
fn invalid_parts_and_batch_shapes_return_errors() {
    assert!(QuantizedMatrix::from_parts(1, 1, 0, vec![1], vec![1.0]).is_err());
    for scale in [0.0, -1.0, f32::NAN, f32::INFINITY] {
        assert!(QuantizedMatrix::from_parts(1, 1, 1, vec![1], vec![scale]).is_err());
    }
    let matrix = QuantizedMatrix::from_parts(1, 1, 1, vec![1], vec![1.0]).unwrap();
    assert!(matrix.matmul(&[], 0, &mut []).is_err());
    assert!(matrix.matmul(&[1.0], 2, &mut [0.0; 2]).is_err());
}
