use minifield_backend_cpu::{CpuBackend, CpuBuffer};
use minifield_engine_api::{AllocationClass, ExecutorError, ResourceLimits, Result, Shape};

fn must<T>(result: Result<T>) -> T {
    match result {
        Ok(value) => value,
        Err(error) => panic!("synthetic packed pair fixture failed: {error:?}"),
    }
}

fn backend() -> CpuBackend {
    CpuBackend::new(
        37,
        ResourceLimits {
            max_allocation_bytes: 4096,
            max_total_bytes: 8192,
            max_pending_operations: 2,
        },
    )
}

fn codes(backend: &mut CpuBackend) -> CpuBuffer {
    must(backend.upload_u8_classified(
        must(Shape::new(&[2, 32])),
        &[0xAA; 64],
        AllocationClass::Weight,
    ))
}

fn scales(backend: &mut CpuBackend) -> CpuBuffer {
    must(backend.upload_f32(must(Shape::new(&[2, 1])), &[1.0, 2.0]))
}

struct PairFixture {
    input: CpuBuffer,
    codes_a: CpuBuffer,
    scales_a: CpuBuffer,
    codes_b: CpuBuffer,
    scales_b: CpuBuffer,
    output: CpuBuffer,
    other_output: CpuBuffer,
}

impl PairFixture {
    fn new(backend: &mut CpuBackend) -> Self {
        Self {
            input: must(backend.upload_f32(must(Shape::new(&[1, 128])), &[0.015; 128])),
            codes_a: codes(backend),
            scales_a: scales(backend),
            codes_b: codes(backend),
            scales_b: scales(backend),
            output: must(backend.upload_f32(must(Shape::new(&[1, 2])), &[101.0, 102.0])),
            other_output: must(backend.upload_f32(must(Shape::new(&[1, 2])), &[201.0, 202.0])),
        }
    }

    fn rejects_b(
        &mut self,
        backend: &CpuBackend,
        bad_codes: Option<&CpuBuffer>,
        bad_scales: Option<&CpuBuffer>,
        expected: ExecutorError,
    ) {
        let codes_b = bad_codes.unwrap_or(&self.codes_b);
        let scales_b = bad_scales.unwrap_or(&self.scales_b);
        assert_eq!(
            backend.packed_swiglu_pair(
                &mut self.output,
                &self.input,
                &self.codes_a,
                &self.scales_a,
                codes_b,
                scales_b,
            ),
            Err(expected.clone()),
        );
        assert_eq!(self.output.as_slice(), &[101.0, 102.0]);
        assert_eq!(
            backend.packed_linear_pair(
                &mut self.output,
                &mut self.other_output,
                &self.input,
                &self.codes_a,
                &self.scales_a,
                codes_b,
                scales_b,
            ),
            Err(expected),
        );
        assert_eq!(self.output.as_slice(), &[101.0, 102.0]);
        assert_eq!(self.other_output.as_slice(), &[201.0, 202.0]);
    }
}

#[test]
fn pairs_reject_wrong_b_dtypes_without_mutating_outputs() {
    let mut backend = backend();
    let mut fixture = PairFixture::new(&mut backend);
    let bad_codes = must(backend.upload_f32(must(Shape::new(&[2, 32])), &[1.0; 64]));
    let bad_scales = must(backend.upload_u8_classified(
        must(Shape::new(&[2, 1])),
        &[1, 2],
        AllocationClass::Weight,
    ));
    fixture.rejects_b(
        &backend,
        Some(&bad_codes),
        None,
        ExecutorError::InvalidDType("expected a u8 operand"),
    );
    fixture.rejects_b(
        &backend,
        None,
        Some(&bad_scales),
        ExecutorError::InvalidDType("expected an f32 operand"),
    );
}

#[test]
fn pairs_reject_foreign_b_buffers_with_colliding_identity_without_mutating_outputs() {
    let mut owner = backend();
    let mut foreign = backend();
    assert_eq!(owner.identity(), foreign.identity());
    let mut fixture = PairFixture::new(&mut owner);
    let bad_codes = codes(&mut foreign);
    let bad_scales = scales(&mut foreign);
    fixture.rejects_b(&owner, Some(&bad_codes), None, ExecutorError::WrongBackend);
    fixture.rejects_b(&owner, None, Some(&bad_scales), ExecutorError::WrongBackend);
}

#[test]
fn pairs_reject_stale_b_buffers_without_mutating_outputs() {
    let mut backend = backend();
    let bad_codes = codes(&mut backend);
    let bad_scales = scales(&mut backend);
    must(backend.advance_generation());
    let mut fixture = PairFixture::new(&mut backend);
    fixture.rejects_b(&backend, Some(&bad_codes), None, ExecutorError::StaleBuffer);
    fixture.rejects_b(
        &backend,
        None,
        Some(&bad_scales),
        ExecutorError::StaleBuffer,
    );
}

#[test]
fn linear_pair_rejects_foreign_second_output_before_mutating_first_output() {
    let mut owner = backend();
    let mut foreign = backend();
    let mut fixture = PairFixture::new(&mut owner);
    let mut bad_output = must(foreign.upload_f32(must(Shape::new(&[1, 2])), &[201.0, 202.0]));
    assert_eq!(
        owner.packed_linear_pair(
            &mut fixture.output,
            &mut bad_output,
            &fixture.input,
            &fixture.codes_a,
            &fixture.scales_a,
            &fixture.codes_b,
            &fixture.scales_b,
        ),
        Err(ExecutorError::WrongBackend),
    );
    assert_eq!(fixture.output.as_slice(), &[101.0, 102.0]);
    assert_eq!(bad_output.as_slice(), &[201.0, 202.0]);
}

#[test]
fn paired_projections_preserve_sequential_projection_and_activation_results() {
    let mut backend = backend();
    let mut fixture = PairFixture::new(&mut backend);
    let mut expected_a = must(backend.allocate_f32(must(Shape::new(&[1, 2]))));
    let mut expected_b = must(backend.allocate_f32(must(Shape::new(&[1, 2]))));
    let mut expected_swiglu = must(backend.allocate_f32(must(Shape::new(&[1, 2]))));
    must(backend.packed_linear(
        &mut expected_a,
        &fixture.input,
        &fixture.codes_a,
        &fixture.scales_a,
    ));
    must(backend.packed_linear(
        &mut expected_b,
        &fixture.input,
        &fixture.codes_b,
        &fixture.scales_b,
    ));
    must(backend.swiglu(&mut expected_swiglu, &expected_a, &expected_b));
    must(backend.packed_linear_pair(
        &mut fixture.output,
        &mut fixture.other_output,
        &fixture.input,
        &fixture.codes_a,
        &fixture.scales_a,
        &fixture.codes_b,
        &fixture.scales_b,
    ));
    assert_eq!(fixture.output.as_slice(), expected_a.as_slice());
    assert_eq!(fixture.other_output.as_slice(), expected_b.as_slice());
    must(backend.packed_swiglu_pair(
        &mut fixture.output,
        &fixture.input,
        &fixture.codes_a,
        &fixture.scales_a,
        &fixture.codes_b,
        &fixture.scales_b,
    ));
    assert_eq!(fixture.output.as_slice(), expected_swiglu.as_slice());
}
