//! Native Metal execution, independent from the WebGPU implementation.
//!
//! Finite operations record into retained-reference Metal command buffers.
//! Fence/readback calls submit, and polling observes status without waiting.
//! Shared buffers are initialized once before encoding; readback uses a private
//! snapshot copy. Allocations remain charged and retained through GPU completion.
#![deny(unsafe_code)]
#![allow(clippy::missing_errors_doc)]

#[allow(unsafe_code)]
mod bridge;
mod completion;
mod encoder;
mod kernels;
mod operations;
mod packed;

pub use completion::{MetalFence, MetalFenceRetirement, MetalReadback};
use kernels::Kernel;
use minifield_engine_api::{
    AllocationClass, BackendCapabilities, BackendIdentity, BackendKind, BackendLease, BufferAccess,
    BufferDescriptor, DType, DTypeSet, ExecutorError, OperationKind, OperationSet, PrecisionPolicy,
    ResourceLimits, ResourceReport, Result, Shape, TensorLayout,
};
use packed::Geometry;
use std::{
    cell::{Cell, RefCell},
    collections::BTreeMap,
    rc::Rc,
};

/// Actual native device identity, with no WebGPU adapter involved.
#[derive(Clone, Debug)]
pub struct MetalDeviceInfo {
    pub name: String,
    pub registry_id: u64,
    pub api: &'static str,
}

#[derive(Clone, Copy)]
enum Bucket {
    Classified(AllocationClass),
    Pending,
}

#[derive(Default)]
struct Accounting {
    report: ResourceReport,
    peak: u64,
}
impl Accounting {
    fn slot(&mut self, bucket: Bucket) -> &mut u64 {
        match bucket {
            Bucket::Classified(AllocationClass::Weight) => &mut self.report.resident_weight_bytes,
            Bucket::Classified(AllocationClass::Cache) => &mut self.report.cache_bytes,
            Bucket::Classified(AllocationClass::Scratch) => &mut self.report.scratch_bytes,
            Bucket::Classified(AllocationClass::Branch) => &mut self.report.staged_branch_bytes,
            Bucket::Pending => &mut self.report.pending_operation_bytes,
        }
    }
    fn add(&mut self, bucket: Bucket, bytes: u64, limits: ResourceLimits) -> Result<()> {
        self.reserve_buffers(bucket, &[bytes], limits).map(|_| ())
    }
    /// Admit independent buffers against their individual cap and the shared
    /// total before changing any accounting. A reservation isn't one allocation.
    fn reserve_buffers(
        &mut self,
        bucket: Bucket,
        sizes: &[u64],
        limits: ResourceLimits,
    ) -> Result<u64> {
        let mut total = self.report.total_owned_bytes()?;
        let mut reserved = 0_u64;
        for &bytes in sizes {
            limits.validate_allocation(bytes, total)?;
            total = total
                .checked_add(bytes)
                .ok_or(ExecutorError::Overflow("Metal resource total overflows"))?;
            reserved = reserved
                .checked_add(bytes)
                .ok_or(ExecutorError::Overflow("Metal reservation total overflows"))?;
        }
        let slot = self.slot(bucket);
        *slot = slot
            .checked_add(reserved)
            .ok_or(ExecutorError::Overflow("Metal accounting overflows"))?;
        self.peak = self.peak.max(total);
        Ok(reserved)
    }
    fn subtract(&mut self, bucket: Bucket, bytes: u64) {
        let slot = self.slot(bucket);
        *slot = slot.saturating_sub(bytes);
    }
}

struct Allocation {
    raw: bridge::Buffer,
    bytes: u64,
    bucket: Bucket,
    accounting: Rc<RefCell<Accounting>>,
}
impl Drop for Allocation {
    fn drop(&mut self) {
        self.accounting
            .borrow_mut()
            .subtract(self.bucket, self.bytes);
    }
}
struct Batch {
    command: bridge::Command,
    retained: Vec<Rc<Allocation>>,
}
struct Submission {
    command: bridge::Command,
    retained: RefCell<Vec<Rc<Allocation>>>,
    _serial: u64,
}
impl Submission {
    fn poll(&self) -> Option<Result<()>> {
        let result = self.command.poll();
        if result.is_some() {
            self.retained.borrow_mut().clear();
        }
        result
    }
}
struct Device {
    raw: bridge::Device,
    info: MetalDeviceInfo,
    identity: Cell<BackendIdentity>,
    lease: BackendLease,
    limits: ResourceLimits,
    accounting: Rc<RefCell<Accounting>>,
    allocation_id: Cell<u64>,
    serial: Cell<u64>,
    batch: RefCell<Option<Batch>>,
    pending: RefCell<Vec<Rc<Submission>>>,
    counts: RefCell<BTreeMap<&'static str, u64>>,
}
impl Device {
    fn reap(&self) {
        let mut pending = self.pending.borrow_mut();
        pending.retain(|submission| submission.poll().is_none());
        self.accounting.borrow_mut().report.pending_operations =
            u32::try_from(pending.len()).unwrap_or(u32::MAX);
    }
    fn alloc(&self, bytes: u64, bucket: Bucket, initial: Option<&[u8]>) -> Result<Rc<Allocation>> {
        self.reap();
        let physical = bytes.max(4);
        let length = usize::try_from(physical)
            .map_err(|_| ExecutorError::ResourceLimit("Metal allocation exceeds address space"))?;
        self.accounting
            .borrow_mut()
            .add(bucket, physical, self.limits)?;
        match self.raw.allocate(length, initial) {
            Ok(raw) => Ok(Rc::new(Allocation {
                raw,
                bytes: physical,
                bucket,
                accounting: Rc::clone(&self.accounting),
            })),
            Err(error) => {
                self.accounting.borrow_mut().subtract(bucket, physical);
                Err(error)
            }
        }
    }
    fn record<T>(
        &self,
        action: impl FnOnce(&bridge::Command) -> Result<T>,
        retained: &[Rc<Allocation>],
    ) -> Result<T> {
        self.reap();
        let mut batch = self.batch.borrow_mut();
        if batch.is_none() {
            *batch = Some(Batch {
                command: self.raw.command()?,
                retained: Vec::new(),
            });
        }
        let current = batch
            .as_mut()
            .ok_or(ExecutorError::BackendFailure("Metal batch missing"))?;
        current.retained.extend(retained.iter().cloned());
        action(&current.command)
    }
    fn submit(&self) -> Result<Rc<Submission>> {
        self.reap();
        if self.accounting.borrow().report.pending_operations >= self.limits.max_pending_operations
        {
            return Err(ExecutorError::ResourceLimit(
                "Metal pending submission limit exceeded",
            ));
        }
        let serial = self
            .serial
            .get()
            .checked_add(1)
            .ok_or(ExecutorError::Overflow("Metal submission serial exhausted"))?;
        let batch = self.batch.borrow_mut().take().map_or_else(
            || {
                self.raw.command().map(|command| Batch {
                    command,
                    retained: Vec::new(),
                })
            },
            Ok,
        )?;
        let submission = Rc::new(Submission {
            command: batch.command,
            retained: RefCell::new(batch.retained),
            _serial: serial,
        });
        self.serial.set(serial);
        submission.command.commit();
        let mut pending = self.pending.borrow_mut();
        pending.push(Rc::clone(&submission));
        self.accounting.borrow_mut().report.pending_operations = u32::try_from(pending.len())
            .map_err(|_| ExecutorError::Overflow("Metal pending count overflows"))?;
        Ok(submission)
    }
    fn dispatch(
        &self,
        kernel: Kernel,
        buffers: &[&MetalBuffer],
        words: &[u32],
        threads: u64,
    ) -> Result<()> {
        if threads == 0 {
            return Ok(());
        }
        let threads = usize::try_from(threads)
            .map_err(|_| ExecutorError::ResourceLimit("Metal grid exceeds address space"))?;
        self.encode(kernel, buffers, words, Geometry::Linear(threads))
    }
    fn encode(
        &self,
        kernel: Kernel,
        buffers: &[&MetalBuffer],
        words: &[u32],
        geometry: Geometry,
    ) -> Result<()> {
        if words.len() > 16 || buffers.len() > 8 {
            return Err(ExecutorError::InvalidArgument(
                "Metal binding limit exceeded",
            ));
        }
        // No command, retention extension or counter change precedes preflight.
        if !geometry.matches_kernel(kernel) {
            return Err(ExecutorError::InvalidArgument(
                "Metal kernel/grid kind differs",
            ));
        }
        if let Geometry::Tile8(grid) = geometry
            && grid.groups().contains(&0)
        {
            return Err(ExecutorError::InvalidArgument("Metal tile8 grid is empty"));
        }
        self.raw.validate_dispatch(kernel, geometry)?;
        let name = kernel.name();
        let next_count = self
            .counts
            .borrow()
            .get(name)
            .copied()
            .unwrap_or(0)
            .checked_add(1)
            .ok_or(ExecutorError::Overflow("Metal dispatch count overflows"))?;
        let mut params = [0_u32; 16];
        params[..words.len()].copy_from_slice(words);
        let retained: Vec<_> = buffers.iter().map(|b| Rc::clone(&b.allocation)).collect();
        let raw: Vec<_> = retained.iter().map(|a| &a.raw).collect();
        self.record(
            |command| self.raw.dispatch(command, kernel, &raw, &params, geometry),
            &retained,
        )?;
        self.counts.borrow_mut().insert(name, next_count);
        Ok(())
    }
}

/// Owned, checked native buffer. Physical storage isn't recycled during a submission.
pub struct MetalBuffer {
    descriptor: BufferDescriptor,
    lease: BackendLease,
    allocation: Rc<Allocation>,
}
impl MetalBuffer {
    #[must_use]
    pub fn descriptor(&self) -> BufferDescriptor {
        self.descriptor
    }
}

/// Native Metal backend with finite F32 arithmetic and portable packed storage.
pub struct MetalBackend {
    device: Rc<Device>,
    retirement: Rc<MetalFenceRetirement>,
    capabilities: BackendCapabilities,
}
impl MetalBackend {
    pub fn new(owner: u64, limits: ResourceLimits) -> Result<Self> {
        let raw = bridge::Device::new()?;
        let (name, registry_id, device_max) = raw.info();
        let identity = BackendIdentity {
            kind: BackendKind::Metal,
            ordinal: 0,
            owner,
            generation: 0,
        };
        let limits = ResourceLimits {
            max_allocation_bytes: limits.max_allocation_bytes.min(device_max),
            ..limits
        };
        let device = Rc::new(Device {
            raw,
            info: MetalDeviceInfo {
                name,
                registry_id,
                api: "metal",
            },
            identity: Cell::new(identity),
            lease: BackendLease::new(identity),
            limits,
            accounting: Rc::new(RefCell::new(Accounting::default())),
            allocation_id: Cell::new(0),
            serial: Cell::new(0),
            batch: RefCell::new(None),
            pending: RefCell::new(Vec::new()),
            counts: RefCell::new(BTreeMap::new()),
        });
        let retirement = Rc::new(MetalFenceRetirement::new(Rc::clone(&device)));
        let mut operations = OperationSet::empty();
        for op in [
            OperationKind::Copy,
            OperationKind::RectCopy2d,
            OperationKind::GatherRows,
            OperationKind::Add,
            OperationKind::Multiply,
            OperationKind::Linear,
            OperationKind::RowRmsNorm,
            OperationKind::Embedding,
            OperationKind::Rotary,
            OperationKind::GroupedQueryAttention,
            OperationKind::GatedShortConvolution,
            OperationKind::SwiGlu,
            OperationKind::LmProjection,
            OperationKind::PackedGatherRows,
            OperationKind::PackedLinear,
            OperationKind::AddRowRmsNorm,
            OperationKind::PackedLinearPair,
            OperationKind::PackedSwigluLinear,
            OperationKind::PackedSwigluPair,
            OperationKind::QkNormRope,
            OperationKind::Argmax,
            OperationKind::GatherColumns,
        ] {
            operations = operations.with(op);
        }
        let capabilities = BackendCapabilities {
            dtypes: DTypeSet::empty().with(DType::F32).with(DType::U8),
            operations,
            precision: PrecisionPolicy {
                weights: DType::F32,
                activations: DType::F32,
                cache: DType::F32,
                accumulation: DType::F32,
            },
            max_rank: 8,
            max_elements: u64::from(u32::MAX),
            max_allocation_bytes: limits.max_allocation_bytes,
            supports_nonblocking_completion: true,
        };
        Ok(Self {
            device,
            retirement,
            capabilities,
        })
    }
    #[must_use]
    pub fn device_info(&self) -> MetalDeviceInfo {
        self.device.info.clone()
    }
    #[must_use]
    pub fn dispatch_counts(&self) -> BTreeMap<&'static str, u64> {
        self.device.counts.borrow().clone()
    }
    fn check(&self, b: &MetalBuffer, dtype: DType) -> Result<()> {
        if !b.lease.same_actual_instance(&self.device.lease) {
            return Err(ExecutorError::WrongBackend);
        }
        b.descriptor.validate_for(self.device.identity.get())?;
        if b.descriptor.layout.dtype() != dtype {
            return Err(ExecutorError::InvalidDType("Metal operand dtype mismatch"));
        }
        if !b.descriptor.layout.is_contiguous()? {
            return Err(ExecutorError::InvalidLayout(
                "Metal requires contiguous operands",
            ));
        }
        Ok(())
    }
    fn output(&self, b: &MetalBuffer, shape: Shape) -> Result<()> {
        self.check(b, DType::F32)?;
        if b.descriptor.layout.shape() != shape {
            return Err(ExecutorError::InvalidShape("Metal output shape mismatch"));
        }
        if b.descriptor.access != BufferAccess::ReadWrite {
            return Err(ExecutorError::InvalidArgument(
                "Metal output isn't writable",
            ));
        }
        Ok(())
    }
    fn distinct(output: &MetalBuffer, inputs: &[&MetalBuffer]) -> Result<()> {
        if inputs
            .iter()
            .any(|input| Rc::ptr_eq(&output.allocation, &input.allocation))
        {
            return Err(ExecutorError::InvalidArgument(
                "Metal output aliases an input",
            ));
        }
        Ok(())
    }
    fn preflight_allocation(&self, shape: Shape, dtype: DType) -> Result<TensorLayout> {
        let layout = TensorLayout::contiguous(dtype, shape)?;
        self.capabilities.validate(
            dtype,
            OperationKind::Copy,
            shape.rank(),
            shape.element_count()?,
            layout.byte_extent(),
        )?;
        self.device.reap();
        self.device.limits.validate_allocation(
            layout.byte_extent().max(4),
            self.device.accounting.borrow().report.total_owned_bytes()?,
        )?;
        Ok(layout)
    }
    fn allocate(
        &self,
        shape: Shape,
        dtype: DType,
        class: AllocationClass,
        initial: Option<&[u8]>,
    ) -> Result<MetalBuffer> {
        let layout = self.preflight_allocation(shape, dtype)?;
        let id = self
            .device
            .allocation_id
            .get()
            .checked_add(1)
            .ok_or(ExecutorError::Overflow(
                "Metal allocation identity exhausted",
            ))?;
        let allocation =
            self.device
                .alloc(layout.byte_extent(), Bucket::Classified(class), initial)?;
        self.device.allocation_id.set(id);
        let identity = self.device.identity.get();
        Ok(MetalBuffer {
            descriptor: BufferDescriptor {
                backend: identity,
                allocation: id,
                layout,
                access: BufferAccess::ReadWrite,
            },
            lease: self.device.lease.with_identity(identity),
            allocation,
        })
    }
    fn stage_u32(&self, values: &[u32]) -> Result<MetalBuffer> {
        let count = u64::try_from(values.len())
            .map_err(|_| ExecutorError::Overflow("Metal selector count overflows"))?;
        let shape = Shape::new(&[product(count, 4)?])?;
        self.preflight_allocation(shape, DType::U8)?;
        let bytes = encode_words(values.iter().copied())?;
        self.allocate(shape, DType::U8, AllocationClass::Scratch, Some(&bytes))
    }
    fn stage_f32(&self, values: &[f32]) -> Result<MetalBuffer> {
        let shape = Shape::new(&[u64::try_from(values.len())
            .map_err(|_| ExecutorError::Overflow("Metal selector count overflows"))?])?;
        self.preflight_allocation(shape, DType::F32)?;
        let bytes = encode_words(values.iter().map(|value| value.to_bits()))?;
        self.allocate(shape, DType::F32, AllocationClass::Scratch, Some(&bytes))
    }
}
fn p(value: u64) -> Result<u32> {
    u32::try_from(value).map_err(|_| ExecutorError::ResourceLimit("Metal dimension exceeds u32"))
}
fn product(a: u64, b: u64) -> Result<u64> {
    a.checked_mul(b)
        .ok_or(ExecutorError::Overflow("Metal geometry product overflows"))
}

fn encode_words(words: impl ExactSizeIterator<Item = u32>) -> Result<Vec<u8>> {
    let bytes = words.len().checked_mul(4).ok_or(ExecutorError::Overflow(
        "Metal staging byte count overflows",
    ))?;
    let mut encoded = Vec::new();
    encoded
        .try_reserve_exact(bytes)
        .map_err(|_| ExecutorError::ResourceLimit("Metal byte staging allocation failed"))?;
    for word in words {
        encoded.extend_from_slice(&word.to_le_bytes());
    }
    Ok(encoded)
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;
    fn limits(total: u64) -> ResourceLimits {
        ResourceLimits {
            max_allocation_bytes: 1024,
            max_total_bytes: total,
            max_pending_operations: 2,
        }
    }
    #[test]
    fn accounting_keeps_classes_and_peak_through_release() {
        let mut a = Accounting::default();
        a.add(
            Bucket::Classified(AllocationClass::Weight),
            128,
            limits(512),
        )
        .expect("weight");
        a.add(Bucket::Classified(AllocationClass::Cache), 64, limits(512))
            .expect("cache");
        a.add(Bucket::Pending, 32, limits(512)).expect("pending");
        assert_eq!(a.report.total_owned_bytes().expect("total"), 224);
        assert_eq!(a.peak, 224);
        a.subtract(Bucket::Classified(AllocationClass::Weight), 128);
        assert_eq!(a.report.resident_weight_bytes, 0);
        assert_eq!(a.peak, 224);
    }
    #[test]
    fn rejected_allocation_preserves_every_counter() {
        let mut a = Accounting::default();
        a.add(
            Bucket::Classified(AllocationClass::Branch),
            128,
            limits(128),
        )
        .expect("branch");
        let previous = a.report;
        assert!(a.add(Bucket::Pending, 1, limits(128)).is_err());
        assert_eq!(a.report, previous);
        assert_eq!(a.peak, 128);
    }
    #[test]
    fn checked_geometry_rejects_overflow_before_dispatch() {
        assert!(product(u64::MAX, 2).is_err());
        assert!(p(u64::from(u32::MAX) + 1).is_err());
        assert_eq!(product(0, u64::MAX).expect("empty"), 0);
    }
    #[test]
    fn readback_reservation_applies_per_buffer_and_aggregate_caps() {
        let limits = ResourceLimits {
            max_allocation_bytes: 16,
            max_total_bytes: 64,
            max_pending_operations: 1,
        };
        let mut a = Accounting::default();
        a.add(Bucket::Classified(AllocationClass::Scratch), 16, limits)
            .expect("source");
        a.add(Bucket::Pending, 16, limits).expect("staging");
        assert_eq!(
            a.reserve_buffers(Bucket::Pending, &[16, 16], limits)
                .expect("byte and F32 result buffers"),
            32
        );
        assert_eq!(a.report.total_owned_bytes().expect("total"), 64);
        assert_eq!(a.peak, 64);
        a.subtract(Bucket::Pending, 48);
        let before = a.report;
        assert!(
            a.reserve_buffers(Bucket::Pending, &[17, 15], limits)
                .is_err()
        );
        assert_eq!(a.report, before);
        let tight = ResourceLimits {
            max_total_bytes: 47,
            ..limits
        };
        assert!(
            a.reserve_buffers(Bucket::Pending, &[16, 16], tight)
                .is_err()
        );
        assert_eq!(a.report, before);
    }
    #[test]
    fn qk_rotary_rejects_mismatched_and_odd_key_heads_during_admission() {
        use minifield_engine_api::PackedHeadSpec;
        let query = PackedHeadSpec::new(2, 2).expect("query");
        for dimension in [3, 4] {
            assert!(
                operations::validate_qk_rotary(
                    query,
                    PackedHeadSpec::new(1, dimension).expect("key"),
                    10000.0
                )
                .is_err()
            );
        }
        let key = PackedHeadSpec::new(1, 2).expect("matching key");
        assert_eq!(
            operations::validate_qk_rotary(query, key, 10000.0)
                .expect("rotary")
                .heads(),
            key
        );
        let odd = PackedHeadSpec::new(1, 3).expect("odd head");
        assert!(operations::validate_qk_rotary(odd, odd, 10000.0).is_err());
    }
    #[test]
    fn byte_staging_preserves_exact_float_bits() {
        let bytes =
            encode_words([0x8000_0000, 0x7f80_0000, 0x7fc0_0001].into_iter()).expect("staging");
        assert_eq!(bytes, [0, 0, 0, 128, 0, 0, 128, 127, 1, 0, 192, 127]);
    }
}
