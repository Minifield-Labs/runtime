//! Device, pooling, batching, and parameter plumbing for the wgpu backend.
//!
//! All portable op calls record into a pending [`PendingOp`] list. The list is
//! replayed into a fresh [`wgpu::CommandEncoder`] only when a fence or readback
//! completion needs a submission boundary, or when the uniform-parameter ring
//! wraps inside a single batch. Consecutive dispatches merge into one compute
//! pass and uniform params land in a single `Queue::write_buffer`. Submitted
//! work is tracked by monotonically increasing local serials;
//! `on_submitted_work_done` callbacks advance a shared confirmation counter
//! that drives pooled-buffer recycling.
//!
//! The buffer pool is safe because a dropped [`wgpu::Buffer`] cannot be
//! referenced by any submission made after its drop. Freed buffers therefore
//! become reusable once the submission serial current at drop time (plus the
//! pending batch that may still reference them) is confirmed complete.

use std::{
    cell::RefCell,
    collections::{HashMap, VecDeque},
    rc::Rc,
    sync::{Arc, Mutex},
};

use minifield_engine_api::{ExecutorError, Result};
// wasm32 lacks native atomics; portable-atomic keeps one code path.
use portable_atomic::{AtomicBool, AtomicU64, Ordering};

use crate::kernels::Kernel;

/// Threads per workgroup for all elementwise and reduction kernels.
pub const WORKGROUP_SIZE: u32 = 256;
/// Uniform-parameter slots available before a ring wrap forces a submission.
const UNIFORM_RING_SLOTS: u64 = 4096;
/// Maximum storage-buffer bindings behind the uniform params binding.
pub const MAX_STORAGE_BINDINGS: u32 = 8;

/// A pooled device buffer plus its size class.
pub struct PooledBuf {
    pub buffer: wgpu::Buffer,
    pub class: u64,
}

/// Round a byte size up to its allocation class: the next power of two below
/// 1 MiB (256 B minimum), else a 1/16 subdivision of the enclosing power of
/// two (max ~12.5% waste). Buffers are created at class size so any same-class
/// request can reuse them.
pub fn size_class(bytes: u64) -> u64 {
    let bytes = bytes.max(4);
    let np2 = bytes.next_power_of_two();
    if np2 <= (1 << 20) {
        np2.max(256)
    } else {
        bytes.div_ceil(np2 / 16) * (np2 / 16)
    }
}

/// Recycling pool keyed by size class.
#[derive(Default)]
pub struct BufferPool {
    free: HashMap<u64, Vec<PooledBuf>>,
}

impl BufferPool {
    fn take(&mut self, class: u64) -> Option<PooledBuf> {
        self.free.get_mut(&class).and_then(Vec::pop)
    }

    fn give(&mut self, buf: PooledBuf) {
        self.free.entry(buf.class).or_default().push(buf);
    }
}

/// Shared submission counters. Callbacks require `Send`, so these are atomics.
pub struct Counters {
    /// Highest submission serial confirmed complete by `on_submitted_work_done`.
    pub confirmed: AtomicU64,
}

/// A staging buffer whose owning readback was consumed or dropped before its
/// map callback ran. It returns to the staging pool once the map resolves.
pub struct Zombie {
    pub buf: PooledBuf,
    pub flag: Arc<AtomicBool>,
    pub outcome: Arc<Mutex<Option<bool>>>,
}

/// Command-recording state guarded by `RefCell`. `op` methods take `&self`, so
/// the pending batch lives behind interior mutability.
pub struct OpCtx {
    /// Recorded operations replayed into a fresh encoder at submit time.
    /// Consecutive dispatches merge into one compute pass; copies, clears,
    /// and deferred maps break the pass.
    ops: Vec<PendingOp>,
    /// Staged uniform-ring contents for the pending batch; one
    /// `Queue::write_buffer` at submit covers every recorded slot.
    uniform_staging: Vec<u8>,
    /// Buffers dropped since the last submission. They may still be referenced
    /// by the pending batch or any in-flight submission.
    pending_free: Vec<PooledBuf>,
    /// Freed buffers tagged with the submission serial that must complete
    /// before they can re-enter the pool.
    awaiting: VecDeque<(u64, Vec<PooledBuf>)>,
    /// Staging buffers awaiting a resolved map before recycling.
    zombies: Vec<Zombie>,
    /// Next free uniform-ring slot offset in bytes.
    uniform_cursor: u64,
    /// Local submission serial counter.
    submissions: u64,
}

/// One recorded operation in the pending batch.
pub enum PendingOp {
    Dispatch {
        pipeline: wgpu::ComputePipeline,
        bind_group: wgpu::BindGroup,
        dynamic_offset: u32,
        grid: (u32, u32, u32),
    },
    Copy {
        source: wgpu::Buffer,
        source_offset: u64,
        destination: wgpu::Buffer,
        destination_offset: u64,
        bytes: u64,
    },
    Clear {
        buffer: wgpu::Buffer,
        bytes: u64,
    },
    MapOnSubmit {
        staging: wgpu::Buffer,
        callback: MapCallback,
    },
}

type MapCallback = Box<dyn FnOnce(std::result::Result<(), wgpu::BufferAsyncError>) + Send>;

/// Dev-facing batching counters. Always collected (cheap `u64` adds); the
/// `*_ns` fields accumulate host wall time only when `MINIFIELD_WGPU_STATS`
/// was set at device init. `encode` covers the host-side `dispatch` path
/// (bind group + record); `submit` covers replay + `queue.submit`; the wait
/// fields measure submit-to-observe latency of each completion kind, which
/// includes GPU execution and callback delivery.
#[derive(Clone, Copy, Debug, Default)]
pub struct DeviceStats {
    pub dispatches: u64,
    pub copies: u64,
    pub clears: u64,
    pub compute_passes: u64,
    pub submits: u64,
    pub fences: u64,
    pub readbacks: u64,
    pub encode_ns: u64,
    pub submit_ns: u64,
    pub fence_wait_ns: u64,
    pub readback_wait_ns: u64,
}

impl DeviceStats {
    /// Read `MINIFIELD_WGPU_STATS` once; any non-empty value enables timing.
    pub fn timing_enabled() -> bool {
        static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ENABLED
            .get_or_init(|| std::env::var_os("MINIFIELD_WGPU_STATS").is_some_and(|v| !v.is_empty()))
    }
}

/// Per-class and aggregate byte accounting, mirroring the CPU tracker.
#[derive(Clone, Copy, Debug, Default)]
pub struct ClassBytes {
    pub weight: u64,
    pub cache: u64,
    pub scratch: u64,
    pub branch: u64,
}

impl ClassBytes {
    pub fn slot(&mut self, class: minifield_engine_api::AllocationClass) -> &mut u64 {
        match class {
            minifield_engine_api::AllocationClass::Weight => &mut self.weight,
            minifield_engine_api::AllocationClass::Cache => &mut self.cache,
            minifield_engine_api::AllocationClass::Scratch => &mut self.scratch,
            minifield_engine_api::AllocationClass::Branch => &mut self.branch,
        }
    }

    pub fn checked_add(
        &mut self,
        class: minifield_engine_api::AllocationClass,
        bytes: u64,
    ) -> Result<()> {
        let slot = self.slot(class);
        *slot = slot.checked_add(bytes).ok_or(ExecutorError::Overflow(
            "wgpu allocation class bytes overflow u64",
        ))?;
        Ok(())
    }

    pub fn saturating_sub(&mut self, class: minifield_engine_api::AllocationClass, bytes: u64) {
        let slot = self.slot(class);
        *slot = slot.saturating_sub(bytes);
    }

    pub fn total(self) -> Result<u64> {
        self.weight
            .checked_add(self.cache)
            .and_then(|value| value.checked_add(self.scratch))
            .and_then(|value| value.checked_add(self.branch))
            .ok_or(ExecutorError::Overflow(
                "wgpu allocation class total overflows u64",
            ))
    }
}

/// Resource accounting mirroring the CPU backend tracker.
pub struct Tracker {
    pub identity: minifield_engine_api::BackendIdentity,
    pub lease: minifield_engine_api::BackendLease,
    pub limits: minifield_engine_api::ResourceLimits,
    pub next_allocation: u64,
    pub live_bytes: u64,
    pub live_by_class: ClassBytes,
    pub pending_retained_bytes: u64,
    pub pending_retained_by_class: ClassBytes,
    pub pending_result_bytes: u64,
    pub pending_operations: u32,
    pub cancellation_requested: bool,
}

/// Bytes currently owned by the backend: live device buffers plus
/// completion-owned host results. Retained buffers are already inside
/// `live_bytes` and are not counted twice.
pub fn tracker_owned_bytes(tracker: &Tracker) -> Result<u64> {
    tracker
        .live_bytes
        .checked_add(tracker.pending_result_bytes)
        .ok_or(ExecutorError::Overflow(
            "wgpu owned resource total overflows u64",
        ))
}

/// Everything ops need from the device: handles, layouts, pools, the pending
/// batch, and resource accounting.
pub struct DeviceInner {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    /// Indexed by storage-binding count; entry 0 is the params-only layout.
    bind_group_layouts: Vec<wgpu::BindGroupLayout>,
    pipeline_layouts: Vec<wgpu::PipelineLayout>,
    pipelines: RefCell<HashMap<Kernel, wgpu::ComputePipeline>>,
    /// `STORAGE|COPY_SRC|COPY_DST` buffers (user buffers and kernel scratch).
    pool: RefCell<BufferPool>,
    /// `MAP_READ|COPY_DST` readback staging buffers.
    staging_pool: RefCell<BufferPool>,
    /// Shared uniform ring for kernel parameter blocks.
    uniform: wgpu::Buffer,
    uniform_stride: u64,
    uniform_ring_bytes: u64,
    ctx: RefCell<OpCtx>,
    pub tracker: RefCell<Tracker>,
    counters: Arc<Counters>,
    /// Batching counters and (env-gated) host timing. See [`DeviceStats`].
    pub stats: RefCell<DeviceStats>,
    stats_timing: bool,
    /// Per-kernel dispatch counts, printed with stats on drop.
    kernels: RefCell<HashMap<Kernel, u64>>,
    /// Limits the device was created with (post adapter clamping).
    pub device_limits: wgpu::Limits,
}

impl Drop for DeviceInner {
    #[allow(clippy::cast_precision_loss)] // diagnostic counters print as ms
    fn drop(&mut self) {
        if self.stats_timing {
            let s = self.stats.borrow();
            eprintln!(
                "wgpu stats: dispatches={} copies={} clears={} passes={} submits={} \
                 fences={} readbacks={} encode={:.2}ms submit={:.2}ms \
                 fence_wait={:.2}ms readback_wait={:.2}ms",
                s.dispatches,
                s.copies,
                s.clears,
                s.compute_passes,
                s.submits,
                s.fences,
                s.readbacks,
                s.encode_ns as f64 / 1e6,
                s.submit_ns as f64 / 1e6,
                s.fence_wait_ns as f64 / 1e6,
                s.readback_wait_ns as f64 / 1e6,
            );
            let mut kernels: Vec<(Kernel, u64)> = self
                .kernels
                .borrow()
                .iter()
                .map(|(k, c)| (*k, *c))
                .collect();
            kernels.sort_by_key(|entry| std::cmp::Reverse(entry.1));
            for (kernel, count) in kernels {
                eprintln!("  kernel {:<24} x{}", kernel.name(), count);
            }
        }
    }
}

impl DeviceInner {
    /// Create a device on the best available adapter. `ordinal` selects among
    /// adapters ranked discrete > integrated > virtual > CPU.
    pub fn new(
        owner: u64,
        ordinal: u32,
        limits: minifield_engine_api::ResourceLimits,
    ) -> Result<Rc<Self>> {
        pollster::block_on(Self::new_async(owner, ordinal, limits))
    }

    #[allow(clippy::too_many_lines)]
    async fn new_async(
        owner: u64,
        ordinal: u32,
        limits: minifield_engine_api::ResourceLimits,
    ) -> Result<Rc<Self>> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapters = instance.enumerate_adapters(wgpu::Backends::all()).await;
        if adapters.is_empty() {
            return Err(ExecutorError::BackendFailure(
                "wgpu: no adapters found (no GPU driver or WebGPU support)",
            ));
        }
        let score = |adapter: &wgpu::Adapter| match adapter.get_info().device_type {
            wgpu::DeviceType::DiscreteGpu => 4_u32,
            wgpu::DeviceType::IntegratedGpu => 3,
            wgpu::DeviceType::VirtualGpu => 2,
            wgpu::DeviceType::Cpu => 1,
            wgpu::DeviceType::Other => 0,
        };
        let mut ranked: Vec<(u32, usize)> = adapters
            .iter()
            .enumerate()
            .map(|(index, adapter)| (score(adapter), index))
            .collect();
        ranked.sort_by(|left, right| right.0.cmp(&left.0).then(left.1.cmp(&right.1)));
        let position = usize::try_from(ordinal).unwrap_or(usize::MAX);
        let index = ranked.get(position).map_or(ranked[0].1, |entry| entry.1);
        let adapter = &adapters[index];

        // Default limits fail a 268 MB f32 lm_head (128 MiB storage binding,
        // 256 MiB buffer). Request the adapter's reported storage/buffer maxima
        // and its exact minimum alignments; every other limit stays at the
        // portable WebGPU default.
        let adapter_limits = adapter.limits();
        let device_limits = wgpu::Limits {
            max_storage_buffer_binding_size: adapter_limits.max_storage_buffer_binding_size,
            max_buffer_size: adapter_limits.max_buffer_size,
            min_uniform_buffer_offset_alignment: adapter_limits.min_uniform_buffer_offset_alignment,
            min_storage_buffer_offset_alignment: adapter_limits.min_storage_buffer_offset_alignment,
            ..wgpu::Limits::default()
        };

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("minifield-wgpu"),
                required_features: wgpu::Features::empty(),
                required_limits: device_limits.clone(),
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
            })
            .await
            .map_err(|_| {
                ExecutorError::BackendFailure("wgpu: device request failed on selected adapter")
            })?;

        // One bind group layout per storage-binding count. Binding 0 is the
        // uniform params block with a dynamic offset; every other binding is a
        // read-write storage buffer, so one layout per count serves all kernels.
        let mut bind_group_layouts = Vec::with_capacity(MAX_STORAGE_BINDINGS as usize + 1);
        let mut pipeline_layouts = Vec::with_capacity(MAX_STORAGE_BINDINGS as usize + 1);
        for storages in 0..=MAX_STORAGE_BINDINGS {
            let mut entries = Vec::with_capacity(storages as usize + 1);
            entries.push(wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: true,
                    min_binding_size: None,
                },
                count: None,
            });
            for binding in 1..=storages {
                entries.push(wgpu::BindGroupLayoutEntry {
                    binding,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                });
            }
            let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("minifield-bgl"),
                entries: &entries,
            });
            let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("minifield-pl"),
                bind_group_layouts: &[Some(&bgl)],
                immediate_size: 0,
            });
            bind_group_layouts.push(bgl);
            pipeline_layouts.push(pl);
        }

        let uniform_stride = device_limits.min_uniform_buffer_offset_alignment.max(256);
        let uniform_ring_bytes = u64::from(uniform_stride)
            .checked_mul(UNIFORM_RING_SLOTS)
            .ok_or(ExecutorError::Overflow(
                "wgpu uniform ring size overflows u64",
            ))?;
        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("minifield-uniform-ring"),
            size: uniform_ring_bytes,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let identity = minifield_engine_api::BackendIdentity {
            kind: minifield_engine_api::BackendKind::Other(1),
            ordinal,
            owner,
            generation: 1,
        };
        Ok(Rc::new(Self {
            device,
            queue,
            bind_group_layouts,
            pipeline_layouts,
            pipelines: RefCell::new(HashMap::new()),
            pool: RefCell::new(BufferPool::default()),
            staging_pool: RefCell::new(BufferPool::default()),
            uniform,
            uniform_stride: u64::from(uniform_stride),
            uniform_ring_bytes,
            ctx: RefCell::new(OpCtx {
                ops: Vec::new(),
                uniform_staging: Vec::new(),
                pending_free: Vec::new(),
                awaiting: VecDeque::new(),
                zombies: Vec::new(),
                uniform_cursor: 0,
                submissions: 0,
            }),
            tracker: RefCell::new(Tracker {
                identity,
                lease: minifield_engine_api::BackendLease::new(identity),
                limits,
                next_allocation: 1,
                live_bytes: 0,
                live_by_class: ClassBytes::default(),
                pending_retained_bytes: 0,
                pending_retained_by_class: ClassBytes::default(),
                pending_result_bytes: 0,
                pending_operations: 0,
                cancellation_requested: false,
            }),
            counters: Arc::new(Counters {
                confirmed: AtomicU64::new(0),
            }),
            stats: RefCell::new(DeviceStats::default()),
            stats_timing: DeviceStats::timing_enabled(),
            kernels: RefCell::new(HashMap::new()),
            device_limits,
        }))
    }

    /// Whether host-side timing accumulation is on (`MINIFIELD_WGPU_STATS`).
    pub fn stats_timing(&self) -> bool {
        self.stats_timing
    }

    /// Highest submission serial confirmed complete.
    pub fn confirmed_serial(&self) -> u64 {
        self.counters.confirmed.load(Ordering::Acquire)
    }

    /// Drive wgpu maintenance without blocking: fires map and
    /// submitted-work-done callbacks for finished submissions.
    pub fn poll_once(&self) {
        drop(self.device.poll(wgpu::PollType::Poll));
    }

    /// Return completed frees to their pools and retire resolved zombie maps.
    /// Called before allocations and dispatches so recycling keeps pace with
    /// submissions even while no completion is being polled.
    pub fn reap(&self) {
        self.poll_once();
        let confirmed = self.confirmed_serial();
        let mut ctx = self.ctx.borrow_mut();
        let mut pool = self.pool.borrow_mut();
        while let Some((serial, _)) = ctx.awaiting.front() {
            if *serial > confirmed {
                break;
            }
            let Some((_, bufs)) = ctx.awaiting.pop_front() else {
                break;
            };
            for buf in bufs {
                pool.give(buf);
            }
        }
        drop(pool);
        let mut staging = self.staging_pool.borrow_mut();
        ctx.zombies.retain_mut(|zombie| {
            if !zombie.flag.load(Ordering::Acquire) {
                return true;
            }
            if *zombie
                .outcome
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                == Some(true)
            {
                zombie.buf.buffer.unmap();
            }
            staging.give(PooledBuf {
                buffer: zombie.buf.buffer.clone(),
                class: zombie.buf.class,
            });
            false
        });
    }

    /// Allocate a pooled storage buffer of at least `bytes`, creating one at
    /// class size on a miss.
    pub fn alloc_storage(&self, bytes: u64) -> Result<PooledBuf> {
        self.reap();
        let class = size_class(bytes);
        let maximum = self
            .device_limits
            .max_buffer_size
            .min(self.device_limits.max_storage_buffer_binding_size);
        if class > maximum {
            return Err(ExecutorError::ResourceLimit(
                "wgpu buffer request exceeds device binding maximum",
            ));
        }
        if let Some(buf) = self.pool.borrow_mut().take(class) {
            return Ok(buf);
        }
        Ok(PooledBuf {
            buffer: self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("minifield-storage"),
                size: class,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            class,
        })
    }

    /// Allocate a pooled `MAP_READ` staging buffer of at least `bytes`.
    pub fn alloc_staging(&self, bytes: u64) -> PooledBuf {
        let class = size_class(bytes.max(4));
        if let Some(buf) = self.staging_pool.borrow_mut().take(class) {
            return buf;
        }
        PooledBuf {
            buffer: self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("minifield-staging"),
                size: class,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            class,
        }
    }

    /// Return a resolved staging buffer to its pool after unmapping.
    pub fn release_staging(&self, buf: PooledBuf, mapped: bool) {
        if mapped {
            buf.buffer.unmap();
        }
        self.staging_pool.borrow_mut().give(buf);
    }

    /// Quarantine a staging buffer whose map callback has not resolved yet.
    pub fn zombie_staging(
        &self,
        buf: PooledBuf,
        flag: Arc<AtomicBool>,
        outcome: Arc<Mutex<Option<bool>>>,
    ) {
        self.ctx
            .borrow_mut()
            .zombies
            .push(Zombie { buf, flag, outcome });
    }

    /// Defer recycling a storage buffer until the pending batch and all
    /// in-flight submissions that could reference it have completed.
    pub fn defer_free(&self, buf: PooledBuf) {
        self.ctx.borrow_mut().pending_free.push(buf);
    }

    /// Submit the pending batch. Flushes staged uniform bytes in one
    /// `write_buffer`, replays recorded operations into a fresh encoder with
    /// consecutive dispatches merged into shared compute passes, registers
    /// the serial-confirmation callback, and moves `pending_free` into
    /// `awaiting` under the new serial.
    fn submit_locked(&self, ctx: &mut OpCtx) -> Result<u64> {
        let start = self.stats_timing.then(std::time::Instant::now);
        if !ctx.uniform_staging.is_empty() {
            self.queue
                .write_buffer(&self.uniform, 0, &ctx.uniform_staging);
            ctx.uniform_staging.clear();
        }
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("minifield-encoder"),
            });
        {
            let mut ops = ctx.ops.drain(..).collect::<VecDeque<_>>();
            while let Some(op) = ops.pop_front() {
                match op {
                    PendingOp::Dispatch { .. } => {
                        let mut run = Vec::new();
                        run.push(op);
                        while let Some(PendingOp::Dispatch { .. }) = ops.front() {
                            let Some(next) = ops.pop_front() else { break };
                            run.push(next);
                        }
                        self.stats.borrow_mut().compute_passes += 1;
                        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                            label: Some("minifield-batch"),
                            timestamp_writes: None,
                        });
                        for op in run {
                            let PendingOp::Dispatch {
                                pipeline,
                                bind_group,
                                dynamic_offset,
                                grid,
                            } = op
                            else {
                                unreachable!("run holds dispatches only");
                            };
                            pass.set_pipeline(&pipeline);
                            pass.set_bind_group(0, &bind_group, &[dynamic_offset]);
                            pass.dispatch_workgroups(grid.0, grid.1, grid.2);
                        }
                    }
                    PendingOp::Copy {
                        source,
                        source_offset,
                        destination,
                        destination_offset,
                        bytes,
                    } => {
                        encoder.copy_buffer_to_buffer(
                            &source,
                            source_offset,
                            &destination,
                            destination_offset,
                            bytes,
                        );
                    }
                    PendingOp::Clear { buffer, bytes } => {
                        encoder.clear_buffer(&buffer, 0, Some(bytes));
                    }
                    PendingOp::MapOnSubmit { staging, callback } => {
                        encoder.map_buffer_on_submit(&staging, wgpu::MapMode::Read, .., callback);
                    }
                }
            }
        }
        let serial = ctx
            .submissions
            .checked_add(1)
            .ok_or(ExecutorError::Overflow(
                "wgpu submission serial overflows u64",
            ))?;
        ctx.submissions = serial;
        {
            let counters = Arc::clone(&self.counters);
            encoder.on_submitted_work_done(move || {
                counters.confirmed.fetch_max(serial, Ordering::AcqRel);
            });
        }
        let _submission_index = self.queue.submit([encoder.finish()]);
        if !ctx.pending_free.is_empty() {
            ctx.awaiting
                .push_back((serial, core::mem::take(&mut ctx.pending_free)));
        }
        ctx.uniform_cursor = 0;
        {
            let mut stats = self.stats.borrow_mut();
            stats.submits += 1;
            if let Some(start) = start {
                stats.submit_ns += u64::try_from(start.elapsed().as_nanos()).unwrap_or(u64::MAX);
            }
        }
        Ok(serial)
    }

    /// Submit the pending batch and return its serial. Used by `fence()` and
    /// `read_f32_async`; also called internally when the uniform ring wraps.
    pub fn submit_pending(&self) -> Result<u64> {
        let mut ctx = self.ctx.borrow_mut();
        self.submit_locked(&mut ctx)
    }

    /// Reserve a uniform-ring slot for one dispatch. A ring wrap inside a batch
    /// submits the pending encoder first, because `Queue::write_buffer` is
    /// queue-ordered between submissions: slots reused across submissions are
    /// safe, but reusing a slot inside one submission would corrupt params of
    /// the earlier dispatch.
    fn uniform_slot(&self, ctx: &mut OpCtx) -> Result<u64> {
        if ctx
            .uniform_cursor
            .checked_add(self.uniform_stride)
            .ok_or(ExecutorError::Overflow("wgpu uniform cursor overflows u64"))?
            > self.uniform_ring_bytes
        {
            self.submit_locked(ctx)?;
        }
        let offset = ctx.uniform_cursor;
        ctx.uniform_cursor += self.uniform_stride;
        Ok(offset)
    }

    fn pipeline(&self, kernel: Kernel) -> wgpu::ComputePipeline {
        if let Some(pipeline) = self.pipelines.borrow().get(&kernel) {
            return pipeline.clone();
        }
        let source = kernel.source();
        let module = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(kernel.name()),
                source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Owned(source)),
            });
        let pipeline = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(kernel.name()),
                layout: Some(&self.pipeline_layouts[kernel.storage_bindings() as usize]),
                module: &module,
                entry_point: Some("main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
        self.pipelines
            .borrow_mut()
            .entry(kernel)
            .or_insert(pipeline)
            .clone()
    }

    /// Record one compute dispatch into the pending batch.
    ///
    /// `params` are little-endian u32/f32 words matching the kernel's uniform
    /// `Params` block. `storages` must match the kernel's declared storage
    /// binding count.
    pub fn dispatch(
        &self,
        kernel: Kernel,
        storages: &[&wgpu::Buffer],
        params: &[u8],
        grid: (u32, u32, u32),
    ) -> Result<()> {
        if grid.0 == 0 || grid.1 == 0 || grid.2 == 0 {
            return Ok(());
        }
        if storages.len() != kernel.storage_bindings() as usize {
            return Err(ExecutorError::BackendFailure(
                "wgpu dispatch storage binding count mismatch",
            ));
        }
        if params.len() as u64 > self.uniform_stride {
            return Err(ExecutorError::BackendFailure(
                "wgpu kernel params exceed uniform slot",
            ));
        }
        let start = self.stats_timing.then(std::time::Instant::now);
        self.reap();
        let pipeline = self.pipeline(kernel);
        let mut ctx = self.ctx.borrow_mut();
        let offset = self.uniform_slot(&mut ctx)?;
        let offset_usize = usize::try_from(offset)
            .map_err(|_| ExecutorError::Overflow("wgpu uniform offset exceeds usize"))?;
        ctx.uniform_staging.resize(offset_usize, 0);
        ctx.uniform_staging.extend_from_slice(params);
        // Keep the staged bytes slot-sized so offsets stay ring-aligned.
        let aligned_end = usize::try_from(ctx.uniform_cursor)
            .map_err(|_| ExecutorError::Overflow("wgpu uniform cursor exceeds usize"))?;
        ctx.uniform_staging.resize(aligned_end, 0);

        let mut entries = Vec::with_capacity(storages.len() + 1);
        entries.push(wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer: &self.uniform,
                offset: 0,
                size: wgpu::BufferSize::new(self.uniform_stride),
            }),
        });
        for (index, buffer) in storages.iter().enumerate() {
            entries.push(wgpu::BindGroupEntry {
                binding: u32::try_from(index + 1)
                    .map_err(|_| ExecutorError::Overflow("wgpu binding index overflows u32"))?,
                resource: (*buffer).as_entire_binding(),
            });
        }
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(kernel.name()),
            layout: &self.bind_group_layouts[kernel.storage_bindings() as usize],
            entries: &entries,
        });

        let dynamic_offset = u32::try_from(offset)
            .map_err(|_| ExecutorError::Overflow("wgpu uniform offset exceeds u32"))?;
        ctx.ops.push(PendingOp::Dispatch {
            pipeline,
            bind_group,
            dynamic_offset,
            grid,
        });
        drop(ctx);
        *self.kernels.borrow_mut().entry(kernel).or_insert(0) += 1;
        let mut stats = self.stats.borrow_mut();
        stats.dispatches += 1;
        if let Some(start) = start {
            stats.encode_ns += u64::try_from(start.elapsed().as_nanos()).unwrap_or(u64::MAX);
        }
        Ok(())
    }

    /// Record a buffer-to-buffer copy into the pending batch.
    pub fn record_copy(
        &self,
        source: &wgpu::Buffer,
        source_offset: u64,
        destination: &wgpu::Buffer,
        destination_offset: u64,
        bytes: u64,
    ) {
        if bytes == 0 {
            return;
        }
        let mut ctx = self.ctx.borrow_mut();
        ctx.ops.push(PendingOp::Copy {
            source: source.clone(),
            source_offset,
            destination: destination.clone(),
            destination_offset,
            bytes,
        });
        self.stats.borrow_mut().copies += 1;
    }

    /// Record a zero fill of `buffer[0..bytes]` into the pending batch.
    pub fn record_clear(&self, buffer: &wgpu::Buffer, bytes: u64) {
        if bytes == 0 {
            return;
        }
        let mut ctx = self.ctx.borrow_mut();
        ctx.ops.push(PendingOp::Clear {
            buffer: buffer.clone(),
            bytes,
        });
        self.stats.borrow_mut().clears += 1;
    }

    /// Attach a deferred map request to the pending batch. The map executes
    /// when the batch is submitted, so the staging copy records first.
    pub fn map_staging_on_submit(
        &self,
        staging: &wgpu::Buffer,
        callback: impl FnOnce(std::result::Result<(), wgpu::BufferAsyncError>) + Send + 'static,
    ) {
        let mut ctx = self.ctx.borrow_mut();
        ctx.ops.push(PendingOp::MapOnSubmit {
            staging: staging.clone(),
            callback: Box::new(callback),
        });
    }
}
