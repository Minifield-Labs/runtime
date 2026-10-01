//! Audited native boundary. The caller validates tensor extents and shader parameters.
//! Shared allocations are initialized before encoding and read only after their
//! snapshot-copy command buffer reaches Completed. No CPU write touches queued storage.

use crate::{
    kernels::Kernel,
    packed::{Geometry, PipelineCaps},
};
use minifield_engine_api::{ExecutorError, Result};

#[cfg(target_os = "macos")]
mod native {
    use super::{ExecutorError, Geometry, Kernel, PipelineCaps, Result};
    use objc2::{rc::Retained, runtime::ProtocolObject};
    use objc2_foundation::NSString;
    use objc2_metal::{
        MTLBlitCommandEncoder, MTLBuffer, MTLCommandBuffer, MTLCommandBufferStatus,
        MTLCommandEncoder, MTLCommandQueue, MTLCompileOptions, MTLComputeCommandEncoder,
        MTLComputePipelineState, MTLCreateSystemDefaultDevice, MTLDevice, MTLLibrary,
        MTLResourceOptions, MTLSize,
    };
    use std::{collections::BTreeMap, ptr::NonNull};

    // MTLCreateSystemDefaultDevice requires CoreGraphics to be loaded.
    #[link(name = "CoreGraphics", kind = "framework")]
    unsafe extern "C" {}

    pub struct Device {
        gpu: Retained<ProtocolObject<dyn MTLDevice>>,
        queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
        pipelines: BTreeMap<Kernel, Pipeline>,
    }
    struct Pipeline {
        raw: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
        caps: PipelineCaps,
    }
    pub struct Buffer(pub Retained<ProtocolObject<dyn MTLBuffer>>);
    pub struct Command(pub Retained<ProtocolObject<dyn MTLCommandBuffer>>);

    impl Device {
        pub fn new() -> Result<Self> {
            let device = MTLCreateSystemDefaultDevice().ok_or(ExecutorError::Unsupported(
                "native Metal device unavailable",
            ))?;
            let queue = device
                .newCommandQueue()
                .ok_or(ExecutorError::BackendFailure(
                    "Metal command queue creation failed",
                ))?;
            let options = MTLCompileOptions::new();
            #[allow(deprecated)]
            options.setFastMathEnabled(false);
            let library = device
                .newLibraryWithSource_options_error(
                    &NSString::from_str(&crate::kernels::library_source()),
                    Some(&options),
                )
                .map_err(|error| {
                    eprintln!("Metal shader compiler: {error}");
                    ExecutorError::BackendFailure("Metal shader compilation failed")
                })?;
            let mut pipelines = BTreeMap::new();
            for kernel in Kernel::ALL {
                let function = library
                    .newFunctionWithName(&NSString::from_str(kernel.name()))
                    .ok_or(ExecutorError::BackendFailure(
                        "Metal shader function missing",
                    ))?;
                let pipeline = device
                    .newComputePipelineStateWithFunction_error(&function)
                    .map_err(|_| ExecutorError::BackendFailure("Metal pipeline creation failed"))?;
                let caps = PipelineCaps {
                    execution_width: pipeline.threadExecutionWidth(),
                    max_threads: pipeline.maxTotalThreadsPerThreadgroup(),
                    static_bytes: pipeline.staticThreadgroupMemoryLength(),
                    device_threadgroup_bytes: device.maxThreadgroupMemoryLength(),
                };
                pipelines.insert(
                    kernel,
                    Pipeline {
                        raw: pipeline,
                        caps,
                    },
                );
            }
            Ok(Self {
                gpu: device,
                queue,
                pipelines,
            })
        }
        pub fn info(&self) -> (String, u64, u64) {
            (
                self.gpu.name().to_string(),
                self.gpu.registryID(),
                self.gpu.maxBufferLength() as u64,
            )
        }
        pub fn pipeline_caps(&self, kernel: Kernel) -> Result<PipelineCaps> {
            self.pipelines
                .get(&kernel)
                .map(|pipeline| pipeline.caps)
                .ok_or(ExecutorError::BackendFailure("Metal pipeline missing"))
        }
        pub fn validate_dispatch(&self, kernel: Kernel, geometry: Geometry) -> Result<()> {
            let caps = self.pipeline_caps(kernel)?;
            match geometry {
                Geometry::Linear(threads) if threads == 0 || caps.max_threads == 0 => {
                    Err(ExecutorError::InvalidArgument("Metal linear grid is empty"))
                }
                Geometry::Tile8(_) if !caps.admits_tile8() => Err(ExecutorError::Unsupported(
                    "Metal tile8 pipeline/grid is incompatible",
                )),
                _ => Ok(()),
            }
        }
        pub fn allocate(&self, length: usize, initial: Option<&[u8]>) -> Result<Buffer> {
            let buffer = self
                .gpu
                .newBufferWithLength_options(length, MTLResourceOptions::StorageModeShared)
                .ok_or(ExecutorError::ResourceLimit(
                    "Metal buffer allocation failed",
                ))?;
            // SAFETY: this is a newly allocated shared buffer, not yet visible to a
            // command encoder. length is its actual allocation extent. Copy source
            // has exactly the checked initialization length and cannot overlap.
            unsafe {
                std::ptr::write_bytes(buffer.contents().as_ptr().cast::<u8>(), 0, length);
                if let Some(bytes) = initial {
                    if bytes.len() > length {
                        return Err(ExecutorError::OutOfBounds(
                            "Metal upload exceeds allocation",
                        ));
                    }
                    std::ptr::copy_nonoverlapping(
                        bytes.as_ptr(),
                        buffer.contents().as_ptr().cast::<u8>(),
                        bytes.len(),
                    );
                }
            }
            Ok(Buffer(buffer))
        }
        pub fn command(&self) -> Result<Command> {
            self.queue
                .commandBuffer()
                .map(Command)
                .ok_or(ExecutorError::BackendFailure(
                    "Metal command buffer creation failed",
                ))
        }
        pub fn dispatch(
            &self,
            command: &Command,
            kernel: Kernel,
            buffers: &[&Buffer],
            params: &[u32; 16],
            geometry: Geometry,
        ) -> Result<()> {
            let pipeline = self
                .pipelines
                .get(&kernel)
                .ok_or(ExecutorError::BackendFailure("Metal pipeline missing"))?;
            let encoder =
                command
                    .0
                    .computeCommandEncoder()
                    .ok_or(ExecutorError::BackendFailure(
                        "Metal compute encoder creation failed",
                    ))?;
            encoder.setComputePipelineState(&pipeline.raw);
            // SAFETY: the finite operation's checked geometry establishes every
            // shader access bound; bindings are <=7, params occupy slot8, and all
            // resources are retained by the batch plus Metal's retained-reference
            // command buffer. setBytes copies the 64-byte parameter slice.
            unsafe {
                for (index, buffer) in buffers.iter().enumerate() {
                    encoder.setBuffer_offset_atIndex(Some(&buffer.0), 0, index);
                }
                encoder.setBytes_length_atIndex(NonNull::from(params).cast(), 64, 8);
                match geometry {
                    Geometry::Linear(threads) => {
                        encoder.dispatchThreads_threadsPerThreadgroup(
                            MTLSize {
                                width: threads,
                                height: 1,
                                depth: 1,
                            },
                            MTLSize {
                                width: pipeline.caps.max_threads.min(256),
                                height: 1,
                                depth: 1,
                            },
                        );
                    }
                    Geometry::Tile8(grid) => {
                        let [columns, tokens, depth] = grid.groups();
                        // SAFETY: the preflight admitted this exact pipeline's
                        // execution width, fixed 256-thread capacity and actual
                        // static storage. Full groups are essential: even tail
                        // threads must reach both shared-memory barriers.
                        encoder.dispatchThreadgroups_threadsPerThreadgroup(
                            MTLSize {
                                width: columns,
                                height: tokens,
                                depth,
                            },
                            MTLSize {
                                width: crate::packed::THREADS,
                                height: 1,
                                depth: 1,
                            },
                        );
                    }
                }
            }
            encoder.endEncoding();
            Ok(())
        }
        #[allow(clippy::unused_self)] // Common bridge API also supports the non-macOS stub.
        pub fn copy(
            &self,
            command: &Command,
            output: &Buffer,
            input: &Buffer,
            length: usize,
        ) -> Result<()> {
            let encoder = command
                .0
                .blitCommandEncoder()
                .ok_or(ExecutorError::BackendFailure(
                    "Metal blit encoder creation failed",
                ))?;
            // SAFETY: callers checked distinct storage and both byte extents.
            unsafe {
                encoder.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                    &input.0, 0, &output.0, 0, length,
                );
            }
            encoder.endEncoding();
            Ok(())
        }
    }
    impl Command {
        pub fn commit(&self) {
            self.0.commit();
        }
        pub fn poll(&self) -> Option<Result<()>> {
            match self.0.status() {
                MTLCommandBufferStatus::Completed => Some(Ok(())),
                MTLCommandBufferStatus::Error => Some(Err(ExecutorError::BackendFailure(
                    "Metal command execution failed",
                ))),
                _ => None,
            }
        }
    }
    impl Buffer {
        pub fn read_completed(&self, length: usize, command: &Command) -> Result<Vec<u8>> {
            match command.poll() {
                Some(Ok(())) => {}
                Some(Err(error)) => return Err(error),
                None => {
                    return Err(ExecutorError::BackendFailure(
                        "Metal readback precedes completion",
                    ));
                }
            }
            if length > self.0.length() {
                return Err(ExecutorError::OutOfBounds(
                    "Metal readback exceeds allocation",
                ));
            }
            // SAFETY: this allocation is private readback staging. The only writer
            // was the snapshot blit, whose command is confirmed complete; future
            // commands cannot reference this allocation. length <= allocation.
            let bytes = unsafe {
                std::slice::from_raw_parts(self.0.contents().as_ptr().cast::<u8>(), length)
            };
            Ok(bytes.to_vec())
        }
    }
}
#[cfg(target_os = "macos")]
pub use native::*;

#[cfg(not(target_os = "macos"))]
// Match the real native bridge signatures while rejecting construction on other targets.
#[allow(clippy::unused_self, clippy::unnecessary_wraps)]
mod unsupported {
    use super::{ExecutorError, Geometry, Kernel, PipelineCaps, Result};
    pub struct Device;
    pub struct Buffer;
    pub struct Command;
    fn unavailable<T>() -> Result<T> {
        Err(ExecutorError::Unsupported("native Metal requires macOS"))
    }
    impl Device {
        pub fn new() -> Result<Self> {
            unavailable()
        }
        pub fn info(&self) -> (String, u64, u64) {
            (String::new(), 0, 0)
        }
        pub fn pipeline_caps(&self, _: Kernel) -> Result<PipelineCaps> {
            unavailable()
        }
        pub fn validate_dispatch(&self, _: Kernel, _: Geometry) -> Result<()> {
            unavailable()
        }
        pub fn allocate(&self, _: usize, _: Option<&[u8]>) -> Result<Buffer> {
            unavailable()
        }
        pub fn command(&self) -> Result<Command> {
            unavailable()
        }
        pub fn dispatch(
            &self,
            _: &Command,
            _: Kernel,
            _: &[&Buffer],
            _: &[u32; 16],
            _: Geometry,
        ) -> Result<()> {
            unavailable()
        }
        pub fn copy(&self, _: &Command, _: &Buffer, _: &Buffer, _: usize) -> Result<()> {
            unavailable()
        }
    }
    impl Command {
        pub fn commit(&self) {}
        pub fn poll(&self) -> Option<Result<()>> {
            Some(unavailable())
        }
    }
    impl Buffer {
        pub fn read_completed(&self, _: usize, _: &Command) -> Result<Vec<u8>> {
            unavailable()
        }
    }
}
#[cfg(not(target_os = "macos"))]
pub use unsupported::*;
