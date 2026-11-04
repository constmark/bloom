//! An interpreter-only, import-free WASM processor boundary.
use anyhow::{Context, Result, anyhow, bail, ensure};
use bloomai_core::{Modality, ProcessorKind, ProcessorSpec};
use serde::{Deserialize, Serialize};
use std::io::{Read, Write};
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::{Duration, Instant};
use wasmi::{
    Config, EnforcedLimits, Engine, Linker, Module, Store, StoreLimits, StoreLimitsBuilder,
    TypedFunc, TypedResumableCall, WasmParams, WasmResults,
};

use crate::io::DataBlock;
use crate::processor::Processor;

pub const WASM_PROCESSOR_ABI_VERSION: u32 = 1;
const ENVELOPE_OBJECT: &str = "bloom.processor.blocks";
const FUEL_QUANTUM: u64 = 10_000;
const MAX_BLOCKS: usize = 64;
const PAGE_BYTES: usize = 65_536;

/// Per-processor bounds. Hard maxima apply even to caller-supplied settings.
#[derive(Debug, Clone)]
pub struct WasmLimits {
    pub module_bytes: usize,
    pub memory_bytes: usize,
    pub io_bytes: usize,
    pub fuel: u64,
    pub execution_timeout: Duration,
    pub concurrent_calls: usize,
}

impl Default for WasmLimits {
    fn default() -> Self {
        Self {
            module_bytes: 2 * 1024 * 1024,
            memory_bytes: 8 * 1024 * 1024,
            io_bytes: 1024 * 1024,
            fuel: 5_000_000,
            execution_timeout: Duration::from_secs(2),
            concurrent_calls: 4,
        }
    }
}

impl WasmLimits {
    fn validate(&self) -> Result<()> {
        ensure!(
            (8..=2 * 1024 * 1024).contains(&self.module_bytes),
            "WASM module limit must be 8 bytes..=2 MiB"
        );
        ensure!(
            (PAGE_BYTES..=64 * 1024 * 1024).contains(&self.memory_bytes),
            "WASM memory limit must be 64 KiB..=64 MiB"
        );
        ensure!(
            (64..=8 * 1024 * 1024).contains(&self.io_bytes) && self.io_bytes <= self.memory_bytes,
            "WASM I/O limit must be 64 bytes..=8 MiB and fit the memory limit"
        );
        ensure!(
            (1..=50_000_000).contains(&self.fuel),
            "WASM fuel limit must be 1..=50000000"
        );
        ensure!(
            !self.execution_timeout.is_zero() && self.execution_timeout <= Duration::from_secs(30),
            "WASM execution timeout must be positive and at most 30 seconds"
        );
        ensure!(
            (1..=32).contains(&self.concurrent_calls),
            "WASM concurrency limit must be 1..=32"
        );
        Ok(())
    }
}

/// A compiled module with no persistent guest state. Each call owns a new
/// Store, memory, fuel budget, and instance; no files, network or clocks are
/// available inside the guest. Cancellation is checked between fuel slices.
pub struct WasmProcessor {
    engine: Engine,
    module: Module,
    spec: ProcessorSpec,
    limits: WasmLimits,
    active_calls: AtomicUsize,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct BlocksEnvelope {
    schema_version: u32,
    object: String,
    #[serde(deserialize_with = "deserialize_blocks")]
    blocks: Vec<DataBlock>,
}

// Stop at the block-count boundary while decoding, before an untrusted array
// can allocate a host-side Vec for every element in its serialized byte limit.
fn deserialize_blocks<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> std::result::Result<Vec<DataBlock>, D::Error> {
    struct BlocksVisitor;
    impl<'de> serde::de::Visitor<'de> for BlocksVisitor {
        type Value = Vec<DataBlock>;
        fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            formatter.write_str("at most 64 inline data blocks")
        }
        fn visit_seq<A: serde::de::SeqAccess<'de>>(
            self,
            mut sequence: A,
        ) -> std::result::Result<Self::Value, A::Error> {
            let mut blocks = Vec::new();
            while let Some(block) = sequence.next_element()? {
                if blocks.len() == MAX_BLOCKS {
                    return Err(serde::de::Error::custom(
                        "WASM processor block count exceeds 64",
                    ));
                }
                blocks.push(block);
            }
            Ok(blocks)
        }
    }
    deserializer.deserialize_seq(BlocksVisitor)
}

struct BoundedWriter {
    bytes: Vec<u8>,
    limit: usize,
}
impl Write for BoundedWriter {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        if bytes.len() > self.limit.saturating_sub(self.bytes.len()) {
            return Err(std::io::Error::other(
                "WASM processor JSON exceeds its I/O limit",
            ));
        }
        self.bytes.extend_from_slice(bytes);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

struct ActiveCall<'a>(&'a AtomicUsize);
impl Drop for ActiveCall<'_> {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::AcqRel);
    }
}

struct CallBudget<'a> {
    not_granted: u64,
    deadline: Instant,
    cancelled: &'a AtomicBool,
}
impl CallBudget<'_> {
    fn check(&self) -> Result<()> {
        ensure!(
            !self.cancelled.load(Ordering::Acquire),
            "WASM processor call cancelled"
        );
        ensure!(
            Instant::now() < self.deadline,
            "WASM processor execution deadline exceeded"
        );
        Ok(())
    }
    fn refill(&mut self, store: &mut Store<StoreLimits>, required: u64) -> Result<()> {
        self.check()?;
        ensure!(self.not_granted > 0, "WASM processor fuel budget exhausted");
        let current = store.get_fuel()?;
        let grant = FUEL_QUANTUM
            .max(required.saturating_sub(current))
            .min(self.not_granted);
        self.not_granted -= grant;
        store.set_fuel(current + grant)?;
        Ok(())
    }
    fn call<P: WasmParams, R: WasmResults>(
        &mut self,
        store: &mut Store<StoreLimits>,
        function: TypedFunc<P, R>,
        params: P,
    ) -> Result<R> {
        self.check()?;
        if store.get_fuel()? == 0 {
            self.refill(store, 1)?;
        }
        let mut invocation = function.call_resumable(&mut *store, params)?;
        loop {
            self.check()?;
            invocation = match invocation {
                TypedResumableCall::Finished(result) => return Ok(result),
                TypedResumableCall::HostTrap(_) => {
                    bail!("WASM processor unexpectedly called a host function")
                }
                TypedResumableCall::OutOfFuel(suspended) => {
                    self.refill(store, suspended.required_fuel())?;
                    suspended.resume(&mut *store)?
                }
            };
        }
    }
}

impl WasmProcessor {
    pub(super) fn load(path: &Path, spec: ProcessorSpec, limits: WasmLimits) -> Result<Self> {
        limits.validate()?;
        ensure!(spec.version == "1", "WASM processor spec version must be 1");
        ensure!(
            !spec.inputs.is_empty() && !spec.outputs.is_empty(),
            "WASM processor must declare input and output modalities"
        );
        // ABI v1 carries DataBlocks, not free-form configuration with implicit
        // host behavior. Parameters must be compiled into the guest for now.
        ensure!(
            spec.parameters.is_empty(),
            "WASM processor ABI v1 does not accept spec parameters"
        );
        let metadata =
            std::fs::symlink_metadata(path).context("WASM module metadata is unavailable")?;
        ensure!(
            metadata.is_file() && !metadata.file_type().is_symlink(),
            "WASM module must be a regular non-symlink file"
        );
        let mut options = std::fs::OpenOptions::new();
        options.read(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK);
        }
        let file = options.open(path).context("WASM module cannot be opened")?;
        ensure!(
            file.metadata()?.is_file(),
            "WASM module must be a regular file"
        );
        let mut bytes = Vec::new();
        file.take(limits.module_bytes as u64 + 1)
            .read_to_end(&mut bytes)?;
        ensure!(
            bytes.len() <= limits.module_bytes,
            "WASM module exceeds the byte limit"
        );
        ensure!(
            bytes.starts_with(b"\0asm\x01\0\0\0"),
            "WASM processor must be a binary core WebAssembly module"
        );
        let mut config = Config::default();
        config
            .consume_fuel(true)
            .allow_start_fn(false)
            .ignore_custom_sections(true)
            .compilation_mode(wasmi::CompilationMode::Eager)
            .enforced_limits(EnforcedLimits::strict())
            .set_max_recursion_depth(128)
            .set_max_stack_height(256 * 1024)
            .set_max_cached_stacks(0)
            .wasm_multi_memory(false)
            .wasm_custom_page_sizes(false)
            .wasm_tail_call(false)
            .wasm_reference_types(false);
        let engine = Engine::new(&config);
        let module = Module::new(&engine, &bytes).context("WASM module validation failed")?;
        ensure!(
            module.imports().next().is_none(),
            "WASM processor imports are forbidden (including WASI)"
        );
        let processor = Self {
            engine,
            module,
            spec,
            limits,
            active_calls: AtomicUsize::new(0),
        };
        // Negotiate and type-check the ABI in a disposable bounded instance.
        let cancelled = AtomicBool::new(false);
        let mut budget = processor.budget(&cancelled);
        let (mut store, instance) = processor.instance()?;
        processor.check_abi(&mut store, instance, &mut budget)?;
        Ok(processor)
    }

    fn budget<'a>(&self, cancelled: &'a AtomicBool) -> CallBudget<'a> {
        CallBudget {
            not_granted: self.limits.fuel,
            deadline: Instant::now() + self.limits.execution_timeout,
            cancelled,
        }
    }

    fn instance(&self) -> Result<(Store<StoreLimits>, wasmi::Instance)> {
        let limits = StoreLimitsBuilder::new()
            .memory_size(self.limits.memory_bytes)
            .instances(1)
            .memories(1)
            .tables(1)
            .table_elements(4096)
            .trap_on_grow_failure(true)
            .build();
        let mut store = Store::new(&self.engine, limits);
        store.limiter(|limits| limits);
        let instance = Linker::new(&self.engine)
            .instantiate_and_start(&mut store, &self.module)
            .context("WASM processor instantiation failed")?;
        Ok((store, instance))
    }

    fn check_abi(
        &self,
        store: &mut Store<StoreLimits>,
        instance: wasmi::Instance,
        budget: &mut CallBudget<'_>,
    ) -> Result<()> {
        instance
            .get_memory(&*store, "memory")
            .ok_or_else(|| anyhow!("WASM processor must export memory"))?;
        instance
            .get_typed_func::<i32, i32>(&*store, "bloom_alloc")
            .context("WASM processor must export bloom_alloc(i32)->i32")?;
        instance
            .get_typed_func::<(i32, i32), i64>(&*store, "bloom_process")
            .context("WASM processor must export bloom_process(i32,i32)->i64")?;
        let version = instance
            .get_typed_func::<(), i32>(&*store, "bloom_plugin_abi_version")
            .context("WASM processor must export bloom_plugin_abi_version()->i32")?;
        ensure!(
            budget.call(store, version, ())? == WASM_PROCESSOR_ABI_VERSION as i32,
            "unsupported WASM processor ABI version"
        );
        Ok(())
    }

    /// Run with cooperative cancellation. No guest state or partial result is
    /// published if cancellation, a trap, a deadline or a limit is encountered.
    pub fn process_cancellable(
        &self,
        blocks: Vec<DataBlock>,
        cancelled: &AtomicBool,
    ) -> Result<Vec<DataBlock>> {
        self.active_calls
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |active| {
                (active < self.limits.concurrent_calls).then_some(active + 1)
            })
            .map_err(|_| anyhow!("WASM processor concurrency limit reached"))?;
        let _active = ActiveCall(&self.active_calls);
        let mut budget = self.budget(cancelled);
        budget.check()?;
        validate_blocks(&blocks, &self.spec.inputs, self.limits.io_bytes)?;
        let envelope = BlocksEnvelope {
            schema_version: WASM_PROCESSOR_ABI_VERSION,
            object: ENVELOPE_OBJECT.to_string(),
            blocks,
        };
        let mut writer = BoundedWriter {
            bytes: Vec::new(),
            limit: self.limits.io_bytes,
        };
        serde_json::to_writer(&mut writer, &envelope)?;
        budget.check()?;
        let input = writer.bytes;
        let (mut store, instance) = self.instance()?;
        self.check_abi(&mut store, instance, &mut budget)?;
        let memory = instance
            .get_memory(&store, "memory")
            .ok_or_else(|| anyhow!("WASM processor memory is missing"))?;
        let alloc = instance.get_typed_func::<i32, i32>(&store, "bloom_alloc")?;
        let input_ptr = budget.call(&mut store, alloc, input.len() as i32)? as u32 as usize;
        memory
            .write(&mut store, input_ptr, &input)
            .context("WASM processor returned an invalid input pointer")?;
        let process = instance.get_typed_func::<(i32, i32), i64>(&store, "bloom_process")?;
        let packed =
            budget.call(&mut store, process, (input_ptr as i32, input.len() as i32))? as u64;
        let output_ptr = (packed & 0xffff_ffff) as usize;
        let output_len = (packed >> 32) as usize;
        ensure!(
            output_len > 0 && output_len <= self.limits.io_bytes,
            "WASM processor output exceeds its byte limit or is empty"
        );
        let end = output_ptr
            .checked_add(output_len)
            .ok_or_else(|| anyhow!("WASM processor output pointer overflow"))?;
        let output = memory
            .data(&store)
            .get(output_ptr..end)
            .ok_or_else(|| anyhow!("WASM processor returned an invalid output range"))?;
        let result: BlocksEnvelope = serde_json::from_slice(output)
            .context("WASM processor returned invalid JSON blocks")?;
        ensure!(
            result.schema_version == WASM_PROCESSOR_ABI_VERSION && result.object == ENVELOPE_OBJECT,
            "unsupported WASM processor output envelope"
        );
        validate_blocks(&result.blocks, &self.spec.outputs, self.limits.io_bytes)?;
        budget.check()?;
        Ok(result.blocks)
    }
}

fn validate_blocks(blocks: &[DataBlock], modalities: &[Modality], byte_limit: usize) -> Result<()> {
    ensure!(
        !blocks.is_empty() && blocks.len() <= MAX_BLOCKS,
        "WASM processor must carry 1..=64 blocks"
    );
    let mut payload_bytes = 0usize;
    for block in blocks {
        let bytes = match block {
            DataBlock::Text(text) => text.len(),
            DataBlock::Tokens(values) => values.len().saturating_mul(4),
            DataBlock::Tensor(values) | DataBlock::Action { values, .. } => {
                values.len().saturating_mul(4)
            }
            DataBlock::AudioPcm { samples, .. } => samples.len().saturating_mul(4),
            DataBlock::Image { bytes, .. } => bytes.len(),
            DataBlock::VideoFrames(frames) => frames
                .iter()
                .fold(0usize, |sum, frame| sum.saturating_add(frame.len())),
            DataBlock::WorldState { latent, .. } => latent.as_ref().map_or(0, Vec::len),
            DataBlock::AudioFile { .. } => {
                bail!("WASM processor file-reference blocks are forbidden; use inline data")
            }
        };
        payload_bytes = payload_bytes.saturating_add(bytes);
        ensure!(
            payload_bytes <= byte_limit,
            "WASM processor inline payload exceeds the I/O limit"
        );
        block.validate().map_err(anyhow::Error::msg)?;
        let modality = match block {
            DataBlock::Text(_) | DataBlock::Tokens(_) => Modality::Text,
            DataBlock::AudioPcm { .. } => Modality::Audio,
            DataBlock::Image { .. } | DataBlock::VideoFrames(_) => Modality::Vision,
            DataBlock::Tensor(_) | DataBlock::WorldState { .. } | DataBlock::Action { .. } => {
                Modality::Multi
            }
            DataBlock::AudioFile { .. } => {
                bail!("WASM processor file-reference blocks are forbidden; use inline data")
            }
        };
        ensure!(
            modalities.contains(&Modality::Multi) || modalities.contains(&modality),
            "WASM processor block modality was not declared"
        );
    }
    Ok(())
}

impl Processor for WasmProcessor {
    fn name(&self) -> &str {
        &self.spec.name
    }
    fn kind(&self) -> ProcessorKind {
        self.spec.kind.clone()
    }
    fn input_modalities(&self) -> Vec<Modality> {
        self.spec.inputs.clone()
    }
    fn output_modalities(&self) -> Vec<Modality> {
        self.spec.outputs.clone()
    }
    fn spec(&self) -> ProcessorSpec {
        self.spec.clone()
    }
    fn process(&self, blocks: Vec<DataBlock>) -> Result<Vec<DataBlock>> {
        self.process_cancellable(blocks, &AtomicBool::new(false))
    }
}

#[cfg(test)]
mod tests;
