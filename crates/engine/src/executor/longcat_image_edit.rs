//! LongCat-Image-Edit adapter.
//!
//! This backend deliberately avoids the upstream Diffusers/PyTorch path. It
//! validates the local model package and delegates execution to a native runner
//! when one is configured. If no native runner is available, it probes TileLang
//! kernels so missing operator coverage is reported explicitly.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{anyhow, bail, Context, Result};
use bloomai_core::{
    constants::{GIB, GIB_F64, MIB_F64},
    DType, DeviceCapability, DeviceClass, DeviceKind, GenerationParams, Modality, ModelFamily,
    ModelFormat,
};

use crate::engine::{
    default_engine_supports, BackendMaturity, Engine, EngineCapability, SupportLevel,
};
use crate::io::{ModelInput, ModelOutput, OutputChunk};
use crate::model::{LoadedModel, ModelMetadata, OutputSink};

#[derive(Debug, Clone)]
struct LongCatProbeReport {
    package_format: &'static str,
    weight_files: usize,
    tensor_count: usize,
    probe_tensor: Option<String>,
    device_backend: String,
    tilelang_report: String,
}

impl std::fmt::Display for LongCatProbeReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "format={}, weights={} files, tensors={}, device={}, probe_tensor={}, tilelang={}",
            self.package_format,
            self.weight_files,
            self.tensor_count,
            self.device_backend,
            self.probe_tensor.as_deref().unwrap_or("metadata-only"),
            self.tilelang_report
        )
    }
}

#[derive(Debug, Clone)]
enum LongCatPackage {
    DiffusersSafetensors { weight_files: Vec<PathBuf> },
    GgufQuantized { weight_files: Vec<PathBuf> },
}

impl LongCatPackage {
    fn weight_files(&self) -> &[PathBuf] {
        match self {
            Self::DiffusersSafetensors { weight_files } | Self::GgufQuantized { weight_files } => {
                weight_files
            }
        }
    }

    fn format_label(&self) -> &'static str {
        match self {
            Self::DiffusersSafetensors { .. } => "diffusers-safetensors",
            Self::GgufQuantized { .. } => "gguf-quantized",
        }
    }

    fn is_quantized(&self) -> bool {
        matches!(self, Self::GgufQuantized { .. })
    }
}

pub struct LongCatImageEditEngine;

impl Engine for LongCatImageEditEngine {
    fn name(&self) -> &'static str {
        "longcat"
    }

    fn supported_modalities(&self) -> Vec<Modality> {
        vec![Modality::Text, Modality::Vision]
    }

    fn supported_devices(&self) -> Vec<DeviceKind> {
        vec![DeviceKind::Gpu]
    }

    fn capability(&self) -> EngineCapability {
        EngineCapability {
            engine_name: self.name(),
            supported_families: vec![ModelFamily::Custom("longcat-image-edit".to_string())],
            supported_dtypes: vec![DType::BF16, DType::F16, DType::F32],
            supported_formats: vec![ModelFormat::Safetensors, ModelFormat::Gguf],
            supported_devices: vec![DeviceClass::IntegratedGpu, DeviceClass::DiscreteGpu],
            supported_modalities: vec![Modality::Text, Modality::Vision],
            supports_streaming: true,
            supports_quantized_models: true,
            supports_embeddings: false,
            supports_rerank: false,
            supports_structured_output: false,
            max_context_tokens: None,
            supported_quant_methods: vec![crate::core::quantization::QuantMethod::Gguf],
            supported_parallel_strategies: vec![crate::core::parallelism::ParallelStrategy::None],
            maturity: BackendMaturity::Experimental,
            diagnostic_tips: vec![
                "Run with --device gpu; CPU execution is intentionally rejected for LongCat.".to_string(),
                "Use BLOOM_LONGCAT_RUNNER or BLOOM_MNN_DIFFUSION_DEMO to point at a native non-PyTorch runner for full image generation.".to_string(),
                "Without a native runner, Bloom validates real safetensors or GGUF quantized weights with GPU startup and probes TileLang attention/matmul/softmax fallback coverage.".to_string(),
            ],
            construction_guide:
                "ModelScope safetensors package or LongCat GGUF quantized package plus a GPU build and optional native LongCat/MNN runner; PyTorch runners are intentionally rejected.".to_string(),
        }
    }

    fn supports(
        &self,
        manifest: &bloomai_core::ModelManifest,
        device_cap: &DeviceCapability,
    ) -> SupportLevel {
        default_engine_supports(&self.capability(), manifest, device_cap)
    }

    fn load(&self, model_path: &Path, device: DeviceKind) -> Result<Box<dyn LoadedModel>> {
        LongCatImageEditModel::load(model_path, device)
    }
}

struct LongCatImageEditModel {
    model_path: PathBuf,
    device: DeviceKind,
    meta: ModelMetadata,
    runner: Option<PathBuf>,
    probe: LongCatProbeReport,
}

impl LongCatImageEditModel {
    fn load(model_path: &Path, device: DeviceKind) -> Result<Box<dyn LoadedModel>> {
        if device != DeviceKind::Gpu {
            bail!("LongCat-Image-Edit requires --device gpu; CPU execution is disabled");
        }
        let package = detect_longcat_package(model_path)?;
        validate_no_pytorch_artifacts(model_path)?;
        let probe = probe_longcat_gpu_package(&package)?;
        let mut manifest = crate::load_manifest(model_path)?;
        if package.is_quantized() {
            manifest.id = model_path
                .file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .into_owned();
            manifest.family = ModelFamily::Custom("longcat-image-edit".to_string());
            manifest.io_schema.inputs = vec![Modality::Text, Modality::Vision];
            manifest.io_schema.outputs = vec![Modality::Vision];
        }
        let runner = resolve_runner();
        let quantized = package.is_quantized();
        Ok(Box::new(Self {
            model_path: model_path.to_path_buf(),
            device,
            meta: ModelMetadata {
                id: manifest.id.clone(),
                modality: Modality::Vision,
                quantized,
                manifest,
            },
            runner,
            probe,
        }))
    }

    fn run_native(&self, prompt: &str, image: &[u8], params: &GenerationParams) -> Result<Vec<u8>> {
        let runner = match self.runner.as_ref() {
            Some(runner) => runner,
            None => {
                #[cfg(feature = "candle-engine")]
                {
                    let device = candle_gpu_device()?;
                    return crate::executor::longcat_native::run_draft_image_edit(
                        &self.model_path,
                        prompt,
                        image,
                        params.max_tokens as u32,
                        &device,
                    );
                }

                #[cfg(not(feature = "candle-engine"))]
                {
                    bail!(
                        "no native LongCat runner configured; set BLOOM_LONGCAT_RUNNER or BLOOM_MNN_DIFFUSION_DEMO"
                    );
                }
            }
        };

        reject_pytorch_runner(runner)?;
        ensure_safe_native_run(&self.model_path)?;

        let run_dir = make_run_dir()?;
        let input_path = run_dir.join("input.png");
        let output_path = run_dir.join("output.png");
        fs::write(&input_path, image).context("failed to write temporary LongCat input image")?;

        crate::core::security::validate_runner(runner)?;
        let mut cmd = Command::new(runner);
        cmd.arg("--model")
            .arg(&self.model_path)
            .arg("--prompt")
            .arg(prompt)
            .arg("--image")
            .arg(&input_path)
            .arg("--output")
            .arg(&output_path)
            .arg("--steps")
            .arg(params.max_tokens.to_string())
            .arg("--device")
            .arg(match self.device {
                DeviceKind::Cpu => "cpu",
                DeviceKind::Gpu => "gpu",
                DeviceKind::Npu => "npu",
            });

        if let Some(seed) = params.seed {
            cmd.arg("--seed").arg(seed.to_string());
        }

        let status = cmd.status().with_context(|| {
            format!(
                "failed to launch native LongCat runner {}",
                runner.display()
            )
        })?;
        if !status.success() {
            bail!("native LongCat runner exited with status {}", status);
        }

        fs::read(&output_path).with_context(|| {
            format!(
                "native LongCat runner did not produce output image at {}",
                output_path.display()
            )
        })
    }
}

impl LoadedModel for LongCatImageEditModel {
    fn metadata(&self) -> &ModelMetadata {
        &self.meta
    }

    fn infer(&self, input: ModelInput, params: &GenerationParams) -> Result<ModelOutput> {
        let (prompt, image) = resolve_edit_input(input)?;
        match self.run_native(&prompt, &image, params) {
            Ok(bytes) => Ok(ModelOutput {
                text: None,
                logits: None,
                image: Some(bytes),
                audio: None,
                video: None,
            }),
            Err(native_err) => {
                Err(anyhow!(
                    "LongCat-Image-Edit package and GPU startup were validated, but full image generation still needs a native non-PyTorch runner or a complete Bloom-native DiT/VAE/text-encoder graph.\nProbe: {}\nNative runner: {native_err}",
                    self.probe
                ))
            }
        }
    }

    fn infer_stream(
        &self,
        input: ModelInput,
        params: &GenerationParams,
        sink: &mut dyn OutputSink,
    ) -> Result<()> {
        sink.on_chunk(OutputChunk::DiffusionProgress {
            step: 0,
            total_steps: params.max_tokens as u32,
        })?;
        let output = self.infer(input, params)?;
        if let Some(image) = output.image {
            sink.on_chunk(OutputChunk::Image(image))?;
        }
        sink.on_chunk(OutputChunk::End)?;
        Ok(())
    }
}

fn resolve_edit_input(input: ModelInput) -> Result<(String, Vec<u8>)> {
    match input {
        ModelInput::Multi {
            text: Some(prompt),
            image: Some(image),
            ..
        } => Ok((prompt, image)),
        ModelInput::Vision { bytes, .. } => Ok(("Edit this image.".to_string(), bytes)),
        ModelInput::Text { .. } => bail!("LongCat-Image-Edit requires --image plus --prompt"),
        _ => bail!("LongCat-Image-Edit expects text + image input"),
    }
}

fn resolve_runner() -> Option<PathBuf> {
    std::env::var_os("BLOOM_LONGCAT_RUNNER")
        .or_else(|| std::env::var_os("BLOOM_MNN_DIFFUSION_DEMO"))
        .map(PathBuf::from)
}

fn reject_pytorch_runner(path: &Path) -> Result<()> {
    let name = path
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or_default()
        .to_lowercase();
    if name.contains("python") || name.contains("torch") || name.contains("diffusers") {
        bail!("refusing PyTorch/Diffusers runner '{}'", path.display());
    }
    Ok(())
}

fn detect_longcat_package(model_path: &Path) -> Result<LongCatPackage> {
    if is_complete_longcat_safetensors_package(model_path) {
        return Ok(LongCatPackage::DiffusersSafetensors {
            weight_files: longcat_safetensors_weight_files(model_path),
        });
    }

    let gguf_files = longcat_gguf_weight_files(model_path)?;
    if !gguf_files.is_empty() {
        return Ok(LongCatPackage::GgufQuantized {
            weight_files: gguf_files,
        });
    }

    validate_longcat_package(model_path)?;
    unreachable!("validate_longcat_package should have returned an error");
}

fn is_complete_longcat_safetensors_package(model_path: &Path) -> bool {
    longcat_required_package_paths(model_path)
        .iter()
        .all(|path| path.exists())
}

fn validate_longcat_package(model_path: &Path) -> Result<()> {
    let required = longcat_required_package_paths(model_path);
    let rels = longcat_required_package_rels();

    for (path, rel) in required.iter().zip(rels.iter()) {
        if !path.exists() {
            bail!("LongCat package is incomplete, missing {}", rel);
        }
    }

    Ok(())
}

fn longcat_required_package_rels() -> [&'static str; 10] {
    [
        "model_index.json",
        "transformer/config.json",
        "transformer/diffusion_pytorch_model.safetensors",
        "vae/config.json",
        "vae/diffusion_pytorch_model.safetensors",
        "text_encoder/config.json",
        "text_encoder/model.safetensors.index.json",
        "tokenizer/tokenizer.json",
        "text_processor/tokenizer.json",
        "scheduler/scheduler_config.json",
    ]
}

fn longcat_required_package_paths(model_path: &Path) -> Vec<PathBuf> {
    longcat_required_package_rels()
        .iter()
        .map(|rel| model_path.join(rel))
        .collect()
}

fn validate_no_pytorch_artifacts(model_path: &Path) -> Result<()> {
    if model_path.is_file() {
        return validate_no_pytorch_artifact_file(model_path);
    }

    let mut stack = vec![model_path.to_path_buf()];
    while let Some(dir) = stack.pop() {
        for entry in
            fs::read_dir(&dir).with_context(|| format!("failed to inspect {}", dir.display()))?
        {
            let entry = entry?;
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
                continue;
            }
            validate_no_pytorch_artifact_file(&path)?;
        }
    }
    Ok(())
}

fn validate_no_pytorch_artifact_file(path: &Path) -> Result<()> {
    let forbidden_extensions = ["pt", "pth", "ckpt"];
    let ext = path
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase();
    if forbidden_extensions.contains(&ext.as_str()) {
        bail!(
            "refusing PyTorch checkpoint artifact in LongCat package: {}",
            path.display()
        );
    }
    Ok(())
}

fn longcat_safetensors_weight_files(model_path: &Path) -> Vec<PathBuf> {
    let mut paths = vec![
        model_path.join("transformer/diffusion_pytorch_model.safetensors"),
        model_path.join("vae/diffusion_pytorch_model.safetensors"),
    ];
    for idx in 1..=5 {
        paths.push(model_path.join(format!("text_encoder/model-{idx:05}-of-00005.safetensors")));
    }
    paths
}

fn longcat_gguf_weight_files(model_path: &Path) -> Result<Vec<PathBuf>> {
    if model_path.is_file() {
        let is_gguf = model_path
            .extension()
            .and_then(|s| s.to_str())
            .map(|ext| ext.eq_ignore_ascii_case("gguf"))
            .unwrap_or(false);
        return Ok(if is_gguf {
            vec![model_path.to_path_buf()]
        } else {
            Vec::new()
        });
    }

    let mut files = Vec::new();
    if !model_path.exists() {
        return Ok(files);
    }
    for entry in fs::read_dir(model_path)
        .with_context(|| format!("failed to inspect {}", model_path.display()))?
    {
        let entry = entry?;
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let is_gguf = path
            .extension()
            .and_then(|s| s.to_str())
            .map(|ext| ext.eq_ignore_ascii_case("gguf"))
            .unwrap_or(false);
        if is_gguf {
            files.push(path);
        }
    }
    files.sort();
    Ok(files)
}

fn longcat_model_weight_files(model_path: &Path) -> Vec<PathBuf> {
    if is_complete_longcat_safetensors_package(model_path) {
        return longcat_safetensors_weight_files(model_path);
    }
    longcat_gguf_weight_files(model_path).unwrap_or_default()
}

fn ensure_safe_native_run(model_path: &Path) -> Result<()> {
    if env_flag("BLOOM_LONGCAT_DISABLE_MEMORY_GUARD") {
        return Ok(());
    }

    let weight_bytes = longcat_model_weight_files(model_path)
        .iter()
        .map(|path| fs::metadata(path).map(|m| m.len()).unwrap_or(0))
        .sum::<u64>();

    if let Some(total_bytes) = total_memory_bytes() {
        let min_total = weight_bytes.saturating_add(4 * GIB);
        if total_bytes < min_total {
            bail!(
                "refusing to start LongCat native runner: system memory {} is below the guarded minimum {} for {} of model weights",
                format_bytes(total_bytes),
                format_bytes(min_total),
                format_bytes(weight_bytes)
            );
        }
    }

    if let Some(available_bytes) = available_memory_bytes() {
        let min_available = longcat_min_available_bytes();
        if available_bytes < min_available {
            bail!(
                "refusing to start LongCat native runner: available/reclaimable memory {} is below guard {}. Close other apps or set BLOOM_LONGCAT_MIN_AVAILABLE_GB to a lower value only if the runner is known to stream/offload safely",
                format_bytes(available_bytes),
                format_bytes(min_available)
            );
        }
    }

    Ok(())
}

fn longcat_min_available_bytes() -> u64 {
    std::env::var("BLOOM_LONGCAT_MIN_AVAILABLE_GB")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .filter(|v| *v > 0.0)
        .map(|v| (v * GIB as f64) as u64)
        .unwrap_or(12 * GIB)
}

fn env_flag(name: &str) -> bool {
    std::env::var(name)
        .map(|v| matches!(v.to_ascii_lowercase().as_str(), "1" | "true" | "yes" | "on"))
        .unwrap_or(false)
}

fn total_memory_bytes() -> Option<u64> {
    #[cfg(target_os = "macos")]
    {
        let output = Command::new("sysctl")
            .arg("-n")
            .arg("hw.memsize")
            .output()
            .ok()?;
        if !output.status.success() {
            return None;
        }
        String::from_utf8_lossy(&output.stdout)
            .trim()
            .parse::<u64>()
            .ok()
    }

    #[cfg(target_os = "linux")]
    {
        let meminfo = fs::read_to_string("/proc/meminfo").ok()?;
        parse_meminfo_kib(&meminfo, "MemTotal").map(|kib| kib * 1024)
    }

    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    {
        None
    }
}

fn available_memory_bytes() -> Option<u64> {
    #[cfg(target_os = "macos")]
    {
        let output = Command::new("vm_stat").output().ok()?;
        if !output.status.success() {
            return None;
        }
        parse_macos_vm_stat_available(&String::from_utf8_lossy(&output.stdout))
    }

    #[cfg(target_os = "linux")]
    {
        let meminfo = fs::read_to_string("/proc/meminfo").ok()?;
        parse_meminfo_kib(&meminfo, "MemAvailable").map(|kib| kib * 1024)
    }

    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    {
        None
    }
}

#[cfg(target_os = "macos")]
fn parse_macos_vm_stat_available(text: &str) -> Option<u64> {
    let page_size = text
        .lines()
        .find_map(|line| {
            line.strip_prefix("Mach Virtual Memory Statistics: (page size of ")
                .and_then(|rest| rest.split_whitespace().next())
                .and_then(|n| n.parse::<u64>().ok())
        })
        .unwrap_or(16_384);

    let mut pages = 0u64;
    for key in [
        "Pages free",
        "Pages inactive",
        "Pages speculative",
        "Pages purgeable",
    ] {
        pages = pages.saturating_add(parse_vm_stat_pages(text, key).unwrap_or(0));
    }

    Some(pages.saturating_mul(page_size))
}

#[cfg(target_os = "macos")]
fn parse_vm_stat_pages(text: &str, key: &str) -> Option<u64> {
    text.lines().find_map(|line| {
        let line = line.trim();
        let rest = line.strip_prefix(key)?.trim_start();
        let rest = rest.strip_prefix(':')?.trim();
        let number = rest.trim_end_matches('.').replace('.', "");
        number.parse::<u64>().ok()
    })
}

#[cfg(target_os = "linux")]
fn parse_meminfo_kib(text: &str, key: &str) -> Option<u64> {
    text.lines().find_map(|line| {
        let rest = line.strip_prefix(key)?.trim_start();
        let rest = rest.strip_prefix(':')?.trim();
        rest.split_whitespace().next()?.parse::<u64>().ok()
    })
}

fn format_bytes(bytes: u64) -> String {
    if bytes as f64 >= GIB_F64 {
        format!("{:.1}GiB", bytes as f64 / GIB_F64)
    } else if bytes as f64 >= MIB_F64 {
        format!("{:.0}MiB", bytes as f64 / MIB_F64)
    } else {
        format!("{}B", bytes)
    }
}

#[cfg(feature = "candle-engine")]
fn probe_longcat_gpu_package(package: &LongCatPackage) -> Result<LongCatProbeReport> {
    use candle_core::{DType as CandleDType, Tensor};

    let device = candle_gpu_device()?;
    let sanity = Tensor::new(&[1.0f32, 2.0, 3.0, 4.0], &device)?.reshape((2, 2))?;
    let sanity_out = sanity.matmul(&sanity)?;
    if sanity_out.device().location() == candle_core::DeviceLocation::Cpu {
        bail!("LongCat GPU probe unexpectedly ran on CPU");
    }

    let (tensor_count, probe_tensor) = match package {
        LongCatPackage::DiffusersSafetensors { weight_files } => {
            let tensors =
                unsafe { candle_core::safetensors::MmapedSafetensors::multi(weight_files)? };
            let tensor_views = tensors.tensors();
            let tensor_count = tensor_views.len();
            let probe_tensor = tensor_views
                .iter()
                .find(|(_, view)| {
                    let elems = view.shape().iter().product::<usize>();
                    elems > 0 && elems <= 4096
                })
                .map(|(name, _)| name.clone());

            if let Some(name) = &probe_tensor {
                let tensor = tensors.load(name, &device)?;
                let tensor = match tensor.dtype() {
                    CandleDType::F32 => tensor,
                    CandleDType::F16 | CandleDType::BF16 => tensor.to_dtype(CandleDType::F32)?,
                    _ => tensor,
                };
                let _ = tensor.flatten_all()?.sum_all()?;
            }
            (tensor_count, probe_tensor)
        }
        LongCatPackage::GgufQuantized { weight_files } => probe_longcat_gguf_headers(weight_files)?,
    };

    Ok(LongCatProbeReport {
        package_format: package.format_label(),
        weight_files: package.weight_files().len(),
        tensor_count,
        probe_tensor,
        device_backend: candle_device_label(&device),
        tilelang_report: probe_tilelang_fallback(),
    })
}

#[cfg(not(feature = "candle-engine"))]
fn probe_longcat_gpu_package(_package: &LongCatPackage) -> Result<LongCatProbeReport> {
    bail!("LongCat GPU validation requires the candle-engine feature")
}

#[cfg(feature = "candle-engine")]
fn probe_longcat_gguf_headers(weight_files: &[PathBuf]) -> Result<(usize, Option<String>)> {
    let mut tensor_count = 0usize;
    let mut probe_tensor = None;

    for path in weight_files {
        let mut file = fs::File::open(path)
            .with_context(|| format!("failed to open GGUF file {}", path.display()))?;
        let content = candle_core::quantized::gguf_file::Content::read(&mut file)
            .with_context(|| format!("failed to read GGUF header {}", path.display()))?;
        let arch = content
            .metadata
            .get("general.architecture")
            .and_then(|value| value.to_string().ok().map(|s| s.to_string()))
            .unwrap_or_default()
            .to_ascii_lowercase();
        let name = path
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or_default()
            .to_ascii_lowercase();
        if !(arch.contains("flux") || arch.contains("longcat") || name.contains("longcat")) {
            bail!(
                "GGUF file {} is not recognized as LongCat/Flux image-edit weights (general.architecture={})",
                path.display(),
                if arch.is_empty() { "missing" } else { &arch }
            );
        }
        tensor_count += content.tensor_infos.len();
        if probe_tensor.is_none() {
            probe_tensor = content.tensor_infos.keys().next().cloned();
        }
    }

    Ok((tensor_count, probe_tensor))
}

#[cfg(feature = "candle-engine")]
fn candle_gpu_device() -> Result<candle_core::Device> {
    #[cfg(feature = "cuda")]
    {
        candle_core::Device::new_cuda(0).context("failed to initialize CUDA device 0 for LongCat")
    }

    #[cfg(all(not(feature = "cuda"), feature = "metal"))]
    {
        candle_core::Device::new_metal(0).context("failed to initialize Metal device 0 for LongCat")
    }

    #[cfg(not(any(feature = "cuda", feature = "metal")))]
    {
        bail!("LongCat GPU validation requires rebuilding with --features metal or --features cuda")
    }
}

#[cfg(feature = "candle-engine")]
fn candle_device_label(device: &candle_core::Device) -> String {
    match device {
        candle_core::Device::Cpu => "cpu".to_string(),
        candle_core::Device::Cuda(_) => match device.location() {
            candle_core::DeviceLocation::Cuda { gpu_id } => format!("cuda:{gpu_id}"),
            _ => "cuda".to_string(),
        },
        candle_core::Device::Metal(_) => "metal".to_string(),
    }
}

fn make_run_dir() -> Result<PathBuf> {
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();
    let dir = std::env::temp_dir().join(format!("bloom-longcat-{stamp}"));
    fs::create_dir_all(&dir)?;
    Ok(dir)
}

fn probe_tilelang_fallback() -> String {
    set_gpu_tilelang_backend_if_unset();
    let compiler = match bloomai_tilelang::TileLangCompiler::new() {
        Ok(c) => c,
        Err(e) => return format!("unavailable ({e})"),
    };

    let mut ok = Vec::new();
    let mut missing = Vec::new();

    match compiler.compile_attention(256, 128) {
        Ok(_) => ok.push("attention"),
        Err(e) => missing.push(format!("attention: {e}")),
    }
    match compiler.compile_softmax(256) {
        Ok(_) => ok.push("softmax"),
        Err(e) => missing.push(format!("softmax: {e}")),
    }
    match compiler.compile_matmul(16, 16, 16) {
        Ok(_) => ok.push("matmul"),
        Err(e) => missing.push(format!("matmul: {e}")),
    }

    if missing.is_empty() {
        format!(
            "ready for {}; remaining native coverage still required for LongCat DiT/VAE image-edit pipeline",
            ok.join(", ")
        )
    } else {
        format!(
            "partial; ready=[{}], missing=[{}]",
            ok.join(", "),
            missing.join("; ")
        )
    }
}

fn set_gpu_tilelang_backend_if_unset() {
    if std::env::var_os("TILELANG_BACKEND").is_some() {
        return;
    }
    if cfg!(target_os = "macos") {
        std::env::set_var("TILELANG_BACKEND", "mlx");
    } else {
        std::env::set_var("TILELANG_BACKEND", "cuda");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn write_required_longcat_layout(dir: &Path) {
        for subdir in [
            "transformer",
            "vae",
            "text_encoder",
            "tokenizer",
            "text_processor",
            "scheduler",
        ] {
            fs::create_dir_all(dir.join(subdir)).unwrap();
        }
        fs::write(dir.join("model_index.json"), "{}").unwrap();
        fs::write(dir.join("transformer/config.json"), "{}").unwrap();
        fs::write(
            dir.join("transformer/diffusion_pytorch_model.safetensors"),
            "",
        )
        .unwrap();
        fs::write(dir.join("vae/config.json"), "{}").unwrap();
        fs::write(dir.join("vae/diffusion_pytorch_model.safetensors"), "").unwrap();
        fs::write(dir.join("text_encoder/config.json"), "{}").unwrap();
        fs::write(dir.join("text_encoder/model.safetensors.index.json"), "{}").unwrap();
        fs::write(dir.join("tokenizer/tokenizer.json"), "{}").unwrap();
        fs::write(dir.join("text_processor/tokenizer.json"), "{}").unwrap();
        fs::write(dir.join("scheduler/scheduler_config.json"), "{}").unwrap();
    }

    #[test]
    fn rejects_pytorch_checkpoint_artifacts() {
        let dir = TempDir::new().unwrap();
        write_required_longcat_layout(dir.path());
        fs::write(dir.path().join("text_encoder/model.pt"), "").unwrap();

        let err = validate_no_pytorch_artifacts(dir.path()).unwrap_err();
        assert!(err
            .to_string()
            .contains("refusing PyTorch checkpoint artifact"));
    }

    #[test]
    fn rejects_pytorch_named_runner() {
        let err = reject_pytorch_runner(Path::new("/tmp/python")).unwrap_err();
        assert!(err
            .to_string()
            .contains("refusing PyTorch/Diffusers runner"));
    }

    #[test]
    fn longcat_backend_is_gpu_only() {
        let engine = LongCatImageEditEngine;
        assert_eq!(engine.supported_devices(), vec![DeviceKind::Gpu]);
        assert!(engine
            .capability()
            .supported_devices
            .iter()
            .all(|device| !matches!(device, DeviceClass::Cpu)));
    }

    #[test]
    fn detects_quantized_gguf_package() {
        let dir = TempDir::new().unwrap();
        let gguf = dir.path().join("LongCat-Image-Edit-Turbo-Q2_K.gguf");
        fs::write(&gguf, b"GGUF").unwrap();

        let package = detect_longcat_package(dir.path()).unwrap();
        assert!(package.is_quantized());
        assert_eq!(package.weight_files(), &[gguf]);
    }

    #[test]
    fn permits_direct_gguf_file_for_artifact_scan() {
        let dir = TempDir::new().unwrap();
        let gguf = dir.path().join("LongCat-Image-Edit-Turbo-Q2_K.gguf");
        fs::write(&gguf, b"GGUF").unwrap();

        validate_no_pytorch_artifacts(&gguf).unwrap();
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn parses_macos_available_memory() {
        let vm_stat = "Mach Virtual Memory Statistics: (page size of 16384 bytes)\n\
            Pages free: 10.\n\
            Pages inactive: 20.\n\
            Pages speculative: 30.\n\
            Pages purgeable: 40.\n";

        assert_eq!(
            parse_macos_vm_stat_available(vm_stat),
            Some((10 + 20 + 30 + 40) * 16_384)
        );
    }
}
