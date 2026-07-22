#![allow(clippy::missing_safety_doc, clippy::needless_return)]
use anyhow::{anyhow, Result};
use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::path::Path;

use bloomai_core::{
    DType, DeviceClass, DeviceKind, GenerationParams, Modality, ModelFamily, ModelFormat,
};
use bloomai_engine::core::engine::{BackendMaturity, Engine, EngineCapability, EngineRegistry};
use bloomai_engine::core::io::{ModelInput, OutputChunk};
use bloomai_engine::model::{EchoTextModel, LoadedModel, OutputSink};
use bloomai_engine::InferencePipeline;

#[cfg(feature = "candle-engine")]
use bloomai_engine::executor::candle::CandleEngine;
use bloomai_engine::executor::coreml::CoreMlEngine;
use bloomai_engine::executor::funasr::FunASREngine;
use bloomai_engine::executor::intel_npu::IntelNpuEngine;
use bloomai_engine::executor::llamacpp::LlamaCppEngine;
use bloomai_engine::executor::longcat_image_edit::LongCatImageEditEngine;
use bloomai_engine::executor::mlx::MlxEngine;
use bloomai_engine::executor::npu_tts::NpuTtsEngine;
use bloomai_engine::executor::onnx::OnnxRuntimeEngine;
use bloomai_engine::executor::openvino::OpenVINOEngine;
use bloomai_engine::executor::qwen3_vl::Qwen3VLEngine;
use bloomai_engine::executor::vulkan::VulkanEngine;
#[cfg(feature = "candle-engine")]
use bloomai_engine::executor::wan::WanEngine;

pub struct BloomPipeline {
    inner: InferencePipeline,
}

pub type BloomStreamCallback =
    unsafe extern "C" fn(user_data: *mut std::ffi::c_void, chunk_json: *const c_char);

struct FfiOutputSink {
    callback: BloomStreamCallback,
    user_data: *mut std::ffi::c_void,
}

unsafe impl Send for FfiOutputSink {}
unsafe impl Sync for FfiOutputSink {}

impl OutputSink for FfiOutputSink {
    fn on_chunk(&mut self, chunk: OutputChunk) -> Result<()> {
        let chunk_json = serde_json::to_string(&chunk).unwrap_or_else(|_| "{}".to_string());
        let chunk_c = CString::new(chunk_json).map_err(|e| anyhow!("{}", e))?;
        unsafe {
            (self.callback)(self.user_data, chunk_c.as_ptr());
        }
        Ok(())
    }
}

pub struct MockEngine;
impl Engine for MockEngine {
    fn name(&self) -> &'static str {
        "mock"
    }
    fn supported_modalities(&self) -> Vec<Modality> {
        vec![Modality::Text]
    }
    fn supported_devices(&self) -> Vec<DeviceKind> {
        vec![DeviceKind::Cpu]
    }
    fn capability(&self) -> EngineCapability {
        EngineCapability {
            engine_name: "mock",
            supported_families: vec![ModelFamily::Llama, ModelFamily::Qwen],
            supported_dtypes: vec![DType::F32, DType::F16],
            supported_formats: vec![ModelFormat::Safetensors],
            supported_devices: vec![DeviceClass::Cpu],
            supported_modalities: vec![Modality::Text],
            supports_streaming: true,
            supports_quantized_models: false,
            supports_embeddings: false,
            supports_rerank: false,
            supports_structured_output: false,
            max_context_tokens: Some(4096),
            supported_quant_methods: vec![],
            supported_parallel_strategies: vec![
                bloomai_engine::core::parallelism::ParallelStrategy::None,
            ],
            maturity: BackendMaturity::Experimental,
            diagnostic_tips: vec![],
            construction_guide: String::new(),
        }
    }
    fn load(&self, _model_path: &Path, _device: DeviceKind) -> Result<Box<dyn LoadedModel>> {
        Ok(Box::new(EchoTextModel::default()))
    }
}

static ENGINE_REGISTRY: once_cell::sync::Lazy<EngineRegistry> = once_cell::sync::Lazy::new(|| {
    let mut registry = EngineRegistry::default();
    registry.register("mock", Box::new(MockEngine));
    #[cfg(feature = "candle-engine")]
    registry.register("candle", Box::new(CandleEngine));
    registry.register("openvino", Box::new(OpenVINOEngine));
    registry.register("funasr", Box::new(FunASREngine));
    registry.register("qwen3_vl", Box::new(Qwen3VLEngine));
    registry.register("longcat", Box::new(LongCatImageEditEngine));
    registry.register("intel-npu", Box::new(IntelNpuEngine));
    registry.register("npu-tts", Box::new(NpuTtsEngine));
    registry.register("onnxruntime", Box::new(OnnxRuntimeEngine));
    registry.register("coreml", Box::new(CoreMlEngine));
    registry.register("mlx", Box::new(MlxEngine));
    registry.register("llamacpp", Box::new(LlamaCppEngine));
    #[cfg(feature = "candle-engine")]
    registry.register("wan", Box::new(WanEngine));
    registry.register("vulkan", Box::new(VulkanEngine));
    registry
});

unsafe fn write_error(err: &str, error_buffer: *mut c_char, error_buffer_len: usize) {
    if error_buffer.is_null() || error_buffer_len == 0 {
        return;
    }
    let err_c = match CString::new(err) {
        Ok(c) => c,
        Err(_) => return,
    };
    let bytes = err_c.as_bytes_with_nul();
    let to_copy = std::cmp::min(bytes.len(), error_buffer_len);
    std::ptr::copy_nonoverlapping(bytes.as_ptr() as *const c_char, error_buffer, to_copy);
    if to_copy > 0 {
        *error_buffer.add(to_copy - 1) = 0;
    }
}

/// Load a model pipeline. Returns NULL on error.
#[no_mangle]
pub unsafe extern "C" fn bloom_pipeline_load(
    model_path: *const c_char,
    engine_name: *const c_char,
    device_name: *const c_char,
    context_size: usize,
    error_buffer: *mut c_char,
    error_buffer_len: usize,
) -> *mut BloomPipeline {
    if model_path.is_null() || engine_name.is_null() || device_name.is_null() {
        write_error(
            "Null argument passed to bloom_pipeline_load",
            error_buffer,
            error_buffer_len,
        );
        return std::ptr::null_mut();
    }

    let model_path_str = match CStr::from_ptr(model_path).to_str() {
        Ok(s) => s,
        Err(e) => {
            write_error(
                &format!("Invalid model_path UTF-8: {}", e),
                error_buffer,
                error_buffer_len,
            );
            return std::ptr::null_mut();
        }
    };

    let engine_name_str = match CStr::from_ptr(engine_name).to_str() {
        Ok(s) => s,
        Err(e) => {
            write_error(
                &format!("Invalid engine_name UTF-8: {}", e),
                error_buffer,
                error_buffer_len,
            );
            return std::ptr::null_mut();
        }
    };

    let device_name_str = match CStr::from_ptr(device_name).to_str() {
        Ok(s) => s,
        Err(e) => {
            write_error(
                &format!("Invalid device_name UTF-8: {}", e),
                error_buffer,
                error_buffer_len,
            );
            return std::ptr::null_mut();
        }
    };

    let device = match device_name_str.to_lowercase().as_str() {
        "cpu" => DeviceKind::Cpu,
        "gpu" => DeviceKind::Gpu,
        "npu" => DeviceKind::Npu,
        other => {
            write_error(
                &format!("Unknown device kind: {}", other),
                error_buffer,
                error_buffer_len,
            );
            return std::ptr::null_mut();
        }
    };

    let engine = match ENGINE_REGISTRY.get(engine_name_str) {
        Ok(eng) => eng,
        Err(e) => {
            write_error(
                &format!("Engine registry failed: {}", e),
                error_buffer,
                error_buffer_len,
            );
            return std::ptr::null_mut();
        }
    };

    let path = Path::new(model_path_str);
    match InferencePipeline::load_standalone_with_context(engine, device, path, context_size) {
        Ok(pipeline) => Box::into_raw(Box::new(BloomPipeline { inner: pipeline })),
        Err(e) => {
            write_error(
                &format!("Failed to load standalone pipeline: {}", e),
                error_buffer,
                error_buffer_len,
            );
            return std::ptr::null_mut();
        }
    }
}

/// Free a loaded pipeline.
#[no_mangle]
pub unsafe extern "C" fn bloom_pipeline_free(pipeline: *mut BloomPipeline) {
    if !pipeline.is_null() {
        drop(Box::from_raw(pipeline));
    }
}

/// Run full inference (non-streaming).
/// Returns a JSON-serialized string of ModelOutput which the caller must free using bloom_string_free.
/// Returns NULL on error.
#[no_mangle]
pub unsafe extern "C" fn bloom_pipeline_run(
    pipeline: *mut BloomPipeline,
    input_json: *const c_char,
    params_json: *const c_char,
    error_buffer: *mut c_char,
    error_buffer_len: usize,
) -> *mut c_char {
    if pipeline.is_null() || input_json.is_null() || params_json.is_null() {
        write_error(
            "Null argument passed to bloom_pipeline_run",
            error_buffer,
            error_buffer_len,
        );
        return std::ptr::null_mut();
    }

    let pipe = &*pipeline;

    let input_json_str = match CStr::from_ptr(input_json).to_str() {
        Ok(s) => s,
        Err(e) => {
            write_error(
                &format!("Invalid input JSON UTF-8: {}", e),
                error_buffer,
                error_buffer_len,
            );
            return std::ptr::null_mut();
        }
    };

    let params_json_str = match CStr::from_ptr(params_json).to_str() {
        Ok(s) => s,
        Err(e) => {
            write_error(
                &format!("Invalid params JSON UTF-8: {}", e),
                error_buffer,
                error_buffer_len,
            );
            return std::ptr::null_mut();
        }
    };

    let input: ModelInput = match serde_json::from_str(input_json_str) {
        Ok(i) => i,
        Err(e) => {
            write_error(
                &format!("Failed to parse input JSON: {}", e),
                error_buffer,
                error_buffer_len,
            );
            return std::ptr::null_mut();
        }
    };

    let params: GenerationParams = match serde_json::from_str(params_json_str) {
        Ok(p) => p,
        Err(e) => {
            write_error(
                &format!("Failed to parse params JSON: {}", e),
                error_buffer,
                error_buffer_len,
            );
            return std::ptr::null_mut();
        }
    };

    match pipe.inner.run(input, &params) {
        Ok(output) => {
            let output_json = match serde_json::to_string(&output) {
                Ok(s) => s,
                Err(e) => {
                    write_error(
                        &format!("Failed to serialize ModelOutput: {}", e),
                        error_buffer,
                        error_buffer_len,
                    );
                    return std::ptr::null_mut();
                }
            };
            match CString::new(output_json) {
                Ok(c) => c.into_raw(),
                Err(e) => {
                    write_error(
                        &format!("CString conversion failed: {}", e),
                        error_buffer,
                        error_buffer_len,
                    );
                    return std::ptr::null_mut();
                }
            }
        }
        Err(e) => {
            write_error(
                &format!("Inference failed: {}", e),
                error_buffer,
                error_buffer_len,
            );
            return std::ptr::null_mut();
        }
    }
}

/// Run streaming inference.
/// Returns 0 on success, negative error code on failure.
#[no_mangle]
pub unsafe extern "C" fn bloom_pipeline_run_stream(
    pipeline: *mut BloomPipeline,
    input_json: *const c_char,
    params_json: *const c_char,
    callback: BloomStreamCallback,
    user_data: *mut std::ffi::c_void,
    error_buffer: *mut c_char,
    error_buffer_len: usize,
) -> i32 {
    if pipeline.is_null() || input_json.is_null() || params_json.is_null() {
        write_error(
            "Null argument passed to bloom_pipeline_run_stream",
            error_buffer,
            error_buffer_len,
        );
        return -1;
    }

    let pipe = &*pipeline;

    let input_json_str = match CStr::from_ptr(input_json).to_str() {
        Ok(s) => s,
        Err(e) => {
            write_error(
                &format!("Invalid input JSON UTF-8: {}", e),
                error_buffer,
                error_buffer_len,
            );
            return -2;
        }
    };

    let params_json_str = match CStr::from_ptr(params_json).to_str() {
        Ok(s) => s,
        Err(e) => {
            write_error(
                &format!("Invalid params JSON UTF-8: {}", e),
                error_buffer,
                error_buffer_len,
            );
            return -3;
        }
    };

    let input: ModelInput = match serde_json::from_str(input_json_str) {
        Ok(i) => i,
        Err(e) => {
            write_error(
                &format!("Failed to parse input JSON: {}", e),
                error_buffer,
                error_buffer_len,
            );
            return -4;
        }
    };

    let params: GenerationParams = match serde_json::from_str(params_json_str) {
        Ok(p) => p,
        Err(e) => {
            write_error(
                &format!("Failed to parse params JSON: {}", e),
                error_buffer,
                error_buffer_len,
            );
            return -5;
        }
    };

    let mut sink = FfiOutputSink {
        callback,
        user_data,
    };
    match pipe.inner.run_stream(input, &params, &mut sink) {
        Ok(()) => 0,
        Err(e) => {
            write_error(
                &format!("Streaming inference failed: {}", e),
                error_buffer,
                error_buffer_len,
            );
            -6
        }
    }
}

/// Free a string returned by bloom_pipeline_run.
#[no_mangle]
pub unsafe extern "C" fn bloom_string_free(s: *mut c_char) {
    if !s.is_null() {
        drop(CString::from_raw(s));
    }
}
