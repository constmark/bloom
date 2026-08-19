#![cfg_attr(not(test), warn(clippy::unwrap_used))]
use anyhow::{Result, anyhow};
use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::Path;

use bloomai_core::{
    DType, DeviceClass, DeviceKind, GenerationParams, Modality, ModelFamily, ModelFormat,
};
use bloomai_engine::InferencePipeline;
use bloomai_engine::core::engine::{BackendMaturity, Engine, EngineCapability, EngineRegistry};
use bloomai_engine::core::io::{ModelInput, OutputChunk};
use bloomai_engine::model::{EchoTextModel, LoadedModel, OutputSink};

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

#[cfg(test)]
struct PanicTestEngine;

#[cfg(test)]
impl Engine for PanicTestEngine {
    fn name(&self) -> &'static str {
        "panic-test"
    }

    fn supported_modalities(&self) -> Vec<Modality> {
        MockEngine.supported_modalities()
    }

    fn supported_devices(&self) -> Vec<DeviceKind> {
        MockEngine.supported_devices()
    }

    fn capability(&self) -> EngineCapability {
        MockEngine.capability()
    }

    fn load(&self, _model_path: &Path, _device: DeviceKind) -> Result<Box<dyn LoadedModel>> {
        panic!("intentional FFI boundary test panic")
    }
}

static ENGINE_REGISTRY: once_cell::sync::Lazy<EngineRegistry> = once_cell::sync::Lazy::new(|| {
    let mut registry = EngineRegistry::default();
    registry.register("mock", Box::new(MockEngine));
    #[cfg(test)]
    registry.register("panic-test", Box::new(PanicTestEngine));
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
    unsafe {
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
}

fn catch_ffi_panic<T>(operation: impl FnOnce() -> T, on_panic: impl FnOnce() -> T) -> T {
    match catch_unwind(AssertUnwindSafe(operation)) {
        Ok(value) => value,
        Err(_) => on_panic(),
    }
}

/// Load a model pipeline. Returns NULL on error.
///
/// # Safety
///
/// Each non-NULL input pointer must reference a valid NUL-terminated C string
/// for the duration of the call. When non-NULL, `error_buffer` must reference
/// `error_buffer_len` writable bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn bloom_pipeline_load(
    model_path: *const c_char,
    engine_name: *const c_char,
    device_name: *const c_char,
    context_size: usize,
    error_buffer: *mut c_char,
    error_buffer_len: usize,
) -> *mut BloomPipeline {
    unsafe {
        catch_ffi_panic(
            || {
                bloom_pipeline_load_impl(
                    model_path,
                    engine_name,
                    device_name,
                    context_size,
                    error_buffer,
                    error_buffer_len,
                )
            },
            || {
                write_error(
                    "Bloom caught an internal panic while loading the pipeline",
                    error_buffer,
                    error_buffer_len,
                );
                std::ptr::null_mut()
            },
        )
    }
}

unsafe fn bloom_pipeline_load_impl(
    model_path: *const c_char,
    engine_name: *const c_char,
    device_name: *const c_char,
    context_size: usize,
    error_buffer: *mut c_char,
    error_buffer_len: usize,
) -> *mut BloomPipeline {
    unsafe {
        if model_path.is_null() || engine_name.is_null() || device_name.is_null() {
            write_error(
                "Null argument passed to bloom_pipeline_load",
                error_buffer,
                error_buffer_len,
            );
            return std::ptr::null_mut();
        }
        if context_size == 0 {
            write_error(
                "context_size must be greater than zero",
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
                std::ptr::null_mut()
            }
        }
    }
}

/// Free a loaded pipeline.
///
/// # Safety
///
/// `pipeline` must be NULL or a live pointer returned by
/// [`bloom_pipeline_load`]. A live pointer must be freed exactly once and must
/// not be in use by another thread during this call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn bloom_pipeline_free(pipeline: *mut BloomPipeline) {
    unsafe {
        if !pipeline.is_null() {
            catch_ffi_panic(|| drop(Box::from_raw(pipeline)), || ());
        }
    }
}

/// Run full inference (non-streaming).
/// Returns a JSON-serialized string of ModelOutput which the caller must free using bloom_string_free.
/// Returns NULL on error.
///
/// # Safety
///
/// `pipeline` must be a live pointer returned by [`bloom_pipeline_load`]. The
/// JSON pointers must reference valid NUL-terminated C strings for the duration
/// of the call. When non-NULL, `error_buffer` must reference
/// `error_buffer_len` writable bytes. The pipeline must not be freed
/// concurrently.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn bloom_pipeline_run(
    pipeline: *mut BloomPipeline,
    input_json: *const c_char,
    params_json: *const c_char,
    error_buffer: *mut c_char,
    error_buffer_len: usize,
) -> *mut c_char {
    unsafe {
        catch_ffi_panic(
            || {
                bloom_pipeline_run_impl(
                    pipeline,
                    input_json,
                    params_json,
                    error_buffer,
                    error_buffer_len,
                )
            },
            || {
                write_error(
                    "Bloom caught an internal panic while running inference",
                    error_buffer,
                    error_buffer_len,
                );
                std::ptr::null_mut()
            },
        )
    }
}

unsafe fn bloom_pipeline_run_impl(
    pipeline: *mut BloomPipeline,
    input_json: *const c_char,
    params_json: *const c_char,
    error_buffer: *mut c_char,
    error_buffer_len: usize,
) -> *mut c_char {
    unsafe {
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
                        std::ptr::null_mut()
                    }
                }
            }
            Err(e) => {
                write_error(
                    &format!("Inference failed: {}", e),
                    error_buffer,
                    error_buffer_len,
                );
                std::ptr::null_mut()
            }
        }
    }
}

/// Run streaming inference.
/// Returns 0 on success, negative error code on failure.
///
/// # Safety
///
/// `pipeline` must be a live pointer returned by [`bloom_pipeline_load`]. The
/// JSON pointers must reference valid NUL-terminated C strings for the duration
/// of the call. `callback` must be non-NULL and must not unwind across the C
/// boundary; each chunk pointer it receives is borrowed only for that callback
/// invocation. When non-NULL, `error_buffer` must reference
/// `error_buffer_len` writable bytes. The pipeline must not be freed
/// concurrently.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn bloom_pipeline_run_stream(
    pipeline: *mut BloomPipeline,
    input_json: *const c_char,
    params_json: *const c_char,
    callback: Option<BloomStreamCallback>,
    user_data: *mut std::ffi::c_void,
    error_buffer: *mut c_char,
    error_buffer_len: usize,
) -> i32 {
    unsafe {
        catch_ffi_panic(
            || {
                bloom_pipeline_run_stream_impl(
                    pipeline,
                    input_json,
                    params_json,
                    callback,
                    user_data,
                    error_buffer,
                    error_buffer_len,
                )
            },
            || {
                write_error(
                    "Bloom caught an internal panic while streaming inference",
                    error_buffer,
                    error_buffer_len,
                );
                -7
            },
        )
    }
}

unsafe fn bloom_pipeline_run_stream_impl(
    pipeline: *mut BloomPipeline,
    input_json: *const c_char,
    params_json: *const c_char,
    callback: Option<BloomStreamCallback>,
    user_data: *mut std::ffi::c_void,
    error_buffer: *mut c_char,
    error_buffer_len: usize,
) -> i32 {
    unsafe {
        let callback = match callback {
            Some(callback) => callback,
            None => {
                write_error(
                    "Null argument passed to bloom_pipeline_run_stream",
                    error_buffer,
                    error_buffer_len,
                );
                return -1;
            }
        };
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
}

/// Free a string returned by bloom_pipeline_run.
///
/// # Safety
///
/// `s` must be NULL or a live pointer returned by [`bloom_pipeline_run`]. A
/// live pointer must be freed exactly once.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn bloom_string_free(s: *mut c_char) {
    unsafe {
        if !s.is_null() {
            catch_ffi_panic(|| drop(CString::from_raw(s)), || ());
        }
    }
}

#[cfg(test)]
mod tests {
    use std::ffi::{CStr, CString};
    use std::os::raw::c_char;

    use super::{bloom_pipeline_load, catch_ffi_panic};

    #[test]
    fn ffi_panic_guard_returns_the_supplied_fallback() {
        let value = catch_ffi_panic(|| panic!("test panic"), || 17);
        assert_eq!(value, 17);
    }

    #[test]
    fn load_boundary_converts_an_engine_panic_to_an_error() {
        let model_path = CString::new(".").unwrap();
        let engine_name = CString::new("panic-test").unwrap();
        let device_name = CString::new("cpu").unwrap();
        let mut error_buffer = [0 as c_char; 128];

        let pipeline = unsafe {
            bloom_pipeline_load(
                model_path.as_ptr(),
                engine_name.as_ptr(),
                device_name.as_ptr(),
                128,
                error_buffer.as_mut_ptr(),
                error_buffer.len(),
            )
        };

        assert!(pipeline.is_null());
        let error = unsafe { CStr::from_ptr(error_buffer.as_ptr()) }.to_string_lossy();
        assert!(error.contains("caught an internal panic"));
    }
}
