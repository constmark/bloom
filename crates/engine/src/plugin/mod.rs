use crate::core::model::{LoadedModel, ModelMetadata};
use crate::engine::Engine;
use crate::{ModelInput, ModelOutput, OutputChunk};
use anyhow::{Context, Result};
use bloomai_core::{BloomError, DeviceKind, Modality};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PluginMetadata {
    pub name: String,
    pub version: String,
    pub description: String,
    pub author: String,
    pub license: String,
    pub homepage: Option<String>,
    pub platforms: Vec<String>,
    pub min_runtime_version: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PluginEntryPoint {
    #[serde(rename = "type")]
    pub entry_type: String, // "native", "wasm", "remote", "subprocess"
    pub path: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PluginManifest {
    pub metadata: PluginMetadata,
    pub entry_point: PluginEntryPoint,
    #[serde(default)]
    pub supported_families: Vec<String>,
    #[serde(default)]
    pub supported_dtypes: Vec<String>,
    #[serde(default)]
    pub supported_formats: Vec<String>,
    #[serde(default)]
    pub supported_devices: Vec<String>,
    #[serde(default)]
    pub supported_modalities: Vec<String>,
    pub supports_streaming: Option<bool>,
    pub supports_quantized_models: Option<bool>,
    pub max_context_tokens: Option<usize>,
    #[serde(default)]
    pub required_backends: Vec<String>,
    #[serde(default)]
    pub example_models: Vec<String>,
    pub device_class: Option<String>,
    pub supports_mmap: Option<bool>,
    pub has_quantization_kernels: Option<bool>,
    pub memory_overhead_bytes: Option<usize>,
    pub probe_script: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PluginEntryValidation {
    NativeLibrary { path: PathBuf },
    WasmModule { path: PathBuf },
    Subprocess { path: PathBuf },
    RemoteEndpoint { url: String },
}

pub struct PluginManager;

impl PluginManager {
    /// Loads and parses a plugin manifest from a JSON file.
    pub fn load_manifest<P: AsRef<Path>>(path: P) -> Result<PluginManifest> {
        let content =
            std::fs::read_to_string(path).context("Failed to read plugin manifest file")?;
        let manifest: PluginManifest =
            serde_json::from_str(&content).context("Failed to parse plugin manifest JSON")?;
        Self::validate_manifest(&manifest)?;
        Ok(manifest)
    }

    /// Validates the plugin manifest fields.
    pub fn validate_manifest(manifest: &PluginManifest) -> Result<()> {
        if manifest.metadata.name.is_empty() {
            return Err(BloomError::Plugin("Plugin name cannot be empty".into()).into());
        }
        crate::core::security::validate_plugin(&manifest.metadata.name)?;
        if manifest.metadata.version.is_empty() {
            return Err(BloomError::Plugin("Plugin version cannot be empty".into()).into());
        }
        if manifest.metadata.platforms.is_empty() {
            return Err(BloomError::Plugin("Plugin platforms list cannot be empty".into()).into());
        }

        // Platform compatibility check
        let current = Self::current_platform();
        if current != "unknown" && !manifest.metadata.platforms.contains(&current.to_string()) {
            return Err(BloomError::Plugin(format!(
                "Plugin '{}' is not compatible with current platform '{}'. Supported platforms: {:?}",
                manifest.metadata.name, current, manifest.metadata.platforms
            )).into());
        }

        // Entry point check
        if manifest.entry_point.path.is_empty() {
            return Err(
                BloomError::Plugin("Plugin entry point path cannot be empty".into()).into(),
            );
        }
        let allowed_types = ["native", "wasm", "remote", "subprocess"];
        if !allowed_types.contains(&manifest.entry_point.entry_type.as_str()) {
            return Err(BloomError::Plugin(format!(
                "Invalid entry point type: {}",
                manifest.entry_point.entry_type
            ))
            .into());
        }

        Ok(())
    }

    /// Validate an entry point without executing untrusted plugin code.
    ///
    /// This is the lightweight boundary check used by mock/plugin CI. Native
    /// libraries can be fully checked with `validate_native_library` when the
    /// test intentionally wants to load a dynamic library.
    pub fn validate_entry_point<P: AsRef<Path>>(
        manifest: &PluginManifest,
        base_dir: P,
    ) -> Result<PluginEntryValidation> {
        match manifest.entry_point.entry_type.as_str() {
            "native" => {
                let path = Self::resolve_entry_path(&manifest.entry_point.path, base_dir);
                if !path.exists() {
                    return Err(BloomError::MissingRequiredFile(format!(
                        "Native plugin entry point does not exist: {:?}",
                        path
                    ))
                    .into());
                }
                Ok(PluginEntryValidation::NativeLibrary { path })
            }
            "wasm" => {
                let path = Self::resolve_entry_path(&manifest.entry_point.path, base_dir);
                if !path.exists() {
                    return Err(BloomError::MissingRequiredFile(format!(
                        "WASM plugin entry point does not exist: {:?}",
                        path
                    ))
                    .into());
                }
                if path.extension().and_then(|s| s.to_str()) != Some("wasm") {
                    return Err(BloomError::InvalidInput(format!(
                        "WASM plugin entry point must end with .wasm: {:?}",
                        path
                    ))
                    .into());
                }
                Ok(PluginEntryValidation::WasmModule { path })
            }
            "subprocess" => {
                let path = Self::resolve_entry_path(&manifest.entry_point.path, base_dir);
                if !path.exists() {
                    return Err(BloomError::MissingRequiredFile(format!(
                        "Subprocess plugin entry point does not exist: {:?}",
                        path
                    ))
                    .into());
                }
                Ok(PluginEntryValidation::Subprocess { path })
            }
            "remote" => {
                let url = manifest.entry_point.path.trim();
                if !(url.starts_with("https://") || url.starts_with("http://127.0.0.1:")) {
                    return Err(BloomError::InvalidInput(
                        "Remote plugin endpoints must use https:// or explicit local http://127.0.0.1: URLs".into()
                    ).into());
                }
                Ok(PluginEntryValidation::RemoteEndpoint {
                    url: url.to_string(),
                })
            }
            other => {
                Err(BloomError::InvalidInput(format!("Invalid entry point type: {}", other)).into())
            }
        }
    }

    fn resolve_entry_path<P: AsRef<Path>>(path: &str, base_dir: P) -> PathBuf {
        let mut entry_path = PathBuf::from(path);
        if entry_path.is_relative() {
            entry_path = base_dir.as_ref().join(entry_path);
        }
        entry_path
    }

    /// Validates the native dynamic library compatibility using libloading.
    pub fn validate_native_library<P: AsRef<Path>>(
        manifest: &PluginManifest,
        base_dir: P,
    ) -> Result<()> {
        Self::validate_manifest(manifest)?;
        if manifest.entry_point.entry_type != "native" {
            return Ok(());
        }
        let lib_path = Self::resolve_entry_path(&manifest.entry_point.path, base_dir);

        if !lib_path.exists() {
            return Err(BloomError::MissingRequiredFile(format!(
                "Native library file does not exist: {:?}",
                lib_path
            ))
            .into());
        }

        unsafe {
            let lib = libloading::Library::new(&lib_path).map_err(|e| {
                BloomError::Plugin(format!(
                    "Failed to load dynamic library {:?}: {:?}",
                    lib_path, e
                ))
            })?;

            // Try to resolve the standard initialization function
            let _init_fn: libloading::Symbol<unsafe extern "C" fn() -> i32> =
                lib.get(b"bloom_plugin_init\0").map_err(|_| {
                    BloomError::Plugin(
                        "Dynamic library does not export 'bloom_plugin_init' initialization symbol"
                            .into(),
                    )
                })?;
        }

        Ok(())
    }

    /// Returns the current platform identifier.
    pub fn current_platform() -> &'static str {
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        return "macos-aarch64";
        #[cfg(all(target_os = "macos", target_arch = "x86_64"))]
        return "macos-x86_64";
        #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
        return "linux-x86_64";
        #[cfg(all(target_os = "linux", target_arch = "aarch64"))]
        return "linux-aarch64";
        #[cfg(all(target_os = "windows", target_arch = "x86_64"))]
        return "windows-x86_64";
        #[cfg(not(any(
            all(target_os = "macos", target_arch = "aarch64"),
            all(target_os = "macos", target_arch = "x86_64"),
            all(target_os = "linux", target_arch = "x86_64"),
            all(target_os = "linux", target_arch = "aarch64"),
            all(target_os = "windows", target_arch = "x86_64")
        )))]
        return "unknown";
    }

    /// Loads a plugin dynamic library and returns a boxed Engine implementation.
    pub fn load_engine_plugin<P: AsRef<Path>>(
        manifest: &PluginManifest,
        base_dir: P,
    ) -> Result<Box<dyn Engine>> {
        Self::validate_manifest(manifest)?;
        if manifest.entry_point.entry_type != "native" {
            return Err(BloomError::Plugin(
                "Only 'native' entry point type is supported for engine plugins".into(),
            )
            .into());
        }
        let lib_path = Self::resolve_entry_path(&manifest.entry_point.path, base_dir);
        let lib = Arc::new(unsafe {
            libloading::Library::new(&lib_path).map_err(|e| {
                BloomError::Plugin(format!(
                    "Failed to load dynamic library {:?}: {:?}",
                    lib_path, e
                ))
            })?
        });

        let init_fn: libloading::Symbol<unsafe extern "C" fn(*mut ffi::CBloomEngine) -> i32> = unsafe {
            lib.get(b"bloom_plugin_init\0").map_err(|_| {
                BloomError::Plugin("Dynamic library does not export 'bloom_plugin_init'".into())
            })?
        };

        // A nullable wire table is valid even if a plugin forgets a callback.
        let mut wire = ffi::NullableEngine::default();
        let res = unsafe { init_fn((&mut wire as *mut ffi::NullableEngine).cast()) };
        if res != 0 {
            return Err(BloomError::Plugin(format!(
                "bloom_plugin_init failed with error code {}",
                res
            ))
            .into());
        }
        let c_engine = wire.validate()?;

        Ok(Box::new(FfiPluginEngine {
            _lib: Some(lib),
            c_engine,
            name: OnceLock::new(),
        }))
    }
}

pub mod ffi {
    use std::os::raw::{c_char, c_void};

    pub type CBloomStreamCallback =
        extern "C" fn(user_data: *mut c_void, chunk_json: *const c_char) -> i32;

    #[repr(C)]
    pub struct CBloomEngine {
        pub name: extern "C" fn() -> *const c_char,
        pub supported_modalities:
            extern "C" fn(out_modalities: *mut i32, out_len: *mut usize) -> i32,
        pub supported_devices: extern "C" fn(out_devices: *mut i32, out_len: *mut usize) -> i32,
        pub load_model: extern "C" fn(
            model_path: *const c_char,
            device_kind: i32,
            out_model: *mut *mut c_void,
        ) -> i32,
        pub free_model: extern "C" fn(model: *mut c_void),
        pub model_metadata: extern "C" fn(model: *mut c_void, out_json: *mut *mut c_char) -> i32,
        pub model_infer: extern "C" fn(
            model: *mut c_void,
            input_json: *const c_char,
            out_json: *mut *mut c_char,
        ) -> i32,
        pub model_infer_stream: extern "C" fn(
            model: *mut c_void,
            input_json: *const c_char,
            callback: CBloomStreamCallback,
            user_data: *mut c_void,
        ) -> i32,
        pub free_string: extern "C" fn(s: *mut c_char),
    }
    #[repr(C)]
    #[derive(Default)]
    pub(super) struct NullableEngine {
        pub name: Option<extern "C" fn() -> *const c_char>,
        pub supported_modalities:
            Option<extern "C" fn(out_modalities: *mut i32, out_len: *mut usize) -> i32>,
        pub supported_devices:
            Option<extern "C" fn(out_devices: *mut i32, out_len: *mut usize) -> i32>,
        pub load_model: Option<
            extern "C" fn(
                model_path: *const c_char,
                device_kind: i32,
                out_model: *mut *mut c_void,
            ) -> i32,
        >,
        pub free_model: Option<extern "C" fn(model: *mut c_void)>,
        pub model_metadata:
            Option<extern "C" fn(model: *mut c_void, out_json: *mut *mut c_char) -> i32>,
        pub model_infer: Option<
            extern "C" fn(
                model: *mut c_void,
                input_json: *const c_char,
                out_json: *mut *mut c_char,
            ) -> i32,
        >,
        pub model_infer_stream: Option<
            extern "C" fn(
                model: *mut c_void,
                input_json: *const c_char,
                callback: CBloomStreamCallback,
                user_data: *mut c_void,
            ) -> i32,
        >,
        pub free_string: Option<extern "C" fn(s: *mut c_char)>,
    }
    impl NullableEngine {
        pub(super) fn validate(self) -> anyhow::Result<CBloomEngine> {
            Ok(CBloomEngine {
                name: self
                    .name
                    .ok_or_else(|| anyhow::anyhow!("plugin callback name is missing"))?,
                supported_modalities: self.supported_modalities.ok_or_else(|| {
                    anyhow::anyhow!("plugin callback supported_modalities is missing")
                })?,
                supported_devices: self.supported_devices.ok_or_else(|| {
                    anyhow::anyhow!("plugin callback supported_devices is missing")
                })?,
                load_model: self
                    .load_model
                    .ok_or_else(|| anyhow::anyhow!("plugin callback load_model is missing"))?,
                free_model: self
                    .free_model
                    .ok_or_else(|| anyhow::anyhow!("plugin callback free_model is missing"))?,
                model_metadata: self
                    .model_metadata
                    .ok_or_else(|| anyhow::anyhow!("plugin callback model_metadata is missing"))?,
                model_infer: self
                    .model_infer
                    .ok_or_else(|| anyhow::anyhow!("plugin callback model_infer is missing"))?,
                model_infer_stream: self.model_infer_stream.ok_or_else(|| {
                    anyhow::anyhow!("plugin callback model_infer_stream is missing")
                })?,
                free_string: self
                    .free_string
                    .ok_or_else(|| anyhow::anyhow!("plugin callback free_string is missing"))?,
            })
        }
    }
}

pub struct FfiPluginEngine {
    _lib: Option<Arc<libloading::Library>>,
    c_engine: ffi::CBloomEngine,
    name: OnceLock<&'static str>,
}

impl Engine for FfiPluginEngine {
    fn name(&self) -> &'static str {
        self.name.get_or_init(|| {
            let ptr = (self.c_engine.name)();
            if ptr.is_null() {
                return "unknown_plugin";
            }
            // Engine::name promises a static lifetime. Copy once so callers can
            // retain the name even after this plugin library has been unloaded.
            let name = unsafe { std::ffi::CStr::from_ptr(ptr) }
                .to_string_lossy()
                .into_owned();
            Box::leak(name.into_boxed_str())
        })
    }

    fn supported_modalities(&self) -> Vec<Modality> {
        let mut out_modalities = vec![0i32; 16];
        let mut out_len = 16usize;
        let res = (self.c_engine.supported_modalities)(out_modalities.as_mut_ptr(), &mut out_len);
        if res != 0 {
            return vec![];
        }
        out_modalities
            .into_iter()
            .take(out_len)
            .filter_map(|m| match m {
                0 => Some(Modality::Text),
                1 => Some(Modality::Vision),
                2 => Some(Modality::Audio),
                _ => None,
            })
            .collect()
    }

    fn supported_devices(&self) -> Vec<DeviceKind> {
        let mut out_devices = vec![0i32; 16];
        let mut out_len = 16usize;
        let res = (self.c_engine.supported_devices)(out_devices.as_mut_ptr(), &mut out_len);
        if res != 0 {
            return vec![];
        }
        out_devices
            .into_iter()
            .take(out_len)
            .filter_map(|d| match d {
                0 => Some(DeviceKind::Cpu),
                1 => Some(DeviceKind::Gpu),
                2 => Some(DeviceKind::Npu),
                _ => None,
            })
            .collect()
    }

    fn load(&self, model_path: &Path, device: DeviceKind) -> Result<Box<dyn LoadedModel>> {
        let path_str = std::ffi::CString::new(model_path.to_string_lossy().as_ref())?;
        let device_i32 = match device {
            DeviceKind::Cpu => 0,
            DeviceKind::Gpu => 1,
            DeviceKind::Npu => 2,
        };

        let mut model_ptr = std::ptr::null_mut();
        let res = (self.c_engine.load_model)(path_str.as_ptr(), device_i32, &mut model_ptr);
        let mut handle = PluginModelHandle {
            ptr: model_ptr,
            free: self.c_engine.free_model,
        };
        if res != 0 || model_ptr.is_null() {
            return Err(BloomError::Plugin(format!(
                "Failed to load model in dynamic plugin: error code {}",
                res
            ))
            .into());
        }

        // Get model metadata
        let mut metadata_json_ptr = std::ptr::null_mut();
        let res = (self.c_engine.model_metadata)(model_ptr, &mut metadata_json_ptr);
        let metadata_json = PluginString {
            ptr: metadata_json_ptr,
            free: self.c_engine.free_string,
        };
        if res != 0 || metadata_json_ptr.is_null() {
            return Err(BloomError::Plugin(format!(
                "Failed to read metadata from plugin model: error code {}",
                res
            ))
            .into());
        }
        let metadata: ModelMetadata = serde_json::from_str(metadata_json.text()?)?;
        handle.ptr = std::ptr::null_mut();

        Ok(Box::new(FfiPluginModel {
            _lib: self._lib.clone(),
            model_ptr,
            free_model_fn: self.c_engine.free_model,
            _model_metadata_fn: self.c_engine.model_metadata,
            model_infer_fn: self.c_engine.model_infer,
            model_infer_stream_fn: self.c_engine.model_infer_stream,
            free_string_fn: self.c_engine.free_string,
            metadata,
            inference_gate: std::sync::Mutex::new(()),
        }))
    }
}

struct PluginModelHandle {
    ptr: *mut std::ffi::c_void,
    free: extern "C" fn(*mut std::ffi::c_void),
}
impl Drop for PluginModelHandle {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            (self.free)(self.ptr);
        }
    }
}
struct PluginString {
    ptr: *mut std::os::raw::c_char,
    free: extern "C" fn(*mut std::os::raw::c_char),
}
impl PluginString {
    fn text(&self) -> Result<&str> {
        anyhow::ensure!(!self.ptr.is_null(), "plugin returned a null string");
        Ok(unsafe { std::ffi::CStr::from_ptr(self.ptr) }.to_str()?)
    }
}
impl Drop for PluginString {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            (self.free)(self.ptr);
        }
    }
}

pub struct FfiPluginModel {
    _lib: Option<Arc<libloading::Library>>,
    model_ptr: *mut std::ffi::c_void,
    free_model_fn: extern "C" fn(*mut std::ffi::c_void),
    _model_metadata_fn: extern "C" fn(*mut std::ffi::c_void, *mut *mut std::os::raw::c_char) -> i32,
    model_infer_fn: extern "C" fn(
        *mut std::ffi::c_void,
        *const std::os::raw::c_char,
        *mut *mut std::os::raw::c_char,
    ) -> i32,
    model_infer_stream_fn: extern "C" fn(
        *mut std::ffi::c_void,
        *const std::os::raw::c_char,
        ffi::CBloomStreamCallback,
        *mut std::ffi::c_void,
    ) -> i32,
    free_string_fn: extern "C" fn(*mut std::os::raw::c_char),
    metadata: ModelMetadata,
    inference_gate: std::sync::Mutex<()>,
}

// Calls on an opaque model are serialized; the plugin ABI requires its handle
// to be movable between threads, but does not require concurrent inference.
unsafe impl Send for FfiPluginModel {}
unsafe impl Sync for FfiPluginModel {}

impl Drop for FfiPluginModel {
    fn drop(&mut self) {
        (self.free_model_fn)(self.model_ptr);
    }
}

impl LoadedModel for FfiPluginModel {
    fn metadata(&self) -> &ModelMetadata {
        &self.metadata
    }

    fn infer(
        &self,
        input: ModelInput,
        params: &bloomai_core::GenerationParams,
    ) -> Result<ModelOutput> {
        #[derive(Serialize)]
        struct FfiInputPayload<'a> {
            input: ModelInput,
            params: &'a bloomai_core::GenerationParams,
        }
        let payload = FfiInputPayload { input, params };
        let payload_str = serde_json::to_string(&payload)?;
        let payload_c = std::ffi::CString::new(payload_str)?;

        let _gate = self
            .inference_gate
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let mut out_json_ptr = std::ptr::null_mut();
        let res = (self.model_infer_fn)(self.model_ptr, payload_c.as_ptr(), &mut out_json_ptr);
        let output_json = PluginString {
            ptr: out_json_ptr,
            free: self.free_string_fn,
        };
        if res != 0 || out_json_ptr.is_null() {
            return Err(BloomError::Plugin(format!(
                "Inference failed in dynamic plugin: error code {}",
                res
            ))
            .into());
        }

        Ok(serde_json::from_str(output_json.text()?)?)
    }

    fn infer_stream(
        &self,
        input: ModelInput,
        params: &bloomai_core::GenerationParams,
        sink: &mut dyn crate::model::OutputSink,
    ) -> Result<()> {
        #[derive(Serialize)]
        struct FfiInputPayload<'a> {
            input: ModelInput,
            params: &'a bloomai_core::GenerationParams,
        }
        let payload = FfiInputPayload { input, params };
        let payload_str = serde_json::to_string(&payload)?;
        let payload_c = std::ffi::CString::new(payload_str)?;

        struct CallbackState<'a> {
            sink: &'a mut dyn crate::model::OutputSink,
            err: Option<anyhow::Error>,
        }

        extern "C" fn stream_callback(
            user_data: *mut std::ffi::c_void,
            chunk_json: *const std::os::raw::c_char,
        ) -> i32 {
            if user_data.is_null() {
                return -1;
            }
            let state = unsafe { &mut *(user_data as *mut CallbackState) };
            if state.err.is_some() {
                return -1;
            }
            if chunk_json.is_null() {
                state.err = Some(anyhow::anyhow!("plugin emitted a null chunk"));
                return -1;
            }
            let chunk_str = unsafe {
                match std::ffi::CStr::from_ptr(chunk_json).to_str() {
                    Ok(s) => s,
                    Err(e) => {
                        state.err = Some(e.into());
                        return -2;
                    }
                }
            };
            let chunk: OutputChunk = match serde_json::from_str(chunk_str) {
                Ok(c) => c,
                Err(e) => {
                    state.err = Some(e.into());
                    return -3;
                }
            };
            match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                state.sink.on_chunk(chunk)
            })) {
                Ok(Ok(())) => 0,
                Ok(Err(error)) => {
                    state.err = Some(error);
                    -4
                }
                Err(_) => {
                    state.err = Some(anyhow::anyhow!("plugin output sink panicked"));
                    -5
                }
            }
        }

        let _gate = self
            .inference_gate
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let mut state = CallbackState { sink, err: None };
        let res = (self.model_infer_stream_fn)(
            self.model_ptr,
            payload_c.as_ptr(),
            stream_callback,
            &mut state as *mut CallbackState as *mut std::ffi::c_void,
        );

        if let Some(err) = state.err {
            return Err(err);
        }
        if res != 0 {
            return Err(BloomError::Plugin(format!(
                "Stream inference failed in dynamic plugin: error code {}",
                res
            ))
            .into());
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn test_manifest(entry_type: &str, path: &str) -> PluginManifest {
        PluginManifest {
            metadata: PluginMetadata {
                name: "test-plugin".to_string(),
                version: "1.0.0".to_string(),
                description: "desc".to_string(),
                author: "author".to_string(),
                license: "MIT".to_string(),
                homepage: None,
                platforms: vec![PluginManager::current_platform().to_string()],
                min_runtime_version: "0.1.0".to_string(),
            },
            entry_point: PluginEntryPoint {
                entry_type: entry_type.to_string(),
                path: path.to_string(),
            },
            supported_families: vec![],
            supported_dtypes: vec![],
            supported_formats: vec![],
            supported_devices: vec![],
            supported_modalities: vec![],
            supports_streaming: None,
            supports_quantized_models: None,
            max_context_tokens: None,
            required_backends: vec![],
            example_models: vec![],
            device_class: None,
            supports_mmap: None,
            has_quantization_kernels: None,
            memory_overhead_bytes: None,
            probe_script: None,
        }
    }

    #[test]
    fn test_load_and_validate_engine_manifest() {
        let manifest_path = Path::new("../../examples/plugins/engine-plugin.json");
        // Check relative path resolution based on cargo test location
        let path = if manifest_path.exists() {
            manifest_path.to_path_buf()
        } else {
            Path::new("examples/plugins/engine-plugin.json").to_path_buf()
        };

        let manifest = PluginManager::load_manifest(&path).unwrap();
        assert_eq!(manifest.metadata.name, "org.community.llama-cpp-engine");
        assert_eq!(manifest.entry_point.entry_type, "native");
        assert_eq!(manifest.entry_point.path, "libllama_engine.so");
        assert_eq!(manifest.supported_families, vec!["Llama", "Qwen", "Gemma"]);
    }

    #[test]
    fn test_load_incompatible_manifest() {
        // Only verify platform incompatibility if the current platform is macos/windows, since backend-plugin.json only supports linux
        let current = PluginManager::current_platform();
        if current.contains("macos") || current.contains("windows") {
            let manifest_path = Path::new("../../examples/plugins/backend-plugin.json");
            let path = if manifest_path.exists() {
                manifest_path.to_path_buf()
            } else {
                Path::new("examples/plugins/backend-plugin.json").to_path_buf()
            };

            let res = PluginManager::load_manifest(&path);
            assert!(res.is_err());
            let err_msg = res.err().unwrap().to_string();
            assert!(err_msg.contains("is not compatible with current platform"));
        }
    }

    #[test]
    fn test_native_library_validation_missing_file() {
        let manifest = test_manifest("native", "non_existent_library.so");

        let temp_dir = tempdir().unwrap();
        let res = PluginManager::validate_native_library(&manifest, temp_dir.path());
        assert!(res.is_err());
        assert!(res.err().unwrap().to_string().contains("does not exist"));
    }

    #[test]
    fn test_native_library_validation_invalid_file() {
        let temp_dir = tempdir().unwrap();
        let invalid_lib_path = temp_dir.path().join("invalid_lib.so");
        std::fs::write(&invalid_lib_path, b"not a valid dynamic library").unwrap();

        let manifest = test_manifest("native", "invalid_lib.so");

        let res = PluginManager::validate_native_library(&manifest, temp_dir.path());
        assert!(res.is_err());
        assert!(
            res.err()
                .unwrap()
                .to_string()
                .contains("Failed to load dynamic library")
        );
    }

    #[test]
    fn test_mock_subprocess_plugin_entry_validation() {
        let temp_dir = tempdir().unwrap();
        let script = temp_dir.path().join("mock-plugin.sh");
        std::fs::write(&script, "#!/bin/sh\nexit 0\n").unwrap();
        let manifest = test_manifest("subprocess", "mock-plugin.sh");

        let validation = PluginManager::validate_entry_point(&manifest, temp_dir.path()).unwrap();
        assert!(matches!(
            validation,
            PluginEntryValidation::Subprocess { .. }
        ));
    }

    #[test]
    fn test_mock_wasm_plugin_entry_validation() {
        let temp_dir = tempdir().unwrap();
        let wasm = temp_dir.path().join("mock-plugin.wasm");
        std::fs::write(&wasm, b"\0asm\x01\0\0\0").unwrap();
        let manifest = test_manifest("wasm", "mock-plugin.wasm");

        let validation = PluginManager::validate_entry_point(&manifest, temp_dir.path()).unwrap();
        assert!(matches!(
            validation,
            PluginEntryValidation::WasmModule { .. }
        ));
    }

    #[test]
    fn test_remote_plugin_requires_https_or_localhost() {
        let manifest = test_manifest("remote", "http://example.com/plugin");
        let temp_dir = tempdir().unwrap();
        let err = PluginManager::validate_entry_point(&manifest, temp_dir.path()).unwrap_err();
        assert!(err.to_string().contains("https://"));

        let manifest = test_manifest("remote", "https://example.com/plugin");
        let validation = PluginManager::validate_entry_point(&manifest, temp_dir.path()).unwrap();
        assert!(matches!(
            validation,
            PluginEntryValidation::RemoteEndpoint { .. }
        ));
    }

    #[test]
    fn test_ffi_plugin_engine_wrapper() {
        use bloomai_core::ModelManifest;
        extern "C" fn name() -> *const std::os::raw::c_char {
            c"mock_ffi_engine".as_ptr()
        }

        extern "C" fn supported_modalities(out_modalities: *mut i32, out_len: *mut usize) -> i32 {
            unsafe {
                *out_modalities = 0;
                *out_len = 1;
            }
            0
        }

        extern "C" fn supported_devices(out_devices: *mut i32, out_len: *mut usize) -> i32 {
            unsafe {
                *out_devices = 0;
                *out_len = 1;
            }
            0
        }

        extern "C" fn load_model(
            _model_path: *const std::os::raw::c_char,
            _device_kind: i32,
            out_model: *mut *mut std::os::raw::c_void,
        ) -> i32 {
            unsafe {
                *out_model = 0x12345678 as *mut _;
            }
            0
        }

        extern "C" fn free_model(model: *mut std::os::raw::c_void) {
            assert_eq!(model as usize, 0x12345678);
        }

        extern "C" fn model_metadata(
            model: *mut std::os::raw::c_void,
            out_json: *mut *mut std::os::raw::c_char,
        ) -> i32 {
            assert_eq!(model as usize, 0x12345678);
            let metadata = ModelMetadata {
                id: "mock-model".to_string(),
                modality: Modality::Text,
                quantized: false,
                manifest: ModelManifest::default(),
            };
            let json = serde_json::to_string(&metadata).unwrap();
            let c_str = std::ffi::CString::new(json).unwrap();
            unsafe {
                *out_json = c_str.into_raw();
            }
            0
        }

        extern "C" fn free_string(s: *mut std::os::raw::c_char) {
            if !s.is_null() {
                unsafe {
                    let _ = std::ffi::CString::from_raw(s);
                }
            }
        }

        extern "C" fn model_infer(
            model: *mut std::os::raw::c_void,
            input_json: *const std::os::raw::c_char,
            out_json: *mut *mut std::os::raw::c_char,
        ) -> i32 {
            assert_eq!(model as usize, 0x12345678);
            let _input_str = unsafe { std::ffi::CStr::from_ptr(input_json).to_str().unwrap() };
            let output = ModelOutput {
                text: Some("hello from FFI".to_string()),
                logits: None,
                image: None,
                audio: None,
                video: None,
            };
            let json = serde_json::to_string(&output).unwrap();
            let c_str = std::ffi::CString::new(json).unwrap();
            unsafe {
                *out_json = c_str.into_raw();
            }
            0
        }

        extern "C" fn model_infer_stream(
            model: *mut std::os::raw::c_void,
            _input_json: *const std::os::raw::c_char,
            callback: ffi::CBloomStreamCallback,
            user_data: *mut std::os::raw::c_void,
        ) -> i32 {
            assert_eq!(model as usize, 0x12345678);
            let chunk1 = OutputChunk::TextDelta("hello ".to_string());
            let json1 = serde_json::to_string(&chunk1).unwrap();
            let c_str1 = std::ffi::CString::new(json1).unwrap();
            let res = callback(user_data, c_str1.as_ptr());
            assert_eq!(res, 0);

            let chunk2 = OutputChunk::End;
            let json2 = serde_json::to_string(&chunk2).unwrap();
            let c_str2 = std::ffi::CString::new(json2).unwrap();
            let res = callback(user_data, c_str2.as_ptr());
            assert_eq!(res, 0);

            0
        }

        let c_engine = ffi::CBloomEngine {
            name,
            supported_modalities,
            supported_devices,
            load_model,
            free_model,
            model_metadata,
            model_infer,
            model_infer_stream,
            free_string,
        };

        let engine = FfiPluginEngine {
            _lib: None,
            c_engine,
            name: OnceLock::new(),
        };

        assert_eq!(engine.name(), "mock_ffi_engine");
        assert_eq!(engine.supported_modalities(), vec![Modality::Text]);
        assert_eq!(engine.supported_devices(), vec![DeviceKind::Cpu]);

        let loaded = engine.load(Path::new("dummy"), DeviceKind::Cpu).unwrap();
        assert_eq!(loaded.metadata().id, "mock-model");

        let input = ModelInput::Text {
            prompt: "prompt".to_string(),
        };
        let params = bloomai_core::GenerationParams::default();
        let output = loaded.infer(input.clone(), &params).unwrap();
        assert_eq!(output.text.unwrap(), "hello from FFI");

        struct MockSink {
            chunks: Vec<OutputChunk>,
        }
        impl crate::model::OutputSink for MockSink {
            fn on_chunk(&mut self, chunk: OutputChunk) -> Result<()> {
                self.chunks.push(chunk);
                Ok(())
            }
        }
        let mut sink = MockSink { chunks: vec![] };
        loaded.infer_stream(input, &params, &mut sink).unwrap();
        assert_eq!(sink.chunks.len(), 2);
        assert!(matches!(&sink.chunks[0], OutputChunk::TextDelta(t) if t == "hello "));
        assert!(matches!(sink.chunks[1], OutputChunk::End));
    }
    thread_local! {
        static FAILURE_MODE: std::cell::Cell<u8> = const { std::cell::Cell::new(0) };
        static MODEL_FREES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
        static STRING_FREES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
        static DYNAMIC_NAME: std::cell::RefCell<Option<std::ffi::CString>> = const { std::cell::RefCell::new(None) };
    }

    fn failing_plugin(mode: u8) -> FfiPluginEngine {
        FAILURE_MODE.set(mode);
        MODEL_FREES.set(0);
        STRING_FREES.set(0);
        extern "C" fn name() -> *const std::os::raw::c_char {
            DYNAMIC_NAME.with(|name| {
                name.borrow()
                    .as_ref()
                    .map_or(c"failure-fixture".as_ptr(), |name| name.as_ptr())
            })
        }
        extern "C" fn supported(_: *mut i32, len: *mut usize) -> i32 {
            unsafe {
                *len = 0;
            }
            0
        }
        extern "C" fn load(
            _: *const std::os::raw::c_char,
            _: i32,
            out: *mut *mut std::ffi::c_void,
        ) -> i32 {
            unsafe {
                *out = Box::into_raw(Box::new(0u8)).cast();
            }
            if FAILURE_MODE.get() == 4 { 1 } else { 0 }
        }
        extern "C" fn free_model(ptr: *mut std::ffi::c_void) {
            unsafe {
                drop(Box::from_raw(ptr.cast::<u8>()));
            }
            MODEL_FREES.set(MODEL_FREES.get() + 1);
        }
        extern "C" fn free_string(ptr: *mut std::os::raw::c_char) {
            unsafe {
                drop(std::ffi::CString::from_raw(ptr));
            }
            STRING_FREES.set(STRING_FREES.get() + 1);
        }
        fn write_string(out: *mut *mut std::os::raw::c_char, bytes: Vec<u8>) {
            unsafe {
                *out = std::ffi::CString::new(bytes).unwrap().into_raw();
            }
        }
        extern "C" fn metadata(
            _: *mut std::ffi::c_void,
            out: *mut *mut std::os::raw::c_char,
        ) -> i32 {
            let bytes = match FAILURE_MODE.get() {
                1 | 3 => b"invalid JSON".to_vec(),
                2 => vec![0xff],
                _ => serde_json::to_vec(&ModelMetadata {
                    id: "fixture".into(),
                    modality: Modality::Text,
                    quantized: false,
                    manifest: Default::default(),
                })
                .unwrap(),
            };
            write_string(out, bytes);
            if FAILURE_MODE.get() == 3 { 1 } else { 0 }
        }
        extern "C" fn infer(
            _: *mut std::ffi::c_void,
            _: *const std::os::raw::c_char,
            out: *mut *mut std::os::raw::c_char,
        ) -> i32 {
            write_string(
                out,
                if FAILURE_MODE.get() == 6 {
                    vec![0xff]
                } else {
                    b"invalid JSON".to_vec()
                },
            );
            if FAILURE_MODE.get() == 7 { 1 } else { 0 }
        }
        extern "C" fn stream(
            _: *mut std::ffi::c_void,
            _: *const std::os::raw::c_char,
            callback: ffi::CBloomStreamCallback,
            data: *mut std::ffi::c_void,
        ) -> i32 {
            let chunk = match FAILURE_MODE.get() {
                8 => std::ptr::null(),
                9 => c"invalid JSON".as_ptr(),
                _ => c"\"End\"".as_ptr(),
            };
            let _ = callback(data, chunk);
            // Deliberately ignore the failure to prove the host retains it.
            let _ = callback(data, c"\"End\"".as_ptr());
            0
        }
        FfiPluginEngine {
            _lib: None,
            name: OnceLock::new(),
            c_engine: ffi::CBloomEngine {
                name,
                supported_modalities: supported,
                supported_devices: supported,
                load_model: load,
                free_model,
                model_metadata: metadata,
                model_infer: infer,
                model_infer_stream: stream,
                free_string,
            },
        }
    }

    #[test]
    fn plugin_failures_release_models_and_strings_exactly_once() {
        for mode in 1..=4 {
            let engine = failing_plugin(mode);
            assert!(engine.load(Path::new("fixture"), DeviceKind::Cpu).is_err());
            assert_eq!(MODEL_FREES.get(), 1);
            assert_eq!(STRING_FREES.get(), usize::from(mode != 4));
        }
        for mode in 5..=7 {
            let engine = failing_plugin(mode);
            let model = engine.load(Path::new("fixture"), DeviceKind::Cpu).unwrap();
            assert!(
                model
                    .infer(
                        ModelInput::Text {
                            prompt: "test".into()
                        },
                        &Default::default()
                    )
                    .is_err()
            );
            assert_eq!(STRING_FREES.get(), 2);
            drop(model);
            assert_eq!(MODEL_FREES.get(), 1);
        }
    }

    #[test]
    fn plugin_callback_retains_null_parse_sink_and_panic_failures() {
        for mode in 8..=11 {
            let engine = failing_plugin(mode);
            let model = engine.load(Path::new("fixture"), DeviceKind::Cpu).unwrap();
            let mut calls = 0;
            let result = model.infer_stream(
                ModelInput::Text {
                    prompt: "test".into(),
                },
                &Default::default(),
                &mut |_| {
                    calls += 1;
                    if mode == 11 {
                        panic!("sink panic fixture");
                    }
                    anyhow::bail!("sink rejected output")
                },
            );
            assert!(result.is_err());
            assert_eq!(calls, usize::from(mode >= 10));
        }
    }

    #[test]
    fn plugin_name_outlives_the_library_owned_string() {
        DYNAMIC_NAME.with(|name| {
            *name.borrow_mut() = Some(std::ffi::CString::new("temporary-plugin-name").unwrap())
        });
        let engine = failing_plugin(0);
        let name = engine.name();
        DYNAMIC_NAME.with(|name| *name.borrow_mut() = None);
        drop(engine);
        assert_eq!(name, "temporary-plugin-name");
    }

    #[test]
    fn plugin_wire_table_rejects_missing_callbacks_before_use() {
        assert_eq!(
            std::mem::size_of::<ffi::CBloomEngine>(),
            std::mem::size_of::<ffi::NullableEngine>()
        );
        assert_eq!(
            std::mem::align_of::<ffi::CBloomEngine>(),
            std::mem::align_of::<ffi::NullableEngine>()
        );
        assert!(ffi::NullableEngine::default().validate().is_err());
    }
}
