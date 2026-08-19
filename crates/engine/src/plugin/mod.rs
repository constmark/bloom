use crate::core::model::{LoadedModel, ModelMetadata};
use crate::engine::Engine;
use crate::{ModelInput, ModelOutput, OutputChunk};
use anyhow::{Context, Result};
use bloomai_core::{BloomError, DeviceKind, Modality};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::Arc;

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

        let mut c_engine = std::mem::MaybeUninit::<ffi::CBloomEngine>::uninit();
        let res = unsafe { init_fn(c_engine.as_mut_ptr()) };
        if res != 0 {
            return Err(BloomError::Plugin(format!(
                "bloom_plugin_init failed with error code {}",
                res
            ))
            .into());
        }
        let c_engine = unsafe { c_engine.assume_init() };

        Ok(Box::new(FfiPluginEngine {
            _lib: Some(lib),
            c_engine,
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
}

pub struct FfiPluginEngine {
    _lib: Option<Arc<libloading::Library>>,
    c_engine: ffi::CBloomEngine,
}

impl Engine for FfiPluginEngine {
    fn name(&self) -> &'static str {
        let ptr = (self.c_engine.name)();
        if ptr.is_null() {
            "unknown_plugin"
        } else {
            unsafe {
                std::ffi::CStr::from_ptr(ptr)
                    .to_str()
                    .unwrap_or("invalid_utf8")
            }
        }
    }

    fn supported_modalities(&self) -> Vec<Modality> {
        let mut out_modalities = vec![0i32; 16];
        let mut out_len = 0usize;
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
        let mut out_len = 0usize;
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
        if res != 0 || metadata_json_ptr.is_null() {
            (self.c_engine.free_model)(model_ptr);
            return Err(BloomError::Plugin(format!(
                "Failed to read metadata from plugin model: error code {}",
                res
            ))
            .into());
        }
        let metadata_str = unsafe { std::ffi::CStr::from_ptr(metadata_json_ptr).to_str()? };
        let metadata: ModelMetadata = serde_json::from_str(metadata_str)?;
        (self.c_engine.free_string)(metadata_json_ptr);

        Ok(Box::new(FfiPluginModel {
            _lib: self._lib.clone(),
            model_ptr,
            free_model_fn: self.c_engine.free_model,
            _model_metadata_fn: self.c_engine.model_metadata,
            model_infer_fn: self.c_engine.model_infer,
            model_infer_stream_fn: self.c_engine.model_infer_stream,
            free_string_fn: self.c_engine.free_string,
            metadata,
        }))
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
}

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

        let mut out_json_ptr = std::ptr::null_mut();
        let res = (self.model_infer_fn)(self.model_ptr, payload_c.as_ptr(), &mut out_json_ptr);
        if res != 0 || out_json_ptr.is_null() {
            return Err(BloomError::Plugin(format!(
                "Inference failed in dynamic plugin: error code {}",
                res
            ))
            .into());
        }

        let out_str = unsafe { std::ffi::CStr::from_ptr(out_json_ptr).to_str()? };
        let output: ModelOutput = serde_json::from_str(out_str)?;
        (self.free_string_fn)(out_json_ptr);

        Ok(output)
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
            let state = unsafe { &mut *(user_data as *mut CallbackState) };
            if chunk_json.is_null() {
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
            if let Err(e) = state.sink.on_chunk(chunk) {
                state.err = Some(e);
                return -4;
            }
            0
        }

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
}
