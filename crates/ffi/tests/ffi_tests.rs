use once_cell::sync::Lazy;
use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::sync::Mutex;

// Import the FFI declarations directly
use bloom_ffi::{
    bloom_pipeline_free, bloom_pipeline_load, bloom_pipeline_run, bloom_pipeline_run_stream,
    bloom_string_free, BloomPipeline,
};

static CALLBACK_DATA: Lazy<Mutex<Vec<String>>> = Lazy::new(|| Mutex::new(Vec::new()));

unsafe extern "C" fn test_stream_callback(
    user_data: *mut std::ffi::c_void,
    chunk_json: *const c_char,
) {
    // Assert user_data is correct
    assert_eq!(user_data as usize, 0x12345678);
    if !chunk_json.is_null() {
        let s = CStr::from_ptr(chunk_json).to_string_lossy().into_owned();
        let mut data = CALLBACK_DATA.lock().unwrap();
        data.push(s);
    }
}

#[test]
fn test_ffi_mock_inference() {
    unsafe {
        // Use "." (current directory) as the path so canonicalization succeeds.
        let model_path = CString::new(".").unwrap();
        let engine_name = CString::new("mock").unwrap();
        let device_name = CString::new("cpu").unwrap();

        let mut err_buf = [0 as c_char; 512];

        // 1. Test load
        let pipeline: *mut BloomPipeline = bloom_pipeline_load(
            model_path.as_ptr(),
            engine_name.as_ptr(),
            device_name.as_ptr(),
            2048,
            err_buf.as_mut_ptr(),
            err_buf.len(),
        );

        let err_str = CStr::from_ptr(err_buf.as_ptr()).to_string_lossy();
        assert!(
            !pipeline.is_null(),
            "Pipeline failed to load. Error: {}",
            err_str
        );

        // 2. Test run (non-streaming)
        let input_json = CString::new(r#"{"Text":{"prompt":"Hello World"}}"#).unwrap();
        let params_json =
            CString::new(r#"{"max_tokens":10,"temperature":0.7,"top_p":0.9,"seed":null}"#).unwrap();

        let mut err_buf_run = [0 as c_char; 512];
        let result_ptr = bloom_pipeline_run(
            pipeline,
            input_json.as_ptr(),
            params_json.as_ptr(),
            err_buf_run.as_mut_ptr(),
            err_buf_run.len(),
        );

        let err_run_str = CStr::from_ptr(err_buf_run.as_ptr()).to_string_lossy();
        assert!(
            !result_ptr.is_null(),
            "Inference run failed. Error: {}",
            err_run_str
        );

        let result_str = CStr::from_ptr(result_ptr).to_string_lossy().into_owned();
        println!("Non-streaming result JSON: {}", result_str);
        assert!(result_str.contains("text"));

        // Free result string
        bloom_string_free(result_ptr);

        // 3. Test run stream
        CALLBACK_DATA.lock().unwrap().clear();
        let user_data = 0x12345678 as *mut std::ffi::c_void;
        let mut err_buf_stream = [0 as c_char; 512];
        let stream_res = bloom_pipeline_run_stream(
            pipeline,
            input_json.as_ptr(),
            params_json.as_ptr(),
            Some(test_stream_callback),
            user_data,
            err_buf_stream.as_mut_ptr(),
            err_buf_stream.len(),
        );

        let err_stream_str = CStr::from_ptr(err_buf_stream.as_ptr()).to_string_lossy();
        assert_eq!(
            stream_res, 0,
            "Streaming inference failed with code {}. Error: {}",
            stream_res, err_stream_str
        );

        let callbacks = CALLBACK_DATA.lock().unwrap();
        println!("Streaming chunks received: {:?}", *callbacks);
        assert!(!callbacks.is_empty(), "No callback chunks received");

        // 4. Clean up pipeline
        bloom_pipeline_free(pipeline);
    }
}

#[test]
fn test_ffi_rejects_a_null_stream_callback() {
    unsafe {
        let model_path = CString::new(".").unwrap();
        let engine_name = CString::new("mock").unwrap();
        let device_name = CString::new("cpu").unwrap();
        let input_json = CString::new(r#"{"Text":{"prompt":"Hello World"}}"#).unwrap();
        let params_json =
            CString::new(r#"{"max_tokens":10,"temperature":0.7,"top_p":0.9,"seed":null}"#).unwrap();
        let mut err_buf = [0 as c_char; 512];
        let pipeline = bloom_pipeline_load(
            model_path.as_ptr(),
            engine_name.as_ptr(),
            device_name.as_ptr(),
            2048,
            err_buf.as_mut_ptr(),
            err_buf.len(),
        );
        assert!(!pipeline.is_null());

        let result = bloom_pipeline_run_stream(
            pipeline,
            input_json.as_ptr(),
            params_json.as_ptr(),
            None,
            std::ptr::null_mut(),
            err_buf.as_mut_ptr(),
            err_buf.len(),
        );

        assert_eq!(result, -1);
        let error = CStr::from_ptr(err_buf.as_ptr()).to_string_lossy();
        assert!(error.contains("Null argument"));
        bloom_pipeline_free(pipeline);
    }
}

#[test]
fn test_ffi_rejects_a_zero_context_size() {
    unsafe {
        let model_path = CString::new(".").unwrap();
        let engine_name = CString::new("mock").unwrap();
        let device_name = CString::new("cpu").unwrap();
        let mut err_buf = [0 as c_char; 512];

        let pipeline = bloom_pipeline_load(
            model_path.as_ptr(),
            engine_name.as_ptr(),
            device_name.as_ptr(),
            0,
            err_buf.as_mut_ptr(),
            err_buf.len(),
        );

        assert!(pipeline.is_null());
        let error = CStr::from_ptr(err_buf.as_ptr()).to_string_lossy();
        assert!(error.contains("context_size must be greater than zero"));
    }
}

#[test]
fn test_ffi_vulkan_routing() {
    unsafe {
        let model_path = CString::new(".").unwrap();
        let engine_name = CString::new("vulkan").unwrap();
        let device_name = CString::new("gpu").unwrap();

        let mut err_buf = [0 as c_char; 512];

        let pipeline: *mut BloomPipeline = bloom_pipeline_load(
            model_path.as_ptr(),
            engine_name.as_ptr(),
            device_name.as_ptr(),
            2048,
            err_buf.as_mut_ptr(),
            err_buf.len(),
        );

        assert!(pipeline.is_null());
        let err_str = CStr::from_ptr(err_buf.as_ptr()).to_string_lossy();
        assert!(
            err_str.contains("SPIR-V")
                || err_str.contains("vulkan")
                || err_str.contains("spv")
                || err_str.contains("Vulkan")
        );
    }
}
