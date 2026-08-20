use once_cell::sync::Lazy;
use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

// Import the FFI declarations directly
use bloom_ffi::{
    BLOOM_ABI_VERSION, BLOOM_STATUS_CANCELLED, BLOOM_STATUS_INVALID_ARGUMENT,
    BLOOM_STATUS_INVALID_INPUT_JSON, BLOOM_STATUS_INVALID_PARAMS_JSON, BLOOM_STATUS_INVALID_UTF8,
    BLOOM_STATUS_OK, BloomCancellationToken, BloomOwnedBuffer, BloomPipeline, BloomSlice,
    bloom_abi_version, bloom_buffer_free, bloom_cancellation_token_cancel,
    bloom_cancellation_token_free, bloom_cancellation_token_new, bloom_pipeline_free,
    bloom_pipeline_load, bloom_pipeline_load_v2, bloom_pipeline_run, bloom_pipeline_run_stream,
    bloom_pipeline_run_stream_v2, bloom_pipeline_run_v2, bloom_string_free,
};

static CALLBACK_DATA: Lazy<Mutex<Vec<String>>> = Lazy::new(|| Mutex::new(Vec::new()));
static V2_CALLBACK_COUNT: AtomicUsize = AtomicUsize::new(0);

fn bloom_slice(bytes: &[u8]) -> BloomSlice {
    BloomSlice {
        data: bytes.as_ptr(),
        len: bytes.len(),
    }
}

unsafe extern "C" fn test_stream_callback(
    user_data: *mut std::ffi::c_void,
    chunk_json: *const c_char,
) {
    unsafe {
        // Assert user_data is correct
        assert_eq!(user_data as usize, 0x12345678);
        if !chunk_json.is_null() {
            let s = CStr::from_ptr(chunk_json).to_string_lossy().into_owned();
            let mut data = CALLBACK_DATA.lock().unwrap();
            data.push(s);
        }
    }
}

unsafe extern "C" fn test_stream_callback_v2(
    _user_data: *mut std::ffi::c_void,
    chunk_json: *const u8,
    chunk_json_len: usize,
) {
    assert!(!chunk_json.is_null());
    let chunk = unsafe { std::slice::from_raw_parts(chunk_json, chunk_json_len) };
    assert!(serde_json::from_slice::<serde_json::Value>(chunk).is_ok());
    V2_CALLBACK_COUNT.fetch_add(1, Ordering::Relaxed);
}

unsafe extern "C" fn cancel_from_stream_callback_v2(
    user_data: *mut std::ffi::c_void,
    chunk_json: *const u8,
    chunk_json_len: usize,
) {
    unsafe { test_stream_callback_v2(std::ptr::null_mut(), chunk_json, chunk_json_len) };
    let token = user_data.cast::<BloomCancellationToken>();
    assert_eq!(
        unsafe { bloom_cancellation_token_cancel(token) },
        BLOOM_STATUS_OK
    );
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
fn test_v2_ffi_uses_length_delimited_buffers() {
    unsafe {
        assert_eq!(bloom_abi_version(), BLOOM_ABI_VERSION);
        let model_path = b".";
        let engine_name = b"mock";
        let device_name = b"cpu";
        let mut err_buf = [0 as c_char; 512];
        let pipeline = bloom_pipeline_load_v2(
            bloom_slice(model_path),
            bloom_slice(engine_name),
            bloom_slice(device_name),
            2048,
            err_buf.as_mut_ptr(),
            err_buf.len(),
        );
        assert!(!pipeline.is_null());

        let input_json = br#"{"Text":{"prompt":"Hello v2"}}"#;
        let params_json = br#"{"max_tokens":10,"temperature":0.7,"top_p":0.9,"seed":null}"#;
        let mut output_buffer = BloomOwnedBuffer {
            data: std::ptr::null_mut(),
            len: 0,
        };
        let status = bloom_pipeline_run_v2(
            pipeline,
            bloom_slice(input_json),
            bloom_slice(params_json),
            &mut output_buffer,
            err_buf.as_mut_ptr(),
            err_buf.len(),
        );
        assert_eq!(status, BLOOM_STATUS_OK);
        assert!(!output_buffer.data.is_null());
        let output_json = std::slice::from_raw_parts(output_buffer.data, output_buffer.len);
        let output: serde_json::Value = serde_json::from_slice(output_json).unwrap();
        assert_eq!(output["text"], "echo: Hello v2");

        bloom_buffer_free(&mut output_buffer);
        assert!(output_buffer.data.is_null());
        assert_eq!(output_buffer.len, 0);
        bloom_pipeline_free(pipeline);
    }
}

#[test]
fn test_v2_stream_cancellation_is_distinct_and_suppresses_callbacks() {
    unsafe {
        let mut err_buf = [0 as c_char; 512];
        let pipeline = bloom_pipeline_load_v2(
            bloom_slice(b"."),
            bloom_slice(b"mock"),
            bloom_slice(b"cpu"),
            2048,
            err_buf.as_mut_ptr(),
            err_buf.len(),
        );
        assert!(!pipeline.is_null());
        let token = bloom_cancellation_token_new();
        assert!(!token.is_null());
        assert_eq!(bloom_cancellation_token_cancel(token), BLOOM_STATUS_OK);
        V2_CALLBACK_COUNT.store(0, Ordering::Relaxed);

        let status = bloom_pipeline_run_stream_v2(
            pipeline,
            bloom_slice(br#"{"Text":{"prompt":"Hello"}}"#),
            bloom_slice(br#"{"max_tokens":10,"temperature":0.7,"top_p":0.9,"seed":null}"#),
            Some(test_stream_callback_v2),
            std::ptr::null_mut(),
            token,
            err_buf.as_mut_ptr(),
            err_buf.len(),
        );
        assert_eq!(status, BLOOM_STATUS_CANCELLED);
        assert_eq!(V2_CALLBACK_COUNT.load(Ordering::Relaxed), 0);
        assert!(
            CStr::from_ptr(err_buf.as_ptr())
                .to_string_lossy()
                .contains("cancelled")
        );

        bloom_cancellation_token_free(token);
        bloom_pipeline_free(pipeline);
    }
}

#[test]
fn test_v2_stream_observes_cancellation_from_an_active_callback() {
    unsafe {
        let mut err_buf = [0 as c_char; 512];
        let pipeline = bloom_pipeline_load_v2(
            bloom_slice(b"."),
            bloom_slice(b"mock"),
            bloom_slice(b"cpu"),
            2048,
            err_buf.as_mut_ptr(),
            err_buf.len(),
        );
        assert!(!pipeline.is_null());
        let token = bloom_cancellation_token_new();
        assert!(!token.is_null());
        V2_CALLBACK_COUNT.store(0, Ordering::Relaxed);

        let status = bloom_pipeline_run_stream_v2(
            pipeline,
            bloom_slice(br#"{"Text":{"prompt":"Hello"}}"#),
            bloom_slice(br#"{"max_tokens":10,"temperature":0.7,"top_p":0.9,"seed":null}"#),
            Some(cancel_from_stream_callback_v2),
            token.cast(),
            token,
            err_buf.as_mut_ptr(),
            err_buf.len(),
        );
        assert_eq!(status, BLOOM_STATUS_CANCELLED);
        assert_eq!(V2_CALLBACK_COUNT.load(Ordering::Relaxed), 1);

        bloom_cancellation_token_free(token);
        bloom_pipeline_free(pipeline);
    }
}

#[test]
fn test_v2_rejects_non_utf8_without_scanning_for_a_terminator() {
    unsafe {
        let mut err_buf = [0 as c_char; 512];
        let pipeline = bloom_pipeline_load_v2(
            bloom_slice(b"."),
            bloom_slice(b"mock"),
            bloom_slice(b"cpu"),
            2048,
            err_buf.as_mut_ptr(),
            err_buf.len(),
        );
        assert!(!pipeline.is_null());
        let mut output = BloomOwnedBuffer {
            data: std::ptr::null_mut(),
            len: 0,
        };
        let status = bloom_pipeline_run_v2(
            pipeline,
            bloom_slice(&[0xff, 0xfe]),
            bloom_slice(b"{}"),
            &mut output,
            err_buf.as_mut_ptr(),
            err_buf.len(),
        );
        assert_eq!(status, BLOOM_STATUS_INVALID_UTF8);
        assert!(output.data.is_null());
        bloom_pipeline_free(pipeline);
    }
}

#[test]
fn test_v2_rejects_oversized_and_malformed_inputs_with_stable_statuses() {
    unsafe {
        let mut err_buf = [0 as c_char; 512];
        let pipeline = bloom_pipeline_load_v2(
            bloom_slice(b"."),
            bloom_slice(b"mock"),
            bloom_slice(b"cpu"),
            2048,
            err_buf.as_mut_ptr(),
            err_buf.len(),
        );
        assert!(!pipeline.is_null());
        let params = bloom_slice(br#"{"max_tokens":10,"temperature":0.7,"top_p":0.9,"seed":null}"#);

        for (input, parameters, expected) in [
            (bloom_slice(b"{"), params, BLOOM_STATUS_INVALID_INPUT_JSON),
            (
                bloom_slice(br#"{"Text":{"prompt":"Hello"}}"#),
                bloom_slice(b"{"),
                BLOOM_STATUS_INVALID_PARAMS_JSON,
            ),
            (
                BloomSlice {
                    data: b"{}".as_ptr(),
                    len: 16 * 1024 * 1024 + 1,
                },
                params,
                BLOOM_STATUS_INVALID_ARGUMENT,
            ),
        ] {
            let mut output = BloomOwnedBuffer {
                data: std::ptr::null_mut(),
                len: 0,
            };
            let status = bloom_pipeline_run_v2(
                pipeline,
                input,
                parameters,
                &mut output,
                err_buf.as_mut_ptr(),
                err_buf.len(),
            );
            assert_eq!(status, expected);
            assert!(output.data.is_null());
        }

        let status = bloom_pipeline_run_stream_v2(
            pipeline,
            bloom_slice(br#"{"Text":{"prompt":"Hello"}}"#),
            params,
            None,
            std::ptr::null_mut(),
            std::ptr::null(),
            err_buf.as_mut_ptr(),
            err_buf.len(),
        );
        assert_eq!(status, BLOOM_STATUS_INVALID_ARGUMENT);
        assert_eq!(
            bloom_cancellation_token_cancel(std::ptr::null_mut()),
            BLOOM_STATUS_INVALID_ARGUMENT
        );
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
