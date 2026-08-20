#ifndef BLOOM_H
#define BLOOM_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Opaque handle representing a loaded Bloom inference pipeline.
 */
typedef struct BloomPipeline BloomPipeline;
typedef struct BloomCancellationToken BloomCancellationToken;

#define BLOOM_ABI_VERSION 2u

typedef enum BloomStatus {
    BLOOM_STATUS_OK = 0,
    BLOOM_STATUS_INVALID_ARGUMENT = -1,
    BLOOM_STATUS_INVALID_UTF8 = -2,
    BLOOM_STATUS_INVALID_INPUT_JSON = -3,
    BLOOM_STATUS_INVALID_PARAMS_JSON = -4,
    BLOOM_STATUS_INFERENCE_ERROR = -5,
    BLOOM_STATUS_OUTPUT_ERROR = -6,
    BLOOM_STATUS_PANIC = -7,
    BLOOM_STATUS_CANCELLED = -8
} BloomStatus;

/** Borrowed bytes. The caller retains ownership for the complete call. */
typedef struct BloomSlice {
    const uint8_t* data;
    size_t len;
} BloomSlice;

/** Owned bytes returned by Bloom. Initialize to {NULL, 0}. */
typedef struct BloomOwnedBuffer {
    uint8_t* data;
    size_t len;
} BloomOwnedBuffer;

/**
 * Callback function type invoked for each chunk during streaming inference.
 *
 * @param user_data User-provided pointer passed to bloom_pipeline_run_stream.
 * @param chunk_json JSON-serialized representation of the OutputChunk event.
 */
typedef void (*BloomStreamCallback)(void* user_data, const char* chunk_json);

/**
 * Length-aware callback used by ABI revision 2.
 * The chunk is borrowed only for the duration of the callback.
 */
typedef void (*BloomStreamCallbackV2)(
    void* user_data,
    const uint8_t* chunk_json,
    size_t chunk_json_len
);

/** Return the newest ABI revision implemented by the loaded library. */
uint32_t bloom_abi_version(void);

/**
 * Load a model pipeline using bounded, length-delimited UTF-8 inputs.
 * Returns NULL on failure and writes a bounded diagnostic to error_buffer.
 */
BloomPipeline* bloom_pipeline_load_v2(
    BloomSlice model_path,
    BloomSlice engine_name,
    BloomSlice device_name,
    size_t context_size,
    char* error_buffer,
    size_t error_buffer_len
);

/**
 * Run full inference using length-delimited input and output buffers.
 * On success, release output with bloom_buffer_free.
 */
int32_t bloom_pipeline_run_v2(
    BloomPipeline* pipeline,
    BloomSlice input_json,
    BloomSlice params_json,
    BloomOwnedBuffer* output,
    char* error_buffer,
    size_t error_buffer_len
);

/** Allocate a thread-safe cooperative cancellation token. */
BloomCancellationToken* bloom_cancellation_token_new(void);

/** Mark a token as cancelled. Safe to call while a stream uses the token. */
int32_t bloom_cancellation_token_cancel(BloomCancellationToken* token);

/**
 * Free a token after the associated stream has returned. NULL is accepted.
 */
void bloom_cancellation_token_free(BloomCancellationToken* token);

/**
 * Run streaming inference using length-delimited JSON and callback chunks.
 * cancellation may be NULL. Cancellation is observed at output boundaries and
 * returns BLOOM_STATUS_CANCELLED.
 */
int32_t bloom_pipeline_run_stream_v2(
    BloomPipeline* pipeline,
    BloomSlice input_json,
    BloomSlice params_json,
    BloomStreamCallbackV2 callback,
    void* user_data,
    const BloomCancellationToken* cancellation,
    char* error_buffer,
    size_t error_buffer_len
);

/** Free and clear a buffer returned by bloom_pipeline_run_v2. */
void bloom_buffer_free(BloomOwnedBuffer* buffer);

/* ABI revision 1 compatibility symbols follow. New consumers should use v2. */

/**
 * Load a model pipeline.
 *
 * @param model_path Path to the model directory or single file (GGUF).
 * @param engine_name Name of the execution engine (e.g. "candle", "openvino", "funasr").
 * @param device_name Name of the target device kind ("cpu", "gpu", "npu").
 * @param context_size Maximum sequence length context; must be greater than zero.
 * @param error_buffer Buffer to write the error message on failure.
 * @param error_buffer_len Capacity of the error buffer in bytes.
 * @return A pointer to the loaded BloomPipeline, or NULL on error.
 */
BloomPipeline* bloom_pipeline_load(
    const char* model_path,
    const char* engine_name,
    const char* device_name,
    size_t context_size,
    char* error_buffer,
    size_t error_buffer_len
);

/**
 * Free a loaded pipeline.
 *
 * @param pipeline Handle to the loaded pipeline.
 */
void bloom_pipeline_free(BloomPipeline* pipeline);

/**
 * Run full non-streaming inference.
 *
 * @param pipeline Handle to the loaded pipeline.
 * @param input_json JSON-serialized ModelInput representation.
 * @param params_json JSON-serialized GenerationParams representation.
 * @param error_buffer Buffer to write error details on failure.
 * @param error_buffer_len Capacity of the error buffer in bytes.
 * @return JSON-serialized string of ModelOutput (must be freed using bloom_string_free), or NULL on failure.
 */
char* bloom_pipeline_run(
    BloomPipeline* pipeline,
    const char* input_json,
    const char* params_json,
    char* error_buffer,
    size_t error_buffer_len
);

/**
 * Run streaming inference.
 *
 * @param pipeline Handle to the loaded pipeline.
 * @param input_json JSON-serialized ModelInput representation.
 * @param params_json JSON-serialized GenerationParams representation.
 * @param callback Non-NULL callback function invoked for each streamed chunk.
 * @param user_data User-provided pointer forwarded to the callback.
 * @param error_buffer Buffer to write error details on failure.
 * @param error_buffer_len Capacity of the error buffer.
 * @return 0 on success, -1 for a NULL argument, -2 through -6 for input or
 *         inference errors, and -7 if Bloom catches an internal panic.
 */
int32_t bloom_pipeline_run_stream(
    BloomPipeline* pipeline,
    const char* input_json,
    const char* params_json,
    BloomStreamCallback callback,
    void* user_data,
    char* error_buffer,
    size_t error_buffer_len
);

/**
 * Free a string returned by bloom_pipeline_run.
 *
 * @param s C-string pointer to free.
 */
void bloom_string_free(char* s);

#ifdef __cplusplus
}
#endif

#endif /* BLOOM_H */
