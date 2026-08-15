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

/**
 * Callback function type invoked for each chunk during streaming inference.
 *
 * @param user_data User-provided pointer passed to bloom_pipeline_run_stream.
 * @param chunk_json JSON-serialized representation of the OutputChunk event.
 */
typedef void (*BloomStreamCallback)(void* user_data, const char* chunk_json);

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
