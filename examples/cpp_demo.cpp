#include "../crates/ffi/bloom.h"
#include <iostream>
#include <vector>
#include <string>

int main() {
    std::cout << "=== Bloom C++ SDK Integration Test ===" << std::endl;

    char err_buf[512] = {0};
    BloomPipeline* pipeline = bloom_pipeline_load(
        ".",      // model_path
        "mock",   // engine_name
        "cpu",    // device_name
        2048,     // context_size
        err_buf,
        sizeof(err_buf)
    );

    if (!pipeline) {
        std::cerr << "Failed to load pipeline: " << err_buf << std::endl;
        return 1;
    }
    std::cout << "Pipeline loaded successfully." << std::endl;

    const char* input_json = "{\"Text\":{\"prompt\":\"Hello from C++!\"}}";
    const char* params_json = "{\"max_tokens\":10,\"temperature\":0.7,\"top_p\":0.9,\"seed\":null}";

    char err_buf_run[512] = {0};
    char* result_json = bloom_pipeline_run(
        pipeline,
        input_json,
        params_json,
        err_buf_run,
        sizeof(err_buf_run)
    );

    if (!result_json) {
        std::cerr << "Inference failed: " << err_buf_run << std::endl;
        bloom_pipeline_free(pipeline);
        return 1;
    }

    std::cout << "Inference result: " << result_json << std::endl;

    // Free resources
    bloom_string_free(result_json);
    bloom_pipeline_free(pipeline);

    std::cout << "C++ integration test completed successfully!" << std::endl;
    return 0;
}
