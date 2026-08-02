#!/usr/bin/env bash
# Run Bloom's immutable official Qwen3 trained-model CPU acceptance profile.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

export BLOOM_TRAINED_MODEL_PROFILE="qwen3"
export BLOOM_TRAINED_COMMAND_NAME="$0"
export BLOOM_TRAINED_MODEL_REPOSITORY="Qwen/Qwen3-0.6B-GGUF"
export BLOOM_TRAINED_MODEL_REVISION="23749fefcc72300e3a2ad315e1317431b06b590a"
export BLOOM_TRAINED_MODEL_FILENAME="Qwen3-0.6B-Q8_0.gguf"
export BLOOM_TRAINED_MODEL_SIZE_BYTES="639446688"
export BLOOM_TRAINED_MODEL_SHA256="9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031"
export BLOOM_TRAINED_MODEL_LICENSE="Apache-2.0"
export BLOOM_TRAINED_LICENSE_SHA256="5de36594c10839788a8c589443a8ef9d8b8d17c65a1b5807206ae037fc36c6bd"
export BLOOM_TRAINED_SEMANTIC_SYSTEM="Follow the user instruction exactly."
export BLOOM_TRAINED_SEMANTIC_PROMPT="Reply with exactly Bloom and no punctuation."
export BLOOM_TRAINED_SEMANTIC_EXPECTED="Bloom"
export BLOOM_TRAINED_MAX_TOKENS="16"
export BLOOM_TRAINED_EXPECTED_DTYPE="Q8"
export BLOOM_TRAINED_CHAT_PROMPT='<|im_start|>system
Follow the user instruction exactly.<|im_end|>
<|im_start|>user
Reply with exactly Bloom and no punctuation.<|im_end|>
<|im_start|>assistant
<think>

</think>

'

exec "${SCRIPT_DIR}/test_trained_gguf_runtime.sh" "$@"
