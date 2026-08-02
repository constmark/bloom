#!/usr/bin/env bash
# Run Bloom's immutable official Qwen2 trained-model CPU acceptance profile.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

export BLOOM_TRAINED_MODEL_PROFILE="qwen2"
export BLOOM_TRAINED_COMMAND_NAME="$0"
export BLOOM_TRAINED_MODEL_REPOSITORY="Qwen/Qwen2-0.5B-Instruct-GGUF"
export BLOOM_TRAINED_MODEL_REVISION="198f08841147e5196a6a69bd0053690fb1fd3857"
export BLOOM_TRAINED_MODEL_FILENAME="qwen2-0_5b-instruct-q4_0.gguf"
export BLOOM_TRAINED_MODEL_SIZE_BYTES="352969408"
export BLOOM_TRAINED_MODEL_SHA256="aca679832ded61145239ce7f5c5ebddb1c57ada786c9c23733899c3888e0596f"
export BLOOM_TRAINED_MODEL_LICENSE="Apache-2.0"
export BLOOM_TRAINED_LICENSE_SHA256="c156170b718ec29139d3653d40ed1986fd92fb7e0959b5c71f3c48f62e6636f4"
export BLOOM_TRAINED_SEMANTIC_SYSTEM="Follow the user instruction exactly."
export BLOOM_TRAINED_SEMANTIC_PROMPT="Reply with only the word Bloom."
export BLOOM_TRAINED_SEMANTIC_EXPECTED="Bloom"
export BLOOM_TRAINED_MAX_TOKENS="6"
export BLOOM_TRAINED_EXPECTED_DTYPE="Q4"
export BLOOM_TRAINED_CHAT_PROMPT='<|im_start|>system
Follow the user instruction exactly.<|im_end|>
<|im_start|>user
Reply with only the word Bloom.<|im_end|>
<|im_start|>assistant
'

exec "${SCRIPT_DIR}/test_trained_gguf_runtime.sh" "$@"
