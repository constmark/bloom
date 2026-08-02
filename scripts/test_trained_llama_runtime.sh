#!/usr/bin/env bash
# Run Bloom's immutable official SmolLM2 Llama-architecture CPU acceptance profile.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

export BLOOM_TRAINED_MODEL_PROFILE="llama"
export BLOOM_TRAINED_COMMAND_NAME="$0"
export BLOOM_TRAINED_MODEL_REPOSITORY="HuggingFaceTB/SmolLM2-360M-Instruct-GGUF"
export BLOOM_TRAINED_MODEL_REVISION="593b5a2e04c8f3e4ee880263f93e0bd2901ad47f"
export BLOOM_TRAINED_MODEL_FILENAME="smollm2-360m-instruct-q8_0.gguf"
export BLOOM_TRAINED_MODEL_SIZE_BYTES="386404992"
export BLOOM_TRAINED_MODEL_SHA256="48ab3034d0dd401fbc721eb1df3217902fee7dab9078992d66431f09b7750201"
export BLOOM_TRAINED_MODEL_LICENSE="Apache-2.0"
export BLOOM_TRAINED_LICENSE_FILENAME="README.md"
export BLOOM_TRAINED_LICENSE_SHA256="5d65a9f761fb64443b565d247593aa3d4ec0f06633a0e16c0231625cdba6a7cd"
export BLOOM_TRAINED_SEMANTIC_SYSTEM="Follow the user instruction exactly."
export BLOOM_TRAINED_SEMANTIC_PROMPT="Reply with exactly Bloom and no punctuation."
export BLOOM_TRAINED_SEMANTIC_EXPECTED="Bloom"
export BLOOM_TRAINED_MAX_TOKENS="2"
export BLOOM_TRAINED_EXPECTED_DTYPE="Q8"
export BLOOM_TRAINED_CHAT_PROMPT='<|im_start|>system
Follow the user instruction exactly.<|im_end|>
<|im_start|>user
Reply with exactly Bloom and no punctuation.<|im_end|>
<|im_start|>assistant
'

exec "${SCRIPT_DIR}/test_trained_gguf_runtime.sh" "$@"
