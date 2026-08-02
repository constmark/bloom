#!/usr/bin/env bash
# Run Bloom's immutable official SmolLM2 Safetensors CPU acceptance profile.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

export BLOOM_TRAINED_MODEL_PROFILE="safetensors"
export BLOOM_TRAINED_COMMAND_NAME="$0"
export BLOOM_TRAINED_MODEL_LAYOUT="directory"
export BLOOM_TRAINED_MODEL_REPOSITORY="HuggingFaceTB/SmolLM2-360M-Instruct"
export BLOOM_TRAINED_MODEL_REVISION="a10cc1512eabd3dde888204e902eca88bddb4951"
export BLOOM_TRAINED_MODEL_FILENAME="model.safetensors"
export BLOOM_TRAINED_MODEL_SIZE_BYTES="723674912"
export BLOOM_TRAINED_MODEL_SHA256="e6bffe7435d7ddc10fd3b9a9efd429dafbacb1cb17015fb5562664e7532bf86e"
export BLOOM_TRAINED_ADDITIONAL_FILES='config.json|846|224f72354f10d617a359cc82ad15a3c96e866b9b2ffadb81997eeea9e88e22ee
tokenizer.json|2104556|9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c
tokenizer_config.json|3764|4ec77d44f62efeb38d7e044a1db318f6a939438425312dfa333b8382dbad98df
special_tokens_map.json|655|2b7379f3ae813529281a5c602bc5a11c1d4e0a99107aaa597fe936c1e813ca52'
export BLOOM_TRAINED_MODEL_LICENSE="Apache-2.0"
export BLOOM_TRAINED_LICENSE_FILENAME="README.md"
export BLOOM_TRAINED_LICENSE_SHA256="6b88794416ac9da8f254ebb0bec228967a2bdd0badf9a2853863928b25facd95"
export BLOOM_TRAINED_SEMANTIC_SYSTEM="Follow the user instruction exactly."
export BLOOM_TRAINED_SEMANTIC_PROMPT="Reply with exactly Bloom and no punctuation."
export BLOOM_TRAINED_SEMANTIC_EXPECTED="Bloom"
export BLOOM_TRAINED_MAX_TOKENS="2"
export BLOOM_TRAINED_EXPECTED_FAMILY="Llama"
export BLOOM_TRAINED_EXPECTED_STORAGE_DTYPE="BF16"
export BLOOM_TRAINED_EXPECTED_DTYPE="F32"
export BLOOM_TRAINED_EXPECTED_CHAT_TEMPLATE_KIND="smollm2"
export BLOOM_TRAINED_EXPECTED_RUNTIME_WEIGHT_BYTES="1447349824"
export BLOOM_TRAINED_CHAT_PROMPT='<|im_start|>system
Follow the user instruction exactly.<|im_end|>
<|im_start|>user
Reply with exactly Bloom and no punctuation.<|im_end|>
<|im_start|>assistant
'

exec "${SCRIPT_DIR}/test_trained_gguf_runtime.sh" "$@"
