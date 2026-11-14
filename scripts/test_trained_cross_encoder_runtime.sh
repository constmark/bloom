#!/usr/bin/env bash
# Verify the immutable MiniLM cross-encoder and execute its native CPU gate.
set -euo pipefail
WORKSPACE_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
exec "${PYTHON:-python3}" "${WORKSPACE_DIR}/scripts/trained_cross_encoder_runtime.py" "$@"
