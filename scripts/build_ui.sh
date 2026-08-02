#!/usr/bin/env bash
# Build production web assets and fail closed on optimizer or artifact errors.

set -euo pipefail

WORKSPACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UI_DIR="${WORKSPACE_DIR}/ui"
PUBLIC_DIR="${UI_DIR}/target/dx/bloom-ui/release/web/public"
BUILD_LOG="$(mktemp)"
trap 'rm -f "$BUILD_LOG"' EXIT

if ! command -v dx >/dev/null 2>&1; then
    echo "Error: dx is required to build the Bloom UI." >&2
    echo "Install it with: cargo install dioxus-cli --version 0.7.10 --locked" >&2
    exit 1
fi

# Dioxus fingerprints assets but does not remove previous fingerprints. Clean
# only its generated public directory so a release cannot retain stale bundles.
rm -rf "$PUBLIC_DIR"

set +e
(
    cd "$UI_DIR"
    dx build --platform web --release --locked --debug-symbols=false
) 2>&1 | tee "$BUILD_LOG"
DX_STATUS=${PIPESTATUS[0]}
set -e

if [ "$DX_STATUS" -ne 0 ]; then
    echo "Error: the Dioxus production build failed." >&2
    exit "$DX_STATUS"
fi
if grep -Fq "wasm-opt failed" "$BUILD_LOG"; then
    echo "Error: Dioxus continued after wasm-opt failed; refusing unoptimized release assets." >&2
    exit 1
fi

if [ ! -s "${PUBLIC_DIR}/index.html" ]; then
    echo "Error: the production UI entry point is missing or empty." >&2
    exit 1
fi
shopt -s nullglob
JS_BUNDLES=("${PUBLIC_DIR}"/assets/bloom-ui-*.js)
WASM_BUNDLES=("${PUBLIC_DIR}"/assets/bloom-ui_bg-*.wasm)
if [ "${#JS_BUNDLES[@]}" -eq 0 ]; then
    echo "Error: the production UI JavaScript bundle is missing or empty." >&2
    exit 1
fi
if [ "${#WASM_BUNDLES[@]}" -eq 0 ]; then
    echo "Error: the production UI WebAssembly payload is missing or empty." >&2
    exit 1
fi
for file in "${JS_BUNDLES[@]}" "${WASM_BUNDLES[@]}"; do
    if [ ! -s "$file" ]; then
        echo "Error: a production UI bundle is empty." >&2
        exit 1
    fi
done
if find "${PUBLIC_DIR}" -type f -size 0 -print -quit | grep -q .; then
    echo "Error: the production UI contains an empty generated file." >&2
    exit 1
fi

rm -rf "${UI_DIR}/dist"
mkdir -p "${UI_DIR}/dist"
cp -R "${PUBLIC_DIR}/." "${UI_DIR}/dist/"

echo "Bloom UI production assets are ready in ${UI_DIR}/dist."
