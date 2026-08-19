#!/usr/bin/env bash
# Install the exact auxiliary binaries used by the Dioxus production build.

set -euo pipefail

BINARYEN_VERSION="127"
BINARYEN_AMD64_SHA256="c8ebe5d00c978601086bcad33b2c80fcfe33d6a8b87b754ba4ea86a9a16cc316"
BINARYEN_ARM64_SHA256="1589778bcedde5ba5ed6b7107f902c4ec6bc4c94147daa79157d778ca08300a2"
WASM_BINDGEN_VERSION="0.2.126"
WASM_BINDGEN_AMD64_SHA256="064948d58e2d6c0a745216477a639ba696216d6309aaa902939d1b865b1d869d"
WASM_BINDGEN_ARM64_SHA256="96864b3992ad45536deb59fc62edc5b845376b7d8b4ac670a6bdab4ab8d2c657"
ESBUILD_VERSION="0.27.3"
ESBUILD_AMD64_SHA256="066e20cdb882994160e18524a552b97e03648eb9aa0c7cdf5680a6493be65ab2"
ESBUILD_ARM64_SHA256="04f0bfb132b8b0800c23b22caa9ad7a7adf41e2434c027fc8571318b9904712f"

INSTALL_DIR="${BLOOM_UI_TOOL_INSTALL_DIR:-/usr/local/bin}"
TEMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TEMP_DIR"' EXIT

case "${TARGETARCH:-$(uname -m)}" in
    amd64 | x86_64)
        BINARYEN_ARCH="x86_64"
        BINARYEN_SHA256="$BINARYEN_AMD64_SHA256"
        WASM_BINDGEN_ARCH="x86_64-unknown-linux-musl"
        WASM_BINDGEN_SHA256="$WASM_BINDGEN_AMD64_SHA256"
        ESBUILD_ARCH="x64"
        ESBUILD_SHA256="$ESBUILD_AMD64_SHA256"
        ;;
    arm64 | aarch64)
        BINARYEN_ARCH="aarch64"
        BINARYEN_SHA256="$BINARYEN_ARM64_SHA256"
        WASM_BINDGEN_ARCH="aarch64-unknown-linux-gnu"
        WASM_BINDGEN_SHA256="$WASM_BINDGEN_ARM64_SHA256"
        ESBUILD_ARCH="arm64"
        ESBUILD_SHA256="$ESBUILD_ARM64_SHA256"
        ;;
    *)
        echo "Error: unsupported Linux UI tool architecture: ${TARGETARCH:-$(uname -m)}" >&2
        exit 1
        ;;
esac

download_and_verify() {
    local url="$1"
    local sha256="$2"
    local destination="$3"
    curl --fail --location --silent --show-error \
        --connect-timeout 15 --max-time 300 --retry 5 --retry-all-errors \
        --output "$destination" "$url"
    printf '%s  %s\n' "$sha256" "$destination" | sha256sum --check --status
}

mkdir -p "$INSTALL_DIR"

BINARYEN_ARCHIVE="${TEMP_DIR}/binaryen.tar.gz"
download_and_verify \
    "https://github.com/WebAssembly/binaryen/releases/download/version_${BINARYEN_VERSION}/binaryen-version_${BINARYEN_VERSION}-${BINARYEN_ARCH}-linux.tar.gz" \
    "$BINARYEN_SHA256" "$BINARYEN_ARCHIVE"
tar -xzf "$BINARYEN_ARCHIVE" -C "$TEMP_DIR" \
    "binaryen-version_${BINARYEN_VERSION}/bin/wasm-opt"
install -m 0755 \
    "${TEMP_DIR}/binaryen-version_${BINARYEN_VERSION}/bin/wasm-opt" \
    "${INSTALL_DIR}/wasm-opt"

WASM_BINDGEN_ARCHIVE="${TEMP_DIR}/wasm-bindgen.tar.gz"
download_and_verify \
    "https://github.com/wasm-bindgen/wasm-bindgen/releases/download/${WASM_BINDGEN_VERSION}/wasm-bindgen-${WASM_BINDGEN_VERSION}-${WASM_BINDGEN_ARCH}.tar.gz" \
    "$WASM_BINDGEN_SHA256" "$WASM_BINDGEN_ARCHIVE"
tar -xzf "$WASM_BINDGEN_ARCHIVE" -C "$TEMP_DIR" \
    "wasm-bindgen-${WASM_BINDGEN_VERSION}-${WASM_BINDGEN_ARCH}/wasm-bindgen"
install -m 0755 \
    "${TEMP_DIR}/wasm-bindgen-${WASM_BINDGEN_VERSION}-${WASM_BINDGEN_ARCH}/wasm-bindgen" \
    "${INSTALL_DIR}/wasm-bindgen"

ESBUILD_ARCHIVE="${TEMP_DIR}/esbuild.tgz"
download_and_verify \
    "https://registry.npmjs.org/@esbuild/linux-${ESBUILD_ARCH}/-/linux-${ESBUILD_ARCH}-${ESBUILD_VERSION}.tgz" \
    "$ESBUILD_SHA256" "$ESBUILD_ARCHIVE"
tar -xzf "$ESBUILD_ARCHIVE" -C "$TEMP_DIR" package/bin/esbuild
install -m 0755 "${TEMP_DIR}/package/bin/esbuild" "${INSTALL_DIR}/esbuild"

test "$(wasm-bindgen --version)" = "wasm-bindgen ${WASM_BINDGEN_VERSION}"
test "$(esbuild --version)" = "$ESBUILD_VERSION"
wasm-opt --version | grep -Eq "version[ _]${BINARYEN_VERSION}"

echo "Installed checksummed Bloom UI build tools in ${INSTALL_DIR}."
