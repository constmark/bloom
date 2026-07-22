#!/usr/bin/env bash
# Automated Release Packaging Script for Bloom
# Builds release binaries and packages them in tar.gz/zip format with SHA-256 checksums.
#
# Usage:
#   ./scripts/package_release.sh [target-triple]
#
# Examples:
#   ./scripts/package_release.sh
#   ./scripts/package_release.sh x86_64-unknown-linux-gnu

set -euo pipefail

TARGET="${1:-}"
WORKSPACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RELEASE_DIR="${WORKSPACE_DIR}/release-artifacts"

echo "=== Bloom Release Packaging ==="
echo "Workspace: $WORKSPACE_DIR"
echo "Artifacts destination: $RELEASE_DIR"

# Clean and create release directory
rm -rf "$RELEASE_DIR"
mkdir -p "$RELEASE_DIR"

# Check rustc / cargo
if ! command -v cargo >/dev/null 2>&1; then
    echo "Error: cargo is not installed or not in PATH." >&2
    exit 1
fi

# Build arguments
CARGO_ARGS=("--release" "--workspace" "--bin" "bloom_infer" "--bin" "bloom_server" "--bin" "bloom_bench" "--bin" "inspect_gguf")

if [ -n "$TARGET" ]; then
    echo "Target triple specified: $TARGET"
    CARGO_ARGS+=("--target" "$TARGET")
fi

echo "Building release binaries..."
cargo build "${CARGO_ARGS[@]}"

# Determine where the compiled binaries are located
if [ -n "$TARGET" ]; then
    BIN_SRC_DIR="${WORKSPACE_DIR}/target/${TARGET}/release"
else
    BIN_SRC_DIR="${WORKSPACE_DIR}/target/release"
fi

# Determine OS suffix / extension
IS_WINDOWS=false
if [[ "$TARGET" == *windows* ]] || [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
    IS_WINDOWS=true
fi

# Create a temporary staging directory
STAGE_DIR=$(mktemp -d)
trap 'rm -rf "$STAGE_DIR"' EXIT

STAGING_NAME="bloom"
if [ -n "$TARGET" ]; then
    STAGING_NAME="bloom-${TARGET}"
else
    # Auto-detect target for staging name
    HOST_TARGET=$(rustc -vV | grep host | cut -d' ' -f2)
    STAGING_NAME="bloom-${HOST_TARGET}"
fi

PKG_DIR="${STAGE_DIR}/${STAGING_NAME}"
mkdir -p "$PKG_DIR"

# Copy binaries to staging directory
echo "Staging binaries from $BIN_SRC_DIR..."
BINARIES=("bloom_infer" "bloom_server" "bloom_bench" "inspect_gguf")
for bin in "${BINARIES[@]}"; do
    if [ "$IS_WINDOWS" = true ]; then
        if [ -f "${BIN_SRC_DIR}/${bin}.exe" ]; then
            cp "${BIN_SRC_DIR}/${bin}.exe" "$PKG_DIR/"
        else
            echo "Warning: ${bin}.exe not found in release directory."
        fi
    else
        if [ -f "${BIN_SRC_DIR}/${bin}" ]; then
            cp "${BIN_SRC_DIR}/${bin}" "$PKG_DIR/"
            chmod +x "${PKG_DIR}/${bin}"
        else
            echo "Warning: ${bin} not found in release directory."
        fi
    fi
done

# Copy docs and license
echo "Staging license and documentation..."
for file in "README.md" "LICENSE" "RELEASE.md" "SECURITY.md"; do
    if [ -f "${WORKSPACE_DIR}/${file}" ]; then
        cp "${WORKSPACE_DIR}/${file}" "$PKG_DIR/"
    fi
done

# Create archives
cd "$STAGE_DIR"
if [ "$IS_WINDOWS" = true ]; then
    ARCHIVE_FILE="${STAGING_NAME}.zip"
    echo "Creating zip archive: ${RELEASE_DIR}/${ARCHIVE_FILE}..."
    if command -v zip >/dev/null 2>&1; then
        zip -r "${RELEASE_DIR}/${ARCHIVE_FILE}" "$STAGING_NAME" >/dev/null
    else
        echo "Error: zip command not found. Cannot package on this system." >&2
        exit 1
    fi
else
    ARCHIVE_FILE="${STAGING_NAME}.tar.gz"
    echo "Creating tar.gz archive: ${RELEASE_DIR}/${ARCHIVE_FILE}..."
    tar -czf "${RELEASE_DIR}/${ARCHIVE_FILE}" "$STAGING_NAME"
fi

# Generate Checksums
cd "$RELEASE_DIR"
echo "Generating SHA-256 checksums..."
if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$ARCHIVE_FILE" > SHA256SUMS
elif command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$ARCHIVE_FILE" > SHA256SUMS
else
    echo "Warning: Could not find sha256sum or shasum tool. Checksums file not generated."
fi

echo "=== Release Packaging Successful ==="
echo "Artifacts location:"
ls -lh "$RELEASE_DIR"
cat "$RELEASE_DIR/SHA256SUMS" || true
