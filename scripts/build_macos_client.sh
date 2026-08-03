#!/usr/bin/env bash
# Build the native SwiftUI client as an ad-hoc signed macOS application bundle.

set -euo pipefail

WORKSPACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PACKAGE_DIR="${WORKSPACE_DIR}/clients/macos"
OUTPUT_ROOT="${WORKSPACE_DIR}/target/macos"
APP_DIR="${OUTPUT_ROOT}/Bloom Desktop.app"

if [ "$(uname -s)" != "Darwin" ]; then
    echo "Error: Bloom Desktop can only be built on macOS." >&2
    exit 1
fi
if ! command -v swift >/dev/null 2>&1; then
    echo "Error: the Swift toolchain is required." >&2
    exit 1
fi

swift build --package-path "${PACKAGE_DIR}" --configuration release --product BloomDesktop
BIN_DIR="$(swift build --package-path "${PACKAGE_DIR}" --configuration release --show-bin-path)"

case "${APP_DIR}" in
    "${WORKSPACE_DIR}"/target/macos/*.app) ;;
    *)
        echo "Error: refusing to replace an application outside target/macos." >&2
        exit 1
        ;;
esac

rm -rf "${APP_DIR}"
mkdir -p "${APP_DIR}/Contents/MacOS" "${APP_DIR}/Contents/Resources"
install -m 755 "${BIN_DIR}/BloomDesktop" "${APP_DIR}/Contents/MacOS/BloomDesktop"
install -m 644 "${PACKAGE_DIR}/Info.plist" "${APP_DIR}/Contents/Info.plist"
plutil -lint "${APP_DIR}/Contents/Info.plist" >/dev/null

if command -v codesign >/dev/null 2>&1; then
    codesign --force --sign - "${APP_DIR}" >/dev/null
fi

echo "Bloom Desktop is ready at ${APP_DIR}"
