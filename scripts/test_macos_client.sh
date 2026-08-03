#!/usr/bin/env bash
# Compile and run native parser/readiness checks without an XCTest dependency.

set -euo pipefail

WORKSPACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CLIENT_DIR="${WORKSPACE_DIR}/clients/macos"
CHECK_DIR="${WORKSPACE_DIR}/target/macos-checks"
CHECK_BIN="${CHECK_DIR}/bloom-desktop-checks"

if [ "$(uname -s)" != "Darwin" ]; then
    echo "Error: Bloom Desktop checks can only run on macOS." >&2
    exit 1
fi

mkdir -p "${CHECK_DIR}"
swiftc \
    "${CLIENT_DIR}/Sources/BloomDesktop/APIModels.swift" \
    "${CLIENT_DIR}/Tests/BloomDesktopChecks/main.swift" \
    -o "${CHECK_BIN}"
"${CHECK_BIN}"
