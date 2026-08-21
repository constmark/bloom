#!/usr/bin/env bash
# Fail closed unless the Dioxus CLI matches Bloom's locked UI dependencies.

set -euo pipefail

REQUIRED_DX_VERSION="0.7.10"

if ! command -v dx >/dev/null 2>&1; then
    echo "Error: dx is required to build the Bloom UI." >&2
    echo "Install it with: cargo install dioxus-cli --version ${REQUIRED_DX_VERSION} --locked" >&2
    exit 1
fi

if ! DX_VERSION_OUTPUT=$(dx --version 2>&1); then
    echo "Error: dx --version failed; refusing to build release UI assets." >&2
    exit 1
fi
read -r DX_PRODUCT DX_VERSION _ <<< "$DX_VERSION_OUTPUT"
if [ "$DX_PRODUCT" != "dioxus" ] ||
    [[ ! "$DX_VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    echo "Error: could not parse the Dioxus CLI version; refusing to build release UI assets." >&2
    exit 1
fi
if [ "$DX_VERSION" != "$REQUIRED_DX_VERSION" ]; then
    echo "Error: Dioxus CLI ${REQUIRED_DX_VERSION} is required, but ${DX_VERSION} is installed." >&2
    echo "Install it with: cargo install dioxus-cli --version ${REQUIRED_DX_VERSION} --locked" >&2
    exit 1
fi

echo "OK: Dioxus CLI ${DX_VERSION} matches the locked Bloom UI dependencies"
