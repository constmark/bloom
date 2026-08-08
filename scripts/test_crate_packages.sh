#!/usr/bin/env bash
# Build every publishable workspace crate archive, then compile the exact
# extracted archives against one another through temporary local patches.
#
# `cargo package --workspace` verifies members independently against crates.io.
# That cannot validate coordinated unpublished workspace changes when an older
# crate with the same version is already published. This gate retains locked
# packaging while checking the actual archive contents as one release set.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
WORKSPACE_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
PACKAGE_VERIFY_ROOT="/tmp/bloom-crate-package-release-set-${UID}"
PACKAGE_VERIFY_DIR="${PACKAGE_VERIFY_ROOT}/source"
if [[ -L "$PACKAGE_VERIFY_ROOT" || -L "$PACKAGE_VERIFY_DIR" ]]; then
    echo "Package verification paths must not be symbolic links" >&2
    exit 1
fi
mkdir -p "$PACKAGE_VERIFY_ROOT"
rm -rf "$PACKAGE_VERIFY_DIR"
mkdir "$PACKAGE_VERIFY_DIR"
trap 'rm -rf "$PACKAGE_VERIFY_DIR"' EXIT

cd "$WORKSPACE_ROOT"

WORKSPACE_PACKAGES=()
while IFS= read -r package; do
    WORKSPACE_PACKAGES+=("$package")
done < <(
    cargo metadata --no-deps --format-version 1 | python3 -c '
import json
import sys

metadata = json.load(sys.stdin)
for package in metadata["packages"]:
    if package.get("publish") != []:
        print("{}\t{}".format(package["name"], package["version"]))
'
)

if [[ "${#WORKSPACE_PACKAGES[@]}" -eq 0 ]]; then
    echo "No publishable workspace packages were discovered" >&2
    exit 1
fi

echo "Packaging publishable workspace crates without registry verification..."
cargo package --workspace --allow-dirty --locked --no-verify

PATCH_ARGS=()
PACKAGE_PATHS=()
for package in "${WORKSPACE_PACKAGES[@]}"; do
    IFS=$'\t' read -r package_name package_version <<<"$package"
    archive="${WORKSPACE_ROOT}/target/package/${package_name}-${package_version}.crate"
    if [[ ! -f "$archive" ]]; then
        echo "Expected crate archive was not produced: $archive" >&2
        exit 1
    fi
    tar -xzf "$archive" -C "$PACKAGE_VERIFY_DIR"
    package_path="${PACKAGE_VERIFY_DIR}/${package_name}-${package_version}"
    if [[ ! -f "${package_path}/Cargo.toml" ]]; then
        echo "Crate archive omitted Cargo.toml: $archive" >&2
        exit 1
    fi
    PACKAGE_PATHS+=("$package_path")
    PATCH_ARGS+=(
        --config
        "patch.crates-io.${package_name}.path=\"${package_path}\""
    )
done

echo "Checking exact crate archive contents as one local release set..."
for package_path in "${PACKAGE_PATHS[@]}"; do
    CARGO_INCREMENTAL=0 CARGO_TARGET_DIR="${WORKSPACE_ROOT}/target" cargo check \
        --quiet \
        --manifest-path "${package_path}/Cargo.toml" \
        --all-targets \
        --offline \
        "${PATCH_ARGS[@]}"
done

echo "OK: every publishable crate archive compiles against this workspace release set"
