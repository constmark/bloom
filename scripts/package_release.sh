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
#
# Environment:
#   BLOOM_PACKAGE_UI=0  Build a server-only archive instead of the default embedded-UI package.
#   SOURCE_DATE_EPOCH   Optional archive timestamp; defaults to the current Git commit time.

set -euo pipefail

TARGET="${1:-}"
WORKSPACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RELEASE_DIR="${WORKSPACE_DIR}/release-artifacts"

echo "=== Bloom Release Packaging ==="
echo "Workspace: $WORKSPACE_DIR"
echo "Artifacts destination: $RELEASE_DIR"
cd "$WORKSPACE_DIR"

# Clean and create release directory
rm -rf "$RELEASE_DIR"
mkdir -p "$RELEASE_DIR"

# Check build and packaging dependencies.
if ! command -v cargo >/dev/null 2>&1; then
    echo "Error: cargo is not installed or not in PATH." >&2
    exit 1
fi
if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN=$(command -v python3)
elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN=$(command -v python)
else
    echo "Error: Python 3 is required for release packaging and validation." >&2
    exit 1
fi
if ! "$PYTHON_BIN" -c 'import sys; raise SystemExit(sys.version_info < (3, 10))'; then
    echo "Error: Python 3.10 or newer is required for release packaging." >&2
    exit 1
fi

if [ -n "${SOURCE_DATE_EPOCH:-}" ]; then
    RELEASE_SOURCE_DATE_EPOCH="$SOURCE_DATE_EPOCH"
elif git -C "$WORKSPACE_DIR" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    RELEASE_SOURCE_DATE_EPOCH=$(git -C "$WORKSPACE_DIR" log -1 --format=%ct)
else
    echo "Error: SOURCE_DATE_EPOCH is required outside a Git checkout." >&2
    exit 1
fi

HOST_TARGET=$(rustc -vV | awk '/^host:/ {print $2}' | tr -d '\r')
EFFECTIVE_TARGET="${TARGET:-$HOST_TARGET}"
PACKAGE_UI="${BLOOM_PACKAGE_UI:-1}"
case "$PACKAGE_UI" in
    1|true|TRUE|yes|YES)
        PACKAGE_UI=true
        ;;
    0|false|FALSE|no|NO)
        PACKAGE_UI=false
        ;;
    *)
        echo "Error: BLOOM_PACKAGE_UI must be 1/true or 0/false." >&2
        exit 1
        ;;
esac

if [ "$PACKAGE_UI" = true ]; then
    if ! command -v dx >/dev/null 2>&1; then
        echo "Error: dx is required for the default embedded-UI release package." >&2
        echo "Install it with: cargo install dioxus-cli --version 0.7.10 --locked" >&2
        exit 1
    fi
    echo "Building embedded Bloom UI..."
    "${WORKSPACE_DIR}/scripts/build_ui.sh"
fi

# Build arguments
CARGO_ARGS=("--locked" "--release" "--workspace" "--bin" "bloom_infer" "--bin" "bloom_server" "--bin" "bloom_bench" "--bin" "inspect_gguf")

if [ "$PACKAGE_UI" = true ]; then
    CARGO_ARGS+=("--features" "serve-ui")
fi

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
if [[ "$EFFECTIVE_TARGET" == *windows* ]]; then
    IS_WINDOWS=true
fi

# Create a temporary staging directory
STAGE_DIR=$(mktemp -d)
trap 'rm -rf "$STAGE_DIR"' EXIT

STAGING_NAME="bloom"
if [ -n "$TARGET" ]; then
    STAGING_NAME="bloom-${TARGET}"
else
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
            echo "Error: ${bin}.exe not found in release directory." >&2
            exit 1
        fi
    else
        if [ -f "${BIN_SRC_DIR}/${bin}" ]; then
            cp "${BIN_SRC_DIR}/${bin}" "$PKG_DIR/"
            chmod +x "${PKG_DIR}/${bin}"
        else
            echo "Error: ${bin} not found in release directory." >&2
            exit 1
        fi
    fi
done

# Run the staged native server without binding a port or loading a model. This
# proves the packaged executable starts and its machine-readable doctor contract
# remains valid. Cross-target archives cannot be executed on the build host.
SELF_CHECK_STATUS="not_run_cross_target"
DOCTOR_REPORT=""
if [ -z "$TARGET" ] || [ "$TARGET" = "$HOST_TARGET" ]; then
    echo "Self-checking staged bloom_server..."
    DOCTOR_HOME="${STAGE_DIR}/doctor-home"
    DOCTOR_REPORT="${STAGE_DIR}/doctor-report.json"
    mkdir -p "$DOCTOR_HOME"
    DOCTOR_CONFIG_HOME="$DOCTOR_HOME"
    if [ "$IS_WINDOWS" = true ] && command -v cygpath >/dev/null 2>&1; then
        DOCTOR_CONFIG_HOME=$(cygpath -w "$DOCTOR_HOME")
    fi
    if [ "$IS_WINDOWS" = true ]; then
        SERVER_BIN="${PKG_DIR}/bloom_server.exe"
    else
        SERVER_BIN="${PKG_DIR}/bloom_server"
    fi
    BLOOM_CONFIG_HOME="$DOCTOR_CONFIG_HOME" "$SERVER_BIN" --doctor=json > "$DOCTOR_REPORT"
    "$PYTHON_BIN" - "$DOCTOR_REPORT" "$PACKAGE_UI" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    report = json.load(handle)
if report.get("schema_version") != 1 or report.get("object") != "bloom.server_doctor":
    raise SystemExit("invalid bloom_server doctor identity")
if report.get("summary", {}).get("failures") != 0:
    raise SystemExit("packaged bloom_server doctor reported a failure")
checks = {check.get("id"): check for check in report.get("checks", [])}
if sys.argv[2] == "true" and checks.get("embedded_ui", {}).get("status") != "pass":
    raise SystemExit("packaged bloom_server does not contain the required embedded UI")
PY
    HTTP_TEST_SERVER_BIN="$SERVER_BIN"
    if [ "$IS_WINDOWS" = true ] && command -v cygpath >/dev/null 2>&1; then
        HTTP_TEST_SERVER_BIN=$(cygpath -w "$SERVER_BIN")
    fi
    echo "Testing staged bloom_server HTTP boundary..."
    BLOOM_TEST_SERVER_BINARY="$HTTP_TEST_SERVER_BIN" \
        BLOOM_EXPECT_EMBEDDED_UI="$PACKAGE_UI" \
        "$PYTHON_BIN" "${WORKSPACE_DIR}/scripts/test_server_http_boundary.py"
    if [ "$IS_WINDOWS" = false ]; then
        echo "Testing staged bloom_server shutdown lifecycle..."
        BLOOM_TEST_SERVER_BINARY="$SERVER_BIN" \
            "$PYTHON_BIN" "${WORKSPACE_DIR}/scripts/test_server_shutdown.py"
    fi
    SELF_CHECK_STATUS="passed"
else
    echo "Skipping executable self-check for cross-target archive ${TARGET}."
fi

# Copy docs and license
echo "Staging license and documentation..."
for file in "README.md" "LICENSE" "RELEASE.md" "SECURITY.md"; do
    if [ -f "${WORKSPACE_DIR}/${file}" ]; then
        cp "${WORKSPACE_DIR}/${file}" "$PKG_DIR/"
    fi
done
cp -R "${WORKSPACE_DIR}/docs" "$PKG_DIR/docs"
cp -R "${WORKSPACE_DIR}/examples" "$PKG_DIR/examples"
if [ "$PACKAGE_UI" = true ]; then
    cp "${WORKSPACE_DIR}/docs/release-quickstart.md" "$PKG_DIR/QUICKSTART.md"
else
    cp "${WORKSPACE_DIR}/docs/release-server-quickstart.md" "$PKG_DIR/QUICKSTART.md"
fi

# Publish machine-readable target, feature, self-check, size, and checksum
# metadata inside the archive. The outer archive checksum remains authoritative
# for transport integrity; these hashes make extracted binaries auditable.
echo "Writing release manifest..."
CARGO_METADATA="${STAGE_DIR}/cargo-metadata.json"
cargo metadata --locked --no-deps --format-version 1 > "$CARGO_METADATA"
"$PYTHON_BIN" - "$PKG_DIR" "$CARGO_METADATA" "$EFFECTIVE_TARGET" "$PACKAGE_UI" "$SELF_CHECK_STATUS" "$DOCTOR_REPORT" <<'PY'
import hashlib
import json
import pathlib
import sys

package_root = pathlib.Path(sys.argv[1])
with open(sys.argv[2], encoding="utf-8") as handle:
    cargo_metadata = json.load(handle)
version = next(
    package["version"]
    for package in cargo_metadata["packages"]
    if package["name"] == "bloomai-engine"
)
target = sys.argv[3]
embedded_ui = sys.argv[4] == "true"
self_check_status = sys.argv[5]
doctor_path = pathlib.Path(sys.argv[6]) if sys.argv[6] else None

doctor_status = None
failures = None
if self_check_status == "passed":
    if doctor_path is None:
        raise SystemExit("native package self-check report is missing")
    with doctor_path.open(encoding="utf-8") as handle:
        doctor = json.load(handle)
    if doctor.get("schema_version") != 1 or doctor.get("object") != "bloom.server_doctor":
        raise SystemExit("native package self-check has an unsupported identity")
    if doctor.get("bloom_version") != version:
        raise SystemExit("native package version does not match cargo metadata")
    doctor_status = doctor["status"]
    failures = doctor["summary"]["failures"]
    if failures != 0:
        raise SystemExit("native package self-check contains failures")
    checks = {check["id"]: check["status"] for check in doctor["checks"]}
    if (checks.get("embedded_ui") == "pass") != embedded_ui:
        raise SystemExit("native package embedded-UI report does not match the build mode")

binary_names = ["bloom_bench", "bloom_infer", "bloom_server", "inspect_gguf"]
binaries = []
for name in binary_names:
    path = package_root / (f"{name}.exe" if (package_root / f"{name}.exe").is_file() else name)
    if not path.is_file():
        raise SystemExit(f"release manifest binary is missing: {name}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    binaries.append(
        {
            "name": path.name,
            "size_bytes": path.stat().st_size,
            "sha256": digest.hexdigest(),
        }
    )

manifest = {
    "schema_version": 1,
    "object": "bloom.release",
    "bloom_version": version,
    "target": target,
    "embedded_ui": embedded_ui,
    "self_check": {
        "status": self_check_status,
        "doctor_status": doctor_status,
        "failures": failures,
    },
    "binaries": sorted(binaries, key=lambda binary: binary["name"]),
}
with (package_root / "BLOOM-RELEASE.json").open("w", encoding="utf-8") as handle:
    json.dump(manifest, handle, indent=2)
    handle.write("\n")
PY

# Create archives
cd "$STAGE_DIR"
if [ "$IS_WINDOWS" = true ]; then
    ARCHIVE_FILE="${STAGING_NAME}.zip"
    echo "Creating zip archive: ${RELEASE_DIR}/${ARCHIVE_FILE}..."
else
    ARCHIVE_FILE="${STAGING_NAME}.tar.gz"
    echo "Creating tar.gz archive: ${RELEASE_DIR}/${ARCHIVE_FILE}..."
fi
"$PYTHON_BIN" "${WORKSPACE_DIR}/scripts/create_release_archive.py" \
    "$PKG_DIR" "${RELEASE_DIR}/${ARCHIVE_FILE}" \
    --source-date-epoch "$RELEASE_SOURCE_DATE_EPOCH"

# Generate Checksums
cd "$RELEASE_DIR"
echo "Generating SHA-256 checksums..."
"$PYTHON_BIN" - "$ARCHIVE_FILE" SHA256SUMS <<'PY'
import hashlib
import pathlib
import sys

archive_path = pathlib.Path(sys.argv[1])
digest = hashlib.sha256()
with archive_path.open("rb") as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(chunk)
pathlib.Path(sys.argv[2]).write_text(
    f"{digest.hexdigest()}  {archive_path.name}\n", encoding="utf-8"
)
PY

VALIDATOR_ARGS=(
    "${RELEASE_DIR}/${ARCHIVE_FILE}"
    "--checksum"
    "${RELEASE_DIR}/SHA256SUMS"
    "--require-deterministic-metadata"
)
if [ "$PACKAGE_UI" = true ]; then
    VALIDATOR_ARGS+=("--require-embedded-ui")
fi
if [ "$SELF_CHECK_STATUS" = "passed" ]; then
    VALIDATOR_ARGS+=("--require-native-self-check")
fi
"$PYTHON_BIN" "${WORKSPACE_DIR}/scripts/validate_release_artifact.py" "${VALIDATOR_ARGS[@]}"

echo "=== Release Packaging Successful ==="
echo "Artifacts location:"
ls -lh "$RELEASE_DIR"
cat "$RELEASE_DIR/SHA256SUMS"
