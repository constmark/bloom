#!/usr/bin/env python3
"""Validate Bloom's immutable and least-privilege Dockerfile baseline."""

from __future__ import annotations

import argparse
import pathlib
import re
import sys


DIGEST = r"sha256:[0-9a-f]{64}"
SYNTAX = re.compile(rf"^# syntax=[^\s@]+@{DIGEST}$")
FROM = re.compile(
    rf"^FROM\s+([^\s@]+)@({DIGEST})\s+AS\s+([A-Za-z0-9_-]+)$",
    re.IGNORECASE,
)
NUMERIC_USER = re.compile(r"^USER\s+([0-9]+):([0-9]+)$", re.IGNORECASE)
UI_TOOLCHAIN_SCRIPT = "scripts/install_ui_toolchain_linux.sh"


def validate_text(content: str) -> list[str]:
    errors: list[str] = []
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    if not lines or SYNTAX.fullmatch(lines[0]) is None:
        errors.append("Dockerfile frontend must be pinned to an immutable SHA-256 digest")

    stages: dict[str, str] = {}
    for line in lines:
        if not line.upper().startswith("FROM "):
            continue
        match = FROM.fullmatch(line)
        if match is None:
            errors.append(f"base image is not tag-and-digest pinned: {line}")
            continue
        image, _digest, stage = match.groups()
        stages[stage.lower()] = image
    for required in ("builder", "runtime"):
        if required not in stages:
            errors.append(f"Dockerfile lacks a digest-pinned {required} stage")

    docker_text = "\n".join(lines)
    for required in (
        f"COPY {UI_TOOLCHAIN_SCRIPT}",
        "cargo install dioxus-cli --version 0.7.10 --locked --features no-downloads",
        "rustup toolchain link bloom-container",
        "RUSTUP_TOOLCHAIN=bloom-container",
        "NO_DOWNLOADS=1",
    ):
        if required not in docker_text:
            errors.append(f"builder lacks fail-closed UI toolchain setting: {required}")

    for required in (
        'cargo metadata --locked --format-version 1 --features serve-ui',
        '--filter-platform "$native_target"',
        '--filter-platform wasm32-unknown-unknown',
        'python3 scripts/generate_sbom.py',
        '--policy config/dependency-policy.json',
        'COPY --from=builder /src/BLOOM-SBOM.cdx.json /usr/share/bloom/BLOOM-SBOM.cdx.json',
        'COPY --from=builder /src/config/dependency-policy.json /usr/share/bloom/BLOOM-DEPENDENCY-POLICY.json',
    ):
        if required not in docker_text:
            errors.append(f"container lacks policy-checked target SBOM: {required}")

    runtime_start = next(
        (
            index
            for index, line in enumerate(lines)
            if (match := FROM.fullmatch(line)) is not None
            and match.group(3).lower() == "runtime"
        ),
        None,
    )
    runtime = lines[runtime_start + 1 :] if runtime_start is not None else []
    user_lines = [line for line in runtime if line.upper().startswith("USER ")]
    final_user = NUMERIC_USER.fullmatch(user_lines[-1]) if user_lines else None
    if final_user is None:
        errors.append("runtime stage must select a fixed numeric UID and GID")
    elif "0" in final_user.groups() or "ENTRYPOINT" not in "\n".join(runtime):
        errors.append("runtime stage must execute its entrypoint as a non-root user")

    runtime_text = "\n".join(runtime)
    for setting in (
        "BLOOM_CONFIG_HOME=/var/lib/bloom",
        "BLOOM_MODELS_DIR=/var/lib/bloom/models",
        "BLOOM_STRICT_MEMORY_BUDGET=1",
        "BLOOM_STRICT_SECURITY=1",
    ):
        if setting not in runtime_text:
            errors.append(f"runtime stage lacks required setting {setting}")
    return errors


def validate_ui_toolchain_text(content: str) -> list[str]:
    errors: list[str] = []
    required_settings = (
        'BINARYEN_VERSION="127"',
        'BINARYEN_AMD64_SHA256="c8ebe5d00c978601086bcad33b2c80fcfe33d6a8b87b754ba4ea86a9a16cc316"',
        'BINARYEN_ARM64_SHA256="1589778bcedde5ba5ed6b7107f902c4ec6bc4c94147daa79157d778ca08300a2"',
        'WASM_BINDGEN_VERSION="0.2.129"',
        'WASM_BINDGEN_AMD64_SHA256="82d12bb940e2d4e72e0d5605387fc1b8ca179044e012b620f0ce4e7440e8320e"',
        'WASM_BINDGEN_ARM64_SHA256="1797c349a0f45d30946e8986b184135e12839ab576c9a04203e85febfc5dcb35"',
        'ESBUILD_VERSION="0.27.3"',
        'ESBUILD_AMD64_SHA256="066e20cdb882994160e18524a552b97e03648eb9aa0c7cdf5680a6493be65ab2"',
        'ESBUILD_ARM64_SHA256="04f0bfb132b8b0800c23b22caa9ad7a7adf41e2434c027fc8571318b9904712f"',
        "sha256sum --check --status",
    )
    for setting in required_settings:
        if setting not in content:
            errors.append(f"UI toolchain installer lacks pinned setting: {setting}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dockerfile",
        type=pathlib.Path,
        default=pathlib.Path(__file__).resolve().parents[1] / "Dockerfile",
        help="Dockerfile to validate.",
    )
    parser.add_argument(
        "--ui-toolchain-script",
        type=pathlib.Path,
        default=pathlib.Path(__file__).resolve().parents[1] / UI_TOOLCHAIN_SCRIPT,
        help="Checksummed Linux UI toolchain installer to validate.",
    )
    args = parser.parse_args()
    try:
        content = args.dockerfile.read_text(encoding="utf-8")
        toolchain_content = args.ui_toolchain_script.read_text(encoding="utf-8")
    except OSError as error:
        print(f"ERROR: cannot read Docker build input: {error}", file=sys.stderr)
        return 1
    errors = validate_text(content) + validate_ui_toolchain_text(toolchain_content)
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print("OK: Dockerfile uses immutable bases and hardened runtime defaults")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
