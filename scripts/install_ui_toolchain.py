#!/usr/bin/env python3
"""Install checksum-pinned Dioxus helpers for official native release builders.

GitHub checksums come from the upstream version_127 / 0.2.129 release assets;
esbuild checksums are SHA-256 of the exact upstream npm 0.27.3 tarballs.
Only the named regular file is copied; archive paths are never extracted.
"""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import tarfile
import tempfile

BINARYEN_VERSION = "127"
WASM_BINDGEN_VERSION = "0.2.129"
ESBUILD_VERSION = "0.27.3"
MAX_BINARY_BYTES = 256 * 1024 * 1024
# target: Binaryen platform, wasm-bindgen triple, npm platform, archive hashes
TARGETS = {
    "linux-x86_64": (
        "x86_64-linux", "x86_64-unknown-linux-musl", "linux-x64",
        "c8ebe5d00c978601086bcad33b2c80fcfe33d6a8b87b754ba4ea86a9a16cc316",
        "82d12bb940e2d4e72e0d5605387fc1b8ca179044e012b620f0ce4e7440e8320e",
        "066e20cdb882994160e18524a552b97e03648eb9aa0c7cdf5680a6493be65ab2",
    ),
    "linux-aarch64": (
        "aarch64-linux", "aarch64-unknown-linux-gnu", "linux-arm64",
        "1589778bcedde5ba5ed6b7107f902c4ec6bc4c94147daa79157d778ca08300a2",
        "1797c349a0f45d30946e8986b184135e12839ab576c9a04203e85febfc5dcb35",
        "04f0bfb132b8b0800c23b22caa9ad7a7adf41e2434c027fc8571318b9904712f",
    ),
    "darwin-x86_64": (
        "x86_64-macos", "x86_64-apple-darwin", "darwin-x64",
        "c814a6aa734f652867de080cf092830de476f2ee2714b3998b14dea7859f3bd7",
        "7e028879a68ec53dae14048b55cc5215f6df75b17a9c1426fac8fa086d3202fd",
        "2b13b0d44be1c9082272e631350820509f2ead61ae77dd36f977949ef49ae9ec",
    ),
    "darwin-aarch64": (
        "arm64-macos", "aarch64-apple-darwin", "darwin-arm64",
        "5b53afbf6b71b3f41e5c896855b41f9f0bcda8d3666e8703d8bd1f65a9b4f767",
        "81d4a23d56b3c3eb8187658329116d50e0b228a93b343825fb71f70179051cd1",
        "a77fecaa4c08a320e6443aeb501932061e499adc93048327d2e64ce2916ed233",
    ),
    "windows-x86_64": (
        "x86_64-windows", "x86_64-pc-windows-msvc", "win32-x64",
        "1f771649683901951531a2753ee95220939841c81bf4331a306c1b3298c02544",
        "79e348c169d0c10c0b287647f1e753d0985353ca96800bc29be8f4261baf13e6",
        "d3f90cb2059c421902580e1b1d8e2b1b0d87195d8f3ecf410cb175d91abb55f1",
    ),
}


def host_target() -> str:
    machine = platform.machine().lower()
    machine = {"amd64": "x86_64", "arm64": "aarch64"}.get(machine, machine)
    target = f"{platform.system().lower()}-{machine}"
    if target not in TARGETS:
        raise ValueError(f"unsupported UI build host: {target}")
    return target


def assets(target: str) -> list[tuple[str, str, str, str]]:
    binaryen, bindgen, npm, binaryen_hash, bindgen_hash, npm_hash = TARGETS[target]
    extension = ".exe" if target.startswith("windows-") else ""
    return [
        (f"wasm-opt{extension}",
         f"https://github.com/WebAssembly/binaryen/releases/download/version_{BINARYEN_VERSION}/binaryen-version_{BINARYEN_VERSION}-{binaryen}.tar.gz",
         binaryen_hash, f"binaryen-version_{BINARYEN_VERSION}/bin/wasm-opt{extension}"),
        (f"wasm-bindgen{extension}",
         f"https://github.com/wasm-bindgen/wasm-bindgen/releases/download/{WASM_BINDGEN_VERSION}/wasm-bindgen-{WASM_BINDGEN_VERSION}-{bindgen}.tar.gz",
         bindgen_hash, f"wasm-bindgen-{WASM_BINDGEN_VERSION}-{bindgen}/wasm-bindgen{extension}"),
        (f"esbuild{extension}",
         f"https://registry.npmjs.org/@esbuild/{npm}/-/{npm}-{ESBUILD_VERSION}.tgz",
         npm_hash, "package/esbuild.exe" if extension else "package/bin/esbuild"),
    ]


def verified_binary(archive: Path, expected_hash: str, member: str, destination: Path) -> None:
    digest = hashlib.sha256()
    with archive.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != expected_hash:
        raise ValueError(f"SHA-256 mismatch for {destination.name}")
    with tarfile.open(archive, "r:gz") as bundle:
        matches = [item for item in bundle if item.name == member]
        if len(matches) != 1:
            raise ValueError(f"archive must contain exactly one {member}")
        info = matches[0]
        if not info.isfile() or not 0 < info.size <= MAX_BINARY_BYTES:
            raise ValueError(f"archive member is not a bounded regular file: {member}")
        source = bundle.extractfile(info)
        if source is None:
            raise ValueError(f"missing file body: {member}")
        with source, destination.open("xb") as output:
            shutil.copyfileobj(source, output, length=1024 * 1024)
        if destination.stat().st_size != info.size:
            raise ValueError(f"truncated archive member: {member}")
    destination.chmod(0o755)


def verify_version(binary: Path) -> None:
    output = subprocess.run([str(binary), "--version"], check=True, capture_output=True,
                            text=True, timeout=15).stdout.strip()
    if binary.stem == "wasm-opt":
        valid = re.fullmatch(r"wasm-opt version[ _]" + BINARYEN_VERSION + r"(?: \([^\r\n]*\))?", output)
    elif binary.stem == "wasm-bindgen":
        valid = output == f"wasm-bindgen {WASM_BINDGEN_VERSION}"
    else:
        valid = output == ESBUILD_VERSION
    if not valid:
        raise ValueError(f"unexpected {binary.name} version: {output!r}")


def install(install_dir: Path) -> None:
    target = host_target()
    selected = assets(target)
    install_dir.mkdir(parents=True, exist_ok=True)
    # Same filesystem enables atomic file replacement. No binary runs until its
    # complete archive has passed the pinned hash and regular-file checks.
    with tempfile.TemporaryDirectory(prefix=".bloom-ui-tools-", dir=install_dir) as temporary:
        stage = Path(temporary)
        (stage / "bin").mkdir()
        (stage / "lib").mkdir()
        for name, url, digest, member in selected:
            archive = stage / f"{name}.tar.gz"
            subprocess.run([
                "curl", "--fail", "--location", "--silent", "--show-error",
                "--proto", "=https", "--proto-redir", "=https",
                "--connect-timeout", "15", "--max-time", "300",
                "--retry", "3", "--retry-all-errors", "--max-filesize", "536870912",
                "--output", str(archive), url,
            ], check=True, timeout=1300)
            binary = stage / "bin" / name
            verified_binary(archive, digest, member, binary)
            # Upstream macOS binaries use @executable_path/../lib. Preserve
            # that layout and verify the dylib from the same pinned archive.
            if name == "wasm-opt" and target.startswith("darwin-"):
                verified_binary(archive, digest,
                                f"binaryen-version_{BINARYEN_VERSION}/lib/libbinaryen.dylib",
                                stage / "lib" / "libbinaryen.dylib")
            verify_version(binary)
        (install_dir / "bin").mkdir(exist_ok=True)
        for library in (stage / "lib").iterdir():
            (install_dir / "lib").mkdir(exist_ok=True)
            os.replace(library, install_dir / "lib" / library.name)
        for name, *_ in selected:
            os.replace(stage / "bin" / name, install_dir / "bin" / name)
    print(f"Installed verified Bloom UI build tools in {install_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--install-dir", type=Path, default=os.environ.get("BLOOM_UI_TOOL_INSTALL_DIR"),
                        help="Private tool prefix; add its bin subdirectory to PATH.")
    args = parser.parse_args()
    if args.install_dir is None:
        parser.error("--install-dir or BLOOM_UI_TOOL_INSTALL_DIR is required")
    try:
        install(args.install_dir.resolve())
    except (OSError, ValueError, tarfile.TarError, subprocess.SubprocessError) as error:
        parser.exit(1, f"ERROR: UI build tool installation failed: {error}\n")


if __name__ == "__main__":
    main()
