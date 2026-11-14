#!/usr/bin/env python3
"""Build a host-specific ctypes SDK wheel containing a validated native library.

Requires requirements/python-package.txt and a built bloomai-ffi library.
Builds the ordinary sdist/wheel first in a private directory, then adds the
library with py3-none-<host> tags, non-pure placement and a complete RECORD.
This does not repair external dependencies or assert manylinux compatibility.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def library_name() -> str:
    return {"darwin": "libbloom_ffi.dylib", "win32": "bloom_ffi.dll"}.get(sys.platform, "libbloom_ffi.so")


def host_platform_tag() -> str:
    # Use the build host, not packaging.tags' first Linux manylinux tag:
    # no dependency repair / oldest-glibc build has been established here.
    if sys.platform == "darwin":
        # Conservative host OS version avoids understating a Rust library's
        # deployment target using the Python interpreter's older minimum.
        version = platform.mac_ver()[0].split(".")
        major, minor = int(version[0]), int(version[1])
        if major >= 11:
            minor = 0
        return f"macosx_{major}_{minor}_{platform.machine().lower()}"
    return sysconfig.get_platform().lower().replace("-", "_").replace(".", "_")


def bundle_wheel(pure: Path, native: Path, output: Path, abi: int, tag: str, sbom: bytes) -> Path:
    from wheel.wheelfile import WheelFile

    if not pure.name.endswith("-py3-none-any.whl"):
        raise ValueError("expected the pure py3-none-any SDK wheel")
    if tag == "any" or not tag or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789_" for c in tag):
        raise ValueError("native wheel requires a single explicit platform tag")
    if abi != 2:
        raise ValueError("bundled SDK libraries must implement ABI revision 2")
    if native.name != library_name() or not native.is_file() or native.is_symlink():
        raise ValueError("native library must be a regular host-named file")
    inventory = json.loads(sbom)
    if inventory.get("bomFormat") != "CycloneDX" or not isinstance(inventory.get("components"), list):
        raise ValueError("expected a CycloneDX build-workspace dependency inventory")
    filename = pure.name.removesuffix("-py3-none-any.whl") + f"-py3-none-{tag}.whl"
    output.mkdir(parents=True, exist_ok=True)
    destination = output / filename
    checksum_path = output / (filename + ".sha256")
    if destination.exists() or destination.is_symlink() or checksum_path.exists() or checksum_path.is_symlink():
        raise FileExistsError(f"wheel or checksum already exists: {destination}")
    manifest = {
        "schema_version": 1,
        "abi_revision": abi,
        "filename": native.name,
        "sha256": hashlib.sha256(native.read_bytes()).hexdigest(),
        "platform_tag": tag,
        "sbom_sha256": hashlib.sha256(sbom).hexdigest(),
        "sbom_scope": "locked build-workspace dependency graph; may include unlinked packages",
        "dependency_policy": "host-native; no manylinux or external-library repair claim",
    }
    with tempfile.TemporaryDirectory(prefix=".bloom-wheel-", dir=output) as temporary:
        staged = Path(temporary) / filename
        with zipfile.ZipFile(pure) as source, WheelFile(staged, "w") as wheel:
            names = source.namelist()
            metadata = [name for name in names if name.endswith(".dist-info/WHEEL")]
            if len(metadata) != 1 or len(set(names)) != len(names):
                raise ValueError("invalid source wheel metadata")
            dist_info = metadata[0].rsplit("/", 1)[0]
            for name in names:
                if name.startswith("/") or ".." in name.split("/") or "\\" in name:
                    raise ValueError("unsafe source wheel member")
                if name.endswith(("/WHEEL", "/RECORD", "/RECORD.jws", "/RECORD.p7s")):
                    continue
                if name.endswith((".so", ".dll", ".dylib")):
                    raise ValueError("source wheel already contains native code")
                wheel.writestr(name, source.read(name))
            wheel.writestr(dist_info + "/WHEEL", "Wheel-Version: 1.0\nGenerator: bloom-native-wheel\nRoot-Is-Purelib: false\nTag: py3-none-" + tag + "\n")
            wheel.writestr(dist_info + "/native-library.json", json.dumps(manifest, sort_keys=True, indent=2) + "\n")
            wheel.writestr(dist_info + "/native-sbom.cdx.json", sbom)
            wheel.writestr(dist_info + "/licenses/LAYA-NOTICE", (ROOT / "crates/laya/NOTICE").read_bytes())
            wheel.write(native, "bloom_sdk/" + native.name)
        # Hard-link publication is atomic and refuses to replace an existing
        # wheel. Both paths are on the destination filesystem.
        os.link(staged, destination)
    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    with checksum_path.open("x", encoding="utf-8") as handle:
        handle.write(f"{digest}  {filename}\n")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-library", type=Path, default=ROOT / "target/debug" / library_name())
    parser.add_argument("--output-dir", type=Path, default=ROOT / "release-artifacts/python")
    args = parser.parse_args()
    # Inspect the actual host-loadable ABI, not a caller-provided architecture
    # or filename claim. Do not accept cross-built binaries on this path.
    native = args.native_library.absolute()
    if native.is_symlink():
        parser.error("native-library must not be a symlink")
    with tempfile.TemporaryDirectory(prefix="bloom-native-wheel-") as temporary:
        work = Path(temporary)
        snapshot = work / native.name
        shutil.copyfile(native, snapshot)
        # A separate process releases the loaded DLL before TemporaryDirectory
        # cleanup on Windows; ctypes does not unload libraries automatically.
        subprocess.run([sys.executable, str(ROOT / "scripts/check_ffi_abi_compatibility.py"),
                        "--library", str(snapshot), "--min-abi", "2", "--max-abi", "2", "--exercise-inference"],
                       check=True, timeout=60, cwd=work)
        abi = 2
        version = subprocess.check_output(["rustc", "-vV"], text=True, cwd=ROOT, timeout=30)
        target = next(line.removeprefix("host: ") for line in version.splitlines() if line.startswith("host: "))
        metadata = work / "cargo-metadata.json"
        with metadata.open("wb") as handle:
            subprocess.run(["cargo", "metadata", "--locked", "--format-version", "1", "--filter-platform", target],
                           stdout=handle, check=True, cwd=ROOT, timeout=120)
        sbom = work / "native-sbom.cdx.json"
        subprocess.run([sys.executable, str(ROOT / "scripts/generate_sbom.py"),
                        "--metadata", str(metadata), "--policy", str(ROOT / "config/dependency-policy.json"),
                        "--target", target, "--embedded-ui", "false", "--output", str(sbom)],
                       check=True, cwd=ROOT, timeout=60)
        source = work / "python"
        shutil.copytree(ROOT / "python", source, ignore=shutil.ignore_patterns(
            "__pycache__", "*.egg-info", "build", "dist", ".venv*", "*.so", "*.dylib", "*.dll"
        ))
        dist = work / "dist"
        subprocess.run([sys.executable, "-m", "build", "--no-isolation", "--outdir", str(dist), str(source)], check=True, timeout=180, cwd=work)
        wheels = list(dist.glob("*.whl"))
        if len(wheels) != 1:
            raise RuntimeError("expected exactly one source-built SDK wheel")
        destination = bundle_wheel(wheels[0], snapshot, args.output_dir, abi, host_platform_tag(), sbom.read_bytes())
        print(f"Built native SDK wheel: {destination}")


if __name__ == "__main__":
    main()
