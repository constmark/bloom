#!/usr/bin/env python3
"""Build an SDK sdist, rebuild its wheel, and exercise a clean installation.

Install requirements/python-package.txt and build bloomai-ffi first. All
packaging and installation happens outside the source tree. No publication.
"""

import argparse
import email
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import venv
import zipfile


ROOT = Path(__file__).resolve().parents[1]


def run(command, *, cwd, env=None):
    subprocess.run(command, cwd=cwd, env=env, check=True, timeout=180)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-library", type=Path)
    args = parser.parse_args()
    name = {"darwin": "libbloom_ffi.dylib", "win32": "bloom_ffi.dll"}.get(
        sys.platform, "libbloom_ffi.so"
    )
    native = (args.native_library or ROOT / "target/debug" / name).resolve()
    if not native.is_file():
        parser.error("native library missing; run cargo build -p bloomai-ffi --locked first")
    with tempfile.TemporaryDirectory(prefix="bloom-python-package-") as directory:
        work = Path(directory)
        source = work / "python"
        shutil.copytree(ROOT / "python", source, ignore=shutil.ignore_patterns(
            "__pycache__", "*.egg-info", "build", "dist", ".venv*"
        ))
        if (source / "LICENSE").read_bytes() != (ROOT / "LICENSE").read_bytes():
            raise AssertionError("SDK license must match the repository license")
        packages = work / "dist"
        # The default build operation creates an sdist and builds the wheel
        # from that sdist, catching missing files masked by a source checkout.
        run([sys.executable, "-m", "build", "--no-isolation", "--outdir", str(packages),
             str(source)], cwd=work)
        sdists = list(packages.glob("*.tar.gz"))
        wheels = list(packages.glob("*.whl"))
        if len(sdists) != 1 or len(wheels) != 1:
            raise AssertionError("expected exactly one SDK sdist and wheel")
        with tarfile.open(sdists[0]) as archive:
            members = {name.split("/", 1)[-1] for name in archive.getnames()}
            required = {"README.md", "LICENSE", "pyproject.toml", "bloom_sdk/_stream.py"}
            if not required <= members:
                raise AssertionError(f"incomplete SDK sdist: {required - members}")
        with zipfile.ZipFile(wheels[0]) as archive:
            names = archive.namelist()
            metadata_name = next(name for name in names if name.endswith(".dist-info/METADATA"))
            metadata = email.message_from_bytes(archive.read(metadata_name))
            if metadata["Name"] != "bloom-sdk" or not metadata.get_payload().strip():
                raise AssertionError("missing SDK metadata or package README")
            license_name = next(name for name in names if name.endswith("/LICENSE"))
            if archive.read(license_name) != (ROOT / "LICENSE").read_bytes():
                raise AssertionError("wheel license does not match")
            if any(name.endswith((".so", ".dylib", ".dll")) for name in names):
                raise AssertionError("pure Python wheel unexpectedly contains a native library")
        environment = work / "installed"
        venv.EnvBuilder(with_pip=True).create(environment)
        python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        run([str(python), "-I", "-m", "pip", "install", "--no-index", "--no-deps",
             str(wheels[0])], cwd=work)
        env = os.environ.copy()
        env["BLOOM_FFI_LIB"] = str(work / "missing-native-library")
        run([str(python), "-I", "-c", "import bloom_sdk"], cwd=work, env=env)
        env["BLOOM_FFI_LIB"] = str(native)
        run([str(python), "-I", "-c", """
from pathlib import Path
import sys
import bloom_sdk
assert Path(sys.prefix) in Path(bloom_sdk.__file__).resolve().parents
with bloom_sdk.BloomPipeline('.', engine='mock') as pipeline:
    assert pipeline._uses_v2
    assert pipeline.generate('installed wheel')['text'] == 'echo: installed wheel'
    chunks = list(pipeline.generate_stream('installed wheel'))
    assert chunks[0] == {'TextDelta': 'echo: installed wheel'}
    assert chunks[-1] == 'End'
"""], cwd=work, env=env)
    print("PASS: SDK sdist -> wheel -> isolated install -> native ABI v2")


if __name__ == "__main__":
    main()
