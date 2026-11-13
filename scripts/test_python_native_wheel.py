#!/usr/bin/env python3
"""Verify a bundled native wheel and run it in a clean, offline installation."""
from __future__ import annotations

import argparse
import base64
import csv
import email
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import venv
import zipfile


def verify_archive(path: Path) -> dict:
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(set(names)) != len(names):
            raise ValueError("duplicate wheel members")
        for name in names:
            if name.startswith("/") or ".." in name.split("/") or "\\" in name:
                raise ValueError("unsafe wheel member")
        wheel_names = [name for name in names if name.endswith(".dist-info/WHEEL")]
        if len(wheel_names) != 1:
            raise ValueError("expected one WHEEL metadata document")
        directory = wheel_names[0].rsplit("/", 1)[0]
        metadata = email.message_from_bytes(archive.read(wheel_names[0]))
        manifest = json.loads(archive.read(directory + "/native-library.json"))
        expected_tag = f"py3-none-{manifest['platform_tag']}"
        if metadata.get("Root-Is-Purelib") != "false" or metadata.get_all("Tag") != [expected_tag]:
            raise ValueError("native wheel purity or compatibility tag mismatch")
        if not path.name.endswith(f"-{expected_tag}.whl") or manifest["platform_tag"] == "any":
            raise ValueError("native wheel filename tag mismatch")
        library = "bloom_sdk/" + manifest["filename"]
        native_names = [name for name in names if name.endswith((".so", ".dylib", ".dll"))]
        if native_names != [library] or manifest["abi_revision"] != 2:
            raise ValueError("native library inventory or ABI mismatch")
        if hashlib.sha256(archive.read(library)).hexdigest() != manifest["sha256"]:
            raise ValueError("native library hash mismatch")
        inventory = archive.read(directory + "/native-sbom.cdx.json")
        if hashlib.sha256(inventory).hexdigest() != manifest["sbom_sha256"] or json.loads(inventory).get("bomFormat") != "CycloneDX":
            raise ValueError("native dependency inventory mismatch")
        record_name = directory + "/RECORD"
        rows = list(csv.reader(io.StringIO(archive.read(record_name).decode())))
        if {row[0] for row in rows} != set(names) or len(rows) != len(names):
            raise ValueError("incomplete RECORD inventory")
        for name, digest, size in rows:
            if name == record_name:
                if digest or size:
                    raise ValueError("RECORD must not hash itself")
                continue
            data = archive.read(name)
            expected = "sha256=" + base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
            if digest != expected or size != str(len(data)):
                raise ValueError(f"RECORD hash/size mismatch: {name}")
        return manifest


SMOKE = r'''
import ctypes
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import bloom_sdk
import bloom_sdk.pipeline as module
package = Path(bloom_sdk.__file__).resolve().parent
assert Path(sys.prefix).resolve() in package.parents, package
manifest = json.loads(importlib.metadata.distribution('bloom-sdk').read_text('native-library.json'))
lib = module._find_lib()
assert Path(lib._name).resolve().parent == package, lib._name
assert hashlib.sha256(Path(lib._name).read_bytes()).hexdigest() == manifest['sha256']
lib.bloom_abi_version.restype = ctypes.c_uint32
assert lib.bloom_abi_version() == 2
with bloom_sdk.BloomPipeline('.', engine='mock') as pipeline:
    assert pipeline.generate('native wheel')['text'] == 'echo: native wheel'
    chunks = list(pipeline.generate_stream('native wheel'))
    assert chunks[0] == {'TextDelta': 'echo: native wheel'}
    assert chunks[-1] == 'End'
    try:
        pipeline.tokenize('text')
    except bloom_sdk.BloomInferenceError as error:
        assert 'tokenizer' in str(error).lower()
    else:
        raise AssertionError('mock must not invent token IDs')
if os.environ.get('BLOOM_SDK_TEST_MODEL'):
    with bloom_sdk.BloomPipeline(os.environ['BLOOM_SDK_TEST_MODEL'], device='cpu') as pipeline:
        original = 'Hello, Bloom! 你好 🌸'
        assert pipeline.detokenize(pipeline.tokenize(original)) == original
        output = pipeline.generate('Hello', max_tokens=4, temperature=0, seed=42)
        assert isinstance(output.get('text'), str) and output['text']
    print('PASS: bundled native model load, multilingual tokenizer and generation')
print('PASS: isolated bundled ABI, buffered/streamed inference, exact-tokenizer rejection')
'''


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    parser.add_argument("--model", type=Path, help="optional existing text model for native execution; no downloads")
    args = parser.parse_args()
    wheel = args.wheel.resolve()
    manifest = verify_archive(wheel)
    with tempfile.TemporaryDirectory(prefix="bloom-native-install-") as temporary:
        work = Path(temporary)
        environment = work / "venv"
        venv.EnvBuilder(with_pip=True).create(environment)
        python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        env = os.environ.copy()
        for name in ("BLOOM_FFI_LIB", "BLOOM_SDK_TEST_MODEL", "PYTHONPATH", "PYTHONHOME"):
            env.pop(name, None)
        if args.model:
            env["BLOOM_SDK_TEST_MODEL"] = str(args.model.resolve())
        subprocess.run([str(python), "-I", "-m", "pip", "install", "--no-index", "--no-deps", str(wheel)], cwd=work, env=env, check=True, timeout=180)
        subprocess.run([str(python), "-I", "-c", SMOKE], cwd=work, env=env, check=True, timeout=60)
    print(f"PASS: {wheel.name} ({manifest['platform_tag']}, ABI 2)")


if __name__ == "__main__":
    main()
