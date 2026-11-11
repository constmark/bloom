#!/usr/bin/env python3
"""Physical ENOSPC in a private 64 MiB image and isolated release-switch drills.

No existing mount, arbitrary mountpoint, sudo, or host-volume fallback is accepted.
Release slots are private and never replace a system service.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import errno
import hashlib
import http.client
import json
import os
from pathlib import Path
import platform
import plistlib
import re
import shutil
import subprocess
import tempfile

from deployment_failure_injection import (
    _model_identity, _post_json, _ready, _request, _sha256_file, _start, _stop, _wait_ready,
)

IMAGE_BYTES = 64 * 1024 * 1024
MAX_FILESYSTEM_BYTES = 80 * 1024 * 1024
MIN_HOST_FREE = 512 * 1024 * 1024
KEY = "isolated-storage-drill-operator"
HEADERS = {"X-API-Key": KEY}
MAX_BODY = 256 * 1024


def command(arguments):
    return subprocess.run(arguments, check=True, capture_output=True, timeout=60).stdout


def validate_volume(mount, host):
    if mount.is_symlink() or not mount.is_dir():
        raise RuntimeError("drill mount must be a real directory")
    if mount.stat().st_dev == host.stat().st_dev or not os.path.ismount(mount):
        raise RuntimeError("refusing to fill a directory on the host filesystem")
    info = os.statvfs(mount)
    capacity = info.f_blocks * info.f_frsize
    if not 8 * 1024 * 1024 <= capacity <= MAX_FILESYSTEM_BYTES:
        raise RuntimeError("isolated filesystem exceeds the hard capacity bound")
    return capacity


@contextmanager
def isolated_volume(root):
    if platform.system() != "Darwin":
        raise RuntimeError("physical ENOSPC requires macOS hdiutil; host-volume fallback is forbidden")
    if shutil.disk_usage(root).free < MIN_HOST_FREE:
        raise RuntimeError("insufficient host headroom for the bounded disk image")
    image, mount = root / "bounded-volume.sparseimage", root / "mount"
    mount.mkdir()
    device = None
    command(["hdiutil", "create", "-size", "64m", "-fs", "HFS+", "-type", "SPARSE",
             "-volname", "BloomIsolatedDrill", str(image)])
    try:
        attached = plistlib.loads(command([
            "hdiutil", "attach", "-nobrowse", "-mountpoint", str(mount), "-plist", str(image),
        ]))
        mounted = [item for item in attached.get("system-entities", [])
                   if item.get("mount-point") and Path(item["mount-point"]).resolve() == mount.resolve()]
        if len(mounted) != 1:
            raise RuntimeError("hdiutil did not mount exactly the requested private image")
        candidate = mounted[0].get("dev-entry", "")
        if re.fullmatch(r"/dev/disk[0-9]+(?:s[0-9]+)*", candidate) is None:
            raise RuntimeError("hdiutil returned an unexpected image device")
        device = candidate
        yield mount, validate_volume(mount, root)
    finally:
        # Exactly our image; never enumerate or detach another mounted image.
        target = device if device is not None else str(mount)
        try:
            command(["hdiutil", "detach", target])
        except subprocess.SubprocessError:
            command(["hdiutil", "detach", "-force", target])
        if os.path.ismount(mount):
            raise RuntimeError("image is still mounted; refusing recursive cleanup")


def fill_volume(mount, host):
    capacity = validate_volume(mount, host)
    filler, written = mount / "drill-fill.bin", 0
    with filler.open("xb", buffering=0) as output:
        block = bytes(range(256)) * 4096
        step = len(block)
        while written <= capacity:
            try:
                count = output.write(block[:min(step, capacity + 1 - written)])
                if not count:
                    raise RuntimeError("filler made no progress")
                written += count
                os.fsync(output.fileno())
            except OSError as error:
                if error.errno != errno.ENOSPC:
                    raise
                if step == 1:
                    return filler, written
                # HFS+ can reject a large extent while smaller free extents
                # remain. Exhaust the last blocks before issuing the import.
                step = max(1, step // 16)
    raise RuntimeError("filesystem did not return ENOSPC within the capacity bound")


def request(port, method, path, body=b"", extra=None):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=15)
    try:
        connection.request(method, path, body=body, headers={
            **HEADERS, "Content-Type": "application/octet-stream", **(extra or {}),
        })
        response = connection.getresponse()
        raw = response.read(MAX_BODY + 1)
        if len(raw) > MAX_BODY:
            raise RuntimeError("HTTP response exceeded bound")
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise RuntimeError("response must be a JSON object")
        return response.status, value
    finally:
        connection.close()


def enospc(binary, root):
    content = bytes(range(256)) * 8192
    digest = hashlib.sha256(content).hexdigest()
    filename = "physical-enospc-fixture.gguf"
    declaration = {"filename": filename, "total_bytes": len(content), "sha256": digest}
    with isolated_volume(root) as (mount, capacity):
        models = mount / "models"
        models.mkdir()
        process, port = _start(
            binary, models, root / "disk-full.log", None, "candle", "cpu",
            extra_args=["--enable-model-imports", "--max-model-import-bytes", str(len(content)),
                        "--max-model-import-chunk-bytes", str(1024 * 1024)],
            environment={"BLOOM_OPERATOR_API_KEY": KEY},
        )
        try:
            if not port:
                raise RuntimeError("disk-full drill server did not start")
            status, _ = _post_json(port, "/v1/model-management/imports", declaration, HEADERS)
            if status != 200:
                raise RuntimeError(f"could not create import before exhaustion: HTTP {status}")
            filler, written = fill_volume(mount, root)
            failure_status, payload = request(
                port, "PUT", f"/v1/model-management/imports/{filename}",
                content[:1024 * 1024], {"Upload-Offset": "0"},
            )
            if not 500 <= failure_status <= 599 or not isinstance(payload.get("error"), dict):
                raise RuntimeError(f"full filesystem did not reject import: HTTP {failure_status}")
            if (models / filename).exists():
                raise RuntimeError("disk-full import published an incomplete model")
            if process.poll() is not None or _request(port, "/health", 5)[0] != 200:
                raise RuntimeError("disk exhaustion killed the service")
            filler.unlink()
            status, resumed = _post_json(port, "/v1/model-management/imports", declaration, HEADERS)
            if status != 200 or not isinstance(resumed, dict):
                raise RuntimeError("import did not resume after freeing space")
            offset = resumed.get("status", {}).get("uploaded_bytes")
            if type(offset) is not int or not 0 <= offset < len(content):
                raise RuntimeError("resumed import returned an invalid offset")
            resumed_offset = offset
            while offset < len(content):
                chunk = content[offset:offset + 1024 * 1024]
                status, _ = request(port, "PUT", f"/v1/model-management/imports/{filename}",
                                    chunk, {"Upload-Offset": str(offset)})
                if status != 200:
                    raise RuntimeError(f"import failed after freeing space: HTTP {status}")
                offset += len(chunk)
            status, completed = _post_json(
                port, f"/v1/model-management/imports/{filename}/complete", {}, HEADERS,
            )
            if status != 200 or completed is None or completed.get("installed") is not True:
                raise RuntimeError("recovered import did not complete")
            if _sha256_file(models / filename) != digest:
                raise RuntimeError("recovered bytes differ from verified fixture")
            if any((models / ".bloom-imports").glob(f"{filename}.*")):
                raise RuntimeError("completed import retained staging residue")
            if _stop(process) != 0:
                raise RuntimeError("disk-full recovery did not drain cleanly")
            process, port = _start(binary, models, root / "disk-restart.log", None, "candle", "cpu",
                                   environment={"BLOOM_OPERATOR_API_KEY": KEY})
            if not port or _request(port, "/health", 5)[0] != 200:
                raise RuntimeError("service could not reacquire recovered catalog")
            if _sha256_file(models / filename) != digest:
                raise RuntimeError("model identity changed after restart")
            return {
                "result": "pass", "filesystem": "HFS+ disk image", "capacity_bytes": capacity,
                "image_logical_limit_bytes": IMAGE_BYTES, "filler_bytes": written,
                "observed_errno": "ENOSPC", "import_failure_status": failure_status,
                "incomplete_model_published": False, "health_during_failure": True,
                "resumed_offset": resumed_offset, "fixture_sha256": digest,
                "fixture_bytes": len(content), "verified_after_restart": True,
            }
        finally:
            if process.poll() is None:
                _stop(process)


def activate(root, slot):
    if slot not in {"previous", "candidate"}:
        raise ValueError("unknown owned release slot")
    temporary = root / "next"
    temporary.symlink_to(slot, target_is_directory=True)
    os.replace(temporary, root / "current")


def serving_probe(process, port, log, model_hash):
    status, payload, elapsed = _wait_ready(process, log, 180)
    if not _ready(status, payload):
        raise RuntimeError("activated release did not become ready")
    status, response = _post_json(port, "/v1/completions", {
        "model": payload["model"], "prompt": "Hello", "max_tokens": 2,
        "temperature": 0, "seed": 7,
    }, HEADERS, timeout=60)
    choices = response.get("choices") if isinstance(response, dict) else None
    if status != 200 or not isinstance(choices, list) or not choices or not isinstance(choices[0].get("text"), str):
        raise RuntimeError("activated release failed real-model inference")
    return {"ready_ms": round(elapsed, 3), "generation_status": status,
            "model_sha256": model_hash,
            "output_sha256": hashlib.sha256(choices[0]["text"].encode()).hexdigest()}


def release_switch(binary, previous, model, root):
    if os.name != "posix":
        raise RuntimeError("release-directory drill requires POSIX atomic symlink replacement")
    releases = root / "releases"
    releases.mkdir()
    for name, source in (("previous", previous), ("candidate", binary)):
        slot = releases / name
        slot.mkdir()
        if source.stat().st_size > 1024 * 1024 * 1024:
            raise RuntimeError("release executable exceeds the 1 GiB drill bound")
        if shutil.disk_usage(root).free < source.stat().st_size + MIN_HOST_FREE:
            raise RuntimeError("insufficient space for isolated release copy")
        shutil.copy2(source, slot / "bloom_server")
    models = root / "release-catalog"
    models.mkdir()
    model_hash, stages = _model_identity(model), []
    for index, slot in enumerate(("previous", "candidate", "previous")):
        activate(releases, slot)
        log = root / f"release-{index}.log"
        process, port = _start(releases / "current/bloom_server", models, log, model, "candle", "cpu",
                               environment={"BLOOM_OPERATOR_API_KEY": KEY})
        try:
            if not port:
                raise RuntimeError("activated release failed to bind")
            evidence = serving_probe(process, port, log, model_hash)
            evidence.update(slot=slot, server_sha256=_sha256_file(releases / slot / "bloom_server"))
            stages.append(evidence)
        finally:
            code = _stop(process)
        if code != 0:
            raise RuntimeError("release did not stop with a clean graceful drain")
    if stages[0]["server_sha256"] != stages[2]["server_sha256"]:
        raise RuntimeError("rollback did not restore the original executable")
    if stages[0]["output_sha256"] != stages[2]["output_sha256"]:
        raise RuntimeError("rollback changed deterministic model output")
    return {
        "result": "pass", "strategy": "atomic relative symlink and graceful process restart",
        "distinct_binaries": stages[0]["server_sha256"] != stages[1]["server_sha256"],
        "stages": stages, "rollback_output_restored": True,
        "service_manager": "isolated subprocess harness",
    }


def run(args):
    if args.server_bin.is_symlink() or (args.previous_server_bin and args.previous_server_bin.is_symlink()):
        raise ValueError("source executables must not be symlinks")
    binary = args.server_bin.resolve(strict=True)
    previous = (args.previous_server_bin or binary).resolve(strict=True)
    model = args.model.resolve(strict=True) if args.model else None
    if not binary.is_file() or not previous.is_file():
        raise ValueError("source executables must be regular files")
    if args.scenario in {"release-switch", "all"} and model is None:
        raise ValueError("release switching requires --model")
    identities = {"server_sha256": _sha256_file(binary), "previous_server_sha256": _sha256_file(previous),
                  "model_sha256": _model_identity(model), "backend": "candle", "device": "cpu"}
    results = {}
    root = Path(tempfile.mkdtemp(prefix="bloom-isolated-storage-drill-")).resolve()
    try:
        if args.scenario in {"physical-enospc", "all"}:
            results["physical_enospc"] = enospc(binary, root)
        if args.scenario in {"release-switch", "all"}:
            results["release_switch"] = release_switch(binary, previous, model, root)
    finally:
        # Never recursively remove a mount if detaching failed.
        if not os.path.ismount(root / "mount"):
            shutil.rmtree(root)
    if identities["server_sha256"] != _sha256_file(binary) or identities["previous_server_sha256"] != _sha256_file(previous) or identities["model_sha256"] != _model_identity(model):
        raise RuntimeError("drill input identity changed during execution")
    return {
        "schema_version": 1, "object": "bloom.deployment_storage_drill", "result": "pass",
        "deployment": identities, "platform": {"os": platform.system(), "machine": platform.machine()},
        "scenarios": results,
        "limitations": [
            "ENOSPC uses a synthetic checksum-controlled import payload, not model-quality evidence.",
            "The isolated disk-image filesystem does not measure the production host filesystem.",
            "Same-binary release switching is a procedure rehearsal, not cross-version compatibility.",
            "The subprocess harness does not prove service-manager or signed-model upgrade recovery.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-bin", required=True, type=Path)
    parser.add_argument("--previous-server-bin", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--scenario", choices=("all", "physical-enospc", "release-switch"), default="all")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        report = run(args)
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as error:
        report = {"schema_version": 1, "object": "bloom.deployment_storage_drill", "result": "fail", "error": str(error)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"deployment storage drill: {report['result']} ({args.output})")
    return 0 if report["result"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
