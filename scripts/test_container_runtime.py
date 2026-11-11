#!/usr/bin/env python3
"""Exercise an existing local Bloom image with a read-only, unprivileged runtime."""
import argparse
import hashlib
import json
import subprocess
import time
import uuid
from pathlib import Path

from readiness_contract import validate_readiness_document
from sbom_contract import SbomError, validate_sbom_document


def docker(*args, check=True):
    return subprocess.run(["docker", *args], check=check, capture_output=True,
                          text=True, timeout=60)


def run(image, require_native_sbom=False):
    suffix = uuid.uuid4().hex
    container, volume = f"bloom-runtime-{suffix}", f"bloom-runtime-data-{suffix}"
    docker("volume", "create", volume)
    try:
        docker("run", "--detach", "--name", container, "--read-only",
               "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
               "--pids-limit", "256", "--memory", "1g",
               "--tmpfs", "/tmp:rw,nosuid,noexec,size=64m",
               "--mount", f"type=volume,src={volume},dst=/var/lib/bloom",
               "-e", "BLOOM_API_KEY=container-drill-inference",
               "-e", "BLOOM_OPERATOR_API_KEY=container-drill-operator",
               image)
        deadline, ready = time.monotonic() + 45, None
        while time.monotonic() < deadline:
            probe = docker("exec", container, "curl", "--max-time", "3", "-sS",
                           "-w", "\n%{http_code}", "http://127.0.0.1:3000/ready", check=False)
            if probe.returncode == 0:
                body, status = probe.stdout.rsplit("\n", 1)
                ready = validate_readiness_document(json.loads(body))
                if status != "503" or ready["status"] != "not_ready" or ready["model"] != "not loaded":
                    raise RuntimeError("empty container must publish validated unavailable-model readiness")
                break
            time.sleep(0.2)
        if ready is None:
            raise RuntimeError("container did not publish readiness before deadline")
        docker("exec", container, "curl", "--max-time", "3", "-fsS", "http://127.0.0.1:3000/health")
        page = docker("exec", container, "curl", "--max-time", "3", "-fsS",
                      "http://127.0.0.1:3000/")
        if "<html" not in page.stdout.lower():
            raise RuntimeError("container does not serve its embedded UI")
        if docker("exec", container, "id", "-u").stdout.strip() != "10001":
            raise RuntimeError("container is not running as its unprivileged UID")
        identity = docker("inspect", "--format", "{{.Image}}", container).stdout.strip()
        native_sbom_hash = None
        if require_native_sbom:
            sbom_raw = docker("exec", container, "head", "-c", "16777217",
                              "/usr/share/bloom/BLOOM-SBOM.cdx.json").stdout
            policy_raw = docker("exec", container, "head", "-c", "65537",
                                "/usr/share/bloom/BLOOM-DEPENDENCY-POLICY.json").stdout
            if len(sbom_raw.encode()) > 16777216 or len(policy_raw.encode()) > 65536:
                raise RuntimeError("bundled dependency evidence exceeds bounds")
            architecture = docker("image", "inspect", "--format", "{{.Architecture}}", identity).stdout.strip()
            target = {"amd64": "x86_64-unknown-linux-gnu", "arm64": "aarch64-unknown-linux-gnu"}.get(architecture)
            if target is None:
                raise RuntimeError("unsupported container platform")
            policy = json.loads(policy_raw)
            expected_policy = json.loads((Path(__file__).resolve().parents[1] / "config/dependency-policy.json").read_text())
            if policy != expected_policy:
                raise RuntimeError("image dependency policy differs from the reviewed policy")
            validate_sbom_document(json.loads(sbom_raw), policy, ready["server_version"], target, True)
            native_sbom_hash = hashlib.sha256(sbom_raw.encode()).hexdigest()
        docker("stop", "--time", "40", container)
        code = docker("inspect", "--format", "{{.State.ExitCode}}", container).stdout.strip()
        if code != "0":
            raise RuntimeError("SIGTERM did not produce a successful graceful drain")
        return {"object": "bloom.container_runtime_check", "schema_version": 1,
                "result": "pass", "image_id": identity, "readonly_root": True,
                "capabilities_dropped": True, "no_new_privileges": True,
                "uid": 10001, "liveness_status": 200, "empty_readiness_status": 503,
                "embedded_ui": True, "sigterm_exit_code": 0,
                "native_sbom_verified": require_native_sbom,
                "native_sbom_sha256": native_sbom_hash}
    finally:
        docker("rm", "--force", container, check=False)
        docker("volume", "rm", volume, check=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-native-sbom", action="store_true")
    args = parser.parse_args()
    try:
        report = run(args.image, args.require_native_sbom)
    except (OSError, RuntimeError, ValueError, SbomError, subprocess.SubprocessError) as error:
        report = {"result": "fail", "error": str(error)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"container runtime: {report['result']} ({args.output})")
    return 0 if report["result"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
