#!/usr/bin/env python3
"""Fail closed if official container publication bypasses its required gates."""
from pathlib import Path
import re
import sys

from check_release_workflow_security import mapping_value, nested_block


def validate_text(content):
    errors = []
    prefix = content.partition("\njobs:")[0]
    permissions = nested_block(prefix, "permissions", 0) or ""
    if mapping_value(permissions, "contents") != "read" or re.search(r":\s*write", permissions):
        errors.append("container workflow must default to read-only permissions")
    checks = nested_block(content, "checks", 2) or ""
    if mapping_value(checks, "uses") != "./.github/workflows/ci.yml":
        errors.append("container checks must reuse same-revision CI")
    for job_name, needs in (("images", "checks"), ("publish", "images")):
        job = nested_block(content, job_name, 2) or ""
        if mapping_value(job, "needs") != needs:
            errors.append(f"{job_name} must require successful {needs}")
        if re.search(r"^    (?:if|continue-on-error):", job, re.MULTILINE):
            errors.append(f"{job_name} must not bypass dependency success")
    if re.search(r"^    (?:if|continue-on-error):", checks, re.MULTILINE):
        errors.append("container checks must not be skipped")
    images = nested_block(content, "images", 2) or ""
    for value in (
        "arch: amd64", "runner: ubuntu-24.04\n", "arch: arm64", "runner: ubuntu-24.04-arm",
        "version: v0.38.2", "syft-version: v1.57.1", "version: v0.76.2",
        "push: false", "load: true", 'exit-code: "1"', "ignore-unfixed: false",
        "severity: HIGH,CRITICAL", "vuln-type: os,library", "upload-release-assets: false",
        "sbom-path: sbom-", "push-to-registry: true",
        "--require-native-sbom",
    ):
        if value not in images:
            errors.append(f"container image gate lacks required setting: {value}")
    steps = (
        "name: Build exact local runtime image", "name: Verify hardened runtime and signal drain",
        "name: Generate final image SBOM", "name: Scan final runtime layer",
        "name: Authenticate after runtime and vulnerability gates",
        "name: Push verified image and record digest", "name: Attest platform build provenance",
        "name: Attest platform SBOM",
    )
    positions = [images.find(step) for step in steps]
    if any(position < 0 for position in positions) or positions != sorted(positions):
        errors.append("container build, test, SBOM, scan, push and attestation order is invalid")
    if "continue-on-error:" in images or re.search(r"^        if:", images, re.MULTILINE):
        errors.append("container image gates must never be optional")
    publish = nested_block(content, "publish", 2) or ""
    for required in (
        "python scripts/container_release.py publish --artifacts artifacts --output container-release.json",
        "name: Attest multi-architecture index", "push-to-registry: true",
    ):
        if required not in publish:
            errors.append("container index lacks digest verification or provenance")
    return errors


def validate_compose(content):
    required = (
        'image: ${BLOOM_IMAGE:?', 'user: "10001:10001"', 'read_only: true',
        'cap_drop: [ALL]', 'security_opt: ["no-new-privileges:true"]',
        'pids_limit: 256', 'mem_limit: 4g', 'cpus: 2',
        '/tmp:rw,nosuid,noexec,size=64m', '"127.0.0.1:3000:3000"',
        'BLOOM_API_KEY: ${BLOOM_API_KEY:?',
        'BLOOM_OPERATOR_API_KEY: ${BLOOM_OPERATOR_API_KEY:?',
        'BLOOM_STRICT_SECURITY: "1"', 'BLOOM_STRICT_MEMORY_BUDGET: "1"',
        'BLOOM_MAX_MODEL_STORAGE_BYTES: "21474836480"',
        'bloom-data:/var/lib/bloom', 'stop_grace_period: 40s',
    )
    return [f"deployment template lacks required hardening: {item}"
            for item in required if item not in content]


def main():
    root = Path(__file__).resolve().parents[1]
    content = (root / ".github/workflows/container-release.yml").read_text()
    errors = validate_text(content)
    errors.extend(validate_compose((root / "deploy/compose.yml").read_text()))
    for error in errors:
        print(f"ERROR: {error}", file=sys.stderr)
    if not errors:
        print("OK: container publication requires CI, native runtime, scan, SBOM and digest attestations")
    return bool(errors)


if __name__ == "__main__":
    raise SystemExit(main())
