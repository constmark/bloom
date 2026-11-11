#!/usr/bin/env python3
"""Validate tagged OCI identities and publish only the two verified platform digests."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
DIGEST = re.compile(r"sha256:[0-9a-f]{64}")
ARCHITECTURES = ("amd64", "arm64")


def digest_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def identity(environment=None):
    env = os.environ if environment is None else environment
    repository = env.get("GITHUB_REPOSITORY", "").lower()
    revision = env.get("GITHUB_SHA", "")
    tag = env.get("GITHUB_REF_NAME", "")
    manifest = (ROOT / "Cargo.toml").read_text()
    version = re.search(r'^version = "([^"]+)"$', manifest, re.MULTILINE).group(1)
    if re.fullmatch(r"[a-z0-9][a-z0-9_.-]*/[a-z0-9][a-z0-9_.-]*", repository) is None:
        raise ValueError("invalid GitHub repository identity")
    if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise ValueError("invalid source revision")
    if tag != f"v{version}" or re.fullmatch(r"v[0-9]+\.[0-9]+\.[0-9]+(?:-[0-9A-Za-z.-]+)?", tag) is None:
        raise ValueError("container tag must exactly match the workspace package version")
    value = {"image": f"ghcr.io/{repository}", "revision": revision, "version": tag}
    if env.get("ARCH"):
        arch, run, attempt = env["ARCH"], env.get("GITHUB_RUN_ID", ""), env.get("GITHUB_RUN_ATTEMPT", "")
        if arch not in ARCHITECTURES or not run.isdecimal() or not attempt.isdecimal():
            raise ValueError("invalid platform build identity")
        value.update(architecture=arch, candidate=f"{value['image']}:build-{run}-{attempt}-{arch}")
    return value


def emit_outputs(values):
    output = os.environ.get("GITHUB_OUTPUT")
    if output:
        with open(output, "a", encoding="utf-8") as destination:
            for key, value in values.items():
                if not isinstance(value, str) or "\n" in value or "\r" in value:
                    raise ValueError("invalid workflow output")
                destination.write(f"{key}={value}\n")


def docker(*arguments):
    return subprocess.run(["docker", *arguments], check=True, capture_output=True,
                          text=True, timeout=300).stdout


def read_json(path, maximum=16 * 1024 * 1024):
    if path.is_symlink() or not path.is_file() or not 0 < path.stat().st_size <= maximum:
        raise ValueError(f"invalid bounded release evidence file: {path.name}")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"release evidence must be a JSON object: {path.name}")
    return value


def validate_records(directory, expected):
    records = []
    for arch in ARCHITECTURES:
        record = read_json(directory / f"{arch}.json", 65536)
        if type(record.get("schema_version")) is not int or record["schema_version"] != 1 or record.get("object") != "bloom.container_platform":
            raise ValueError("unknown container platform evidence contract")
        for key in ("image", "revision", "version"):
            if record.get(key) != expected[key]:
                raise ValueError(f"container platform {key} does not match this release")
        if record.get("architecture") != arch or DIGEST.fullmatch(record.get("digest", "")) is None:
            raise ValueError("invalid platform architecture or image digest")
        for label, filename in (("runtime", f"runtime-{arch}.json"), ("scan", f"scan-{arch}.json"),
                                ("sbom", f"sbom-{arch}.spdx.json")):
            evidence = directory / filename
            read_json(evidence)
            if record.get(f"{label}_sha256") != digest_file(evidence):
                raise ValueError(f"platform {label} evidence was replaced")
        records.append(record)
    if records[0]["digest"] == records[1]["digest"]:
        raise ValueError("two native architectures cannot share one image manifest digest")
    return records


def validate_index(index, records):
    manifests = index.get("manifests")
    if index.get("schemaVersion") != 2 or not isinstance(manifests, list) or len(manifests) != 2:
        raise ValueError("published index must contain exactly two platform manifests")
    actual = {}
    for manifest in manifests:
        platform = manifest.get("platform", {})
        arch = platform.get("architecture")
        if platform.get("os") != "linux" or arch not in ARCHITECTURES or arch in actual:
            raise ValueError("published index contains an unexpected platform")
        actual[arch] = manifest.get("digest")
    if actual != {record["architecture"]: record["digest"] for record in records}:
        raise ValueError("published index differs from verified platform digests")


def record_platform(path):
    expected = identity()
    arch = expected["architecture"]
    local = json.loads(docker("image", "inspect", expected["candidate"]))[0]
    runtime_path, scan_path, sbom_path = Path(f"runtime-{arch}.json"), Path(f"scan-{arch}.json"), Path(f"sbom-{arch}.spdx.json")
    runtime, scan, sbom = read_json(runtime_path), read_json(scan_path), read_json(sbom_path)
    if runtime.get("result") != "pass" or runtime.get("image_id") != local.get("Id") or runtime.get("native_sbom_verified") is not True:
        raise ValueError("pushed image does not match the runtime-tested local image")
    if local.get("Architecture") != arch or local.get("Os") != "linux":
        raise ValueError("local image does not match its declared native platform")
    if not isinstance(scan.get("Results"), list) or not scan["Results"]:
        raise ValueError("final image vulnerability scan has no package results")
    for result in scan["Results"]:
        if any(item.get("Severity") in {"HIGH", "CRITICAL"} for item in (result.get("Vulnerabilities") or [])):
            raise ValueError("final image has an unresolved high or critical vulnerability")
    if not isinstance(sbom.get("packages"), list) or not sbom["packages"] or not str(sbom.get("spdxVersion", "")).startswith("SPDX-"):
        raise ValueError("final image SBOM has no package inventory")
    digests = [item.split("@", 1)[1] for item in local.get("RepoDigests", [])
               if item.startswith(expected["image"] + "@")]
    if len(digests) != 1 or DIGEST.fullmatch(digests[0]) is None:
        raise ValueError("pushed image did not receive exactly one registry manifest digest")
    record = {"schema_version": 1, "object": "bloom.container_platform", **expected, "digest": digests[0],
              "runtime_sha256": digest_file(runtime_path), "scan_sha256": digest_file(scan_path),
              "sbom_sha256": digest_file(sbom_path)}
    path.write_text(json.dumps(record, indent=2) + "\n")
    emit_outputs({"digest": digests[0]})


def publish(directory, output):
    expected = identity()
    records = validate_records(directory, expected)
    references = [f"{expected['image']}@{record['digest']}" for record in records]
    target = f"{expected['image']}:{expected['version']}"
    docker("buildx", "imagetools", "create", "--tag", target, *references)
    # Resolve the mutable tag once, then validate exactly the immutable object
    # used by the release record and subsequent provenance attestation.
    descriptor = json.loads(docker("buildx", "imagetools", "inspect", "--format", "{{json .Manifest}}", target))
    digest = descriptor.get("digest", "")
    if DIGEST.fullmatch(digest) is None:
        raise ValueError("registry did not return a valid index digest")
    immutable_reference = f"{expected['image']}@{digest}"
    index = json.loads(docker("buildx", "imagetools", "inspect", "--raw", immutable_reference))
    validate_index(index, records)
    report = {"schema_version": 1, "object": "bloom.container_release", **expected, "digest": digest,
              "immutable_reference": immutable_reference, "platforms": records}
    output.write_text(json.dumps(report, indent=2) + "\n")
    emit_outputs({"image": expected["image"], "digest": digest})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("identity", "record", "publish"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--artifacts", type=Path)
    args = parser.parse_args()
    try:
        if args.action == "identity":
            emit_outputs(identity())
        elif args.action == "record":
            if args.output is None:
                parser.error("record requires --output")
            record_platform(args.output)
        else:
            if args.output is None or args.artifacts is None:
                parser.error("publish requires --artifacts and --output")
            publish(args.artifacts, args.output)
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as error:
        parser.exit(1, f"ERROR: container release rejected: {error}\n")


if __name__ == "__main__":
    main()
