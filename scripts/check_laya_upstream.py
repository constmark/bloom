#!/usr/bin/env python3
"""Verify the exact reviewed source and provenance of Bloom's Laya fork."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys


def verify(root: Path) -> None:
    package = root / "crates/laya"
    record = json.loads((package / "UPSTREAM.json").read_text(encoding="utf-8"))
    policy = json.loads((root / "config/dependency-policy.json").read_text(encoding="utf-8"))
    expected = policy["reviewed_workspace_forks"]["bloomai-laya"]
    if (record["schema_version"] != 1 or record["object"] != "bloom.upstream_source"
            or record["package"] != "bloomai-laya" or record["version"] != expected["version"]
            or record["upstream"] != expected["upstream"]):
        raise ValueError("Laya provenance does not match reviewed dependency policy")
    files = {}
    for path in sorted(package.rglob("*")):
        if path.is_symlink():
            raise ValueError("Laya source contains a symlink")
        if path.is_file() and path != package / "UPSTREAM.json":
            files[path.relative_to(package).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    if files != record["files"]:
        raise ValueError("Laya source differs from its reviewed file checksums")
    omitted = set(record["omitted_upstream_files"])
    renamed = record["renamed_upstream_files"]
    changed = set(record["modified_upstream_files"])
    if omitted != {".cargo_vcs_info.json", "Cargo.lock"} or changed != {"Cargo.toml", "README.md"}:
        raise ValueError("Laya changes exceed the reviewed manifest/documentation migration")
    if renamed != {"Cargo.toml.orig": "UPSTREAM-Cargo.toml"}:
        raise ValueError("Laya original manifest was not preserved")
    for path, digest in record["upstream_files"].items():
        if path not in omitted | changed and files.get(renamed.get(path, path)) != digest:
            raise ValueError(f"Laya upstream code changed without review: {path}")
    expected_paths = {renamed.get(path, path) for path in record["upstream_files"] if path not in omitted}
    if set(files) != expected_paths | {"LICENSE", "NOTICE"}:
        raise ValueError("Laya source contains unreviewed additions or omissions")
    if not (package / "LICENSE").read_text().lstrip().startswith("Apache License"):
        raise ValueError("Laya Apache-2.0 license is missing")


def main() -> int:
    try:
        verify(Path(__file__).resolve().parents[1])
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"Laya upstream source: FAIL: {error}", file=sys.stderr)
        return 1
    print("Laya upstream source: PASS (original code, reviewed Candle migration, license and SHA-256)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
