#!/usr/bin/env python3
"""Bind the reviewed Candle/GEMM memory and GPU ABI assumptions to Cargo's graph."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REGISTRY = "registry+https://github.com/rust-lang/crates.io-index"
CANDLE_VERSION = "0.11.1"
GEMM_VERSION = "0.19.0"


def verify(metadata: dict) -> None:
    packages = metadata["packages"]
    by_name = {}
    for package in packages:
        by_name.setdefault(package["name"], []).append(package)
        if package["name"].startswith("candle-"):
            if package["version"] != CANDLE_VERSION or package["source"] != REGISTRY:
                raise ValueError("Candle must use the reviewed official 0.11.1 release")
        if package["name"].startswith("gemm"):
            if package["version"] != GEMM_VERSION or package["source"] != REGISTRY:
                raise ValueError("GEMM must preserve the reviewed 0.19.0 packing/TLS bounds")
        if package["name"] == "laya-decision":
            raise ValueError("the old Laya dependency must not reintroduce another Candle/ZIP graph")
        if package["name"] == "zip":
            # Pin the exact audited resolution. A future update needs a new
            # audit and this deliberate review, not a loose version alias.
            if package["version"] != "9.5.1" or package["source"] != REGISTRY:
                raise ValueError("ZIP must use the audited official 9.5.1 release")

    nodes = {node["id"]: node for node in metadata["resolve"]["nodes"]}

    def identity(name: str) -> str:
        matching = by_name.get(name, [])
        if len(matching) != 1 or matching[0]["id"] not in nodes:
            raise ValueError(f"expected exactly one resolved {name}")
        return matching[0]["id"]

    for owner, dependency in (
        ("bloomai-engine", "bloomai-laya"),
        ("bloomai-engine", "gemm-common"),
        ("bloomai-engine", "candle-core"),
        ("bloomai-laya", "candle-core"),
        ("candle-core", "gemm"),
        ("candle-core", "zip"),
        ("gemm", "gemm-common"),
    ):
        edges = {dependency["pkg"] for dependency in nodes[identity(owner)]["deps"]}
        if identity(dependency) not in edges:
            raise ValueError(f"reviewed runtime dependency changed: {owner} -> {dependency}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", required=True, type=Path,
                        help="Full Cargo metadata produced with --locked for the release target")
    args = parser.parse_args()
    try:
        verify(json.loads(args.metadata.read_text(encoding="utf-8")))
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"Runtime dependencies: FAIL: {error}", file=sys.stderr)
        return 1
    print("Runtime dependencies: PASS (official Candle/ZIP, shared GEMM bounds and Laya graph)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
