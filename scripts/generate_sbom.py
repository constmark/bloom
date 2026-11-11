#!/usr/bin/env python3
"""Generate Bloom's deterministic, policy-checked CycloneDX release SBOM."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

from sbom_contract import SbomError, build_sbom
from check_laya_upstream import verify as verify_laya_upstream
from check_runtime_dependencies import verify as verify_runtime_dependencies


def read_json(path: pathlib.Path, label: str) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SbomError(f"{label} is not readable UTF-8 JSON: {path}") from error


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metadata",
        required=True,
        type=pathlib.Path,
        action="append",
        help="Cargo metadata JSON; repeat for independently built workspaces",
    )
    parser.add_argument("--policy", required=True, type=pathlib.Path)
    parser.add_argument("--target", required=True)
    parser.add_argument("--embedded-ui", choices=("true", "false"), required=True)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    args = parser.parse_args()
    try:
        verify_laya_upstream(pathlib.Path(__file__).resolve().parents[1])
        documents = [read_json(path, "cargo metadata") for path in args.metadata]
        # The first metadata document is the native release graph; optional UI
        # metadata comes from its separate workspace. Both are obtained locked.
        verify_runtime_dependencies(documents[0])
        sbom = build_sbom(
            documents,
            read_json(args.policy, "dependency policy"),
            args.target,
            args.embedded_ui == "true",
        )
        args.output.write_text(
            json.dumps(sbom, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    except (OSError, SbomError, ValueError, KeyError, TypeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(f"OK: wrote policy-checked CycloneDX SBOM to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
