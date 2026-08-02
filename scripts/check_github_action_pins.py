#!/usr/bin/env python3
"""Require immutable full-SHA references for every external GitHub Action."""

from __future__ import annotations

import argparse
import pathlib
import re
import sys


USES_LINE = re.compile(r"^\s*(?:-\s*)?uses:\s*([^\s#]+)(?:\s+#\s*(.+?))?\s*$")
FULL_COMMIT_SHA = re.compile(r"^[0-9a-f]{40}$")


def workflow_files(root: pathlib.Path) -> list[pathlib.Path]:
    workflow_root = root / ".github" / "workflows"
    return sorted([*workflow_root.glob("*.yml"), *workflow_root.glob("*.yaml")])


def validate(root: pathlib.Path) -> tuple[int, list[str]]:
    root = root.resolve()
    checked = 0
    errors: list[str] = []
    for workflow in workflow_files(root):
        try:
            lines = workflow.read_text(encoding="utf-8").splitlines()
        except OSError as error:
            errors.append(f"cannot read {workflow.relative_to(root)}: {error}")
            continue
        for line_number, line in enumerate(lines, start=1):
            match = USES_LINE.match(line)
            if match is None:
                continue
            reference, version_comment = match.groups()
            if reference.startswith("./"):
                continue
            checked += 1
            if reference.startswith("docker://"):
                if "@sha256:" not in reference:
                    errors.append(
                        f"{workflow.relative_to(root)}:{line_number}: "
                        "external container action is not digest-pinned"
                    )
                continue
            if "@" not in reference:
                errors.append(
                    f"{workflow.relative_to(root)}:{line_number}: "
                    f"external action has no revision: {reference}"
                )
                continue
            _action, revision = reference.rsplit("@", maxsplit=1)
            if FULL_COMMIT_SHA.fullmatch(revision) is None:
                errors.append(
                    f"{workflow.relative_to(root)}:{line_number}: "
                    f"external action is not pinned to a full commit SHA: {reference}"
                )
            if version_comment is None or not version_comment.strip():
                errors.append(
                    f"{workflow.relative_to(root)}:{line_number}: "
                    "SHA-pinned action lacks an inline release/branch comment"
                )
    if checked == 0:
        errors.append("no external GitHub Actions were found")
    return checked, errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=pathlib.Path,
        default=pathlib.Path(__file__).resolve().parents[1],
        help="Repository root to validate (defaults to the script's parent repository).",
    )
    args = parser.parse_args()
    if not args.root.is_dir():
        parser.error(f"repository root is not a directory: {args.root}")

    checked, errors = validate(args.root)
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        print("FAILED: GitHub Action references are not immutable", file=sys.stderr)
        return 1
    print(f"OK: validated {checked} full-SHA GitHub Action references")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
