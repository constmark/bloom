#!/usr/bin/env python3
"""Fail when a repository Markdown document links to a missing local path."""

from __future__ import annotations

import argparse
import pathlib
import re
import sys
import urllib.parse


EXCLUDED_DIRECTORIES = {
    ".git",
    ".venv",
    "node_modules",
    "release-artifacts",
    "target",
}
INLINE_LINK = re.compile(r"!?\[[^\]]*\]\(([^)\n]+)\)")
REFERENCE_LINK = re.compile(r"^\s*\[[^\]]+\]:\s*(\S+)")


def markdown_files(root: pathlib.Path) -> list[pathlib.Path]:
    return sorted(
        path
        for path in root.rglob("*.md")
        if not EXCLUDED_DIRECTORIES.intersection(path.relative_to(root).parts)
    )


def link_target(raw_target: str) -> str | None:
    value = raw_target.strip()
    if value.startswith("<"):
        closing = value.find(">")
        if closing < 0:
            return value
        value = value[1:closing]
    else:
        value = value.split(maxsplit=1)[0]
    if not value or value.startswith("#") or value.startswith("//"):
        return None
    parsed = urllib.parse.urlsplit(value)
    if parsed.scheme:
        return None
    return urllib.parse.unquote(parsed.path)


def document_targets(path: pathlib.Path) -> list[tuple[int, str]]:
    targets: list[tuple[int, str]] = []
    fence: str | None = None
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        stripped = line.lstrip()
        marker = stripped[:3]
        if marker in {"```", "~~~"}:
            if fence is None:
                fence = marker
            elif marker == fence:
                fence = None
            continue
        if fence is not None:
            continue
        raw_targets = INLINE_LINK.findall(line)
        reference = REFERENCE_LINK.match(line)
        if reference is not None:
            raw_targets.append(reference.group(1))
        for raw_target in raw_targets:
            target = link_target(raw_target)
            if target:
                targets.append((line_number, target))
    return targets


def validate(root: pathlib.Path) -> tuple[int, list[str]]:
    root = root.resolve()
    files = markdown_files(root)
    checked_links = 0
    errors: list[str] = []
    for document in files:
        for line_number, target in document_targets(document):
            checked_links += 1
            target_path = pathlib.Path(target)
            candidate = (
                root / target.lstrip("/")
                if target_path.is_absolute()
                else document.parent / target_path
            ).resolve()
            try:
                candidate.relative_to(root)
            except ValueError:
                errors.append(
                    f"{document.relative_to(root)}:{line_number}: "
                    f"local link escapes the repository: {target}"
                )
                continue
            if not candidate.exists():
                errors.append(
                    f"{document.relative_to(root)}:{line_number}: "
                    f"local link target does not exist: {target}"
                )
    return checked_links, errors


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

    checked_links, errors = validate(args.root)
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        print(
            f"FAILED: {len(errors)} invalid local Markdown link(s)", file=sys.stderr
        )
        return 1
    document_count = len(markdown_files(args.root.resolve()))
    print(
        f"OK: validated {checked_links} local Markdown links "
        f"across {document_count} documents"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
