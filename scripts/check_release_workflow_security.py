#!/usr/bin/env python3
"""Validate immutable provenance and least-privilege release workflow policy."""

from __future__ import annotations

import argparse
import pathlib
import re
import sys


FULL_SHA_ACTION = r"[0-9a-f]{40}"
DOWNLOAD_ACTION = re.compile(rf"uses:\s*actions/download-artifact@{FULL_SHA_ACTION}")
ATTEST_ACTION = re.compile(rf"uses:\s*actions/attest@{FULL_SHA_ACTION}")
PUBLISH_ACTION = re.compile(rf"uses:\s*softprops/action-gh-release@{FULL_SHA_ACTION}")
REQUIRED_JOB_PERMISSIONS = {
    "artifact-metadata": "write",
    "attestations": "write",
    "contents": "write",
    "id-token": "write",
}
REQUIRED_SUBJECTS = {
    "artifacts/*.tar.gz",
    "artifacts/*.zip",
    "artifacts/*.sha256",
}


def mapping_value(block: str, key: str) -> str | None:
    match = re.search(rf"^\s*{re.escape(key)}:\s*([^\s#]+)\s*$", block, re.MULTILINE)
    return match.group(1) if match is not None else None


def nested_block(
    content: str, key: str, indent: int, header_value: str = ""
) -> str | None:
    """Return an indentation-delimited YAML block without importing PyYAML."""
    lines = content.splitlines()
    header = re.compile(
        rf"^{' ' * indent}{re.escape(key)}:\s*{re.escape(header_value)}\s*(?:#.*)?$"
    )
    for index, line in enumerate(lines):
        if header.fullmatch(line) is None:
            continue
        body: list[str] = []
        for candidate in lines[index + 1 :]:
            stripped = candidate.lstrip()
            if stripped and not stripped.startswith("#"):
                candidate_indent = len(candidate) - len(stripped)
                if candidate_indent <= indent:
                    break
            body.append(candidate)
        return "\n".join(body)
    return None


def action_step(job: str, action: re.Match[str]) -> str | None:
    """Return the complete workflow step containing an action match."""
    step_starts = [
        match.start() for match in re.finditer(r"^      - ", job, re.MULTILINE)
    ]
    preceding = [position for position in step_starts if position <= action.start()]
    if not preceding:
        return None
    start = max(preceding)
    end = next((position for position in step_starts if position > action.start()), len(job))
    return job[start:end]


def unique_action(
    job: str, pattern: re.Pattern[str], description: str, errors: list[str]
) -> re.Match[str] | None:
    matches = list(pattern.finditer(job))
    if not matches:
        errors.append(f"github-release job lacks a full-SHA {description} step")
        return None
    if len(matches) != 1:
        errors.append(f"github-release job must contain exactly one {description} step")
        return None
    return matches[0]


def validate_text(content: str) -> list[str]:
    errors: list[str] = []
    workflow_prefix, separator, jobs = content.partition("\njobs:")
    if not separator:
        return ["release workflow does not contain a jobs mapping"]
    workflow_permissions = nested_block(workflow_prefix, "permissions", 0)
    if (
        workflow_permissions is None
        or mapping_value(workflow_permissions, "contents") != "read"
    ):
        errors.append("release workflow must default contents permission to read")

    job_marker = "\n  github-release:\n"
    _before_job, separator, release_job = ("\n" + jobs).partition(job_marker)
    if not separator:
        return errors + ["release workflow does not contain the github-release job"]
    next_job = re.search(r"^  [A-Za-z0-9_-]+:\s*$", release_job, re.MULTILINE)
    if next_job is not None:
        release_job = release_job[: next_job.start()]

    release_permissions = nested_block(release_job, "permissions", 4)
    for permission, expected in REQUIRED_JOB_PERMISSIONS.items():
        actual = (
            mapping_value(release_permissions, permission)
            if release_permissions is not None
            else None
        )
        if actual != expected:
            errors.append(
                f"github-release permission {permission} must be {expected}, got {actual}"
            )

    download = unique_action(
        release_job, DOWNLOAD_ACTION, "download-artifact", errors
    )
    attest = unique_action(release_job, ATTEST_ACTION, "actions/attest", errors)
    publish = unique_action(
        release_job, PUBLISH_ACTION, "release publication", errors
    )
    if download is not None and attest is not None and publish is not None:
        if not download.start() < attest.start() < publish.start():
            errors.append("release order must be download, attest, then publish")

    if attest is not None:
        attest_step = action_step(release_job, attest)
        attest_with = (
            nested_block(attest_step, "with", 8) if attest_step is not None else None
        )
        subject_block = (
            nested_block(attest_with, "subject-path", 10, "|")
            if attest_with is not None
            else None
        )
        configured_subjects = (
            {
                line.strip()
                for line in subject_block.splitlines()
                if line.strip() and not line.lstrip().startswith("#")
            }
            if subject_block is not None
            else set()
        )
        missing_subjects = sorted(REQUIRED_SUBJECTS - configured_subjects)
        if missing_subjects:
            errors.append(
                "release attestation omits subject path(s): "
                + ", ".join(missing_subjects)
            )
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workflow",
        type=pathlib.Path,
        default=pathlib.Path(__file__).resolve().parents[1]
        / ".github"
        / "workflows"
        / "release.yml",
        help="Release workflow to validate.",
    )
    args = parser.parse_args()
    try:
        content = args.workflow.read_text(encoding="utf-8")
    except OSError as error:
        print(f"ERROR: cannot read release workflow: {error}", file=sys.stderr)
        return 1
    errors = validate_text(content)
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        print("FAILED: release workflow security contract is invalid", file=sys.stderr)
        return 1
    print("OK: release workflow is least-privilege and provenance-attested")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
