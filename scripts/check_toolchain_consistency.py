#!/usr/bin/env python3
"""Require one exact tested Rust toolchain across local, CI, and Docker builds."""

from __future__ import annotations

import argparse
import pathlib
import re
import subprocess
import sys


EXACT_VERSION = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+$")
TOOLCHAIN_CHANNEL = re.compile(
    r'^\s*channel\s*=\s*"([^"]+)"\s*(?:#.*)?$', re.MULTILINE
)
ACTION_REFERENCE = re.compile(r"dtolnay/rust-toolchain@([0-9a-f]{40})")
TOOLCHAIN_INPUT = re.compile(r'^\s*toolchain:\s*["\']?([^"\'\s]+)["\']?\s*$', re.MULTILINE)
DOCKER_RUST_IMAGE = re.compile(r"^\s*FROM\s+rust:([^\s@]+)", re.MULTILINE)
WORKSPACE_RUST_VERSION = re.compile(
    r'^\s*rust-version\s*=\s*"([^"]+)"\s*(?:#.*)?$', re.MULTILINE
)
INHERITED_RUST_VERSION = re.compile(
    r"^\s*rust-version\.workspace\s*=\s*true\s*(?:#.*)?$", re.MULTILINE
)


def validate(root: pathlib.Path) -> tuple[str | None, list[str]]:
    root = root.resolve()
    errors: list[str] = []
    toolchain_path = root / "rust-toolchain.toml"
    try:
        toolchain = toolchain_path.read_text(encoding="utf-8")
    except OSError as error:
        return None, [f"cannot read rust-toolchain.toml: {error}"]
    channel = TOOLCHAIN_CHANNEL.search(toolchain)
    version = channel.group(1) if channel is not None else None
    if version is None or EXACT_VERSION.fullmatch(version) is None:
        return None, ["rust-toolchain.toml must select one major.minor.patch release"]

    workspace_manifest_path = root / "Cargo.toml"
    try:
        workspace_manifest = workspace_manifest_path.read_text(encoding="utf-8")
    except OSError as error:
        errors.append(f"cannot read Cargo.toml: {error}")
    else:
        declared_version = WORKSPACE_RUST_VERSION.search(workspace_manifest)
        if declared_version is None:
            errors.append("workspace.package must declare rust-version")
        elif declared_version.group(1) != version:
            errors.append(
                f"workspace rust-version is {declared_version.group(1)}; expected {version}"
            )

    for crate_manifest_path in sorted((root / "crates").glob("*/Cargo.toml")):
        try:
            crate_manifest = crate_manifest_path.read_text(encoding="utf-8")
        except OSError as error:
            errors.append(
                f"cannot read {crate_manifest_path.relative_to(root)}: {error}"
            )
            continue
        if INHERITED_RUST_VERSION.search(crate_manifest) is None:
            errors.append(
                f"{crate_manifest_path.relative_to(root)} must inherit workspace rust-version"
            )

    ui_manifest_path = root / "ui" / "Cargo.toml"
    try:
        ui_manifest = ui_manifest_path.read_text(encoding="utf-8")
    except OSError as error:
        errors.append(f"cannot read ui/Cargo.toml: {error}")
    else:
        ui_version = WORKSPACE_RUST_VERSION.search(ui_manifest)
        if ui_version is None:
            errors.append("ui/Cargo.toml must declare rust-version")
        elif ui_version.group(1) != version:
            errors.append(
                f"ui rust-version is {ui_version.group(1)}; expected {version}"
            )

    workflow_paths = sorted((root / ".github" / "workflows").glob("*.yml"))
    action_references: list[tuple[pathlib.Path, str]] = []
    action_toolchains: list[tuple[pathlib.Path, str]] = []
    for workflow_path in workflow_paths:
        try:
            content = workflow_path.read_text(encoding="utf-8")
        except OSError as error:
            errors.append(f"cannot read {workflow_path.relative_to(root)}: {error}")
            continue
        action_references.extend(
            (workflow_path, reference)
            for reference in ACTION_REFERENCE.findall(content)
        )
        action_toolchains.extend(
            (workflow_path, toolchain)
            for toolchain in TOOLCHAIN_INPUT.findall(content)
        )
    if not action_references:
        errors.append("no dtolnay/rust-toolchain workflow references were found")
    if len(action_references) != len(action_toolchains):
        errors.append(
            "each SHA-pinned dtolnay/rust-toolchain step must declare one explicit "
            "toolchain input"
        )
    for workflow_path, selected_toolchain in action_toolchains:
        if selected_toolchain != version:
            errors.append(
                f"{workflow_path.relative_to(root)} selects Rust {selected_toolchain}; "
                f"expected {version}"
            )

    dockerfile_path = root / "Dockerfile"
    try:
        dockerfile = dockerfile_path.read_text(encoding="utf-8")
    except OSError as error:
        errors.append(f"cannot read Dockerfile: {error}")
    else:
        image = DOCKER_RUST_IMAGE.search(dockerfile)
        if image is None:
            errors.append("Dockerfile does not declare a tagged rust builder image")
        elif not image.group(1).startswith(f"{version}-"):
            errors.append(
                f"Dockerfile selects rust:{image.group(1)}; expected Rust {version}"
            )

    try:
        active_output = subprocess.run(
            ["rustc", "--version"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        errors.append(f"cannot query the active rustc: {error}")
    else:
        active_parts = active_output.split()
        active_version = active_parts[1] if len(active_parts) >= 2 else "unknown"
        if active_version != version:
            errors.append(
                f"active rustc is {active_version}; expected repository toolchain {version}"
            )

    return version, errors


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

    version, errors = validate(args.root)
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        print("FAILED: Rust toolchain selection is inconsistent", file=sys.stderr)
        return 1
    print(f"OK: local, CI, and Docker builds select Rust {version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
