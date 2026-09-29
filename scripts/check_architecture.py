#!/usr/bin/env python3
"""Enforce Bloom's crate dependency direction and application/engine boundaries."""

from __future__ import annotations

import json
import pathlib
import re
import subprocess
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]
ALLOWED_INTERNAL = {
    "bloomai-core": set(),
    "bloomai-backend": {"bloomai-core"},
    "bloomai-tilelang": set(),
    "bloomai-engine": {"bloomai-core", "bloomai-backend", "bloomai-tilelang"},
    "bloomai-app": {"bloomai-core", "bloomai-engine"},
    "bloomai-server": {"bloomai-core", "bloomai-backend", "bloomai-engine", "bloomai-app"},
    "bloomai-ffi": {"bloomai-core", "bloomai-engine"},
    "bloom-ui": set(),
}
TRANSPORT = {"axum", "tower", "tower-http", "rust-embed", "mime_guess"}
PRESENTATION = {"dioxus", "web-sys", "js-sys", "wasm-bindgen", "gloo-net", "gloo-timers"}
ENGINE_ONLY = {"tokenizers", "cudarc", "metal", "objc2-metal"}
CLI = {"clap", "tracing-subscriber"}
APPLICATION_IMPORTS = {
    "application", "metrics", "model_download", "model_import", "model_index",
    "model_integrity", "model_manager", "model_preflight", "model_storage",
}


def metadata_packages(manifest: pathlib.Path) -> list[dict]:
    result = subprocess.run(
        ["cargo", "metadata", "--manifest-path", str(manifest), "--no-deps",
         "--format-version", "1", "--locked", "--offline"],
        check=True, capture_output=True, text=True,
    )
    metadata = json.loads(result.stdout)
    members = set(metadata["workspace_members"])
    return [package for package in metadata["packages"] if package["id"] in members]


def validate_packages(packages: list[dict]) -> list[str]:
    errors = []
    names = {package["name"] for package in packages}
    if names != set(ALLOWED_INTERNAL):
        errors.append(f"update the explicit layer map for workspace packages: {sorted(names ^ set(ALLOWED_INTERNAL))}")
    for package in packages:
        name = package["name"]
        for dependency in package["dependencies"]:
            target = dependency["name"]  # Cargo resolves aliases and target-specific tables.
            if target in names or target.startswith("bloomai-"):
                if target not in ALLOWED_INTERNAL.get(name, set()):
                    errors.append(f"forbidden layer dependency: {name} -> {target}")
            if target in {"bloomai-engine", "bloomai-app"} and dependency["uses_default_features"]:
                errors.append(f"{name} must explicitly forward features to {target}")
            if name != "bloom-ui" and target in PRESENTATION:
                errors.append(f"browser dependency outside UI: {name} -> {target}")
            if name not in {"bloomai-server", "bloom-ui"} and target in TRANSPORT:
                errors.append(f"transport dependency below HTTP: {name} -> {target}")
            if name == "bloomai-server" and (target.startswith("candle-") or target in ENGINE_ONLY):
                errors.append(f"HTTP must use engine contracts, not {target}")
            if name in {"bloomai-core", "bloomai-backend", "bloomai-engine", "bloomai-ffi", "bloomai-tilelang"}:
                if dependency["kind"] != "dev" and target in CLI:
                    errors.append(f"process dependency below application layer: {name} -> {target}")
        if name == "bloomai-engine" and any("bin" in target["kind"] for target in package["targets"]):
            errors.append("native CLI binaries belong in bloomai-app")
    return errors


def validate_sources(root: pathlib.Path) -> list[str]:
    errors = []
    boundaries = (
        ("crates/server/src", r"\b(?:candle_core|candle_nn|tokenizers)::|\b(?:QwenModelWrapper|ServerKvHook)\b|\.create_wrapper\("),
        ("crates/engine/src", r"\b(?:clap|tracing_subscriber)::|\b(?:ServerConfig|InferConfig|BenchConfig)\b"),
    )
    for directory, pattern in boundaries:
        for source in sorted((root / directory).rglob("*.rs")):
            for number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
                if re.search(pattern, line):
                    errors.append(f"{source.relative_to(root)}:{number}: implementation crosses layer boundary")
    # These contracts must remain usable without importing a concrete backend.
    for name in ("core/model.rs", "core/pipeline.rs", "batching.rs"):
        source = root / "crates/engine/src" / name
        if not source.exists():
            continue
        for number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
            if re.search(r"\b(?:candle_core|candle_nn|tokenizers)::|crate::executor::|\.downcast\b|\bAny\b", line):
                errors.append(f"{source.relative_to(root)}:{number}: backend detail in model contract")
    for source in sorted((root / "crates/server/src/application").rglob("*.rs")):
        content = source.read_text(encoding="utf-8")
        # Unit tests may import their owner. Production cannot hide dependencies
        # behind parent/root wildcard imports. All test modules live at EOF.
        production = re.split(r"#\[cfg\(test\)\]\s*mod tests\s*\{", content, maxsplit=1)[0]
        for number, line in enumerate(production.splitlines(), 1):
            if line.lstrip().startswith("//"):
                continue
            if re.search(r"\b(?:axum|tower|tower_http|clap)::|\b(?:ServerState|ApiError|Args)\b|(?:crate|super)::\*|super::super::", line):
                errors.append(f"{source.relative_to(root)}:{number}: transport or root dependency in application service")
            for module in re.findall(r"\bcrate::(\w+)", line):
                if module not in APPLICATION_IMPORTS:
                    errors.append(f"{source.relative_to(root)}:{number}: application dependency is not an approved service/infrastructure module: {module}")
    return errors


def main() -> int:
    try:
        packages = metadata_packages(ROOT / "Cargo.toml") + metadata_packages(ROOT / "ui/Cargo.toml")
    except (OSError, subprocess.CalledProcessError, ValueError) as error:
        print(f"Cannot inspect architecture: {error}", file=sys.stderr)
        return 1
    errors = validate_packages(packages) + validate_sources(ROOT)
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print("OK: crate direction, feature ownership, transport/application and model capability boundaries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
