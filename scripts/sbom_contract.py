#!/usr/bin/env python3
"""Build and validate Bloom's deterministic CycloneDX dependency SBOM."""

from __future__ import annotations

import json
import re
import uuid
from typing import Any
from urllib.parse import quote


MAX_COMPONENTS = 4096
MAX_STRING_BYTES = 4096
TARGET_RE = re.compile(r"^[A-Za-z0-9_.-]{1,128}$")
CYCLONEDX_SCHEMA = "http://cyclonedx.org/schema/bom-1.5.schema.json"
POLICY_KEYS = {
    "schema_version",
    "object",
    "allowed_registry_sources",
    "allowed_license_expressions",
    "license_expression_normalizations",
}


class SbomError(Exception):
    """Dependency metadata, policy, or SBOM violates Bloom's contract."""


def _bounded_string(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value.encode("utf-8")) > MAX_STRING_BYTES
        or "\x00" in value
    ):
        raise SbomError(f"{label} must be a non-empty bounded string")
    return value


def _string_set(value: object, label: str) -> set[str]:
    if not isinstance(value, list) or not value or len(value) > MAX_COMPONENTS:
        raise SbomError(f"{label} must be a non-empty bounded list")
    strings = [_bounded_string(item, f"{label} entry") for item in value]
    if strings != sorted(strings) or len(strings) != len(set(strings)):
        raise SbomError(f"{label} must be unique and sorted")
    return set(strings)


def validate_policy(value: object) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != POLICY_KEYS:
        raise SbomError("dependency policy has unknown or missing fields")
    if (
        isinstance(value["schema_version"], bool)
        or value["schema_version"] != 1
        or value["object"] != "bloom.dependency_policy"
    ):
        raise SbomError("dependency policy has an unsupported identity")
    _string_set(value["allowed_registry_sources"], "allowed_registry_sources")
    allowed_licenses = _string_set(
        value["allowed_license_expressions"], "allowed_license_expressions"
    )
    normalizations = value["license_expression_normalizations"]
    if not isinstance(normalizations, dict) or len(normalizations) > MAX_COMPONENTS:
        raise SbomError("license_expression_normalizations must be a bounded object")
    if list(normalizations) != sorted(normalizations):
        raise SbomError("license_expression_normalizations must be sorted")
    for raw, normalized in normalizations.items():
        raw = _bounded_string(raw, "license normalization source")
        normalized = _bounded_string(normalized, "license normalization result")
        if raw not in allowed_licenses or normalized not in allowed_licenses or raw == normalized:
            raise SbomError("license normalization must map reviewed distinct expressions")
    return value


def _cargo_purl(name: str, version: str) -> str:
    return f"pkg:cargo/{quote(name, safe='.-_~')}@{quote(version, safe='.-_~')}"


def _source_for_package(
    package: dict[str, object], workspace_members: set[str], allowed_sources: set[str]
) -> str:
    package_id = _bounded_string(package.get("id"), "cargo package id")
    source = package.get("source")
    if package_id in workspace_members:
        if source is not None:
            raise SbomError(f"workspace package unexpectedly has a source: {package_id}")
        return "workspace"
    if not isinstance(source, str) or source not in allowed_sources:
        raise SbomError(f"dependency uses an unreviewed source: {package_id}")
    return source


def merge_cargo_metadata(value: object) -> dict[str, object]:
    """Merge one or more target-specific Cargo metadata documents.

    Application archives can contain both the native workspace and the
    independently resolved wasm UI workspace. Cargo does not expose those two
    workspaces through one metadata invocation, so the SBOM contract merges
    their resolved graphs before component generation.
    """
    documents = value if isinstance(value, list) else [value]
    if not documents or len(documents) > 8:
        raise SbomError("cargo metadata document count is outside the supported bound")

    workspace_members: set[str] = set()
    packages: dict[str, dict[str, object]] = {}
    dependency_ids: dict[str, set[str]] = {}
    for document in documents:
        if not isinstance(document, dict):
            raise SbomError("cargo metadata must be a JSON object")
        raw_packages = document.get("packages")
        raw_workspace = document.get("workspace_members")
        resolve = document.get("resolve")
        if (
            not isinstance(raw_packages, list)
            or not isinstance(raw_workspace, list)
            or not isinstance(resolve, dict)
            or not isinstance(resolve.get("nodes"), list)
        ):
            raise SbomError(
                "cargo metadata is missing packages, workspace members, or resolve nodes"
            )
        for member in raw_workspace:
            workspace_members.add(_bounded_string(member, "workspace member"))
        for raw_package in raw_packages:
            if not isinstance(raw_package, dict):
                raise SbomError("cargo package entry must be an object")
            package_id = _bounded_string(raw_package.get("id"), "cargo package id")
            previous = packages.get(package_id)
            if previous is not None and any(
                previous.get(field) != raw_package.get(field)
                for field in ("name", "version", "license", "source")
            ):
                raise SbomError(f"conflicting cargo package metadata: {package_id}")
            packages[package_id] = raw_package
        for raw_node in resolve["nodes"]:
            if not isinstance(raw_node, dict):
                raise SbomError("cargo resolve node must be an object")
            package_id = _bounded_string(raw_node.get("id"), "cargo resolve id")
            raw_deps = raw_node.get("deps")
            if not isinstance(raw_deps, list):
                raise SbomError(f"cargo resolve node has no dependency list: {package_id}")
            merged_deps = dependency_ids.setdefault(package_id, set())
            for raw_dep in raw_deps:
                if not isinstance(raw_dep, dict):
                    raise SbomError(f"cargo dependency edge is malformed: {package_id}")
                merged_deps.add(
                    _bounded_string(raw_dep.get("pkg"), "cargo dependency id")
                )

    if not packages or len(packages) > MAX_COMPONENTS:
        raise SbomError("cargo metadata package count is outside the supported bound")
    if not dependency_ids or len(dependency_ids) > MAX_COMPONENTS:
        raise SbomError("resolved dependency count is outside the supported bound")
    if any(package_id not in packages for package_id in dependency_ids):
        raise SbomError("cargo resolve graph references a package without metadata")
    if not workspace_members or any(
        member not in dependency_ids for member in workspace_members
    ):
        raise SbomError("cargo resolve graph omits a workspace member")
    if any(
        dependency_id not in dependency_ids
        for dependencies in dependency_ids.values()
        for dependency_id in dependencies
    ):
        raise SbomError("cargo dependency edge is unresolved")

    return {
        "workspace_members": sorted(workspace_members),
        "packages": [packages[package_id] for package_id in sorted(packages)],
        "resolve": {
            "nodes": [
                {
                    "id": package_id,
                    "deps": [
                        {"pkg": dependency_id}
                        for dependency_id in sorted(dependency_ids[package_id])
                    ],
                }
                for package_id in sorted(dependency_ids)
            ]
        },
    }


def build_sbom(
    metadata: object,
    policy_value: object,
    target: str,
    embedded_ui: bool,
) -> dict[str, object]:
    policy = validate_policy(policy_value)
    metadata = merge_cargo_metadata(metadata)
    if not TARGET_RE.fullmatch(target):
        raise SbomError("target is not a bounded Rust target triple")
    if not isinstance(embedded_ui, bool):
        raise SbomError("embedded_ui must be a boolean")

    raw_packages = metadata.get("packages")
    raw_workspace = metadata.get("workspace_members")
    resolve = metadata.get("resolve")
    if (
        not isinstance(raw_packages, list)
        or not isinstance(raw_workspace, list)
        or not isinstance(resolve, dict)
        or not isinstance(resolve.get("nodes"), list)
    ):
        raise SbomError("cargo metadata is missing packages, workspace members, or resolve nodes")
    if not raw_packages or len(raw_packages) > MAX_COMPONENTS:
        raise SbomError("cargo metadata package count is outside the supported bound")

    workspace_members = {
        _bounded_string(member, "workspace member") for member in raw_workspace
    }
    packages: dict[str, dict[str, object]] = {}
    for raw_package in raw_packages:
        if not isinstance(raw_package, dict):
            raise SbomError("cargo package entry must be an object")
        package_id = _bounded_string(raw_package.get("id"), "cargo package id")
        if package_id in packages:
            raise SbomError(f"duplicate cargo package id: {package_id}")
        packages[package_id] = raw_package

    nodes: dict[str, dict[str, object]] = {}
    for raw_node in resolve["nodes"]:
        if not isinstance(raw_node, dict):
            raise SbomError("cargo resolve node must be an object")
        package_id = _bounded_string(raw_node.get("id"), "cargo resolve id")
        if package_id not in packages or package_id in nodes:
            raise SbomError(f"cargo resolve node is missing or duplicated: {package_id}")
        nodes[package_id] = raw_node
    if not nodes or len(nodes) > MAX_COMPONENTS:
        raise SbomError("resolved dependency count is outside the supported bound")

    allowed_sources = _string_set(
        policy["allowed_registry_sources"], "allowed_registry_sources"
    )
    allowed_licenses = _string_set(
        policy["allowed_license_expressions"], "allowed_license_expressions"
    )
    license_normalizations = policy["license_expression_normalizations"]
    assert isinstance(license_normalizations, dict)
    components: list[dict[str, object]] = []
    refs_by_id: dict[str, str] = {}
    workspace_versions: set[str] = set()
    for package_id in sorted(nodes):
        package = packages[package_id]
        name = _bounded_string(package.get("name"), f"{package_id} name")
        version = _bounded_string(package.get("version"), f"{package_id} version")
        declared_license = _bounded_string(
            package.get("license"), f"{name} {version} license"
        )
        if declared_license not in allowed_licenses:
            raise SbomError(
                f"dependency has an unreviewed license expression: "
                f"{name} {version}: {declared_license}"
            )
        license_expression = license_normalizations.get(declared_license, declared_license)
        assert isinstance(license_expression, str)
        source = _source_for_package(package, workspace_members, allowed_sources)
        if package_id in workspace_members:
            workspace_versions.add(version)
        reference = _cargo_purl(name, version)
        if reference in refs_by_id.values():
            raise SbomError(f"dependency package URL is not unique: {reference}")
        refs_by_id[package_id] = reference
        components.append(
            {
                "type": "library",
                "bom-ref": reference,
                "name": name,
                "version": version,
                "licenses": [{"expression": license_expression}],
                "purl": reference,
                "properties": [
                    {"name": "bloom:declared_license", "value": declared_license},
                    {"name": "bloom:source", "value": source},
                ],
            }
        )
    if len(workspace_versions) != 1:
        raise SbomError("workspace packages do not share one Bloom version")
    bloom_version = next(iter(workspace_versions))

    dependencies: list[dict[str, object]] = []
    for package_id in sorted(nodes, key=lambda item: refs_by_id[item]):
        raw_deps = nodes[package_id].get("deps")
        if not isinstance(raw_deps, list):
            raise SbomError(f"cargo resolve node has no dependency list: {package_id}")
        dependency_refs: set[str] = set()
        for raw_dep in raw_deps:
            if not isinstance(raw_dep, dict):
                raise SbomError(f"cargo dependency edge is malformed: {package_id}")
            dependency_id = _bounded_string(raw_dep.get("pkg"), "cargo dependency id")
            if dependency_id not in refs_by_id:
                raise SbomError(f"cargo dependency edge is unresolved: {dependency_id}")
            dependency_refs.add(refs_by_id[dependency_id])
        dependencies.append(
            {"ref": refs_by_id[package_id], "dependsOn": sorted(dependency_refs)}
        )

    root_ref = (
        f"pkg:generic/bloom@{quote(bloom_version, safe='.-_~')}"
        f"?target={quote(target, safe='.-_~')}"
    )
    root_dependencies = sorted(
        refs_by_id[member] for member in workspace_members if member in refs_by_id
    )
    if not root_dependencies:
        raise SbomError("no workspace package is present in the resolved dependency graph")
    dependencies.insert(0, {"ref": root_ref, "dependsOn": root_dependencies})
    components.sort(key=lambda component: str(component["bom-ref"]))
    has_embedded_ui_component = any(
        component["name"] == "bloom-ui"
        and component["properties"][1]["value"] == "workspace"
        for component in components
    )
    if has_embedded_ui_component != embedded_ui:
        raise SbomError(
            "embedded-UI identity does not match the resolved bloom-ui workspace"
        )
    identity = json.dumps(
        {
            "target": target,
            "embedded_ui": embedded_ui,
            "components": components,
            "dependencies": dependencies,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    serial = uuid.uuid5(uuid.NAMESPACE_URL, f"https://bloom.local/sbom/{identity}")
    sbom = {
        "$schema": CYCLONEDX_SCHEMA,
        "bomFormat": "CycloneDX",
        "specVersion": "1.5",
        "serialNumber": f"urn:uuid:{serial}",
        "version": 1,
        "metadata": {
            "component": {
                "type": "application",
                "bom-ref": root_ref,
                "name": "Bloom",
                "version": bloom_version,
            },
            "properties": [
                {"name": "bloom:embedded_ui", "value": str(embedded_ui).lower()},
                {"name": "bloom:target", "value": target},
            ],
        },
        "components": components,
        "dependencies": dependencies,
    }
    validate_sbom_document(sbom, policy, bloom_version, target, embedded_ui)
    return sbom


def _exact_keys(value: object, keys: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise SbomError(f"{label} has unknown or missing fields")
    return value


def validate_sbom_document(
    value: object,
    policy_value: object,
    bloom_version: str,
    target: str,
    embedded_ui: bool,
) -> None:
    policy = validate_policy(policy_value)
    sbom = _exact_keys(
        value,
        {
            "$schema",
            "bomFormat",
            "specVersion",
            "serialNumber",
            "version",
            "metadata",
            "components",
            "dependencies",
        },
        "SBOM",
    )
    if (
        sbom["$schema"] != CYCLONEDX_SCHEMA
        or sbom["bomFormat"] != "CycloneDX"
        or sbom["specVersion"] != "1.5"
        or isinstance(sbom["version"], bool)
        or sbom["version"] != 1
    ):
        raise SbomError("SBOM has an unsupported CycloneDX identity")
    serial = _bounded_string(sbom["serialNumber"], "SBOM serial number")
    try:
        uuid.UUID(serial.removeprefix("urn:uuid:"))
    except (ValueError, AttributeError) as error:
        raise SbomError("SBOM serial number is not a UUID URN") from error
    if not serial.startswith("urn:uuid:"):
        raise SbomError("SBOM serial number is not a UUID URN")

    metadata = _exact_keys(sbom["metadata"], {"component", "properties"}, "SBOM metadata")
    root = _exact_keys(
        metadata["component"], {"type", "bom-ref", "name", "version"}, "SBOM root"
    )
    if root["type"] != "application" or root["name"] != "Bloom":
        raise SbomError("SBOM root component is not Bloom")
    if root["version"] != bloom_version:
        raise SbomError("SBOM Bloom version does not match the release")
    root_ref = _bounded_string(root["bom-ref"], "SBOM root reference")
    expected_root_ref = (
        f"pkg:generic/bloom@{quote(bloom_version, safe='.-_~')}"
        f"?target={quote(target, safe='.-_~')}"
    )
    if root_ref != expected_root_ref:
        raise SbomError("SBOM root reference does not match the release")
    expected_properties = [
        {"name": "bloom:embedded_ui", "value": str(embedded_ui).lower()},
        {"name": "bloom:target", "value": target},
    ]
    if metadata["properties"] != expected_properties:
        raise SbomError("SBOM target or embedded-UI metadata does not match the release")

    components = sbom["components"]
    if not isinstance(components, list) or not (1 <= len(components) <= MAX_COMPONENTS):
        raise SbomError("SBOM component count is outside the supported bound")
    allowed_sources = _string_set(
        policy["allowed_registry_sources"], "allowed_registry_sources"
    ) | {"workspace"}
    allowed_licenses = _string_set(
        policy["allowed_license_expressions"], "allowed_license_expressions"
    )
    license_normalizations = policy["license_expression_normalizations"]
    assert isinstance(license_normalizations, dict)
    component_refs: set[str] = set()
    has_embedded_ui_component = False
    last_ref = ""
    for index, raw_component in enumerate(components):
        component = _exact_keys(
            raw_component,
            {"type", "bom-ref", "name", "version", "licenses", "purl", "properties"},
            f"SBOM component {index}",
        )
        reference = _bounded_string(component["bom-ref"], "SBOM component reference")
        if (
            reference <= last_ref
            or component["purl"] != reference
            or not reference.startswith("pkg:cargo/")
        ):
            raise SbomError("SBOM component references must be unique sorted Cargo purls")
        last_ref = reference
        component_refs.add(reference)
        if component["type"] != "library":
            raise SbomError("SBOM dependency component must be a library")
        name = _bounded_string(component["name"], "SBOM component name")
        version = _bounded_string(component["version"], "SBOM component version")
        if reference != _cargo_purl(name, version):
            raise SbomError("SBOM component purl does not match its name and version")
        properties = component["properties"]
        if (
            not isinstance(properties, list)
            or len(properties) != 2
            or any(
                not isinstance(item, dict) or set(item) != {"name", "value"}
                for item in properties
            )
            or properties[0].get("name") != "bloom:declared_license"
            or properties[1].get("name") != "bloom:source"
        ):
            raise SbomError("SBOM component properties are incomplete")
        declared_license = _bounded_string(
            properties[0]["value"], "SBOM component declared license"
        )
        source = _bounded_string(properties[1]["value"], "SBOM component source")
        if declared_license not in allowed_licenses:
            raise SbomError("SBOM component has an unreviewed declared license")
        if source not in allowed_sources:
            raise SbomError("SBOM component has an unreviewed dependency source")
        if component["name"] == "bloom-ui" and source == "workspace":
            has_embedded_ui_component = True
        expected_license = license_normalizations.get(declared_license, declared_license)
        licenses = component["licenses"]
        if (
            not isinstance(licenses, list)
            or len(licenses) != 1
            or not isinstance(licenses[0], dict)
            or set(licenses[0]) != {"expression"}
            or licenses[0]["expression"] != expected_license
        ):
            raise SbomError("SBOM component has an unreviewed license expression")
    if has_embedded_ui_component != embedded_ui:
        raise SbomError("SBOM embedded-UI component does not match the release")

    dependencies = sbom["dependencies"]
    if not isinstance(dependencies, list) or len(dependencies) != len(component_refs) + 1:
        raise SbomError("SBOM dependency graph is incomplete")
    dependency_refs: set[str] = set()
    valid_refs = component_refs | {root_ref}
    for index, raw_dependency in enumerate(dependencies):
        dependency = _exact_keys(
            raw_dependency, {"ref", "dependsOn"}, f"SBOM dependency {index}"
        )
        reference = _bounded_string(dependency["ref"], "SBOM dependency reference")
        depends_on = dependency["dependsOn"]
        if reference not in valid_refs or reference in dependency_refs:
            raise SbomError("SBOM dependency graph has an unknown or duplicate node")
        if not isinstance(depends_on, list):
            raise SbomError("SBOM dependency graph has invalid edges")
        bounded_dependencies = [
            _bounded_string(item, "SBOM dependency edge") for item in depends_on
        ]
        if (
            bounded_dependencies != sorted(bounded_dependencies)
            or len(bounded_dependencies) != len(set(bounded_dependencies))
            or any(
                item not in component_refs or item == reference
                for item in bounded_dependencies
            )
        ):
            raise SbomError("SBOM dependency graph has invalid edges")
        dependency_refs.add(reference)
    if dependency_refs != valid_refs:
        raise SbomError("SBOM dependency graph does not cover every component")

    identity = json.dumps(
        {
            "target": target,
            "embedded_ui": embedded_ui,
            "components": components,
            "dependencies": dependencies,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    expected_serial = uuid.uuid5(
        uuid.NAMESPACE_URL, f"https://bloom.local/sbom/{identity}"
    )
    if serial != f"urn:uuid:{expected_serial}":
        raise SbomError("SBOM serial number does not match its deterministic contents")
