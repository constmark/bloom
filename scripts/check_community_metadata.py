#!/usr/bin/env python3
"""Validate Bloom's structured, privacy-safe GitHub issue intake."""

from __future__ import annotations

import argparse
import pathlib
import re
import sys
from typing import Any

try:
    import yaml
except ModuleNotFoundError as error:  # pragma: no cover - exercised by clean hosts
    raise SystemExit(
        "PyYAML is required; install requirements/schema-validation.txt"
    ) from error


MAX_TEMPLATE_BYTES = 64 * 1024
ID_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
HTTPS_RE = re.compile(r"^https://[^\s]{1,2040}$")
SECURITY_ADVISORY_URL = "https://github.com/constmark/bloom/security/advisories/new"
SUPPORT_MATRIX_URL = (
    "https://github.com/constmark/bloom/blob/main/docs/support-matrix.md"
)
REQUIRED_GUIDANCE = (
    "API keys",
    "Authorization headers",
    "local paths",
    "private model data",
    "prompts and responses",
    "security/advisories/new",
    "written in English",
    "Code of Conduct",
)
TOP_LEVEL_KEYS = {"name", "description", "title", "labels", "assignees", "body"}
FORM_CONTRACTS: dict[str, dict[str, Any]] = {
    "bug_report.yml": {
        "title": "[Bug]: ",
        "labels": ["bug"],
        "fields": {
            "surface": "dropdown",
            "current_behavior": "textarea",
            "expected_behavior": "textarea",
            "reproduction": "textarea",
            "environment": "textarea",
            "version": "input",
            "model_reference": "input",
            "logs": "textarea",
            "privacy": "checkboxes",
            "security": "checkboxes",
            "english": "checkboxes",
            "conduct": "checkboxes",
        },
        "optional": {"model_reference", "logs"},
    },
    "feature_request.yml": {
        "title": "[Feature]: ",
        "labels": ["enhancement"],
        "fields": {
            "surface": "dropdown",
            "problem": "textarea",
            "outcome": "textarea",
            "proposal": "textarea",
            "alternatives": "textarea",
            "acceptance": "textarea",
            "constraints": "textarea",
            "privacy": "checkboxes",
            "security": "checkboxes",
            "english": "checkboxes",
            "conduct": "checkboxes",
        },
        "optional": {"proposal", "alternatives", "constraints"},
    },
    "model_support.yml": {
        "title": "[Support]: ",
        "labels": ["enhancement"],
        "fields": {
            "request_type": "dropdown",
            "model_family": "input",
            "public_source": "input",
            "license": "input",
            "format": "input",
            "device": "input",
            "workflow": "textarea",
            "current_result": "textarea",
            "evidence": "checkboxes",
            "privacy": "checkboxes",
            "security": "checkboxes",
            "english": "checkboxes",
            "conduct": "checkboxes",
        },
        "optional": {"current_result"},
    },
}


class MetadataError(Exception):
    """Community metadata is not valid YAML or violates the local contract."""


class UniqueKeyLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects mappings with duplicate keys."""


def construct_unique_mapping(
    loader: UniqueKeyLoader, node: yaml.nodes.MappingNode, deep: bool = False
) -> dict[Any, Any]:
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in mapping
        except TypeError as error:
            raise MetadataError("YAML mapping keys must be scalar values") from error
        if duplicate:
            raise MetadataError(f"duplicate YAML key: {key!r}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, construct_unique_mapping
)


def load_yaml_text(text: str, label: str) -> object:
    try:
        return yaml.load(text, Loader=UniqueKeyLoader)
    except (yaml.YAMLError, MetadataError) as error:
        raise MetadataError(f"{label} is not valid duplicate-free YAML: {error}") from error


def bounded_string(value: object, label: str, limit: int = 4_096) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or len(value.encode("utf-8")) > limit
    ):
        raise MetadataError(f"{label} must be a non-empty string of at most {limit} bytes")
    return value


def exact_mapping(value: object, keys: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise MetadataError(f"{label} must contain exactly: {', '.join(sorted(keys))}")
    return value


def validate_attributes(element_type: str, attributes: object, label: str) -> list[str]:
    errors: list[str] = []
    allowed = {
        "markdown": {"value"},
        "input": {"label", "description", "placeholder", "value"},
        "textarea": {"label", "description", "placeholder", "value", "render"},
        "dropdown": {"label", "description", "options", "multiple", "default"},
        "checkboxes": {"label", "description", "options"},
    }[element_type]
    if not isinstance(attributes, dict) or not set(attributes).issubset(allowed):
        return [f"{label}.attributes contains unsupported or malformed keys"]
    required_key = "value" if element_type == "markdown" else "label"
    try:
        bounded_string(attributes.get(required_key), f"{label}.attributes.{required_key}")
    except MetadataError as error:
        errors.append(str(error))
    for key in ("description", "placeholder", "value"):
        if key in attributes and key != required_key:
            try:
                bounded_string(attributes[key], f"{label}.attributes.{key}")
            except MetadataError as error:
                errors.append(str(error))
    if "render" in attributes:
        render = attributes["render"]
        if not isinstance(render, str) or re.fullmatch(r"[A-Za-z0-9_.+-]{1,32}", render) is None:
            errors.append(f"{label}.attributes.render is invalid")
    if "multiple" in attributes and not isinstance(attributes["multiple"], bool):
        errors.append(f"{label}.attributes.multiple must be a boolean")
    if element_type in {"dropdown", "checkboxes"}:
        options = attributes.get("options")
        if not isinstance(options, list) or not (1 <= len(options) <= 20):
            return errors + [f"{label}.attributes.options must contain 1 through 20 items"]
        if element_type == "dropdown":
            if not all(isinstance(option, str) and option.strip() for option in options):
                errors.append(f"{label}.attributes.options contains an invalid choice")
            elif len(set(options)) != len(options):
                errors.append(f"{label}.attributes.options contains duplicate choices")
            if "default" in attributes:
                default = attributes["default"]
                if (
                    isinstance(default, bool)
                    or not isinstance(default, int)
                    or not (0 <= default < len(options))
                ):
                    errors.append(f"{label}.attributes.default is outside the choices")
        else:
            for index, option in enumerate(options):
                if not isinstance(option, dict) or set(option) != {"label", "required"}:
                    errors.append(f"{label}.attributes.options[{index}] is malformed")
                    continue
                try:
                    bounded_string(option["label"], f"{label}.attributes.options[{index}].label")
                except MetadataError as error:
                    errors.append(str(error))
                if option["required"] is not True:
                    errors.append(f"{label}.attributes.options[{index}] must be required")
    return errors


def validate_form(filename: str, document: object, raw_text: str) -> list[str]:
    errors: list[str] = []
    contract = FORM_CONTRACTS[filename]
    try:
        form = exact_mapping(document, TOP_LEVEL_KEYS, filename)
        bounded_string(form["name"], f"{filename}.name", 128)
        bounded_string(form["description"], f"{filename}.description", 256)
    except MetadataError as error:
        return [str(error)]
    if form["title"] != contract["title"]:
        errors.append(f"{filename}.title must be {contract['title']!r}")
    if form["labels"] != contract["labels"]:
        errors.append(f"{filename}.labels must be {contract['labels']!r}")
    if form["assignees"] != []:
        errors.append(f"{filename}.assignees must remain empty")
    body = form["body"]
    if not isinstance(body, list) or not (1 <= len(body) <= 24):
        return errors + [f"{filename}.body must contain 1 through 24 elements"]

    actual_fields: dict[str, str] = {}
    required_fields: set[str] = set()
    for index, raw_element in enumerate(body):
        label = f"{filename}.body[{index}]"
        if not isinstance(raw_element, dict):
            errors.append(f"{label} must be an object")
            continue
        element_type = raw_element.get("type")
        if element_type == "markdown":
            if set(raw_element) != {"type", "attributes"}:
                errors.append(f"{label} markdown element is malformed")
                continue
            errors.extend(validate_attributes("markdown", raw_element["attributes"], label))
            continue
        if element_type not in {"input", "textarea", "dropdown", "checkboxes"}:
            errors.append(f"{label}.type is unsupported")
            continue
        if set(raw_element) != {"type", "id", "attributes", "validations"}:
            errors.append(f"{label} interactive element is malformed")
            continue
        field_id = raw_element["id"]
        if not isinstance(field_id, str) or ID_RE.fullmatch(field_id) is None:
            errors.append(f"{label}.id is invalid")
            continue
        if field_id in actual_fields:
            errors.append(f"{filename} contains duplicate field id {field_id!r}")
        actual_fields[field_id] = element_type
        errors.extend(validate_attributes(element_type, raw_element["attributes"], label))
        validations = raw_element["validations"]
        if not isinstance(validations, dict) or set(validations) != {"required"}:
            errors.append(f"{label}.validations must contain exactly required")
        elif not isinstance(validations["required"], bool):
            errors.append(f"{label}.validations.required must be a boolean")
        elif validations["required"]:
            required_fields.add(field_id)

    if actual_fields != contract["fields"]:
        errors.append(f"{filename} field IDs or types do not match the maintained contract")
    expected_required = set(contract["fields"]) - set(contract["optional"])
    if required_fields != expected_required:
        errors.append(f"{filename} required fields do not match the maintained contract")
    for guidance in REQUIRED_GUIDANCE:
        if guidance not in raw_text:
            errors.append(f"{filename} omits required guidance: {guidance}")
    return errors


def validate_config(document: object) -> list[str]:
    errors: list[str] = []
    try:
        config = exact_mapping(
            document,
            {"blank_issues_enabled", "contact_links"},
            "config.yml",
        )
    except MetadataError as error:
        return [str(error)]
    if config["blank_issues_enabled"] is not False:
        errors.append("config.yml must disable public blank issues")
    links = config["contact_links"]
    if not isinstance(links, list) or not (1 <= len(links) <= 8):
        return errors + ["config.yml contact_links must contain 1 through 8 links"]
    urls: list[str] = []
    for index, raw_link in enumerate(links):
        label = f"config.yml.contact_links[{index}]"
        try:
            link = exact_mapping(raw_link, {"name", "url", "about"}, label)
            bounded_string(link["name"], f"{label}.name", 128)
            bounded_string(link["about"], f"{label}.about", 256)
        except MetadataError as error:
            errors.append(str(error))
            continue
        url = link["url"]
        if not isinstance(url, str) or HTTPS_RE.fullmatch(url) is None:
            errors.append(f"{label}.url must be a bounded HTTPS URL")
        else:
            urls.append(url)
    if len(set(urls)) != len(urls):
        errors.append("config.yml contact links must be unique")
    if SECURITY_ADVISORY_URL not in urls:
        errors.append("config.yml omits the private security advisory link")
    if SUPPORT_MATRIX_URL not in urls:
        errors.append("config.yml omits the support matrix link")
    return errors


def validate_repository(root: pathlib.Path) -> tuple[int, list[str]]:
    template_root = root / ".github" / "ISSUE_TEMPLATE"
    expected = {*FORM_CONTRACTS, "config.yml"}
    actual = {
        path.name
        for pattern in ("*.yml", "*.yaml", "*.md")
        for path in template_root.glob(pattern)
    }
    errors: list[str] = []
    form_names: set[str] = set()
    if actual != expected:
        errors.append(
            "issue template files must be exactly: " + ", ".join(sorted(expected))
        )
    for filename in sorted(expected & actual):
        path = template_root / filename
        try:
            raw = path.read_bytes()
        except OSError as error:
            errors.append(f"cannot read {path.relative_to(root)}: {error}")
            continue
        if not (1 <= len(raw) <= MAX_TEMPLATE_BYTES):
            errors.append(f"{path.relative_to(root)} is empty or exceeds 64 KiB")
            continue
        try:
            text = raw.decode("utf-8")
            document = load_yaml_text(text, filename)
        except (UnicodeDecodeError, MetadataError) as error:
            errors.append(str(error))
            continue
        if filename == "config.yml":
            errors.extend(validate_config(document))
        else:
            errors.extend(validate_form(filename, document, text))
            if isinstance(document, dict) and isinstance(document.get("name"), str):
                name = document["name"]
                if name in form_names:
                    errors.append(f"issue form name is duplicated: {name!r}")
                form_names.add(name)
    return len(FORM_CONTRACTS), errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=pathlib.Path,
        default=pathlib.Path(__file__).resolve().parents[1],
        help="Repository root to validate.",
    )
    args = parser.parse_args()
    if not args.root.is_dir():
        parser.error(f"repository root is not a directory: {args.root}")
    count, errors = validate_repository(args.root.resolve())
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        print("FAILED: community metadata contract is invalid", file=sys.stderr)
        return 1
    print(f"OK: validated {count} structured, privacy-safe issue forms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
