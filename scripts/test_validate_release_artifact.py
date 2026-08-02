#!/usr/bin/env python3
"""Regression tests for the standard-library release archive validator."""

from __future__ import annotations

import hashlib
import json
import pathlib
import sys
import tarfile
import tempfile
import unittest
import zipfile


sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from validate_release_artifact import (  # noqa: E402
    EXPECTED_BINARIES,
    REQUIRED_FILES,
    ValidationError,
    validate_release,
)
from create_release_archive import create_archive  # noqa: E402

REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[1]


class ReleaseArtifactValidatorTests(unittest.TestCase):
    def make_package(
        self,
        parent: pathlib.Path,
        target: str,
        *,
        embedded_ui: bool = True,
        native_self_check: bool = True,
        schema_version: object = 1,
    ) -> pathlib.Path:
        root = parent / f"bloom-{target}"
        root.mkdir()
        for relative_name in REQUIRED_FILES:
            if relative_name == "BLOOM-RELEASE.json":
                continue
            path = root / relative_name
            path.parent.mkdir(parents=True, exist_ok=True)
            if relative_name in {
                "examples/readiness.json",
                "examples/readiness.schema.json",
            }:
                path.write_bytes((REPOSITORY_ROOT / relative_name).read_bytes())
            else:
                path.write_text(f"fixture for {relative_name}\n", encoding="utf-8")

        windows = "windows" in target
        binaries = []
        for base_name in EXPECTED_BINARIES:
            name = f"{base_name}.exe" if windows else base_name
            payload = f"Bloom test executable: {name}\n".encode()
            path = root / name
            path.write_bytes(payload)
            if not windows:
                path.chmod(0o755)
            binaries.append(
                {
                    "name": name,
                    "size_bytes": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
            )

        manifest = {
            "schema_version": schema_version,
            "object": "bloom.release",
            "bloom_version": "0.1.0-test",
            "target": target,
            "embedded_ui": embedded_ui,
            "self_check": (
                {
                    "status": "passed",
                    "doctor_status": "warn",
                    "failures": 0,
                }
                if native_self_check
                else {
                    "status": "not_run_cross_target",
                    "doctor_status": None,
                    "failures": None,
                }
            ),
            "binaries": sorted(binaries, key=lambda item: item["name"]),
        }
        (root / "BLOOM-RELEASE.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        return root

    @staticmethod
    def make_tar(root: pathlib.Path, destination: pathlib.Path) -> None:
        with tarfile.open(destination, mode="w:gz") as archive:
            archive.add(root, arcname=root.name)

    @staticmethod
    def make_zip(root: pathlib.Path, destination: pathlib.Path) -> None:
        with zipfile.ZipFile(
            destination,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=9,
        ) as archive:
            for path in sorted(root.rglob("*")):
                if path.is_file():
                    archive.write(
                        path,
                        (pathlib.Path(root.name) / path.relative_to(root)).as_posix(),
                    )

    @staticmethod
    def write_checksum(archive: pathlib.Path) -> pathlib.Path:
        checksum = archive.with_suffix(archive.suffix + ".sha256")
        checksum.write_text(
            f"{hashlib.sha256(archive.read_bytes()).hexdigest()}  {archive.name}\n",
            encoding="utf-8",
        )
        return checksum

    def test_accepts_native_unix_tar_and_windows_zip_contracts(self) -> None:
        with tempfile.TemporaryDirectory() as raw_temp:
            temp = pathlib.Path(raw_temp)
            unix_root = self.make_package(temp, "x86_64-unknown-linux-gnu")
            unix_archive = temp / "bloom-linux.tar.gz"
            self.make_tar(unix_root, unix_archive)
            unix_manifest = validate_release(
                unix_archive,
                self.write_checksum(unix_archive),
                require_embedded_ui=True,
                require_native_self_check=True,
            )
            self.assertEqual(unix_manifest["target"], "x86_64-unknown-linux-gnu")

        with tempfile.TemporaryDirectory() as raw_temp:
            temp = pathlib.Path(raw_temp)
            windows_root = self.make_package(temp, "x86_64-pc-windows-msvc")
            windows_archive = temp / "bloom-windows.zip"
            self.make_zip(windows_root, windows_archive)
            windows_manifest = validate_release(
                windows_archive,
                self.write_checksum(windows_archive),
                require_embedded_ui=True,
                require_native_self_check=True,
            )
            self.assertEqual(
                [item["name"] for item in windows_manifest["binaries"]],
                [f"{name}.exe" for name in EXPECTED_BINARIES],
            )

    def test_rejects_unsafe_zip_members(self) -> None:
        with tempfile.TemporaryDirectory() as raw_temp:
            temp = pathlib.Path(raw_temp)
            root = self.make_package(temp, "x86_64-pc-windows-msvc")
            archive_path = temp / "unsafe.zip"
            self.make_zip(root, archive_path)
            with zipfile.ZipFile(archive_path, mode="a") as archive:
                archive.writestr("../outside.txt", "not allowed")

            with self.assertRaisesRegex(ValidationError, "unsafe archive member"):
                validate_release(archive_path, None, False, False)

    def test_feature_and_native_self_check_requirements_are_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as raw_temp:
            temp = pathlib.Path(raw_temp)
            root = self.make_package(
                temp,
                "x86_64-pc-windows-msvc",
                embedded_ui=False,
                native_self_check=False,
            )
            archive_path = temp / "cross-target-server.zip"
            self.make_zip(root, archive_path)

            manifest = validate_release(archive_path, None, False, False)
            self.assertFalse(manifest["embedded_ui"])
            with self.assertRaisesRegex(ValidationError, "embedded UI"):
                validate_release(archive_path, None, True, False)
            with self.assertRaisesRegex(ValidationError, "native executable"):
                validate_release(archive_path, None, False, True)

    def test_rejects_checksum_mismatches_before_archive_contents(self) -> None:
        with tempfile.TemporaryDirectory() as raw_temp:
            temp = pathlib.Path(raw_temp)
            root = self.make_package(temp, "x86_64-pc-windows-msvc")
            archive_path = temp / "checksum.zip"
            self.make_zip(root, archive_path)
            checksum_path = archive_path.with_suffix(".zip.sha256")
            checksum_path.write_text(
                f"{'0' * 64}  {archive_path.name}\n", encoding="utf-8"
            )

            with self.assertRaisesRegex(ValidationError, "archive SHA-256"):
                validate_release(archive_path, checksum_path, False, False)

    def test_rejects_binary_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as raw_temp:
            temp = pathlib.Path(raw_temp)
            root = self.make_package(temp, "x86_64-unknown-linux-gnu")
            (root / "bloom_server").write_bytes(b"tampered executable\n")
            (root / "bloom_server").chmod(0o755)
            archive_path = temp / "tampered.tar.gz"
            self.make_tar(root, archive_path)

            with self.assertRaisesRegex(ValidationError, "binary size"):
                validate_release(archive_path, None, False, False)

    def test_rejects_boolean_schema_versions(self) -> None:
        with tempfile.TemporaryDirectory() as raw_temp:
            temp = pathlib.Path(raw_temp)
            root = self.make_package(
                temp, "x86_64-pc-windows-msvc", schema_version=True
            )
            archive_path = temp / "invalid-schema.zip"
            self.make_zip(root, archive_path)

            with self.assertRaisesRegex(ValidationError, "schema_version"):
                validate_release(archive_path, None, False, False)

    def test_rejects_stale_and_inconsistent_packaged_readiness_examples(self) -> None:
        mutations = (
            (
                "stale identity",
                lambda value: value.__setitem__("schema_version", 2),
                "readiness document identity",
            ),
            (
                "inverted protocol range",
                lambda value: value.update(
                    {
                        "minimum_ui_protocol_version": 4,
                        "maximum_ui_protocol_version": 3,
                    }
                ),
                "compatibility range",
            ),
            (
                "inconsistent ready state",
                lambda value: value.update({"status": "ready", "progress": 0}),
                "internally inconsistent",
            ),
        )
        for label, mutate, expected_error in mutations:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as raw_temp:
                temp = pathlib.Path(raw_temp)
                root = self.make_package(temp, "x86_64-unknown-linux-gnu")
                readiness_path = root / "examples/readiness.json"
                readiness = json.loads(readiness_path.read_text(encoding="utf-8"))
                mutate(readiness)
                readiness_path.write_text(
                    json.dumps(readiness, indent=2) + "\n", encoding="utf-8"
                )
                archive_path = temp / "invalid-readiness.tar.gz"
                self.make_tar(root, archive_path)
                with self.assertRaisesRegex(ValidationError, expected_error):
                    validate_release(archive_path, None, False, False)

    def test_rejects_a_stale_packaged_readiness_schema(self) -> None:
        with tempfile.TemporaryDirectory() as raw_temp:
            temp = pathlib.Path(raw_temp)
            root = self.make_package(temp, "x86_64-unknown-linux-gnu")
            schema_path = root / "examples/readiness.schema.json"
            schema = json.loads(schema_path.read_text(encoding="utf-8"))
            schema["properties"]["schema_version"] = {"const": 2}
            schema_path.write_text(json.dumps(schema, indent=2) + "\n", encoding="utf-8")
            archive_path = temp / "stale-readiness-schema.tar.gz"
            self.make_tar(root, archive_path)
            with self.assertRaisesRegex(ValidationError, "schema version identity"):
                validate_release(archive_path, None, False, False)

    def test_accepts_deterministic_tar_and_zip_metadata(self) -> None:
        cases = (
            ("x86_64-unknown-linux-gnu", "tar.gz"),
            ("x86_64-pc-windows-msvc", "zip"),
        )
        for target, extension in cases:
            with self.subTest(extension=extension), tempfile.TemporaryDirectory() as raw_temp:
                temp = pathlib.Path(raw_temp)
                root = self.make_package(temp, target)
                archive_path = temp / f"release.{extension}"
                create_archive(root, archive_path, 1_700_000_000)
                validate_release(
                    archive_path,
                    None,
                    False,
                    False,
                    require_deterministic_metadata=True,
                )

    def test_rejects_non_deterministic_archive_metadata_when_required(self) -> None:
        with tempfile.TemporaryDirectory() as raw_temp:
            temp = pathlib.Path(raw_temp)
            root = self.make_package(temp, "x86_64-unknown-linux-gnu")
            archive_path = temp / "legacy.tar.gz"
            self.make_tar(root, archive_path)
            with self.assertRaisesRegex(ValidationError, "archive"):
                validate_release(
                    archive_path,
                    None,
                    False,
                    False,
                    require_deterministic_metadata=True,
                )


if __name__ == "__main__":
    unittest.main()
