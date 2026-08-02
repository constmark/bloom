#!/usr/bin/env python3
"""Validate a Bloom tar.gz or zip release archive without extracting it."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
import stat
import sys
import tarfile
import zipfile
from contextlib import contextmanager
from dataclasses import dataclass
from typing import BinaryIO, Iterator

from readiness_contract import (
    READINESS_SERVER_PROTOCOL_VERSION,
    ReadinessContractError,
    validate_readiness_document,
    validate_readiness_schema_document,
)

MAX_ENTRY_COUNT = 10_000
MAX_UNCOMPRESSED_BYTES = 2 * 1024 * 1024 * 1024
MAX_MANIFEST_BYTES = 1024 * 1024
MAX_READINESS_ARTIFACT_BYTES = 64 * 1024
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
TARGET_RE = re.compile(r"^[A-Za-z0-9_.-]{1,128}$")
EXPECTED_BINARIES = ("bloom_bench", "bloom_infer", "bloom_server", "inspect_gguf")
REQUIRED_FILES = (
    "BLOOM-RELEASE.json",
    "LICENSE",
    "QUICKSTART.md",
    "README.md",
    "RELEASE.md",
    "SECURITY.md",
    "docs/release-manifest.md",
    "docs/release-quickstart.md",
    "docs/release-server-quickstart.md",
    "docs/readiness-contract.md",
    "examples/readiness.json",
    "examples/readiness.schema.json",
    "examples/release-manifest.schema.json",
)


class ValidationError(Exception):
    """A release archive violates the public packaging contract."""


@dataclass(frozen=True)
class ArchiveEntry:
    name: str
    size: int
    is_dir: bool
    mode: int | None
    source: tarfile.TarInfo | zipfile.ZipInfo


def normalized_member_name(raw_name: str) -> str:
    if not raw_name or "\x00" in raw_name or "\\" in raw_name:
        raise ValidationError(f"unsafe archive member name: {raw_name!r}")
    trimmed = raw_name.rstrip("/")
    path = pathlib.PurePosixPath(trimmed)
    if (
        not trimmed
        or path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValidationError(f"unsafe archive member name: {raw_name!r}")
    return path.as_posix()


class ReleaseArchive:
    def __init__(self, path: pathlib.Path):
        self.path = path
        self.kind = ""
        self.gzip_mtime: int | None = None
        self.handle: tarfile.TarFile | zipfile.ZipFile
        self.entries: dict[str, ArchiveEntry] = {}

    def __enter__(self) -> "ReleaseArchive":
        if zipfile.is_zipfile(self.path):
            self.kind = "zip"
            self.handle = zipfile.ZipFile(self.path)
            raw_entries = self.handle.infolist()
            for info in raw_entries:
                mode = info.external_attr >> 16
                file_type = stat.S_IFMT(mode)
                if file_type not in {0, stat.S_IFREG, stat.S_IFDIR}:
                    raise ValidationError(
                        f"special entries and links are not allowed: {info.filename}"
                    )
                self._add_entry(
                    ArchiveEntry(
                        name=normalized_member_name(info.filename),
                        size=info.file_size,
                        is_dir=info.is_dir(),
                        mode=mode or None,
                        source=info,
                    )
                )
        else:
            with self.path.open("rb") as raw_archive:
                gzip_header = raw_archive.read(10)
            if len(gzip_header) == 10 and gzip_header[:2] == b"\x1f\x8b":
                self.gzip_mtime = int.from_bytes(gzip_header[4:8], byteorder="little")
            try:
                self.handle = tarfile.open(self.path, mode="r:gz")
            except tarfile.TarError as error:
                raise ValidationError(
                    "archive is neither a readable zip nor tar.gz file"
                ) from error
            self.kind = "tar.gz"
            raw_entries = self.handle.getmembers()
            for info in raw_entries:
                if not (info.isfile() or info.isdir()):
                    raise ValidationError(f"special entries and links are not allowed: {info.name}")
                self._add_entry(
                    ArchiveEntry(
                        name=normalized_member_name(info.name),
                        size=info.size,
                        is_dir=info.isdir(),
                        mode=info.mode,
                        source=info,
                    )
                )

        if len(self.entries) > MAX_ENTRY_COUNT:
            raise ValidationError(f"archive contains more than {MAX_ENTRY_COUNT} entries")
        total_size = sum(entry.size for entry in self.entries.values() if not entry.is_dir)
        if total_size > MAX_UNCOMPRESSED_BYTES:
            raise ValidationError("archive exceeds the uncompressed size limit")
        return self

    def validate_deterministic_metadata(self) -> None:
        """Require metadata emitted by create_release_archive.py."""
        if self.kind == "tar.gz":
            timestamps: set[int] = set()
            for entry in self.entries.values():
                assert isinstance(entry.source, tarfile.TarInfo)
                info = entry.source
                if (
                    isinstance(info.mtime, bool)
                    or not isinstance(info.mtime, (int, float))
                    or not float(info.mtime).is_integer()
                    or not (0 <= int(info.mtime) <= 0xFFFF_FFFF)
                ):
                    raise ValidationError(
                        "tar archive has a non-canonical entry timestamp"
                    )
                timestamps.add(int(info.mtime))
                if info.uid != 0 or info.gid != 0 or info.uname or info.gname:
                    raise ValidationError(
                        "tar archive has non-canonical ownership metadata"
                    )
                expected_modes = {0o755} if entry.is_dir else {0o644, 0o755}
                if stat.S_IMODE(info.mode) not in expected_modes:
                    raise ValidationError("tar archive has a non-canonical entry mode")
                unsupported_pax = set(info.pax_headers) - {"path"}
                if unsupported_pax:
                    raise ValidationError(
                        "tar archive has non-canonical extended metadata"
                    )
            if len(timestamps) != 1 or self.gzip_mtime not in timestamps:
                raise ValidationError(
                    "tar archive does not use one deterministic timestamp"
                )
            return

        timestamps: set[tuple[int, int, int, int, int, int]] = set()
        for entry in self.entries.values():
            assert isinstance(entry.source, zipfile.ZipInfo)
            info = entry.source
            timestamps.add(info.date_time)
            raw_mode = info.external_attr >> 16
            if info.create_system != 3 or stat.S_IFMT(raw_mode) != stat.S_IFREG:
                raise ValidationError("zip archive has non-canonical file metadata")
            if stat.S_IMODE(raw_mode) not in {0o644, 0o755}:
                raise ValidationError("zip archive has a non-canonical entry mode")
        if len(timestamps) != 1:
            raise ValidationError("zip archive does not use one deterministic timestamp")

    def __exit__(self, *_args: object) -> None:
        self.handle.close()

    def _add_entry(self, entry: ArchiveEntry) -> None:
        if entry.name in self.entries:
            raise ValidationError(f"duplicate archive member: {entry.name}")
        self.entries[entry.name] = entry

    @contextmanager
    def open_file(self, name: str) -> Iterator[BinaryIO]:
        entry = self.entries.get(name)
        if entry is None or entry.is_dir:
            raise ValidationError(f"required archive file is missing: {name}")
        if self.kind == "zip":
            assert isinstance(self.handle, zipfile.ZipFile)
            assert isinstance(entry.source, zipfile.ZipInfo)
            with self.handle.open(entry.source, mode="r") as stream:
                yield stream
        else:
            assert isinstance(self.handle, tarfile.TarFile)
            assert isinstance(entry.source, tarfile.TarInfo)
            stream = self.handle.extractfile(entry.source)
            if stream is None:
                raise ValidationError(f"archive file cannot be read: {name}")
            with stream:
                yield stream


def require_exact_keys(value: object, expected: set[str], label: str) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != expected:
        raise ValidationError(f"{label} must contain exactly: {', '.join(sorted(expected))}")
    return value


def require_integer(value: object, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValidationError(f"{label} must be an integer of at least {minimum}")
    return value


def read_manifest(archive: ReleaseArchive, name: str) -> dict[str, object]:
    entry = archive.entries.get(name)
    if entry is None or entry.size > MAX_MANIFEST_BYTES:
        raise ValidationError("release manifest is missing or exceeds its size limit")
    with archive.open_file(name) as stream:
        raw = stream.read(MAX_MANIFEST_BYTES + 1)
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValidationError("release manifest is not valid UTF-8 JSON") from error
    return require_exact_keys(
        value,
        {
            "schema_version",
            "object",
            "bloom_version",
            "target",
            "embedded_ui",
            "self_check",
            "binaries",
        },
        "release manifest",
    )


def read_json_artifact(
    archive: ReleaseArchive, name: str, maximum_bytes: int, label: str
) -> object:
    entry = archive.entries.get(name)
    if entry is None or entry.is_dir or entry.size == 0 or entry.size > maximum_bytes:
        raise ValidationError(f"{label} is missing, empty, or exceeds its size limit")
    with archive.open_file(name) as stream:
        raw = stream.read(maximum_bytes + 1)
    try:
        return json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValidationError(f"{label} is not valid UTF-8 JSON") from error


def stream_size_and_sha256(stream: BinaryIO) -> tuple[int, str]:
    size = 0
    digest = hashlib.sha256()
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        size += len(chunk)
        digest.update(chunk)
    return size, digest.hexdigest()


def validate_checksum(archive_path: pathlib.Path, checksum_path: pathlib.Path) -> None:
    if checksum_path.stat().st_size > MAX_MANIFEST_BYTES:
        raise ValidationError("checksum file exceeds its size limit")
    expected: str | None = None
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) >= 2 and pathlib.Path(fields[-1].lstrip("*")).name == archive_path.name:
            expected = fields[0].lower()
            break
    if expected is None or not SHA256_RE.fullmatch(expected):
        raise ValidationError(f"checksum file has no valid entry for {archive_path.name}")
    with archive_path.open("rb") as stream:
        _size, actual = stream_size_and_sha256(stream)
    if actual != expected:
        raise ValidationError("archive SHA-256 does not match the checksum file")


def validate_release(
    archive_path: pathlib.Path,
    checksum_path: pathlib.Path | None,
    require_embedded_ui: bool,
    require_native_self_check: bool,
    require_deterministic_metadata: bool = False,
) -> dict[str, object]:
    if checksum_path is not None:
        validate_checksum(archive_path, checksum_path)

    with ReleaseArchive(archive_path) as archive:
        if require_deterministic_metadata:
            archive.validate_deterministic_metadata()
        roots = {pathlib.PurePosixPath(name).parts[0] for name in archive.entries}
        if len(roots) != 1:
            raise ValidationError("archive must contain exactly one top-level directory")
        root = next(iter(roots))
        manifest = read_manifest(archive, f"{root}/BLOOM-RELEASE.json")

        schema_version = require_integer(manifest["schema_version"], "schema_version", 1)
        if schema_version != 1 or manifest["object"] != "bloom.release":
            raise ValidationError("unsupported release manifest identity")
        version = manifest["bloom_version"]
        if not isinstance(version, str) or not version or len(version) > 128:
            raise ValidationError("bloom_version must be a non-empty bounded string")
        target = manifest["target"]
        if not isinstance(target, str) or not TARGET_RE.fullmatch(target):
            raise ValidationError("target is not a valid Rust target triple")
        if root != f"bloom-{target}":
            raise ValidationError("top-level directory does not match the manifest target")
        embedded_ui = manifest["embedded_ui"]
        if not isinstance(embedded_ui, bool):
            raise ValidationError("embedded_ui must be a boolean")
        if require_embedded_ui and not embedded_ui:
            raise ValidationError("release does not contain the required embedded UI")

        self_check = require_exact_keys(
            manifest["self_check"], {"status", "doctor_status", "failures"}, "self_check"
        )
        status = self_check["status"]
        if status == "passed":
            failures = require_integer(self_check["failures"], "self_check.failures")
            doctor_status = self_check["doctor_status"]
            if not isinstance(doctor_status, str) or doctor_status not in {"pass", "warn"}:
                raise ValidationError("passed self-check has an invalid doctor status")
            if failures != 0:
                raise ValidationError("passed self-check has inconsistent doctor results")
        elif status == "not_run_cross_target":
            if self_check["doctor_status"] is not None or self_check["failures"] is not None:
                raise ValidationError("cross-target self-check must not claim doctor results")
        else:
            raise ValidationError("self_check.status is unsupported")
        if require_native_self_check and status != "passed":
            raise ValidationError("release did not pass a native executable self-check")

        windows_target = "windows" in target
        expected_names = [f"{name}.exe" if windows_target else name for name in EXPECTED_BINARIES]
        binaries = manifest["binaries"]
        if not isinstance(binaries, list) or len(binaries) != len(expected_names):
            raise ValidationError("release manifest must describe the complete executable set")
        actual_names: list[str] = []
        for index, raw_binary in enumerate(binaries):
            binary = require_exact_keys(
                raw_binary, {"name", "size_bytes", "sha256"}, f"binaries[{index}]"
            )
            name = binary["name"]
            if not isinstance(name, str) or pathlib.PurePosixPath(name).name != name:
                raise ValidationError(f"binaries[{index}].name is unsafe")
            actual_names.append(name)
            declared_size = require_integer(
                binary["size_bytes"], f"binaries[{index}].size_bytes", 1
            )
            declared_sha = binary["sha256"]
            if not isinstance(declared_sha, str) or not SHA256_RE.fullmatch(declared_sha):
                raise ValidationError(f"binaries[{index}].sha256 is invalid")
            archive_name = f"{root}/{name}"
            entry = archive.entries.get(archive_name)
            if entry is None or entry.is_dir:
                raise ValidationError(f"manifest binary is missing from the archive: {name}")
            if not windows_target and (entry.mode is None or entry.mode & 0o111 == 0):
                raise ValidationError(f"Unix release binary is not executable: {name}")
            with archive.open_file(archive_name) as stream:
                actual_size, actual_sha = stream_size_and_sha256(stream)
            if actual_size != declared_size or entry.size != declared_size:
                raise ValidationError(f"binary size does not match the manifest: {name}")
            if actual_sha != declared_sha:
                raise ValidationError(f"binary SHA-256 does not match the manifest: {name}")

        if actual_names != sorted(expected_names):
            raise ValidationError("release binaries are incomplete, duplicated, or not sorted")
        for relative_name in REQUIRED_FILES:
            entry = archive.entries.get(f"{root}/{relative_name}")
            if entry is None or entry.is_dir or entry.size == 0:
                raise ValidationError(f"required release file is missing or empty: {relative_name}")

        readiness_name = f"{root}/examples/readiness.json"
        readiness = read_json_artifact(
            archive,
            readiness_name,
            MAX_READINESS_ARTIFACT_BYTES,
            "packaged readiness example",
        )
        readiness_schema = read_json_artifact(
            archive,
            f"{root}/examples/readiness.schema.json",
            MAX_READINESS_ARTIFACT_BYTES,
            "packaged readiness schema",
        )
        try:
            validate_readiness_document(
                readiness,
                expected_server_protocol_version=READINESS_SERVER_PROTOCOL_VERSION,
            )
            validate_readiness_schema_document(readiness_schema)
        except ReadinessContractError as error:
            raise ValidationError(f"packaged readiness contract is invalid: {error}") from error

    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=pathlib.Path, help="Bloom .tar.gz or .zip archive")
    parser.add_argument("--checksum", type=pathlib.Path, help="SHA-256 file to verify first")
    parser.add_argument("--require-embedded-ui", action="store_true")
    parser.add_argument("--require-native-self-check", action="store_true")
    parser.add_argument("--require-deterministic-metadata", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        manifest = validate_release(
            args.archive,
            args.checksum,
            args.require_embedded_ui,
            args.require_native_self_check,
            args.require_deterministic_metadata,
        )
    except Exception as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 1
    print(
        f"valid Bloom release: {args.archive} "
        f"(version {manifest['bloom_version']}, target {manifest['target']}, "
        f"embedded_ui={str(manifest['embedded_ui']).lower()}, "
        f"self_check={manifest['self_check']['status']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
