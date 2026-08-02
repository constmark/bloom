#!/usr/bin/env python3
"""Regression tests for deterministic Bloom release archives."""

from __future__ import annotations

import gzip
import os
import pathlib
import stat
import sys
import tarfile
import tempfile
import unittest
import zipfile
from unittest import mock


sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from create_release_archive import (  # noqa: E402
    ArchiveError,
    create_archive,
    parse_source_date_epoch,
)

SOURCE_EPOCH = 1_700_000_000


class DeterministicReleaseArchiveTests(unittest.TestCase):
    @staticmethod
    def make_package(parent: pathlib.Path) -> pathlib.Path:
        root = parent / "bloom-test-target"
        (root / "docs").mkdir(parents=True)
        (root / "README.md").write_text("Bloom release fixture\n", encoding="utf-8")
        executable = root / "bloom_server"
        executable.write_bytes(b"test executable\n")
        executable.chmod(0o755)
        (root / "docs" / "guide.md").write_text("# Guide\n", encoding="utf-8")
        return root

    @staticmethod
    def change_source_times(root: pathlib.Path, timestamp: int) -> None:
        for path in [*root.rglob("*"), root]:
            os.utime(path, (timestamp, timestamp), follow_symlinks=False)

    def test_tar_gz_is_byte_reproducible_and_metadata_normalized(self) -> None:
        with tempfile.TemporaryDirectory() as raw_temp:
            temp = pathlib.Path(raw_temp)
            root = self.make_package(temp)
            first = temp / "first.tar.gz"
            second = temp / "second.tar.gz"
            create_archive(root, first, SOURCE_EPOCH)
            self.change_source_times(root, SOURCE_EPOCH + 10_000)
            create_archive(root, second, SOURCE_EPOCH)

            self.assertEqual(first.read_bytes(), second.read_bytes())
            self.assertEqual(
                int.from_bytes(first.read_bytes()[4:8], byteorder="little"),
                SOURCE_EPOCH,
            )
            with gzip.open(first, mode="rb") as compressed:
                with tarfile.open(fileobj=compressed, mode="r:") as archive:
                    members = archive.getmembers()
            self.assertTrue(members)
            self.assertTrue(all(member.mtime == SOURCE_EPOCH for member in members))
            self.assertTrue(all(member.uid == 0 and member.gid == 0 for member in members))
            modes = {member.name: member.mode for member in members}
            self.assertEqual(modes["bloom-test-target/bloom_server"], 0o755)
            self.assertEqual(modes["bloom-test-target/README.md"], 0o644)

    def test_zip_is_byte_reproducible_and_metadata_normalized(self) -> None:
        with tempfile.TemporaryDirectory() as raw_temp:
            temp = pathlib.Path(raw_temp)
            root = self.make_package(temp)
            first = temp / "first.zip"
            second = temp / "second.zip"
            create_archive(root, first, SOURCE_EPOCH)
            self.change_source_times(root, SOURCE_EPOCH + 10_000)
            create_archive(root, second, SOURCE_EPOCH)

            self.assertEqual(first.read_bytes(), second.read_bytes())
            with zipfile.ZipFile(first) as archive:
                entries = {info.filename: info for info in archive.infolist()}
            executable = entries["bloom-test-target/bloom_server"]
            readme = entries["bloom-test-target/README.md"]
            self.assertEqual(stat.S_IMODE(executable.external_attr >> 16), 0o755)
            self.assertEqual(stat.S_IMODE(readme.external_attr >> 16), 0o644)
            self.assertEqual(executable.date_time, (2023, 11, 14, 22, 13, 20))

    def test_rejects_symlinks_and_destinations_inside_the_package(self) -> None:
        with tempfile.TemporaryDirectory() as raw_temp:
            temp = pathlib.Path(raw_temp)
            root = self.make_package(temp)
            link = root / "unsafe-link"
            try:
                link.symlink_to("README.md")
            except (NotImplementedError, OSError):
                self.skipTest("symbolic links are not available on this host")
            with self.assertRaisesRegex(ArchiveError, "symbolic links"):
                create_archive(root, temp / "unsafe.tar.gz", SOURCE_EPOCH)
            link.unlink()
            with self.assertRaisesRegex(ArchiveError, "outside the package root"):
                create_archive(root, root / "unsafe.zip", SOURCE_EPOCH)

    def test_source_date_epoch_is_strict_and_bounded(self) -> None:
        self.assertEqual(parse_source_date_epoch("0"), 0)
        self.assertEqual(parse_source_date_epoch(str(SOURCE_EPOCH)), SOURCE_EPOCH)
        for invalid in ("", "-1", "+1", "01", "1.0", " 1", str(1 << 32)):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ArchiveError):
                    parse_source_date_epoch(invalid)

    def test_failed_creation_preserves_an_existing_destination(self) -> None:
        with tempfile.TemporaryDirectory() as raw_temp:
            temp = pathlib.Path(raw_temp)
            root = self.make_package(temp)
            destination = temp / "release.tar.gz"
            destination.write_bytes(b"existing archive")
            with mock.patch(
                "create_release_archive.create_tar_gz",
                side_effect=ArchiveError("injected archive failure"),
            ):
                with self.assertRaisesRegex(ArchiveError, "injected archive failure"):
                    create_archive(root, destination, SOURCE_EPOCH)
            self.assertEqual(destination.read_bytes(), b"existing archive")
            self.assertEqual(list(temp.glob(".release.tar.gz.*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
