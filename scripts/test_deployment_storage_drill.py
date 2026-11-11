#!/usr/bin/env python3
"""Safety regressions for the physical-storage drill; no mounts or disk filling."""
import errno
import io
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest import mock

import deployment_storage_drill as drill


class StorageDrillSafetyTests(unittest.TestCase):
    def test_refuses_host_filesystem_before_writing(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            mount = root / "mount"
            mount.mkdir()
            with self.assertRaisesRegex(RuntimeError, "host filesystem"):
                drill.fill_volume(mount, root)
            self.assertEqual(list(mount.iterdir()), [])

    def test_refuses_symlink_mount(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            mount = root / "mount"
            mount.symlink_to(root, target_is_directory=True)
            with self.assertRaisesRegex(RuntimeError, "real directory"):
                drill.fill_volume(mount, root)
            self.assertFalse((root / "drill-fill.bin").exists())

    def test_refuses_large_distinct_filesystem(self):
        mount, host = mock.Mock(), mock.Mock()
        mount.is_symlink.return_value = False
        mount.is_dir.return_value = True
        mount.stat.return_value.st_dev = 10
        host.stat.return_value.st_dev = 11
        with mock.patch.object(drill.os.path, "ismount", return_value=True), \
                mock.patch.object(drill.os, "statvfs", return_value=types.SimpleNamespace(
                    f_blocks=drill.MAX_FILESYSTEM_BYTES + 1, f_frsize=1)):
            with self.assertRaisesRegex(RuntimeError, "capacity bound"):
                drill.validate_volume(mount, host)

    def test_unsupported_platform_never_invokes_mount_commands(self):
        with mock.patch.object(drill.platform, "system", return_value="Linux"), \
                mock.patch.object(drill, "command") as command:
            with self.assertRaisesRegex(RuntimeError, "fallback is forbidden"):
                with drill.isolated_volume(Path("/unused")):
                    self.fail("unsupported host entered the mount context")
            command.assert_not_called()

    def test_insufficient_host_headroom_never_creates_image(self):
        with mock.patch.object(drill.platform, "system", return_value="Darwin"), \
                mock.patch.object(drill.shutil, "disk_usage", return_value=types.SimpleNamespace(free=1)), \
                mock.patch.object(drill, "command") as command:
            with self.assertRaisesRegex(RuntimeError, "headroom"):
                with drill.isolated_volume(Path("/unused")):
                    self.fail("unsafe host entered the mount context")
            command.assert_not_called()

    def test_never_exceeds_bounded_filler_when_filesystem_does_not_fill(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            with mock.patch.object(drill, "validate_volume", return_value=17):
                with self.assertRaisesRegex(RuntimeError, "capacity bound"):
                    drill.fill_volume(root, root)
            self.assertEqual((root / "drill-fill.bin").stat().st_size, 18)

    def test_small_free_extents_are_exhausted_after_large_write_failure(self):
        output = mock.MagicMock()
        output.__enter__.return_value = output
        output.fileno.return_value = 8
        attempts = []
        def write(data):
            attempts.append(len(data))
            if len(data) > 1 or len(attempts) > 6:
                raise OSError(errno.ENOSPC, "isolated fixture is full")
            return len(data)
        output.write.side_effect = write
        mount = mock.MagicMock()
        filler = mock.Mock()
        filler.open.return_value = output
        mount.__truediv__ = mock.Mock(return_value=filler)
        with mock.patch.object(drill, "validate_volume", return_value=2 * 1024 * 1024), \
                mock.patch.object(drill.os, "fsync"):
            _, written = drill.fill_volume(mount, Path("/unused"))
        self.assertEqual(attempts[:6], [1048576, 65536, 4096, 256, 16, 1])
        self.assertEqual(written, 1)

    def test_atomic_slot_replacement_and_unknown_slot_rejection(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            drill.activate(root, "previous")
            self.assertEqual(os.readlink(root / "current"), "previous")
            drill.activate(root, "candidate")
            self.assertEqual(os.readlink(root / "current"), "candidate")
            self.assertFalse((root / "next").exists())
            with self.assertRaisesRegex(ValueError, "unknown owned release slot"):
                drill.activate(root, "../../other")
            self.assertEqual(os.readlink(root / "current"), "candidate")


if __name__ == "__main__":
    unittest.main()
