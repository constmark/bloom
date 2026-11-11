#!/usr/bin/env python3
"""Process-level regressions for durable soak ownership and cancellation."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import subprocess
import tempfile
import time
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_deployment_soak as soak
import run_with_timeout as deadline


class SoakTests(unittest.TestCase):
    def test_darwin_zombie_group_permission_race_preserves_cleanup(self):
        with mock.patch.object(deadline.sys, "platform", "darwin"), \
             mock.patch.object(deadline.os, "killpg", create=True,
                               side_effect=[PermissionError(1, "exiting group"),
                                            ProcessLookupError(3, "reaped group")]) as send, \
             mock.patch.object(deadline.time, "sleep") as pause:
            deadline._signal_group(12345, 9)
            self.assertEqual(send.call_count, 2)
            pause.assert_called_once_with(.01)

    def test_persistent_signal_permission_error_is_not_suppressed(self):
        with mock.patch.object(deadline.sys, "platform", "darwin"), \
             mock.patch.object(deadline.os, "killpg", create=True,
                               side_effect=PermissionError(1, "not permitted")), \
             mock.patch.object(deadline.time, "monotonic", side_effect=[0, 1]), \
             mock.patch.object(deadline.time, "sleep") as pause:
            with self.assertRaises(PermissionError):
                deadline._signal_group(12345, 9)
            pause.assert_not_called()

    @unittest.skipUnless(os.name == "posix", "POSIX parent reaping")
    def test_guard_reaps_phase_after_runner_is_killed(self):
        with tempfile.TemporaryDirectory() as raw:
            marker = Path(raw) / "phase-pid"
            phase = ("import os,time,pathlib; pathlib.Path(" + repr(str(marker))
                     + ").write_text(str(os.getpid())); time.sleep(30)")
            owner_code = (
                "import subprocess,sys,os,time; subprocess.Popen([sys.executable,"
                + repr(str(Path(deadline.__file__)))
                + ",'--parent-pid',str(os.getpid()),'20',sys.executable,'-c',"
                + repr(phase) + "]); time.sleep(30)"
            )
            owner = subprocess.Popen([sys.executable, "-c", owner_code])
            try:
                limit = time.monotonic() + 5
                while not marker.exists() and time.monotonic() < limit:
                    time.sleep(.01)
                self.assertTrue(marker.exists())
                phase_pid = int(marker.read_text())
                owner.kill()
                owner.wait(timeout=5)
                limit = time.monotonic() + 5
                while time.monotonic() < limit:
                    try:
                        os.kill(phase_pid, 0)
                    except ProcessLookupError:
                        break
                    time.sleep(.05)
                else:
                    self.fail("orphan phase survived its owner")
            finally:
                if owner.poll() is None:
                    owner.kill()
                    owner.wait(timeout=5)

    @unittest.skipUnless(os.name == "posix", "POSIX process-group cleanup")
    def test_completed_command_does_not_leave_a_worker_or_output_pipe(self):
        # The command exits normally/nonzero after spawning a worker which
        # keeps stdout open. communicate() must still see EOF immediately.
        for status in (0, 7):
            with self.subTest(status=status):
                command = ("import subprocess,sys; "
                           "subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)']); "
                           f"raise SystemExit({status})")
                wrapper = subprocess.Popen(
                    [sys.executable, str(Path(deadline.__file__)), "20", sys.executable, "-c", command],
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                try:
                    stdout, stderr = wrapper.communicate(timeout=5)
                    self.assertEqual(wrapper.returncode, status,
                                     f"stdout={stdout!r}; stderr={stderr!r}")
                finally:
                    if wrapper.poll() is None:
                        wrapper.terminate()
                        wrapper.wait(timeout=5)

    @unittest.skipUnless(os.name == "posix", "POSIX process-group cleanup")
    def test_terminating_deadline_wrapper_reaps_its_child(self):
        with tempfile.TemporaryDirectory() as raw:
            marker = Path(raw) / "pid"
            child_code = "import os,time,pathlib; pathlib.Path(" + repr(str(marker)) + ").write_text(str(os.getpid())); time.sleep(30)"
            wrapper = subprocess.Popen([sys.executable, str(Path(deadline.__file__)), "20",
                                        sys.executable, "-c", child_code])
            try:
                limit = time.monotonic() + 5
                while not marker.exists() and time.monotonic() < limit:
                    time.sleep(.01)
                self.assertTrue(marker.exists())
                child_pid = int(marker.read_text())
                wrapper.terminate()
                self.assertEqual(wrapper.wait(timeout=5), 143)
                with self.assertRaises(ProcessLookupError):
                    os.kill(child_pid, 0)
            finally:
                if wrapper.poll() is None:
                    wrapper.kill()
                    wrapper.wait()

    def test_timeout_reaps_the_process_and_returns_124(self):
        started = time.monotonic()
        self.assertEqual(deadline.run([sys.executable, "-c", "import time; time.sleep(20)"], .1), 124)
        self.assertLess(time.monotonic() - started, 5)

    def test_child_exit_status_is_not_hidden(self):
        self.assertEqual(deadline.run([sys.executable, "-c", "raise SystemExit(7)"], 2), 7)

    def test_cancellation_stops_owned_process(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            (root / "cancel").touch()
            started = time.monotonic()
            with self.assertRaises(InterruptedError):
                soak.execute([sys.executable, "-c", "import time; time.sleep(20)"],
                             root, "cancel-test", 30, os.environ.copy())
            self.assertLess(time.monotonic() - started, 5)

    def test_pending_cancellation_does_not_launch_another_phase(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            (root / "cancel").touch()
            with mock.patch.object(soak.subprocess, "Popen") as launch:
                with self.assertRaises(InterruptedError):
                    soak.execute(["unused"], root, "cancelled", 30, {})
                launch.assert_not_called()

    def test_cancellation_at_child_exit_is_not_reported_as_success(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            child = mock.Mock(returncode=0)
            def completed():
                (root / "cancel").touch()
                return 0
            child.poll.side_effect = completed
            with mock.patch.object(soak.subprocess, "Popen", return_value=child), \
                 mock.patch.object(soak, "terminate") as stop:
                with self.assertRaises(InterruptedError):
                    soak.execute(["unused"], root, "cancelled", 30, {})
            stop.assert_called_once_with(child, grace=4)

    def test_budget_changes_cannot_silently_relax_the_acceptance_limits(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            path = root / "budgets.json"
            path.write_text('{"latency_p95_ms": 10}\n')
            state = {"budgets_sha256": soak.sha256_file(path)}
            soak.verify_budgets(root, state)
            path.write_text('{"latency_p95_ms": 100000}\n')
            with self.assertRaisesRegex(RuntimeError, "budgets changed"):
                soak.verify_budgets(root, state)

    def test_deadline_cannot_be_disabled_by_nonfinite_values(self):
        for value in (0, -1, float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                deadline.run([sys.executable, "-c", "pass"], value)

    def test_failed_command_is_a_failed_phase(self):
        with tempfile.TemporaryDirectory() as raw:
            with self.assertRaisesRegex(RuntimeError, "exit 7"):
                soak.execute([sys.executable, "-c", "raise SystemExit(7)"],
                             Path(raw), "failure", 30, os.environ.copy())

    def test_missing_or_invalid_budgets_do_not_launch_a_worker(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            model = root / "model"
            model.write_bytes(b"model")
            budget_path = root / "budgets.json"
            args = argparse.Namespace(server_bin=Path(sys.executable), model=model,
                expected_model_sha256=soak.model_identity(model)["sha256"],
                budgets=budget_path, run_dir=root / "run", smoke_seconds=None, threads=2)
            budgets = dict(cold_start_ms=10, latency_p95_ms=10, first_content_p95_ms=10,
                           recovery_ms=10, peak_rss_bytes=10, rss_growth_bytes=0)
            for bad in ({}, {**budgets, "recovery_ms": float("inf")},
                        {**budgets, "rss_growth_bytes": True}):
                budget_path.write_text(json.dumps(bad))
                with mock.patch.object(soak.subprocess, "Popen") as start:
                    with self.assertRaises(ValueError):
                        soak.start(args)
                    start.assert_not_called()
                    self.assertFalse(args.run_dir.exists())


if __name__ == "__main__":
    unittest.main()
