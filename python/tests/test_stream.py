import threading
import unittest
from unittest import mock

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bloom_sdk import _stream


class StreamBufferTests(unittest.TestCase):
    def test_byte_budget_blocks_until_consumed_and_preserves_order(self):
        with mock.patch.object(_stream, "MAX_STREAM_BYTES", 10):
            buffer = _stream.StreamBuffer()
            buffer.put(b"123456")
            started = threading.Event()
            finished = threading.Event()

            def produce():
                started.set()
                buffer.put(b"abcdef")
                buffer.finish()
                finished.set()

            worker = threading.Thread(target=produce)
            worker.start()
            try:
                self.assertTrue(started.wait(timeout=1))
                self.assertFalse(finished.wait(timeout=0.05))
                self.assertEqual(buffer.receive(), b"123456")
                self.assertTrue(finished.wait(timeout=1))
                self.assertEqual(buffer.receive(), b"abcdef")
                self.assertIsNone(buffer.receive())
            finally:
                buffer.stop()
                worker.join(timeout=1)

    def test_terminal_error_does_not_need_queue_capacity_or_get_overwritten(self):
        with mock.patch.object(_stream, "MAX_STREAM_CHUNKS", 1):
            buffer = _stream.StreamBuffer()
            buffer.put(b"first")
            buffer.finish(ValueError("first error"))
            buffer.finish()
            buffer.put(b"ignored after error")
            self.assertEqual(buffer.receive(), b"first")
            with self.assertRaisesRegex(ValueError, "first error"):
                buffer.receive()

    def test_stop_wakes_a_waiting_consumer(self):
        buffer = _stream.StreamBuffer()
        values = []
        worker = threading.Thread(target=lambda: values.append(buffer.receive()))
        worker.start()
        buffer.stop()
        worker.join(timeout=1)
        self.assertFalse(worker.is_alive())
        self.assertEqual(values, [None])

    def test_receive_timeout_does_not_wait_for_native_completion(self):
        buffer = _stream.StreamBuffer()
        with self.assertRaisesRegex(TimeoutError, "waiting for streaming output"):
            buffer.receive(timeout=0.01)
        buffer.stop()

    def test_receive_rejects_invalid_timeout(self):
        buffer = _stream.StreamBuffer()
        for timeout in (True, -1, float("nan"), float("inf"), "1"):
            with self.subTest(timeout=timeout):
                with self.assertRaises(ValueError):
                    buffer.receive(timeout=timeout)
        buffer.stop()


if __name__ == "__main__":
    unittest.main()
