"""Bounded handoff from native callbacks to a Python stream consumer."""

from collections import deque
import threading


MAX_STREAM_CHUNKS = 64
MAX_STREAM_BYTES = 16 * 1024 * 1024


class StreamBuffer:
    """Backpressure by both chunk count and serialized bytes.

    Terminal state has its own slot so completion and errors never need space
    in a full queue. Stopping a consumer releases blocked native callbacks.
    """

    def __init__(self):
        self._condition = threading.Condition()
        self._chunks = deque()
        self._bytes = 0
        self._finished = False
        self._error = None
        self.stopped = threading.Event()

    def put(self, chunk: bytes):
        if len(chunk) > MAX_STREAM_BYTES:
            raise ValueError("Streaming chunk exceeds the 16 MiB limit")
        with self._condition:
            while not self.stopped.is_set() and not self._finished:
                if (len(self._chunks) < MAX_STREAM_CHUNKS
                        and self._bytes + len(chunk) <= MAX_STREAM_BYTES):
                    self._chunks.append(chunk)
                    self._bytes += len(chunk)
                    self._condition.notify_all()
                    return
                self._condition.wait()

    def receive(self):
        with self._condition:
            while True:
                if self.stopped.is_set():
                    return None
                if self._chunks:
                    chunk = self._chunks.popleft()
                    self._bytes -= len(chunk)
                    self._condition.notify_all()
                    return chunk
                if self._finished:
                    if self._error is not None:
                        raise self._error
                    return None
                self._condition.wait()

    def finish(self, error=None):
        with self._condition:
            if not self._finished:
                self._finished = True
                self._error = error
            self._condition.notify_all()

    def stop(self):
        with self._condition:
            self.stopped.set()
            self._chunks.clear()
            self._bytes = 0
            self._condition.notify_all()
