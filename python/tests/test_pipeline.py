import ctypes
import os
import subprocess
import sys
import threading
import time
import unittest
from unittest import mock
from pathlib import Path


PYTHON_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PYTHON_ROOT.parent
sys.path.insert(0, str(PYTHON_ROOT))

import bloom_sdk.pipeline as pipeline_module


class FakeFunction:
    def __init__(self, implementation):
        self.implementation = implementation
        self.argtypes = None
        self.restype = None

    def __call__(self, *args):
        return self.implementation(*args)


class FakeLibrary:
    def __init__(self):
        self.pipeline_storage = pipeline_module.BloomPipelineOpaque()
        self.pipeline_pointer = ctypes.pointer(self.pipeline_storage)
        self.output_buffer = ctypes.create_string_buffer(b'{"text":"ok"}')
        self.freed_pipelines = 0
        self.freed_strings = 0
        self.freed_buffers = 0
        self.freed_tokens = 0
        self.cancel_calls = 0
        self.stream_started = threading.Event()
        self.stream_release = threading.Event()
        self.stream_cancelled = threading.Event()
        self.block_stream = False

        self.bloom_abi_version = FakeFunction(lambda: 2)
        self.bloom_pipeline_load = FakeFunction(self._load)
        self.bloom_pipeline_load_v2 = FakeFunction(self._load)
        self.bloom_pipeline_free = FakeFunction(self._free_pipeline)
        self.bloom_pipeline_run = FakeFunction(self._run)
        self.bloom_pipeline_run_v2 = FakeFunction(self._run_v2)
        self.bloom_pipeline_run_stream = FakeFunction(self._run_stream)
        self.bloom_pipeline_run_stream_v2 = FakeFunction(self._run_stream_v2)
        self.bloom_string_free = FakeFunction(self._free_string)
        self.bloom_buffer_free = FakeFunction(self._free_buffer)
        self.bloom_cancellation_token_new = FakeFunction(self._new_token)
        self.bloom_cancellation_token_cancel = FakeFunction(self._cancel_token)
        self.bloom_cancellation_token_free = FakeFunction(self._free_token)

    def _load(self, *_args):
        return self.pipeline_pointer

    def _free_pipeline(self, _pipeline):
        self.freed_pipelines += 1

    def _run(self, *_args):
        return ctypes.cast(self.output_buffer, ctypes.c_void_p).value

    def _run_v2(
        self,
        _pipeline,
        _input_json,
        _params_json,
        output,
        _error_buffer,
        _error_buffer_len,
    ):
        output = ctypes.cast(
            output, ctypes.POINTER(pipeline_module.BloomOwnedBuffer)
        ).contents
        output.data = ctypes.cast(
            self.output_buffer, ctypes.POINTER(ctypes.c_uint8)
        )
        output.len = len(self.output_buffer.value)
        return 0

    def _run_stream(
        self,
        _pipeline,
        _input_json,
        _params_json,
        callback,
        _user_data,
        _error_buffer,
        _error_buffer_len,
    ):
        callback(None, b'{"TextDelta":"hello"}')
        self.stream_started.set()
        if self.block_stream:
            self.stream_release.wait(timeout=5)
        return 0

    def _run_stream_v2(
        self,
        _pipeline,
        _input_json,
        _params_json,
        callback,
        _user_data,
        _token,
        _error_buffer,
        _error_buffer_len,
    ):
        chunk = ctypes.create_string_buffer(b'{"TextDelta":"hello"}')
        callback(
            None,
            ctypes.cast(chunk, ctypes.POINTER(ctypes.c_uint8)),
            len(chunk.value),
        )
        self.stream_started.set()
        if self.block_stream:
            while not self.stream_release.wait(timeout=0.01):
                if self.stream_cancelled.is_set():
                    return pipeline_module.BLOOM_STATUS_CANCELLED
        return 0

    def _free_string(self, _pointer):
        self.freed_strings += 1

    def _free_buffer(self, output):
        output = ctypes.cast(
            output, ctypes.POINTER(pipeline_module.BloomOwnedBuffer)
        ).contents
        output.data = ctypes.POINTER(ctypes.c_uint8)()
        output.len = 0
        self.freed_buffers += 1

    def _new_token(self):
        self.stream_cancelled.clear()
        return ctypes.c_void_p(0x1234)

    def _cancel_token(self, _token):
        self.cancel_calls += 1
        self.stream_cancelled.set()
        return 0

    def _free_token(self, _token):
        self.freed_tokens += 1


class BloomPipelineTests(unittest.TestCase):
    def setUp(self):
        self.previous_lib = pipeline_module._lib
        self.fake_lib = pipeline_module._configure_lib(FakeLibrary())
        pipeline_module._lib = self.fake_lib

    def tearDown(self):
        self.fake_lib.stream_release.set()
        pipeline_module._lib = self.previous_lib

    def test_package_import_does_not_require_a_native_library(self):
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(PYTHON_ROOT)
        environment["BLOOM_FFI_LIB"] = str(
            REPOSITORY_ROOT / "definitely-missing-bloom-ffi-library"
        )
        result = subprocess.run(
            [sys.executable, "-c", "import bloom_sdk"],
            cwd=REPOSITORY_ROOT,
            env=environment,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_generate_decodes_and_frees_native_output(self):
        pipeline = pipeline_module.BloomPipeline(".", engine="mock")

        self.assertEqual(pipeline.generate("hello"), {"text": "ok"})
        self.assertEqual(self.fake_lib.freed_buffers, 1)
        self.assertEqual(self.fake_lib.freed_strings, 0)

        pipeline.close()
        pipeline.close()
        self.assertEqual(self.fake_lib.freed_pipelines, 1)

    def test_invalid_native_json_is_freed_and_wrapped(self):
        self.fake_lib.output_buffer = ctypes.create_string_buffer(b"not-json")
        pipeline = pipeline_module.BloomPipeline(".", engine="mock")

        with self.assertRaisesRegex(
            pipeline_module.BloomInferenceError, "invalid JSON"
        ):
            pipeline.generate("hello")
        self.assertEqual(self.fake_lib.freed_buffers, 1)
        pipeline.close()

    def test_generation_parameters_are_validated_before_native_calls(self):
        pipeline = pipeline_module.BloomPipeline(".", engine="mock")

        invalid_arguments = [
            {"max_tokens": -1},
            {"temperature": float("nan")},
            {"temperature": True},
            {"top_p": 0},
            {"top_p": 1.1},
            {"top_p": True},
            {"seed": -1},
        ]
        for arguments in invalid_arguments:
            with self.subTest(arguments=arguments):
                with self.assertRaises(ValueError):
                    pipeline.generate("hello", **arguments)
        with self.assertRaises(TypeError):
            pipeline.generate(["not", "a", "model", "input"])
        with self.assertRaises(ValueError):
            pipeline.generate({"Text": {"prompt": float("nan")}})
        pipeline.close()

    def test_constructor_rejects_nul_terminated_identifiers(self):
        for arguments in (
            {"model_path": "model\0hidden"},
            {"model_path": ".", "engine": "mock\0hidden"},
            {"model_path": ".", "device": "cpu\0hidden"},
        ):
            with self.subTest(arguments=arguments):
                with self.assertRaisesRegex(ValueError, "NUL"):
                    pipeline_module.BloomPipeline(**arguments)

    def test_revision_one_library_remains_supported(self):
        legacy_lib = FakeLibrary()
        for symbol in (
            "bloom_abi_version",
            "bloom_pipeline_load_v2",
            "bloom_pipeline_run_v2",
            "bloom_pipeline_run_stream_v2",
            "bloom_buffer_free",
            "bloom_cancellation_token_new",
            "bloom_cancellation_token_cancel",
            "bloom_cancellation_token_free",
        ):
            delattr(legacy_lib, symbol)
        pipeline_module._lib = pipeline_module._configure_lib(legacy_lib)

        pipeline = pipeline_module.BloomPipeline(".", engine="mock")
        self.assertFalse(pipeline._uses_v2)
        self.assertEqual(pipeline.generate("hello"), {"text": "ok"})
        self.assertEqual(legacy_lib.freed_strings, 1)
        pipeline.close()

    def test_declared_revision_two_requires_the_complete_symbol_set(self):
        partial_lib = FakeLibrary()
        delattr(partial_lib, "bloom_buffer_free")

        with self.assertRaisesRegex(
            RuntimeError, "declares ABI revision 2.*bloom_buffer_free"
        ):
            pipeline_module._configure_lib(partial_lib)

    def test_close_waits_for_active_stream_before_freeing_pipeline(self):
        # Legacy workers cannot be interrupted, but close must still wait for
        # the native call before releasing its handle.
        self.fake_lib._bloom_uses_v2 = False
        self.fake_lib.block_stream = True
        pipeline = pipeline_module.BloomPipeline(".", engine="mock")
        stream = pipeline.generate_stream("hello")

        self.assertEqual(next(stream), {"TextDelta": "hello"})
        self.assertTrue(self.fake_lib.stream_started.is_set())

        close_finished = threading.Event()

        def close_pipeline():
            pipeline.close()
            close_finished.set()

        close_thread = threading.Thread(target=close_pipeline)
        close_thread.start()
        time.sleep(0.05)
        self.assertFalse(close_finished.is_set())
        self.assertEqual(self.fake_lib.freed_pipelines, 0)

        self.fake_lib.stream_release.set()
        with self.assertRaises(StopIteration):
            next(stream)
        close_thread.join(timeout=2)
        self.assertTrue(close_finished.is_set())
        self.assertEqual(self.fake_lib.freed_pipelines, 1)

    def test_closing_a_stream_cancels_the_v2_native_worker(self):
        self.fake_lib.block_stream = True
        pipeline = pipeline_module.BloomPipeline(".", engine="mock")
        stream = pipeline.generate_stream("hello")

        self.assertEqual(next(stream), {"TextDelta": "hello"})
        self.assertTrue(self.fake_lib.stream_started.is_set())
        stream.close()

        self.assertTrue(self.fake_lib.stream_cancelled.wait(timeout=1))
        deadline = time.monotonic() + 1
        while self.fake_lib.freed_tokens == 0 and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertGreaterEqual(self.fake_lib.cancel_calls, 1)
        self.assertEqual(self.fake_lib.freed_tokens, 1)
        pipeline.close()

    def _install_burst(self, count, *, legacy=False, status=0):
        self.burst_blocked = threading.Event()
        self.burst_finished = threading.Event()
        self.burst_progress = []

        def emit(callback, v2):
            try:
                for index in range(count):
                    if index == 65:
                        self.burst_blocked.set()
                    value = ('{"TextDelta":"%d"}' % index).encode()
                    if v2:
                        chunk = ctypes.create_string_buffer(value)
                        callback(None, ctypes.cast(
                            chunk, ctypes.POINTER(ctypes.c_uint8)
                        ), len(value))
                    else:
                        callback(None, value)
                    self.burst_progress.append(index)
                    if v2 and self.fake_lib.stream_cancelled.is_set():
                        return pipeline_module.BLOOM_STATUS_CANCELLED
                return status
            finally:
                self.burst_finished.set()

        if legacy:
            self.fake_lib._bloom_uses_v2 = False
            self.fake_lib.bloom_pipeline_run_stream = FakeFunction(
                lambda _p, _i, _a, cb, *_rest: emit(cb, False)
            )
        else:
            self.fake_lib.bloom_pipeline_run_stream_v2 = FakeFunction(
                lambda _p, _i, _a, cb, *_rest: emit(cb, True)
            )

    def test_slow_stream_consumer_backpressures_without_losing_output(self):
        self._install_burst(200)
        pipeline = pipeline_module.BloomPipeline(".", engine="mock")
        stream = pipeline.generate_stream("hello")
        try:
            self.assertEqual(next(stream), {"TextDelta": "0"})
            self.assertTrue(self.burst_blocked.wait(timeout=1))
            self.assertFalse(self.burst_finished.wait(timeout=0.05))
            self.assertLessEqual(len(self.burst_progress), 65)
            self.assertEqual(list(stream), [
                {"TextDelta": str(index)} for index in range(1, 200)
            ])
        finally:
            stream.close()
            pipeline.close()
        self.assertEqual(self.fake_lib.freed_tokens, 1)

    def test_close_releases_backpressure_for_both_abi_revisions(self):
        for legacy in (False, True):
            with self.subTest(legacy=legacy):
                self._install_burst(200, legacy=legacy)
                pipeline = pipeline_module.BloomPipeline(".", engine="mock")
                stream = pipeline.generate_stream("hello")
                self.assertEqual(next(stream), {"TextDelta": "0"})
                self.assertTrue(self.burst_blocked.wait(timeout=1))
                closer = threading.Thread(target=pipeline.close)
                closer.start()
                try:
                    closer.join(timeout=2)
                    self.assertFalse(closer.is_alive())
                    self.assertTrue(self.burst_finished.is_set())
                    self.assertEqual(list(stream), [])
                finally:
                    stream.close()
                    closer.join(timeout=2)
                self.assertFalse(pipeline._stream_cancellations)

    def test_full_stream_delivers_terminal_native_error_after_buffered_chunks(self):
        self._install_burst(100, status=-4)
        with pipeline_module.BloomPipeline(".", engine="mock") as pipeline:
            stream = pipeline.generate_stream("hello")
            for index in range(100):
                self.assertEqual(next(stream), {"TextDelta": str(index)})
            with self.assertRaisesRegex(pipeline_module.BloomInferenceError, "code -4"):
                next(stream)
        self.assertEqual(self.fake_lib.freed_tokens, 1)

    def test_v2_oversized_chunk_is_rejected_before_copying_native_memory(self):
        def oversized(_p, _i, _a, callback, *_rest):
            byte = ctypes.c_uint8(1)
            callback(None, ctypes.pointer(byte), pipeline_module.MAX_STREAM_BYTES + 1)
            return 0

        self.fake_lib.bloom_pipeline_run_stream_v2 = FakeFunction(oversized)
        with pipeline_module.BloomPipeline(".", engine="mock") as pipeline:
            with mock.patch.object(ctypes, "string_at", side_effect=AssertionError("unsafe read")):
                with self.assertRaisesRegex(pipeline_module.BloomInferenceError, "16 MiB"):
                    list(pipeline.generate_stream("hello"))
        self.assertEqual(self.fake_lib.freed_tokens, 1)

    def test_invalid_stream_encoding_or_json_cancels_native_worker(self):
        for value in (b"\xff", b"not-json"):
            with self.subTest(value=value):
                def invalid(_p, _i, _a, callback, *_rest):
                    chunk = ctypes.create_string_buffer(value)
                    callback(None, ctypes.cast(chunk, ctypes.POINTER(ctypes.c_uint8)), len(value))
                    self.fake_lib.stream_cancelled.wait(timeout=2)
                    return pipeline_module.BLOOM_STATUS_CANCELLED

                self.fake_lib.bloom_pipeline_run_stream_v2 = FakeFunction(invalid)
                with pipeline_module.BloomPipeline(".", engine="mock") as pipeline:
                    with self.assertRaisesRegex(pipeline_module.BloomInferenceError, "Invalid streaming chunk"):
                        list(pipeline.generate_stream("hello"))
                self.assertTrue(self.fake_lib.stream_cancelled.is_set())

    def test_worker_start_failure_releases_token_and_registration(self):
        with pipeline_module.BloomPipeline(".", engine="mock") as pipeline:
            with mock.patch.object(threading.Thread, "start", side_effect=RuntimeError("no thread")):
                with self.assertRaisesRegex(RuntimeError, "no thread"):
                    next(pipeline.generate_stream("hello"))
            self.assertFalse(pipeline._stream_cancellations)
        self.assertEqual(self.fake_lib.freed_tokens, 1)

    def test_stream_after_pipeline_close_releases_token(self):
        pipeline = pipeline_module.BloomPipeline(".", engine="mock")
        pipeline.close()
        with self.assertRaisesRegex(pipeline_module.BloomError, "closed"):
            next(pipeline.generate_stream("hello"))
        self.assertEqual(self.fake_lib.freed_tokens, 1)


@unittest.skipUnless(
    os.environ.get("BLOOM_TEST_NATIVE_FFI") == "1",
    "set BLOOM_TEST_NATIVE_FFI=1 after building bloomai-ffi",
)
class NativeFfiIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.previous_lib = pipeline_module._lib
        pipeline_module._lib = None

    def tearDown(self):
        pipeline_module._lib = self.previous_lib

    def test_python_wrapper_crosses_the_native_mock_engine(self):
        with pipeline_module.BloomPipeline(".", engine="mock") as pipeline:
            output = pipeline.generate("hello", max_tokens=4)
            chunks = list(pipeline.generate_stream("hello", max_tokens=4))

        self.assertEqual(output["text"], "echo: hello")
        self.assertEqual(chunks[0], {"TextDelta": "echo: hello"})
        self.assertEqual(chunks[-1], "End")


if __name__ == "__main__":
    unittest.main()
