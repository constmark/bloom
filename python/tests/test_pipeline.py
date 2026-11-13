import ctypes
import json
import os
import subprocess
import sys
import threading
import tempfile
import time
import unittest
from unittest import mock
from pathlib import Path


PYTHON_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PYTHON_ROOT.parent
sys.path.insert(0, str(PYTHON_ROOT))

import bloom_sdk.pipeline as pipeline_module


class BundledLibraryTests(unittest.TestCase):
    def test_bundled_runtime_precedes_workspace_and_system(self):
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory) / "bloom_sdk"
            package.mkdir()
            name = {"win32": "bloom_ffi.dll", "darwin": "libbloom_ffi.dylib"}.get(sys.platform, "libbloom_ffi.so")
            native = package / name
            native.write_bytes(b"test fixture")
            with mock.patch.object(pipeline_module, "__file__", str(package / "pipeline.py")), mock.patch.dict(os.environ, {}, clear=True), mock.patch.object(ctypes, "CDLL") as load:
                expected = object()
                load.return_value = expected
                self.assertIs(pipeline_module._find_lib(), expected)
                load.assert_called_once_with(str(native.resolve()))

    def test_broken_bundle_cannot_silently_fall_back(self):
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory) / "bloom_sdk"
            package.mkdir()
            name = {"win32": "bloom_ffi.dll", "darwin": "libbloom_ffi.dylib"}.get(sys.platform, "libbloom_ffi.so")
            (package / name).write_bytes(b"test fixture")
            with mock.patch.object(pipeline_module, "__file__", str(package / "pipeline.py")), mock.patch.dict(os.environ, {}, clear=True), mock.patch.object(ctypes, "CDLL", side_effect=OSError("bad dependency")) as load:
                with self.assertRaisesRegex(RuntimeError, "bundled Bloom"):
                    pipeline_module._find_lib()
                self.assertEqual(load.call_count, 1)

    def test_explicit_native_override_retains_precedence(self):
        with mock.patch.dict(os.environ, {"BLOOM_FFI_LIB": "/chosen/library"}), mock.patch.object(ctypes, "CDLL") as load:
            pipeline_module._find_lib()
            load.assert_called_once_with("/chosen/library")


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
        self.bloom_pipeline_tokenize_v2 = FakeFunction(self._tokenize_v2)
        self.bloom_pipeline_detokenize_v2 = FakeFunction(self._detokenize_v2)
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

    def _tokenize_v2(self, pipeline, text, output, error, error_len):
        text = ctypes.string_at(text.data, text.len).decode("utf-8")
        self.output_buffer = ctypes.create_string_buffer(json.dumps([ord(c) for c in text]).encode())
        return self._run_v2(pipeline, None, None, output, error, error_len)

    def _detokenize_v2(self, pipeline, tokens, output, error, error_len):
        tokens = json.loads(ctypes.string_at(tokens.data, tokens.len))
        self.output_buffer = ctypes.create_string_buffer(json.dumps("".join(chr(t) for t in tokens)).encode())
        return self._run_v2(pipeline, None, None, output, error, error_len)

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

    def test_exact_tokenization_round_trips_unicode_and_empty_input(self):
        with pipeline_module.BloomPipeline(".", engine="mock") as pipeline:
            for text in ("", "hello", "中文 🌸\0"):
                tokens = pipeline.tokenize(text)
                self.assertEqual(tokens, [ord(c) for c in text])
                self.assertEqual(pipeline.detokenize(tokens), text)
        self.assertEqual(self.fake_lib.freed_buffers, 6)

    def test_tokenization_validates_inputs_before_native_calls(self):
        with pipeline_module.BloomPipeline(".", engine="mock") as pipeline:
            for text in (1, None, b"hello"):
                with self.assertRaises(TypeError):
                    pipeline.tokenize(text)
            with self.assertRaises(ValueError):
                pipeline.tokenize("\ud800")
            for tokens in ([True], [-1], [1.5], [1 << 32], ["1"]):
                with self.assertRaises(ValueError):
                    pipeline.detokenize(tokens)
            for tokens in (None, "123", iter([1])):
                with self.assertRaises(TypeError):
                    pipeline.detokenize(tokens)
            with mock.patch.object(pipeline_module, "MAX_TEXT_BYTES", 3):
                with self.assertRaisesRegex(ValueError, "UTF-8 limit"):
                    pipeline.tokenize("🌸")
            with mock.patch.object(pipeline_module, "MAX_TOKEN_COUNT", 1):
                with self.assertRaisesRegex(ValueError, "token limit"):
                    pipeline.detokenize([1, 2])
            with mock.patch.object(pipeline_module, "MAX_JSON_BYTES", 3):
                with self.assertRaisesRegex(ValueError, "JSON limit"):
                    pipeline.detokenize([100])
        self.assertEqual(self.fake_lib.freed_buffers, 0)

    def test_tokenization_is_optional_for_old_libraries_and_checks_closed_state(self):
        del self.fake_lib.bloom_pipeline_tokenize_v2
        del self.fake_lib.bloom_pipeline_detokenize_v2
        pipeline_module._configure_lib(self.fake_lib)
        with pipeline_module.BloomPipeline(".", engine="mock") as pipeline:
            self.assertEqual(pipeline.generate("hello"), {"text": "ok"})
            for call in (lambda: pipeline.tokenize("hello"), lambda: pipeline.detokenize([1])):
                with self.assertRaisesRegex(pipeline_module.BloomInferenceError, "does not support exact tokenization"):
                    call()
            pipeline.close()
            with self.assertRaisesRegex(pipeline_module.BloomError, "closed"):
                pipeline.tokenize("hello")

    def test_tokenization_errors_free_native_buffers_and_reject_invalid_results(self):
        def error(*args):
            self.fake_lib._run_v2(args[0], None, None, *args[2:])
            return -9
        self.fake_lib.bloom_pipeline_tokenize_v2.implementation = error
        with pipeline_module.BloomPipeline(".", engine="mock") as pipeline:
            with self.assertRaisesRegex(pipeline_module.BloomInferenceError, "code -9"):
                pipeline.tokenize("hello")
            self.fake_lib.bloom_pipeline_tokenize_v2.implementation = (
                lambda *args: self.fake_lib._run_v2(args[0], None, None, *args[2:])
            )
            for output in (b"not-json", b"[true]", b"[4294967296]", b"{}", b"\xff"):
                self.fake_lib.output_buffer = ctypes.create_string_buffer(output)
                with self.assertRaises(pipeline_module.BloomInferenceError):
                    pipeline.tokenize("hello")
        self.assertEqual(self.fake_lib.freed_buffers, 6)

    def test_sdk_json_and_identifier_limits_apply_before_native_execution(self):
        with mock.patch.object(pipeline_module, "MAX_IDENTIFIER_BYTES", 3):
            for name in ("engine", "device"):
                with self.assertRaisesRegex(ValueError, "UTF-8 limit"):
                    pipeline_module.BloomPipeline(".", **{name: "🌸"})
        with pipeline_module.BloomPipeline(".", engine="mock") as pipeline:
            with mock.patch.object(pipeline_module, "MAX_JSON_BYTES", 100):
                with self.assertRaisesRegex(ValueError, "generation parameters.*JSON limit"):
                    pipeline.generate("a")
                with self.assertRaisesRegex(ValueError, "input.*JSON limit"):
                    next(pipeline.generate_stream("🌸" * 30))
        self.assertFalse(self.fake_lib.stream_started.is_set())

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
            {"max_tokens": 0},
            {"max_tokens": True},
            {"max_tokens": 1 << (8 * ctypes.sizeof(ctypes.c_size_t))},
            {"temperature": float("nan")},
            {"temperature": True},
            {"temperature": 2.1},
            {"temperature": 10 ** 400},
            {"top_p": 0},
            {"top_p": 1.1},
            {"top_p": True},
            {"top_p": 10 ** 400},
            {"seed": -1},
            {"top_k": 0},
            {"top_k": True},
            {"top_k": 32769},
            {"top_k": 2.5},
            {"presence_penalty": float("inf")},
            {"presence_penalty": -2.1},
            {"frequency_penalty": True},
            {"frequency_penalty": 2.1},
            {"frequency_penalty": 10 ** 400},
            {"logit_bias": []},
            {"logit_bias": {True: 1}},
            {"logit_bias": {1: float("nan")}},
            {"logit_bias": {1: 101}},
            {"logit_bias": {1: 10 ** 400}},
            {"logit_bias": {1000001: 1}},
            {"logit_bias": {"01": 1}},
            {"logit_bias": {"1": 1, 1: 2}},
            {"logit_bias": {"1" * 5000: 1}},
            {"logit_bias": dict.fromkeys(range(1025), 0)},
            {"logprobs": True},
            {"logprobs": False},
            {"logprobs": -1},
            {"logprobs": 21},
            {"logprobs": 1.5},
            {"logprobs": "1"},
            {"logprobs": 0, "max_tokens": 4097},
        ]
        for arguments in invalid_arguments:
            with self.subTest(arguments=arguments):
                with self.assertRaises(ValueError):
                    pipeline.generate("hello", **arguments)
                with self.assertRaises(ValueError):
                    next(pipeline.generate_stream("hello", **arguments))
        self.assertFalse(self.fake_lib.stream_started.is_set())
        self.assertEqual(self.fake_lib.freed_buffers, 0)
        with self.assertRaises(TypeError):
            pipeline.generate(["not", "a", "model", "input"])
        with self.assertRaises(ValueError):
            pipeline.generate({"Text": {"prompt": float("nan")}})
        pipeline.close()

    def test_sampling_controls_reach_both_native_abi_revisions_and_modes(self):
        expected = {
            "top_k": 40,
            "presence_penalty": -0.5,
            "frequency_penalty": 0.75,
            "logit_bias": {"0": -100, "1000000": 100},
        }
        for v2 in (False, True):
            with self.subTest(v2=v2):
                captured = []
                self.fake_lib._bloom_uses_v2 = v2
                suffix = "_v2" if v2 else ""
                for function in ("bloom_pipeline_run", "bloom_pipeline_run_stream"):
                    native_function = getattr(self.fake_lib, function + suffix)
                    original = native_function.implementation

                    def capture(*args, original=original):
                        params = args[2]
                        if v2:
                            params = ctypes.string_at(params.data, params.len)
                        captured.append(json.loads(params))
                        return original(*args)

                    native_function.implementation = capture
                with pipeline_module.BloomPipeline(".", engine="mock") as pipeline:
                    arguments = dict(expected, logit_bias={0: -100, "1000000": 100})
                    pipeline.generate("hello", **arguments)
                    list(pipeline.generate_stream("hello", **arguments))
                self.assertEqual(len(captured), 2)
                for params in captured:
                    for name, value in expected.items():
                        self.assertEqual(params[name], value)

    def test_logprobs_reach_both_abis_and_preserve_buffered_and_stream_results(self):
        records = [
            {"token": {"token_id": 10, "token": "�", "bytes": [228], "logprob": -0.1}, "top_logprobs": []},
            {"token": {"token_id": 11, "token": "��", "bytes": [189, 160], "logprob": -0.2}, "top_logprobs": []},
        ]
        output = {"text": "你", "logprobs": records}
        events = [{"TokenLogprobs": row} for row in records] + [{"TextDelta": "你"}, "End"]
        self.fake_lib.output_buffer = ctypes.create_string_buffer(json.dumps(output).encode())
        for v2 in (False, True):
            self.fake_lib._bloom_uses_v2 = v2
            captured = []
            suffix = "_v2" if v2 else ""
            native_run = getattr(self.fake_lib, "bloom_pipeline_run" + suffix)
            original_run = native_run.implementation

            def capture_params(params):
                if v2:
                    params = ctypes.string_at(params.data, params.len)
                captured.append(json.loads(params))

            def run(*args):
                capture_params(args[2])
                return original_run(*args)

            def stream(_p, _i, params, callback, *_rest):
                capture_params(params)
                for event in events:
                    encoded = json.dumps(event).encode()
                    if v2:
                        chunk = ctypes.create_string_buffer(encoded)
                        callback(None, ctypes.cast(chunk, ctypes.POINTER(ctypes.c_uint8)), len(encoded))
                    else:
                        callback(None, encoded)
                return 0

            native_run.implementation = run
            getattr(self.fake_lib, "bloom_pipeline_run_stream" + suffix).implementation = stream
            with pipeline_module.BloomPipeline(".", engine="mock") as pipeline:
                for count in (None, 0, 20):
                    with self.subTest(v2=v2, logprobs=count):
                        # Requests without metadata retain the earlier token
                        # budget contract, including values above 4096.
                        arguments = {"logprobs": count, "max_tokens": 4097 if count is None else 4096}
                        self.assertEqual(pipeline.generate("hello", **arguments), output)
                        self.assertEqual(list(pipeline.generate_stream("hello", **arguments)), events)
                        for params in captured[-2:]:
                            if count is None:
                                self.assertNotIn("logprobs", params)
                            else:
                                self.assertEqual(params["logprobs"], count)

    def test_response_format_rejects_ignored_semantics(self):
        invalid_formats = [
            {"type": "text", "json_schema": {"type": "object"}},
            {"type": "json_object", "strict": True},
            {"type": "json_schema", "unknown": 1, "json_schema": {}},
            {"type": "json_schema", "json_schema": {"schema": {}, "unknown": 1}},
            {"type": "json_schema", "json_schema": {"schema": {}, "strict": "true"}},
            {"type": "json_schema", "json_schema": {"schema": {}, "name": "名字"}},
            {"type": "json_schema", "json_schema": {"schema": {}, "description": 1}},
        ]
        for response_format in invalid_formats:
            with self.subTest(response_format=response_format):
                with self.assertRaises(ValueError):
                    pipeline_module._normalize_response_format(response_format)

    def test_stream_preserves_native_end_variant_for_both_abi_revisions(self):
        for v2 in (False, True):
            with self.subTest(v2=v2):
                self.fake_lib._bloom_uses_v2 = v2

                def stream(_p, _i, _a, callback, *_rest):
                    for value in (b'{"TextDelta":"hello"}', b'"End"'):
                        if v2:
                            chunk = ctypes.create_string_buffer(value)
                            callback(None, ctypes.cast(chunk, ctypes.POINTER(ctypes.c_uint8)), len(value))
                        else:
                            callback(None, value)
                    return 0

                function = "bloom_pipeline_run_stream_v2" if v2 else "bloom_pipeline_run_stream"
                getattr(self.fake_lib, function).implementation = stream
                with pipeline_module.BloomPipeline(".", engine="mock") as pipeline:
                    self.assertEqual(
                        list(pipeline.generate_stream("hello")),
                        [{"TextDelta": "hello"}, "End"],
                    )

    def test_constructor_rejects_nul_terminated_identifiers(self):
        for arguments in (
            {"model_path": "model\0hidden"},
            {"model_path": ".", "engine": "mock\0hidden"},
            {"model_path": ".", "device": "cpu\0hidden"},
        ):
            with self.subTest(arguments=arguments):
                with self.assertRaisesRegex(ValueError, "NUL"):
                    pipeline_module.BloomPipeline(**arguments)

    def test_constructor_rejects_model_paths_over_native_utf8_limit(self):
        oversized = "模型" * (
            pipeline_module.MAX_MODEL_PATH_BYTES // len("模型".encode("utf-8")) + 1
        )
        with self.assertRaisesRegex(ValueError, "UTF-8 limit"):
            pipeline_module.BloomPipeline(oversized, engine="mock")

    def test_constructor_accepts_pathlike_and_localizes_diagnostics(self):
        pipeline = pipeline_module.BloomPipeline(Path("模型/问答.gguf"), engine="mock", locale="zh_CN")
        self.assertEqual(pipeline.locale, "zh-CN")
        pipeline.close()

        with pipeline_module.BloomPipeline(Path("模型/问答.gguf"), engine="mock", locale="zh-CN") as closed:
            closed.close()
            with self.assertRaisesRegex(pipeline_module.BloomError, "推理管道已关闭"):
                closed.generate("你好")

    def test_unicode_prompt_and_response_format_are_utf8_json(self):
        pipeline = pipeline_module.BloomPipeline(Path("模型/问答.gguf"), engine="mock")
        input_bytes, params_bytes = pipeline._prepare_input_params(
            "请用中文回答 🌸", 8, 0.7, 0.9, None,
            {"type": "json_object"},
        )
        self.assertIn("请用中文回答 🌸".encode("utf-8"), input_bytes)
        self.assertIn(b'"response_format": {"type": "json_object"}', params_bytes)
        self.assertEqual(pipeline.generate("你好", response_format={"type": "json_object"}), {"text": "ok"})
        _, schema_params = pipeline._prepare_input_params(
            "schema", 8, 0.7, 0.9, None,
            {"type": "json_schema", "json_schema": {"name": "answer", "schema": {"type": "object"}}},
        )
        self.assertIn(
            b'"response_format": {"type": "json_schema", "json_schema": {"type": "object"}}',
            schema_params,
        )
        pipeline.close()

    def test_response_format_shorthands_match_rust_serde_shape(self):
        self.assertEqual(
            pipeline_module._normalize_response_format("text"),
            {"type": "text"},
        )
        self.assertEqual(
            pipeline_module._normalize_response_format("json_object"),
            {"type": "json_object"},
        )
        self.assertEqual(
            pipeline_module._normalize_response_format(
                {"json_schema": {"type": "object"}}
            ),
            {"type": "json_schema", "json_schema": {"type": "object"}},
        )
        with self.assertRaises(ValueError):
            pipeline_module._normalize_response_format("json_schema")

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
        self.assertTrue(self.fake_lib.stream_started.wait(timeout=1))

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
        self.assertTrue(self.fake_lib.stream_started.wait(timeout=1))
        stream.close()

        self.assertTrue(self.fake_lib.stream_cancelled.wait(timeout=1))
        deadline = time.monotonic() + 1
        while self.fake_lib.freed_tokens == 0 and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertGreaterEqual(self.fake_lib.cancel_calls, 1)
        self.assertEqual(self.fake_lib.freed_tokens, 1)
        pipeline.close()

    def test_stream_timeout_cancels_a_blocked_v2_native_worker(self):
        self.fake_lib.block_stream = True
        pipeline = pipeline_module.BloomPipeline(".", engine="mock")
        stream = pipeline.generate_stream("hello", timeout=0.01)

        self.assertEqual(next(stream), {"TextDelta": "hello"})
        with self.assertRaisesRegex(TimeoutError, "waiting for streaming output"):
            next(stream)

        self.assertTrue(self.fake_lib.stream_cancelled.wait(timeout=1))
        self.assertGreaterEqual(self.fake_lib.cancel_calls, 1)
        self.assertEqual(self.fake_lib.freed_tokens, 1)
        pipeline.close()

    def test_stream_timeout_is_validated_before_starting_native_worker(self):
        pipeline = pipeline_module.BloomPipeline(".", engine="mock")
        for timeout in (True, -1, float("nan"), float("inf"), "1"):
            with self.subTest(timeout=timeout):
                with self.assertRaises(ValueError):
                    next(pipeline.generate_stream("hello", timeout=timeout))
        self.assertFalse(self.fake_lib.stream_started.is_set())
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
        for value in (b"\xff", b"not-json", b"null", b'"unknown"', b"[]", b"42"):
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

    def test_native_model_without_tokenizer_fails_explicitly(self):
        with pipeline_module.BloomPipeline(".", engine="mock") as pipeline:
            for call in (lambda: pipeline.tokenize("hello"), lambda: pipeline.detokenize([1])):
                with self.assertRaisesRegex(pipeline_module.BloomInferenceError, "code -9.*exact tokenizer"):
                    call()


if __name__ == "__main__":
    unittest.main()
