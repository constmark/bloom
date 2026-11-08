#!/usr/bin/env python3
"""Real CPU ONNX/tokenizers regression; no downloaded model or hardware claim.

Install the documented optional dependencies, then run this file. The
--write-fixture mode is shared by Rust's opt-in end-to-end adapter test.
"""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import onnxruntime as ort
from tokenizers import Tokenizer, models, pre_tokenizers, processors

import onnxruntime_infer as bridge


CONFIG = {"pooling": "mean", "output_name": "hidden_states", "max_length": 16,
          "pad_token_id": 0, "normalize": False, "truncate": False}


def write_fixture(directory: Path, static_width=None, mask=True, pooled=False, int32=False):
    directory.mkdir(parents=True, exist_ok=True)
    vocabulary = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3,
                  "hello": 4, "world": 5, "café": 6, "play": 7, "##ing": 8}
    tokenizer = Tokenizer(models.WordPiece(vocab=vocabulary, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 2), ("[SEP]", 3)])
    tokenizer.save(str(directory / "tokenizer.json"))
    shape = ["batch", static_width or "sequence"]
    integer_type = TensorProto.INT32 if int32 else TensorProto.INT64
    inputs = [helper.make_tensor_value_info("input_ids", integer_type, shape)]
    if mask:
        inputs.append(helper.make_tensor_value_info("attention_mask", integer_type, shape))
    weights = np.asarray([[i, i * i] for i in range(len(vocabulary))], dtype=np.float32)
    nodes = [helper.make_node("Gather", ["weights", "input_ids"], ["hidden_states"], axis=0)]
    initializers = [numpy_helper.from_array(weights, "weights")]
    output_name = "hidden_states"
    output_shape = [*shape, 2]
    if pooled:
        nodes.append(helper.make_node("Gather", ["hidden_states", "cls_index"], ["pooled"], axis=1))
        initializers.append(numpy_helper.from_array(np.asarray(0, dtype=np.int64), "cls_index"))
        output_name, output_shape = "pooled", ["batch", 2]
    graph = helper.make_graph(nodes, "tiny-real-token-encoder", inputs,
                              [helper.make_tensor_value_info(output_name, TensorProto.FLOAT, output_shape)],
                              initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=9)
    onnx.checker.check_model(model)
    onnx.save(model, directory / "model.onnx")
    manifest = {"id": "tiny-onnx-encoder", "family": {"Custom": "onnx"}, "version": "1",
                "license": "MIT", "io_schema": {"inputs": ["Text"], "outputs": ["Text"]},
                "memory_profile": {"min_ram_bytes": 0, "min_vram_bytes": 0,
                                   "recommended_ram_bytes": 0, "recommended_vram_bytes": 0},
                "files": [{"name": "model.onnx", "format": "Onnx", "size_bytes": 0,
                           "hash_sha256": None, "required": True}],
                "parameters": {"bloom_task": "embedding", "onnx_embedding": CONFIG},
                "runtime_hints": {"preferred_backends": ["onnxruntime"]}, "primary_dtype": "F32"}
    (directory / "bloom.json").write_text(json.dumps(manifest))


class EncoderUsageTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name)
        write_fixture(self.directory)
        self.model = str(self.directory / "model.onnx")
        self.session = ort.InferenceSession(self.model, providers=["CPUExecutionProvider"])

    def tearDown(self):
        self.temporary.cleanup()

    def run_encoder(self, inputs, **changes):
        return bridge._embed(self.session, self.model,
                             {"inputs": inputs, "config": {**CONFIG, **changes}})

    def test_real_ids_include_specials_subwords_and_unicode_but_exclude_padding(self):
        output = self.run_encoder(["hello", "playing café"])
        self.assertEqual(output["prompt_tokens"], 8)  # [2,4,3] + [2,7,8,6,3]
        self.assertEqual(output["usage_source"], "input_ids_attention_mask")
        np.testing.assert_allclose(output["embeddings"], [[3, 29/3], [26/5, 162/5]])
        for text, row in zip(["hello", "playing café"], output["embeddings"]):
            np.testing.assert_allclose(self.run_encoder([text])["embeddings"][0], row)

    def test_explicit_truncation_counts_only_the_ids_actually_forwarded(self):
        with self.assertRaisesRegex(ValueError, "exceeds max_length"):
            self.run_encoder(["playing café"], max_length=4)
        output = self.run_encoder(["hello", "playing café"], max_length=4, truncate=True)
        self.assertEqual(output["prompt_tokens"], 7)
        np.testing.assert_allclose(output["embeddings"], [[3, 29/3], [5, 31.5]])

    def test_pooling_and_normalization_do_not_change_token_usage(self):
        output = self.run_encoder(["hello", "playing café"], pooling="cls", normalize=True)
        self.assertEqual(output["prompt_tokens"], 8)
        np.testing.assert_allclose(np.linalg.norm(output["embeddings"], axis=1), [1, 1])
        with self.assertRaisesRegex(ValueError, "rank-2"):
            self.run_encoder(["hello"], pooling="none")
        with self.assertRaisesRegex(ValueError, "output_name"):
            self.run_encoder(["hello"], output_name="not_hidden_states")

    def test_static_padding_is_not_charged_and_unmasked_padding_is_rejected(self):
        write_fixture(self.directory, static_width=8)
        self.session = ort.InferenceSession(self.model, providers=["CPUExecutionProvider"])
        self.assertEqual(self.run_encoder(["hello"])["prompt_tokens"], 3)
        write_fixture(self.directory, mask=False)
        self.session = ort.InferenceSession(self.model, providers=["CPUExecutionProvider"])
        with self.assertRaisesRegex(ValueError, "attention_mask"):
            self.run_encoder(["hello", "playing café"])

    def test_actual_feed_and_usage_are_one_operation(self):
        session = self.session
        feeds = []
        class RecordingSession:
            def get_inputs(self):
                return session.get_inputs()
            def get_outputs(self):
                return session.get_outputs()
            def run(self, names, feed):
                feeds.append(copy.deepcopy(feed))
                return session.run(names, feed)
        output = bridge._embed(RecordingSession(), self.model,
                               {"inputs": ["hello", "playing café"], "config": CONFIG})
        self.assertEqual(len(feeds), 1)
        np.testing.assert_array_equal(feeds[0]["input_ids"], [[2,4,3,0,0], [2,7,8,6,3]])
        self.assertEqual(output["prompt_tokens"], int(feeds[0]["attention_mask"].sum()))

    def test_already_pooled_int32_graph_retains_exact_input_count(self):
        write_fixture(self.directory, pooled=True, int32=True)
        self.session = ort.InferenceSession(self.model, providers=["CPUExecutionProvider"])
        output = self.run_encoder(["hello", "playing café"], pooling="none", output_name="pooled")
        self.assertEqual(output["prompt_tokens"], 8)
        np.testing.assert_array_equal(output["embeddings"], [[2, 4], [2, 4]])

    def test_bundled_process_protocol_with_real_cpu_runtime(self):
        command = [sys.executable, str(Path(bridge.__file__)), "--model-path", self.model,
                   "--device", "CPU", "--provider", "CPUExecutionProvider"]
        probe = subprocess.run(command + ["--probe", "--embedding-config", json.dumps(CONFIG)],
                               capture_output=True, text=True, check=True)
        self.assertEqual(json.loads(probe.stdout)["provider"], "CPUExecutionProvider")
        result = subprocess.run(command, input=json.dumps({"operation": "embed", "inputs": ["hello", "playing café"],
                                                           "config": CONFIG}),
                                capture_output=True, text=True, check=True)
        self.assertEqual(json.loads(result.stdout)["prompt_tokens"], 8)


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--write-fixture":
        write_fixture(Path(sys.argv[2]))
    else:
        unittest.main()
