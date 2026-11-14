#!/usr/bin/env python3
"""Fail-closed validator regressions; actual quality requires the trained gate."""
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import trained_cross_encoder_runtime as gate


class RankingContractTests(unittest.TestCase):
    def payload(self):
        return {
            "id": "rerank-fixture", "object": "rerank", "model": "model",
            "results": [{"index": 1, "relevance_score": 8.5, "document": {"text": "relevant"}},
                        {"index": 0, "relevance_score": -4.25, "document": {"text": "other"}}],
            "usage": {"prompt_tokens": 12, "total_tokens": 12},
        }

    def validate(self, payload):
        return gate.validate_ranking(payload, "model", ["other", "relevant"], 2)

    def test_raw_logits_preserve_input_identity_and_are_not_probability_bounded(self):
        self.assertEqual(self.validate(self.payload()), [-4.25, 8.5])

    def test_model_identity_cardinality_document_and_order_corruption_fail(self):
        changes = [
            (lambda p: p.update(model="other")),
            (lambda p: p.update(object="embedding")),
            (lambda p: p.update(id="")),
            (lambda p: p["results"].pop()),
            (lambda p: p["results"].reverse()),
            (lambda p: p["results"][0].update(index=0)),
            (lambda p: p["results"][0].update(index=True)),
            (lambda p: p["results"][0].update(index=2)),
            (lambda p: p["results"][0].update(document={"text": "rewritten"})),
        ]
        for mutate in changes:
            payload = self.payload()
            mutate(payload)
            with self.subTest(payload=payload), self.assertRaises(AssertionError):
                self.validate(payload)

    def test_nonfinite_boolean_scores_and_invalid_usage_fail(self):
        for score in (float("nan"), float("inf"), True, "8.5", None):
            payload = self.payload()
            payload["results"][0]["relevance_score"] = score
            with self.subTest(score=score), self.assertRaises(AssertionError):
                self.validate(payload)
        for usage in (None, {}, {"prompt_tokens": 0, "total_tokens": 0},
                      {"prompt_tokens": True, "total_tokens": 1},
                      {"prompt_tokens": 1, "total_tokens": True},
                      {"prompt_tokens": 1, "total_tokens": 2}):
            payload = self.payload()
            payload["usage"] = usage
            with self.subTest(usage=usage), self.assertRaises(AssertionError):
                self.validate(payload)

    def test_quality_requires_correct_top_result_and_a_real_margin(self):
        self.assertEqual(gate.quality_margin([-4.25, 8.5], 1), 12.75)
        for scores, relevant in (([1.0, 0.0], 0), ([9.0, 2.0], 1),
                                 ([8.0, 8.0], 0), ([float("nan"), 4.0], 1),
                                 ([False, 8.0], 1), ([8.0], 0), ([8.0, 0.0], -1)):
            with self.subTest(scores=scores, relevant=relevant), self.assertRaises(AssertionError):
                gate.quality_margin(scores, relevant)

    def test_equal_scores_require_stable_input_index_order(self):
        payload = self.payload()
        payload["results"][0]["relevance_score"] = -4.25
        with self.assertRaises(AssertionError):
            self.validate(payload)
        payload["results"].reverse()
        self.assertEqual(self.validate(payload), [-4.25, -4.25])


class PinnedAcquisitionTests(unittest.TestCase):
    def test_cached_artifact_is_rehashed_and_corruption_is_not_downloaded_over(self):
        card = b"---\nlicense: apache-2.0\n---\n"
        digest = hashlib.sha256(card).hexdigest()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "README.md"
            path.write_bytes(card)
            with mock.patch.object(gate, "FILES", {"README.md": (len(card), digest)}), \
                    mock.patch.object(gate.urllib.request, "urlopen") as network:
                self.assertEqual(gate.acquire_model(root)["README.md"]["sha256"], digest)
                path.write_bytes(card.replace(b"apache", b"broken"))
                with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                    gate.acquire_model(root)
                network.assert_not_called()

    def test_unpinned_files_and_symlinked_model_artifacts_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "bloom.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "unpinned files"):
                gate.acquire_model(root)
            artifact = root / "bloom.json"
            link = root / "weights-link"
            try:
                link.symlink_to(artifact)
            except OSError:
                self.skipTest("symlink creation is unavailable")
            with self.assertRaisesRegex(ValueError, "non-symlink"):
                gate.verify_file(link, 2, hashlib.sha256(b"{}").hexdigest())


if __name__ == "__main__":
    unittest.main()
