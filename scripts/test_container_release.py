#!/usr/bin/env python3
"""Offline adversarial release identity, evidence and OCI index tests."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import container_release as release
from check_container_workflow import validate_compose, validate_text


class ContainerReleaseTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        version = release.re.search(r'^version = "([^"]+)"$', (release.ROOT / "Cargo.toml").read_text(), release.re.MULTILINE).group(1)
        self.environment = {"GITHUB_REPOSITORY": "Example/Bloom", "GITHUB_SHA": "a" * 40,
                            "GITHUB_REF_NAME": f"v{version}", "ARCH": "amd64",
                            "GITHUB_RUN_ID": "100", "GITHUB_RUN_ATTEMPT": "1"}
        self.identity = release.identity(self.environment)
        self.records = []
        for arch, digit in (("amd64", "1"), ("arm64", "2")):
            record = {"schema_version": 1, "object": "bloom.container_platform",
                      **self.identity, "architecture": arch, "digest": "sha256:" + digit * 64}
            for label, name in (("runtime", f"runtime-{arch}.json"), ("scan", f"scan-{arch}.json"),
                                ("sbom", f"sbom-{arch}.spdx.json")):
                path = self.root / name
                path.write_text('{"fixture": true}')
                record[f"{label}_sha256"] = release.digest_file(path)
            (self.root / f"{arch}.json").write_text(json.dumps(record))
            self.records.append(record)

    def test_exact_release_identity_and_candidate_are_safe(self):
        self.assertEqual(self.identity["image"], "ghcr.io/example/bloom")
        self.assertEqual(self.identity["candidate"], "ghcr.io/example/bloom:build-100-1-amd64")
        for key, value in (("GITHUB_REF_NAME", "v99.0.0"), ("GITHUB_SHA", "main"),
                           ("GITHUB_REPOSITORY", "../bad\nname"), ("ARCH", "unknown"),
                           ("GITHUB_RUN_ATTEMPT", "1\nforged=yes")):
            with self.subTest(key=key), self.assertRaises(ValueError):
                release.identity({**self.environment, key: value})

    def test_both_platform_evidence_and_index_are_exact(self):
        records = release.validate_records(self.root, self.identity)
        index = {"schemaVersion": 2, "manifests": [
            {"digest": item["digest"], "platform": {"os": "linux", "architecture": item["architecture"]}}
            for item in records
        ]}
        release.validate_index(index, records)
        for change in ("wrong_digest", "wrong_os", "duplicate_platform", "missing_platform"):
            broken = copy.deepcopy(index)
            if change == "wrong_digest":
                broken["manifests"][0]["digest"] = "sha256:" + "0" * 64
            elif change == "wrong_os":
                broken["manifests"][0]["platform"]["os"] = "windows"
            elif change == "duplicate_platform":
                broken["manifests"][1] = broken["manifests"][0]
            else:
                broken["manifests"].pop()
            with self.subTest(change=change), self.assertRaises(ValueError):
                release.validate_index(broken, records)

    def test_rejects_wrong_revision_architecture_or_digest(self):
        path = self.root / "amd64.json"
        original = path.read_text()
        for key, value in (("revision", "b" * 40), ("image", "ghcr.io/other/image"),
                           ("architecture", "arm64"), ("digest", "sha256:not-a-digest"),
                           ("schema_version", True)):
            record = json.loads(original)
            record[key] = value
            path.write_text(json.dumps(record))
            with self.subTest(key=key), self.assertRaises(ValueError):
                release.validate_records(self.root, self.identity)

    def test_replaced_evidence_cannot_be_promoted(self):
        (self.root / "scan-arm64.json").write_text('{"replaced": true}')
        with self.assertRaisesRegex(ValueError, "scan evidence was replaced"):
            release.validate_records(self.root, self.identity)

    def test_missing_or_symlink_evidence_cannot_be_promoted(self):
        path = self.root / "amd64.json"
        path.unlink()
        with self.assertRaises(ValueError):
            release.validate_records(self.root, self.identity)
        path.symlink_to(self.root / "arm64.json")
        with self.assertRaises(ValueError):
            release.validate_records(self.root, self.identity)

    def test_invalid_evidence_never_calls_registry(self):
        (self.root / "amd64.json").unlink()
        with mock.patch.object(release, "identity", return_value=self.identity), \
                mock.patch.object(release, "docker") as docker:
            with self.assertRaises(ValueError):
                release.publish(self.root, self.root / "output.json")
            docker.assert_not_called()

    def test_publish_validates_only_the_resolved_immutable_index(self):
        digest = "sha256:" + "3" * 64
        immutable = f"{self.identity['image']}@{digest}"
        target = f"{self.identity['image']}:{self.identity['version']}"
        index = {"schemaVersion": 2, "manifests": [
            {"digest": item["digest"], "platform": {"os": "linux", "architecture": item["architecture"]}}
            for item in self.records
        ]}
        tag_reads = 0

        def registry(*arguments):
            nonlocal tag_reads
            if arguments[:3] == ("buildx", "imagetools", "create"):
                return ""
            if arguments[-1] == target:
                tag_reads += 1
                self.assertEqual(arguments[3:5], ("--format", "{{json .Manifest}}"))
                # A second tag resolution would identify an unrelated index.
                return json.dumps({"digest": digest if tag_reads == 1 else "sha256:" + "4" * 64})
            self.assertEqual(arguments, ("buildx", "imagetools", "inspect", "--raw", immutable))
            return json.dumps(index)

        output = self.root / "output.json"
        with mock.patch.object(release, "identity", return_value=self.identity), \
                mock.patch.object(release, "docker", side_effect=registry), \
                mock.patch.object(release, "emit_outputs") as emit:
            release.publish(self.root, output)
            emit.assert_called_once_with({"image": self.identity["image"], "digest": digest})
        self.assertEqual(tag_reads, 1)
        self.assertEqual(json.loads(output.read_text())["immutable_reference"], immutable)

    def test_publish_rejects_invalid_resolved_digest_before_fetching_index(self):
        output = self.root / "output.json"
        with mock.patch.object(release, "identity", return_value=self.identity), \
                mock.patch.object(release, "docker", side_effect=["", '{"digest":"sha256:invalid"}']) as docker, \
                mock.patch.object(release, "emit_outputs") as emit:
            with self.assertRaisesRegex(ValueError, "valid index digest"):
                release.publish(self.root, output)
            self.assertEqual(docker.call_count, 2)
            emit.assert_not_called()
        self.assertFalse(output.exists())

    def test_publish_rejects_unverified_pinned_index_without_emitting_evidence(self):
        digest = "sha256:" + "3" * 64
        output = self.root / "output.json"
        wrong_index = {"schemaVersion": 2, "manifests": [
            {"digest": "sha256:" + "9" * 64,
             "platform": {"os": "linux", "architecture": item["architecture"]}}
            for item in self.records
        ]}
        with mock.patch.object(release, "identity", return_value=self.identity), \
                mock.patch.object(release, "docker", side_effect=[
                    "", json.dumps({"digest": digest}), json.dumps(wrong_index),
                ]) as docker, mock.patch.object(release, "emit_outputs") as emit:
            with self.assertRaisesRegex(ValueError, "differs from verified platform digests"):
                release.publish(self.root, output)
            self.assertEqual(docker.call_args.args[-1], f"{self.identity['image']}@{digest}")
            emit.assert_not_called()
        self.assertFalse(output.exists())

    def test_current_workflow_and_mutated_gate_rejection(self):
        content = (release.ROOT / ".github/workflows/container-release.yml").read_text()
        self.assertEqual(validate_text(content), [])
        for old, new in (("    needs: checks", "    needs: []"),
                         ("    needs: images", "    needs: []"),
                         ('exit-code: "1"', 'exit-code: "0"'),
                         ("ignore-unfixed: false", "ignore-unfixed: true"),
                         ("version: v0.76.2", "version: latest"),
                         (" --require-native-sbom", ""),
                         ("          push: false", "          push: true"),
                         ("    needs: images", "    needs: images\n    if: always()")):
            with self.subTest(old=old):
                self.assertTrue(validate_text(content.replace(old, new, 1)))

    def test_deployment_template_requires_hardened_runtime(self):
        content = (release.ROOT / "deploy/compose.yml").read_text()
        self.assertEqual(validate_compose(content), [])
        for old, new in (("read_only: true", "read_only: false"),
                         ("127.0.0.1:3000:3000", "0.0.0.0:3000:3000"),
                         ('user: "10001:10001"', 'user: "0:0"'),
                         ('BLOOM_STRICT_SECURITY: "1"', 'BLOOM_STRICT_SECURITY: "0"'),
                         ("stop_grace_period: 40s", "stop_grace_period: 1s"),
                         ("${BLOOM_OPERATOR_API_KEY:?", "${BLOOM_OPERATOR_API_KEY:-")):
            with self.subTest(old=old):
                self.assertTrue(validate_compose(content.replace(old, new, 1)))


if __name__ == "__main__":
    unittest.main()
