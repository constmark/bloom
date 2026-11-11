#!/usr/bin/env python3
"""Regressions for dependency graph changes that invalidate audited runtime bounds."""
import copy
import unittest

from check_runtime_dependencies import REGISTRY, verify


def metadata():
    versions = {"bloomai-engine": "0.1.0", "bloomai-laya": "0.2.20-bloom.1",
                "candle-core": "0.11.1", "gemm": "0.19.0", "gemm-common": "0.19.0",
                "zip": "9.5.1"}
    edges = {"bloomai-engine": ["bloomai-laya", "gemm-common", "candle-core"],
             "bloomai-laya": ["candle-core"], "candle-core": ["gemm", "zip"],
             "gemm": ["gemm-common"]}
    return {"packages": [{"id": name, "name": name, "version": version,
                          "source": None if name.startswith("bloomai-") else REGISTRY}
                         for name, version in versions.items()],
            "resolve": {"nodes": [{"id": name, "deps": [{"pkg": dep} for dep in edges.get(name, [])]}
                                   for name in versions]}}


class RuntimeDependencyTests(unittest.TestCase):
    def test_reviewed_dependency_graph(self):
        verify(metadata())

    def test_version_aliases_duplicates_and_changed_sources_fail(self):
        for name, field, value in [("candle-core", "version", "0.12.0"),
                                   ("gemm", "version", "0.20.0"),
                                   ("gemm-common", "source", None),
                                   ("zip", "version", "7.4.0"),
                                   ("zip", "source", "git+https://example.com/zip")]:
            changed = metadata()
            next(package for package in changed["packages"] if package["name"] == name)[field] = value
            with self.subTest(name=name, field=field), self.assertRaises(ValueError):
                verify(changed)
        changed = metadata()
        changed["packages"].append(copy.deepcopy(changed["packages"][2]))
        with self.assertRaises(ValueError):
            verify(changed)

    def test_missing_nodes_and_edges_fail(self):
        original = metadata()
        for index, node in enumerate(original["resolve"]["nodes"]):
            changed = copy.deepcopy(original)
            changed["resolve"]["nodes"].pop(index)
            with self.subTest(node=node["id"]), self.assertRaises(ValueError):
                verify(changed)
            for edge_index in range(len(node["deps"])):
                changed = copy.deepcopy(original)
                changed["resolve"]["nodes"][index]["deps"].pop(edge_index)
                with self.subTest(node=node["id"], edge=edge_index), self.assertRaises(ValueError):
                    verify(changed)


if __name__ == "__main__":
    unittest.main()
