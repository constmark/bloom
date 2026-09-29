#!/usr/bin/env python3
"""Regression checks for architecture violations, including Cargo aliases."""

import copy
import pathlib
import tempfile
import unittest

from check_architecture import ROOT, metadata_packages, validate_packages, validate_sources


class ArchitectureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.packages = metadata_packages(ROOT / "Cargo.toml") + metadata_packages(ROOT / "ui/Cargo.toml")

    def add_dependency(self, owner, dependency, **changes):
        packages = copy.deepcopy(self.packages)
        package = next(package for package in packages if package["name"] == owner)
        package["dependencies"].append({
            "name": dependency, "kind": None, "uses_default_features": False, **changes,
        })
        return validate_packages(packages)

    def test_current_layers(self):
        self.assertEqual(validate_packages(self.packages), [])
        self.assertEqual(validate_sources(ROOT), [])

    def test_reverse_dependencies_include_optional_target_aliases(self):
        errors = self.add_dependency("bloomai-engine", "bloomai-server", rename="http", optional=True, target="cfg(windows)")
        self.assertTrue(any("forbidden layer" in error for error in errors))

    def test_frontend_cannot_link_native_runtime(self):
        self.assertTrue(self.add_dependency("bloom-ui", "bloomai-engine"))

    def test_transport_and_tensor_dependencies_are_owned(self):
        self.assertTrue(self.add_dependency("bloomai-engine", "axum"))
        self.assertTrue(self.add_dependency("bloomai-server", "candle-core"))
        self.assertTrue(self.add_dependency("bloomai-app", "dioxus"))

    def test_cli_dev_tools_do_not_leak_into_library(self):
        self.assertTrue(self.add_dependency("bloomai-engine", "clap"))
        self.assertEqual(self.add_dependency("bloomai-engine", "clap", kind="dev"), [])

    def test_inherited_defaults_cannot_enable_an_optional_engine(self):
        errors = self.add_dependency("bloomai-ffi", "bloomai-engine", uses_default_features=True)
        self.assertTrue(any("explicitly forward features" in error for error in errors))

    def test_reexported_wrappers_cannot_bypass_manifest_checks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            source = root / "crates/server/src/scheduling.rs"
            source.parent.mkdir(parents=True)
            source.write_text("let model = wrapper.downcast::<QwenModelWrapper>();\n")
            self.assertEqual(len(validate_sources(root)), 1)

    def check_source(self, relative, text):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            source = root / relative
            source.parent.mkdir(parents=True)
            source.write_text(text)
            return validate_sources(root)

    def test_model_contract_rejects_concrete_backend_types_and_downcasts(self):
        for code in [
            "fn device(&self) -> candle_core::Device;",
            "use tokenizers::Tokenizer;",
            "use crate::executor::candle::CandleEngine;",
            "fn wrapper(&self) -> Box<dyn std::any::Any>;",
        ]:
            with self.subTest(code=code):
                self.assertTrue(self.check_source("crates/engine/src/core/model.rs", code))

    def test_runtime_service_rejects_http_cli_and_root_imports(self):
        for code in [
            "use axum::Json;",
            "use crate::helpers::RequestedModelError;",
            "use crate::cli::Args;",
            "use crate::{ServerState};",
            "use super::*;",
            "use super::super::handlers::handle_cancel;",
        ]:
            with self.subTest(code=code):
                self.assertTrue(self.check_source("crates/server/src/application/runtime_service.rs", code))

    def test_application_unit_tests_can_import_their_owner(self):
        self.assertEqual(self.check_source(
            "crates/server/src/application/inference.rs",
            "use super::runtime::RuntimeRequestLease;\n#[cfg(test)]\nmod tests {\nuse super::*;\n}\n",
        ), [])


if __name__ == "__main__":
    unittest.main()
