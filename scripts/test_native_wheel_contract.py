#!/usr/bin/env python3
"""Offline metadata, integrity and no-overwrite regressions for native wheels."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import zipfile

from build_python_native_wheel import bundle_wheel, host_platform_tag, library_name
from test_python_native_wheel import isolated_install_environment, verify_archive


class NativeWheelTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.pure = self.root / 'bloom_sdk-0.1.0-py3-none-any.whl'
        with zipfile.ZipFile(self.pure, 'w') as archive:
            archive.writestr('bloom_sdk/__init__.py', '# fixture')
            archive.writestr('bloom_sdk-0.1.0.dist-info/METADATA', 'Name: bloom-sdk\nVersion: 0.1.0\n')
            archive.writestr('bloom_sdk-0.1.0.dist-info/WHEEL', 'Root-Is-Purelib: true\nTag: py3-none-any\n')
            archive.writestr('bloom_sdk-0.1.0.dist-info/RECORD', 'old inventory')
        self.native = self.root / library_name()
        self.native.write_bytes(b'archive-assembly-fixture-not-executable')
        self.output = self.root / 'dist'

    def bundle(self, **kwargs):
        return bundle_wheel(self.pure, self.native, self.output, kwargs.get('abi', 2), kwargs.get('tag', 'linux_x86_64'), b'{"bomFormat":"CycloneDX","components":[]}')

    def test_complete_record_and_native_placement(self):
        path = self.bundle()
        report = verify_archive(path)
        self.assertEqual(report['filename'], library_name())
        self.assertEqual(report['abi_revision'], 2)
        self.assertTrue(path.with_suffix('.whl.sha256').exists())

    def test_existing_output_is_never_overwritten(self):
        path = self.bundle()
        original = path.read_bytes()
        with self.assertRaises(FileExistsError):
            self.bundle()
        self.assertEqual(path.read_bytes(), original)

    def test_rejects_invalid_platform_and_abi(self):
        for tag in ('any', '../linux', 'manylinux.unsafe', ''):
            with self.subTest(tag=tag), self.assertRaises(ValueError):
                self.bundle(tag=tag)
        with self.assertRaises(ValueError):
            self.bundle(abi=1)

    def test_corruption_fails_hash_validation(self):
        path = self.bundle()
        altered = self.root / path.name
        with zipfile.ZipFile(path) as source, zipfile.ZipFile(altered, 'w') as destination:
            for name in source.namelist():
                data = source.read(name)
                if name.endswith('/native-library.json'):
                    value = json.loads(data)
                    value['sha256'] = '0' * 64
                    data = json.dumps(value).encode()
                destination.writestr(name, data)
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            verify_archive(altered)

    def test_rejects_source_with_native_payload(self):
        with zipfile.ZipFile(self.pure, 'a') as archive:
            archive.writestr('bloom_sdk/other.so', b'bad')
        with self.assertRaises(ValueError):
            self.bundle()
        self.assertEqual(list(self.output.glob('*.whl')), [])

    def test_host_tags_do_not_claim_linux_portability(self):
        with mock.patch('build_python_native_wheel.sys.platform', 'linux'), mock.patch('build_python_native_wheel.sysconfig.get_platform', return_value='linux-x86_64'):
            self.assertEqual(host_platform_tag(), 'linux_x86_64')
        with mock.patch('build_python_native_wheel.sys.platform', 'darwin'), mock.patch('build_python_native_wheel.platform.mac_ver', return_value=('26.1', (), '')), mock.patch('build_python_native_wheel.platform.machine', return_value='arm64'):
            self.assertEqual(host_platform_tag(), 'macosx_26_0_arm64')

    def test_clean_install_cannot_inherit_pip_destinations_or_config(self):
        source = {
            'PATH': '/runtime/bin', 'SYSTEMROOT': 'C:\\Windows',
            'PIP_TARGET': '/outside/target', 'PIP_PREFIX': '/outside/prefix',
            'PIP_USER': '1', 'PIP_CONFIG_FILE': '/outside/pip.ini',
            'PIP_REQUIRE_VIRTUALENV': 'false', 'pip_index_url': 'https://example.invalid',
            'PYTHONPATH': '/outside/python', 'PYTHONHOME': '/outside/home',
            'BLOOM_FFI_LIB': '/outside/library', 'BLOOM_SDK_TEST_MODEL': '/outside/model',
            'BLOOM_SDK_REQUIRE_GENERATION_METADATA': '1',
            'BLOOM_SDK_CHECK_LOGPROBS': '1',
        }
        import os
        self.assertEqual(isolated_install_environment(source), {
            'PATH': '/runtime/bin', 'SYSTEMROOT': 'C:\\Windows',
            'PIP_CONFIG_FILE': os.devnull, 'PIP_REQUIRE_VIRTUALENV': 'true',
        })
        self.assertEqual(source['PIP_TARGET'], '/outside/target')


if __name__ == '__main__':
    unittest.main()
