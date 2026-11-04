#!/usr/bin/env python3
"""Keep documentation validation independent from transient build directories."""
import os
import pathlib
import tempfile
import unittest
from unittest import mock

import check_markdown_links as links


class MarkdownDiscoveryTests(unittest.TestCase):
    def test_excluded_build_trees_are_never_enumerated(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            (root / 'docs').mkdir()
            (root / 'docs' / 'guide.md').write_text('[readme](../README.md)')
            (root / 'README.md').write_text('[guide](docs/guide.md)')
            (root / 'target').mkdir()
            (root / 'target' / 'transient.md').write_text('[missing](missing.md)')
            original = os.scandir

            def guarded(path):
                if pathlib.Path(path).name == 'target':
                    raise FileNotFoundError('Cargo removed this transient directory')
                return original(path)

            with mock.patch.object(links.os, 'scandir', side_effect=guarded):
                self.assertEqual(links.validate(root), (2, []))

    def test_real_document_directories_are_checked_and_errors_not_hidden(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            (root / 'build-target').mkdir()
            (root / 'build-target' / 'guide.md').write_text('[broken](absent.md)')
            count, errors = links.validate(root)
            self.assertEqual(count, 1)
            self.assertEqual(len(errors), 1)
            with mock.patch.object(links.os, 'scandir', side_effect=PermissionError('unreadable docs')):
                with self.assertRaises(PermissionError):
                    links.markdown_files(root)


if __name__ == '__main__':
    unittest.main()
