#!/usr/bin/env python3
"""Exercise exact tokenizer or numeric HTTP operations against a real local model.

Example: --model /path/to/tiny-qwen2 --text hello
Example: --model /path/to/affine.mlmodel --backend coreml --input '[1,2,3]' --expected-output '[4.5]'
Uses existing model files, an isolated configuration, and a temporary server.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import tempfile
import time
import urllib.error
import urllib.request

from readiness_contract import validate_readiness_document


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--server-bin', type=Path, default=Path('target/debug/bloom_server'))
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--backend', default='candle')
    parser.add_argument('--text', default='hello')
    parser.add_argument('--input', help='Numeric JSON input; otherwise run tokenizer checks')
    parser.add_argument('--expected-output', help='Expected flat numeric output JSON array')
    args = parser.parse_args()
    if args.input is not None and args.expected_output is None:
        parser.error('--input requires --expected-output')
    key = 'local-native-operation-smoke'
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    url = f'http://127.0.0.1:{port}'

    def request(path: str, body: object | None = None, *, authenticated: bool = True):
        headers = {'Content-Type': 'application/json'}
        if authenticated:
            headers['Authorization'] = f'Bearer {key}'
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(url + path, data=data, headers=headers)
        try:
            response = urllib.request.urlopen(req, timeout=30)
        except urllib.error.HTTPError as error:
            response = error
        with response:
            assert response.headers.get('Cache-Control') == 'no-store'
            assert response.headers.get('x-request-id')
            raw = response.read(32 * 1024 * 1024 + 1)
            assert len(raw) <= 32 * 1024 * 1024
            return response.status, json.loads(raw)

    with tempfile.TemporaryDirectory(prefix='bloom-native-operations-') as directory:
        base = Path(directory)
        config = base / 'config.json'
        config.write_text('{}', encoding='utf-8')
        environment = {name: value for name, value in os.environ.items() if not name.startswith('BLOOM_')}
        environment['BLOOM_API_KEY'] = key
        with (base / 'server.log').open('w+') as log:
            process = subprocess.Popen([
                str(args.server_bin.resolve()), '--model', str(args.model.resolve()),
                '--backend', args.backend, '--device', 'cpu', '--host', '127.0.0.1',
                '--port', str(port), '--config', str(config), '--models-dir', str(base / 'models'),
                '--context-size', '128', '--disable-memory-prealloc',
            ], stdout=log, stderr=log, env=environment)
            try:
                deadline = time.monotonic() + 90
                while time.monotonic() < deadline:
                    if process.poll() is not None:
                        raise AssertionError('server exited before readiness')
                    try:
                        status, ready = request('/ready')
                        validate_readiness_document(ready)
                        if ready.get('load_error'):
                            raise AssertionError('model failed startup preflight or loading')
                        if status == 200:
                            break
                    except (OSError, urllib.error.URLError):
                        pass
                    time.sleep(.1)
                else:
                    raise AssertionError('server did not become ready')
                model = ready['model']
                if args.input is None:
                    status, encoded = request('/v1/tokenize', {'model': 'default', 'text': args.text})
                    assert status == 200 and encoded['object'] == 'bloom.tokenization', encoded
                    assert encoded['model'] == model and encoded['count'] == len(encoded['tokens'])
                    status, decoded = request('/v1/detokenize', {'model': model, 'tokens': encoded['tokens']})
                    assert status == 200 and decoded['text'] == args.text, decoded
                    assert decoded['model'] == model
                    assert request('/v1/detokenize', {'tokens': [2**32 - 1]})[0] == 422
                    assert request('/v1/tensors/infer', {'input': [1]})[0] == 422
                    path, body = '/v1/tokenize', {'text': args.text}
                    result = {'operation': 'tokenizer_round_trip', 'token_count': encoded['count']}
                else:
                    body = {'model': 'default', 'input': json.loads(args.input)}
                    path = '/v1/tensors/infer'
                    status, output = request(path, body)
                    assert status == 200 and output['object'] == 'bloom.tensor_inference', output
                    actual, expected = output['output']['data'], json.loads(args.expected_output)
                    assert output['model'] == model and output['output']['element_count'] == len(actual)
                    assert len(actual) == len(expected)
                    assert all(math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-6) for a, b in zip(actual, expected))
                    assert request('/v1/tokenize', {'text': args.text})[0] == 422
                    result = {'operation': 'numeric_inference', 'elements': len(actual)}
                assert request(path, body, authenticated=False)[0] == 401
                assert request(path + '?unsupported=true', body)[0] == 400
                assert request(path, dict(body, unsupported=True))[0] == 422
                print(json.dumps(dict(result, status='ok', model=model)))
            finally:
                if process.poll() is None:
                    process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)


if __name__ == '__main__':
    main()
