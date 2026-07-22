# Production Checklist

Bloom is pre-1.0. Only deploy model/backend combinations with your own
real-model and hardware validation.

## Server

- Set `BLOOM_API_KEY` or `--api-key`.
- Set `BLOOM_CORS_ALLOW_ORIGIN` to an exact trusted origin.
- Set `--max-body-bytes`, `--max-concurrent`, and `--timeout`.
- Put TLS, rate limiting, and request logging policy in a reverse proxy.
- Keep `/metrics`, `/health`, and `/ready` on an internal network or proxy ACL.

Verify authentication before deployment:

```bash
BLOOM_MODEL_PATH=/path/to/model.gguf \
./scripts/openai_compat_smoke.py --api-key change-me
```

## Memory

- Set `BLOOM_STRICT_MEMORY_BUDGET=1`.
- Tune `--memory-utilization`, context size, and concurrency on the target host.
- Record peak host memory and device memory with the production model.
- Configure operating-system or container memory limits as a final guard.

## Models and plugins

- Use fixed, read-only model directories.
- Include `bloom.json`, model license information, and SHA-256 hashes.
- Allowlist every external script, runner, and plugin.
- Do not expose model-path selection directly to API users.
- Do not deploy `skeleton` paths as executable backends.

## Release gate

- Complete the checks in [RELEASE.md](../RELEASE.md).
- Run the OpenAI compatibility smoke test with authentication enabled.
- Run a real-model benchmark on the target hardware.
- Confirm startup, readiness, graceful shutdown, and request cancellation.
- Document supported model hashes, runtime versions, and known limits.
