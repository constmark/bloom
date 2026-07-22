# Security Policy

Bloom loads local model files, can call optional external runtimes, and can load
third-party plugin entry points. Treat model directories, plugin manifests, and
external scripts as executable supply-chain inputs.

## Supported Versions

| Version | Supported |
| --- | --- |
| `main` | Best-effort security fixes before the first stable release |
| `< 1.0` releases | Best-effort fixes for the latest minor release |

## Reporting a Vulnerability

Please report vulnerabilities privately to the maintainers before opening a
public issue. Include:

- Affected commit or release.
- Reproduction steps.
- Whether the issue requires a malicious model, plugin, request payload, or
  network exposure.
- Any observed data exposure, code execution, denial of service, or path escape.

Until a dedicated security contact is published, use a private GitHub security
advisory on the repository.

## Production Baseline

Before exposing `bloom_server` outside localhost:

- Set `BLOOM_API_KEY` or pass `--api-key`.
- Set `BLOOM_CORS_ALLOW_ORIGIN` to an exact trusted origin instead of `*`.
- Keep `/metrics` behind an internal network or reverse proxy ACL.
- Pin model directories and verify `bloom.json` file hashes where available.
- Load only allowlisted plugins and external runners.
- Keep request payload limits small with `--max-body-bytes`.

## Supply Chain

Default CI must not download opaque model weights or execute unknown external
assets. Real-model and hardware validation should use pinned model sources,
recorded licenses, and SHA-256 hashes.
