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
- Keep the default `same-origin` browser policy for the embedded UI, or set
  `BLOOM_CORS_ALLOW_ORIGIN` to the separately hosted UI's one exact HTTP(S)
  origin. Never deploy the explicit `*` wildcard without a reviewed exception.
- Keep `/metrics` behind an internal network or reverse proxy ACL.
- Pin model directories and verify `bloom.json` file hashes where available.
- Load only allowlisted plugins and external runners.
- Keep request payload limits small with `--max-body-bytes`.

## Supply Chain

Default CI must not download opaque model weights or execute unknown external
assets. Real-model and hardware validation should use pinned model sources,
recorded licenses, and SHA-256 hashes.

Dependabot monitors both Cargo workspaces, pinned Python requirement sets, the
Python package, the Docker base image, and GitHub Actions. Dependency update
pull requests must pass the same locked, model-free, and pinned-model gates as
maintainer-authored changes; automated updates are not trusted implicitly.
The Rust compiler patch release is updated separately as one reviewed change
across `rust-toolchain.toml`, CI, and the Docker builder so automated ecosystem
updates cannot silently split the tested build environment.

CI and release workflows grant read-only repository access by default and pin
every external Action to a full upstream commit SHA. Only the release-publishing
job receives scoped content, OIDC, attestation, and artifact-metadata write
permissions; inline version comments remain available to Dependabot and
reviewers without making the executable reference mutable. Hosted archives and
checksum files receive signed GitHub/Sigstore build-provenance attestations.
Release archive entry order, timestamps, owners, and modes are normalized from
`SOURCE_DATE_EPOCH`, and a failed compression never replaces an existing
archive. This archive-layer guarantee does not imply reproducible compiler or
linker output.
