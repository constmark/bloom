# Server Doctor

`bloom_server --doctor` inspects the effective deployment configuration and
host capabilities without loading a model, creating the model catalog, binding
a port, or starting background tasks. It is safe to run in build pipelines and
on CPU-only hosts.

Use the human-readable report during setup:

```bash
bloom_server --doctor
```

Use the versioned JSON contract in packaging and deployment automation:

```bash
bloom_server --doctor=json > bloom-server-doctor.json
```

Warnings describe a usable but incomplete or risky deployment and retain a
zero exit status. A failed check prints the complete report and exits non-zero.
Normal server startup applies the same blocking argument validation before it
loads an engine, mutates storage, or binds a listener.

## Checks

The report currently covers:

- configuration parsing and effective numeric/feature invariants, including
  request and bounded graceful-shutdown timeouts plus the target runtime's
  maximum admission-semaphore capacity;
- listener authentication and browser-origin posture, including exact-origin
  validation, wildcard warnings, and strict-security rejection;
- runtime-engine registration and maturity;
- selected CPU, GPU, or NPU backend availability;
- safe, bounded model-catalog discovery;
- startup-model metadata, routing, device compatibility, and memory planning;
- writable-catalog quota, stale-staging policy, and acquisition license
  allowlisting;
- signed model-index source/key pairing, bounded keyring uniqueness and
  strength, URL policy, and refresh interval, without fetching the index;
- persistent model-index rollback-state structure and bounds, without exposing
  its path or source identity; and
- presence of the embedded browser UI.

Model checks inspect metadata only. They do not load weights and do not replace
a pinned real-model smoke test on the deployment hardware.

## Privacy and automation

The report does not serialize either API key, absolute model paths, source URLs,
prompts, responses, or raw loader errors. It may contain selected engine and
device names, model counts, Bloom's version, and general remediation guidance.
Review it as deployment metadata before publishing it.

Consumers must check both `schema_version` and `object` before reading fields.
Within schema version `1`, new check IDs may be added, so automation should key
checks by `id` and tolerate unknown IDs. Summary counts must match the check
statuses, and the top-level status is `fail` if any check failed, `warn` if no
check failed but at least one warned, and `pass` otherwise.

See the [JSON Schema](../examples/server-doctor.schema.json) and
[example report](../examples/server-doctor.json).

## Release integration

Official release archives and the Docker image build the Dioxus assets into
`bloom_server`. Release automation executes the staged native binary with
`--doctor=json`, rejects any failed check, and verifies that `embedded_ui`
passes. The Docker builder performs the same embedded-UI assertion with the
text report. Local packaging has the same default; set `BLOOM_PACKAGE_UI=0`
only when intentionally producing a server-only archive.

Every archive also contains `BLOOM-RELEASE.json`, which records its target,
embedded-UI state, self-check result, and SHA-256 digest for each executable.
See the [release manifest guide](release-manifest.md),
[schema](../examples/release-manifest.schema.json), and
[example](../examples/release-manifest.json).

Building the Dioxus CLI from source requires native TLS discovery tooling. On
Debian or Ubuntu, install `libssl-dev` and `pkg-config` before running the
documented `cargo install dioxus-cli` command. The Docker and release builds
install these dependencies explicitly.
