# Container publication and hardened deployment

The tag-triggered `container-release.yml` workflow defines the official CPU
container contract. The repository must enable GHCR package publication and
GitHub artifact attestations for the workflow token. Adding this workflow does
not publish an image; it runs only when a version tag is pushed.

## Publication gates

The exact tag must equal `v` plus the workspace package version. The workflow
first executes the full CI from the same source revision. It then builds native
Linux AMD64 and ARM64 images on separate runners, with immutable action
references and explicit Buildx, Syft and Trivy versions.

Each image is built once and loaded locally. Before a registry login or push:

1. A real container must run with a read-only root, all capabilities removed,
   no new privileges, UID 10001, bounded memory/PIDs and a private writable data
   volume. Liveness, validated empty-model readiness, embedded UI and clean
   SIGTERM drain must pass. The image must also contain a validated native/UI
   CycloneDX dependency inventory and the exact reviewed dependency policy.
2. Syft inventories the final runtime image as SPDX JSON.
3. Trivy scans OS and library packages. Any high or critical vulnerability fails
   the build, including vulnerabilities without an available fix. Scanner or
   database failures also fail the gate. Native and UI Rust dependency audits
   are separate mandatory checks in the preceding full CI.

The Docker builder runs the same dependency-policy, Laya-source and runtime
dependency checks as native archive packaging, using its own exact native
target plus the separate WASM graph. The resulting `BLOOM-SBOM.cdx.json` and
policy are bundled under `/usr/share/bloom/`. This preserves target-specific
Rust dependency coverage alongside Syft's final runtime-package inventory.

Only the tested local image is pushed; it is never rebuilt after scanning.
A descriptor binds its registry digest to the tested image ID and the SHA-256
of the runtime report, scan result and SBOM. GitHub signs and publishes build
provenance and SBOM attestations for each platform digest.

The final job accepts exactly one validated AMD64 descriptor and one ARM64
descriptor for the same image, tag and source revision. It constructs the
versioned OCI index using those immutable digests, verifies the registry index
contains exactly those two Linux platforms, and attests the index. The
`container-release.json` artifact supplies the immutable deployment reference.
The per-platform evidence remains in workflow artifacts.

Temporary `build-RUN-ATTEMPT-ARCH` tags can remain if later attestation or index
promotion fails. They are candidates, not promoted releases. There is no
automatically advanced `latest` tag. Deploy by the final digest and verify its
provenance against this repository.

The runtime smoke test covers an empty catalog, security context and signal
handling. It does not replace trained-model quality, production-memory,
saturation, long-soak or orchestrator-specific deployment-cell evidence.

## Deployment

Obtain `immutable_reference` from a successful release's `container-release.json`
and verify its attestation before deployment:

```bash
export BLOOM_IMAGE="$(python3 -c 'import json; print(json.load(open("container-release.json"))["immutable_reference"])')"
gh attestation verify "oci://$BLOOM_IMAGE" --repo constmark/bloom
export BLOOM_API_KEY="$(openssl rand -hex 32)"
export BLOOM_OPERATOR_API_KEY="$(openssl rand -hex 32)"
docker compose -f deploy/compose.yml up -d
```

The [Compose reference](../deploy/compose.yml) binds only host loopback, keeps
configuration/catalog data on a persistent named volume, enforces distinct
credentials through strict server startup, caps memory/PIDs/CPU and storage,
and grants forty seconds for the server's thirty-second graceful drain.
Adjust resource budgets for the validated model before enabling inference.
Store credentials in the deployment's secret manager instead of committing a
populated environment file. Containers with an empty catalog are live but
correctly return HTTP 503 from `/ready`.

For remote clients, put a TLS reverse proxy with explicit rate and request
limits in front of loopback. Configure the server's exact browser origin as
needed. Health, readiness and metrics need the deployment's network ACL;
this reference intentionally exposes no host-wide listener.

## Local verification and limits

```bash
python3 scripts/test_container_release.py
python3 scripts/check_container_workflow.py
python3 scripts/test_container_runtime.py \
  --image bloom:local --output /tmp/bloom-container-runtime.json --require-native-sbom
```

The first two checks are offline. The third tests an already built local image,
creates its own container and named volume, and cleans both on exit.
On 2026-11-11 it passed against the existing local `bloom:production-readiness`
image (`sha256:7c925cbb4643ec0c827729580f1f075e1853188a625ec30ef02114392ce1f414`).
That validates the runtime-check procedure, not a newly published image.

The current-source Linux ARM64 builds were attempted with explicit process
deadlines and bounded retries. The pinned Docker frontend and base images resolved, but Debian's
bookworm InRelease endpoint returned HTTP 502 in the runtime stage and HTTP 500
with an unexpected EOF in the builder on retry. Bounded APT retries and a
thirty-second HTTP timeout did not resolve the upstream transfer failure. APT
signature checks remained enabled and both attempts exited 100; no new image
was produced or pushed. The separate Linux ARM64 native/WASM Cargo metadata
successfully passed the bundled dependency-policy/SBOM generator. These results
do not establish a successful current-source Docker build or vulnerability scan.

The multi-architecture workflow, live vulnerability database results, registry
publication and remote attestation verification must still execute successfully
for a real tag before an official image exists. Repository-side contract tests
must not be presented as evidence of completed remote publication.
