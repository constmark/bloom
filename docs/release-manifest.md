# Release Manifest

Every Bloom application archive contains `BLOOM-RELEASE.json`. It identifies
the build and lets users or automation verify extracted executables without
depending on archive filenames. The same archive contains the deterministic
CycloneDX 1.5 `BLOOM-SBOM.cdx.json` and the exact reviewed
`BLOOM-DEPENDENCY-POLICY.json` used to admit its Cargo dependency graph.

Current archives use:

- `schema_version: 2`; and
- `object: "bloom.release"`.

The offline validator continues to accept legacy schema-version-1 archives,
which predate the required packaged dependency policy and SBOM. Version 2
requires both inventory files and validates them as part of the archive
contract.

Unknown schema versions must be rejected. Within version `1`, consumers should
require the complete target-specific executable set and validate every declared
size and hash.

## Fields

| Field | Meaning |
| --- | --- |
| `bloom_version` | Workspace version compiled into `bloom_server` |
| `target` | Rust target triple used for the native executables |
| `embedded_ui` | Whether `bloom_server` contains the browser application |
| `self_check.status` | `passed` for a native doctor run, or `not_run_cross_target` when the build host could not execute the target |
| `self_check.doctor_status` | Doctor `pass` or `warn` result when executed; otherwise `null` |
| `self_check.failures` | Zero for a passed native self-check; otherwise `null` |
| `binaries` | Sorted executable names, byte sizes, and lowercase SHA-256 digests |

Windows targets use `.exe` binary names. Linux and macOS targets use
extensionless binary names with executable archive modes.

A doctor warning is compatible with a passed package self-check. For example,
an intentionally empty model catalog warns because the application is not
ready for inference, while still proving that the executable, CPU backend,
configuration defaults, and embedded UI are usable.

Cross-target packages are explicitly marked rather than claiming execution
evidence that the build host could not produce. Official GitHub Release jobs
build natively and therefore require `self_check.status: "passed"`.

## Integrity boundary

The release manifest is inside the archive. Its hashes detect accidental
changes after extraction and support local inventory tooling, but the manifest
cannot authenticate itself. Verify the separately published archive checksum
before trusting its contents. Official GitHub archives and checksum files also
receive signed build-provenance attestations before their draft release is
created. Follow the bundled quickstart to bind a downloaded archive to this
repository and workflow; local archives do not receive that hosted attestation.

The manifest intentionally excludes build timestamps and absolute paths. This
keeps it stable across equivalent builds and prevents local build information
from entering published artifacts.

The packaging script normalizes tar.gz and zip entry ordering, timestamps,
owners, and modes from `SOURCE_DATE_EPOCH`. If the variable is absent in a Git
checkout, the current commit time is used. Archives are written to a same-
directory temporary file and atomically replaced only after compression
succeeds. This makes the archive container reproducible for an identical staged
tree; it does not claim bit-for-bit reproducibility across separate native
compiler or linker executions.

The SBOM merges target-filtered locked Cargo metadata for the native workspace
with the independent `wasm32-unknown-unknown` UI workspace when the browser is
embedded. Policy validation runs before either workspace is built. It lists workspace and registry packages,
their declared license expressions, reviewed source, and resolved dependency
edges without local paths or timestamps. The policy rejects undeclared or new
license expressions, non-crates.io registries, Git dependencies, and external
path dependencies until a maintainer explicitly reviews and updates the policy.
Known legacy slash-separated Cargo declarations are preserved in a component
property and explicitly normalized to an SPDX `OR` expression for CycloneDX
consumers; every such normalization is itself part of the reviewed policy.
This is a reproducible inventory and drift gate, not a legal conclusion about
license compatibility or proof that every resolved package contributed machine
code to every binary.

From a Bloom source checkout, validate either supported archive format without
extracting it:

```bash
./scripts/validate_release_artifact.py /path/to/bloom-target.tar.gz \
  --checksum /path/to/bloom-target.tar.gz.sha256 \
  --require-embedded-ui \
  --require-native-self-check \
  --require-deterministic-metadata
```

The validator also accepts Windows `.zip` archives. It rejects unsafe paths,
links, duplicate members, oversized archives, incomplete documentation,
incorrect executable modes, inconsistent self-check metadata, and binary hash
or size mismatches. It also parses the bounded SBOM and packaged policy,
requires their target, version, and embedded-UI identity to match the release,
and rejects incomplete dependency graphs or unreviewed sources and license
expressions. The release-only metadata flag additionally requires one
normalized archive timestamp, canonical ownership and modes, and restricted
extended metadata. The validator also parses the packaged readiness example and
schema within a 64 KiB budget, requires the current Bloom identity and server
protocol, validates the positive ordered UI compatibility range and complete
example invariants, and checks the compatibility-critical Draft-07 schema
structure. A stale v2 document or a nonempty placeholder therefore cannot pass
as a current release contract.

See the [Draft-07 JSON Schema](../examples/release-manifest.schema.json),
[example](../examples/release-manifest.json), and bundled
[application quickstart](release-quickstart.md).
