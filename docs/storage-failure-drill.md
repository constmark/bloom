# Physical storage failure and release rollback drills

`deployment_storage_drill.py` supplies process-level evidence that a quota
rejection cannot supply: an operating system `ENOSPC`, failed HTTP import,
verified recovery, and restart against the recovered catalog. It also runs a
separate release-directory replacement and rollback against a real model.

## Physical disk exhaustion

On macOS the script creates a private **64 MiB HFS+ sparse disk image** through
`hdiutil`. It refuses insufficient host headroom, a symlink mount, a mount on the
host device, and any filesystem exceeding its 80 MiB safety ceiling. The
filesystem is filled only after these checks; total filler writes have another
capacity bound. It never accepts an existing mountpoint or existing image and
never falls back to filling the host disk. Other operating systems fail
explicitly instead of requesting privileges or mounting anything.

The scenario begins a checksum-controlled 2 MiB synthetic import, exhausts the
image with actual writes and `fsync`, and requires all of these results:

- The operating system returns `ENOSPC`, including for the final small extent.
- The authenticated chunk upload returns JSON HTTP 5xx without publishing a
  partial model, and the health endpoint remains available.
- After removing the filler, the same import resumes from its reported byte
  offset and completes with the exact original SHA-256.
- Staging files disappear, graceful shutdown succeeds, and a restarted process
  reacquires the catalog with the same installed bytes.

The synthetic payload exercises acquisition, not model parsing or quality.
This is evidence for HFS+ inside a disk image, not automatically for APFS,
ext4, network volumes, container storage drivers, or the production mount.

```bash
python3 scripts/deployment_storage_drill.py \
  --server-bin target/debug/bloom_server \
  --scenario physical-enospc \
  --output /tmp/bloom-physical-enospc.json
```

Every exit attempts to detach exactly the created image. If detach fails and the
mount remains active, the script preserves its temporary directory rather than
recursively removing an active mount. It never enumerates or detaches unrelated
user images.

## Release directory switching and rollback

The scenario copies the two supplied executables into private `previous` and
`candidate` directories. It activates each through atomic replacement of a
relative `current` symlink, starts the binary against one shared catalog and
real model, validates readiness and an actual deterministic completion, and
requires a clean graceful drain. It switches back and checks the original
executable hash and deterministic model output.

```bash
python3 scripts/deployment_storage_drill.py \
  --server-bin /path/to/candidate/bloom_server \
  --previous-server-bin /path/to/previous/bloom_server \
  --model /path/to/validated/model \
  --scenario release-switch \
  --output /tmp/bloom-release-switch.json
```

Omitting `--previous-server-bin` uses the same binary in both slots. The report
marks `distinct_binaries: false`; this is a procedure rehearsal, not cross-version
compatibility evidence. Executables are copied rather than hard-linked, capped
at 1 GiB each, and checked for identity changes before and after the drill.
Credentials, configuration, ports, logs and catalogs are isolated.

The harness does not exercise a system service manager, reverse proxy,
signed-model upgrade transaction, distributed incident response, or zero
downtime rollout. Repeat through the actual launchd/systemd/orchestrator before
claiming that environment is supported.

Use `--scenario all --model ...` to run both scenarios. Reports bind both
executable hashes, the model content digest, CPU backend/device, and platform.
Failed or unsupported scenarios exit nonzero. Safety regressions never mount
or fill a filesystem:

```bash
python3 scripts/test_deployment_storage_drill.py
```

## Recorded local execution

On 2026-11-11 the macOS ARM64 run passed both scenarios using the current debug
server and a deterministic tiny Qwen2 fixture. Image capacity was 67,067,904
bytes; 65,495,040 filler bytes caused `ENOSPC`. Upload returned HTTP 500, health
stayed available, no partial model was published, and the 2,097,152-byte import
recovered with verified bytes after restart. Three release activations generated
HTTP 200 completions; rollback restored the output. Both slots used the same
executable, so this is a procedure rehearsal. Retain the generated JSON with
the exact candidate's evidence; rebuilding invalidates the executable binding.

A second run on the same date used the retained Candle 0.10.2 debug executable
(SHA-256 prefix `33ef65d3d4ed`) as the previous build and the Candle 0.11.1 release
executable (`bcc9e6eac8b1`) as the candidate, with the pinned trained Qwen2 Q4_0
model (`aca679832ded`). Physical ENOSPC recovery passed again. Previous,
candidate, and rollback all returned HTTP 200 from real-model completion and
produced the same deterministic output digest; `distinct_binaries` was true.
Both source packages still identify as 0.1.0, so this establishes a cross-build
rollback for that exact model and host, not compatibility between two published
version tags or service managers.
