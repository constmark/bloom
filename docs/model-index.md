# Signed Model Discovery Index

Bloom can expose a curated model list without trusting mutable repository
branches or unsigned catalog metadata. The server reads a bounded local file or
HTTPS response, verifies an Ed25519 signature against an operator-configured,
bounded trust set, validates every entry, and gives the authenticated Models
drawer a safe discovery snapshot.

This feature is optional. Manual source inspection and checksum-verified
downloads remain available when no index is configured.

## Trust boundary

An index signature authenticates the exact payload bytes and the configured
publisher. It does not prove that a model is safe, useful, free of malicious
behavior, or correctly licensed. The publisher supplies size, license, and
checksum metadata. Bloom independently enforces its configured size and license
admission policies, downloads only an exact Hugging Face commit, and verifies
every acquired byte against signed SHA-256 metadata before installation. A
version 1 entry describes one catalog file. A version 2 package entry describes
one contained Transformers directory and every file that may appear below it.

Keep signing seeds offline and configure only public keys on Bloom hosts.
TLS still protects request privacy and availability for a remote index; the
signature protects payload authenticity.

## Server configuration

Choose exactly one source and provide one to eight trusted public keys:

```bash
bloom_server \
  --models-dir /srv/bloom/models \
  --enable-model-downloads \
  --model-index-file /etc/bloom/model-index.signed.json \
  --model-index-public-key ea4a6c63e29c520abef5507b132ec5f9954776aebebe7b92421eea691446d22c
```

For a remote source, replace `--model-index-file` with an HTTPS URL:

```bash
--model-index-url https://models.example.org/bloom/index.json
```

The equivalent environment variables are:

- `BLOOM_MODEL_INDEX_FILE`
- `BLOOM_MODEL_INDEX_URL`
- `BLOOM_MODEL_INDEX_PUBLIC_KEY`
- `BLOOM_MODEL_INDEX_PUBLIC_KEYS` (comma-separated additional rotation keys)
- `BLOOM_MODEL_INDEX_REFRESH_SECONDS` (default `300`, range `1` to `86400`)
- `BLOOM_MODEL_INDEX_STATE_DIR` (private persistent rollback state)

Each public key is exactly 32 bytes encoded as 64 hexadecimal characters or
unpadded base64url. Bloom accepts one to eight unique, non-weak keys. The
singular option remains the backward-compatible primary setting; the plural
option is additive and enables a bounded overlap window during rotation. Local
index paths must be regular, non-symlink files. Remote sources require
default-port HTTPS without credentials, query strings, fragments, or loopback
hosts. Redirects are limited and must preserve those rules.

The rollback state directory defaults to `model-index-watermarks` beside the
effective Bloom config file. Override it with `--model-index-state-dir` for
service deployments. Bloom creates it only after the first index generation is
successfully verified. The directory must be private, durable, writable by the
server, and backed up with its configuration. Do not place unrelated files in
it. On Unix, Bloom creates the directory with mode `0700` and rejects a state
directory that is writable by group or other users.

Run `bloom_server --doctor` with the effective configuration before deployment.
For one key the report exposes a short key ID; for a rotation set it exposes
only the key count and a deterministic trust-set fingerprint. It never exposes
the source path, URL, or public-key material.

## Publisher workflow

Start from the version 1 [single-file payload example](../examples/model-index-payload.json)
or the version 2 [package payload example](../examples/model-index-payload-v2.json),
then validate it with the shared
[Draft-07 payload schema](../examples/model-index-payload.schema.json). Every
download URL must use this shape:

```text
https://huggingface.co/OWNER/REPOSITORY/resolve/40_OR_64_HEX_COMMIT/PATH/FILENAME
```

Create a 32-byte Ed25519 seed with an audited random-number generator and
protect it as a secret. One OpenSSL-based example is:

```bash
umask 077
openssl rand -hex 32 > model-index-signing-key.hex
```

From a Bloom source checkout, sign the raw payload bytes with the repository
helper. The output path must not already exist, which prevents accidental
replacement. Application archives include the format documentation and
schemas, but not the Cargo-based publisher helper:

```bash
cargo run -p bloomai-engine --example sign_model_index -- \
  --payload model-index-payload.json \
  --private-key model-index-signing-key.hex \
  --output model-index.signed.json
```

The helper prints the public key and its SHA-256 key ID. Publish the signed
envelope and distribute the public key through a separate trusted deployment
channel. Never publish or copy the seed to a Bloom server.

The bundled [signed example](../examples/model-index.signed.json) uses a public
demonstration key and fictitious model metadata. It is only a protocol fixture,
not a trusted or downloadable catalog, and its validity window ends in July
2027.

## Wire format

The envelope follows the
[envelope schema](../examples/model-index-envelope.schema.json):

```json
{
  "schema_version": 1,
  "object": "bloom.signed_model_index",
  "algorithm": "ed25519",
  "key_id": "lowercase SHA-256 of the 32-byte public key",
  "payload": "unpadded base64url of the raw JSON payload bytes",
  "signature": "unpadded base64url Ed25519 signature"
}
```

The envelope and decoded payload must use the same schema version. The signed
message is one of these byte concatenations:

```text
schema 1: "bloom.model_index.v1\0" || raw_payload_bytes
schema 2: "bloom.model_index.v2\0" || raw_payload_bytes
```

Domain separation prevents a valid signature from being reused as another
Bloom object type. The server uses strict Ed25519 verification, rejects weak
keys and malformed signatures, selects the configured key by the envelope's
`key_id`, and rejects unknown fields.

### Version 1 single-file entries

A version 1 entry has one top-level `download_url`, `filename`, `size_bytes`,
and `sha256`. Its destination is one direct catalog child ending in `.gguf`,
`.onnx`, or `.mlmodel`. Version 2 may also carry this single-file shape for
forward-compatible publishers, but it does not change the installation
semantics.

### Version 2 model packages

A multi-file entry is valid only in version 2. Its top-level `filename` is a
non-hidden catalog directory name, `size_bytes` is the exact sum of all file
sizes, and top-level `download_url` and input `sha256` are absent. The `files`
array is the complete signed manifest. It contains between 2 and 256 unique,
case-insensitive relative paths, must include root `config.json`, and permits
only `.json`, `.safetensors`, `.txt`, `.model`, and `.tiktoken` data files. The
weight layout is either consolidated root `model.safetensors` or a canonical,
complete `model-00001-of-00002.safetensors` sequence accompanied by
`model.safetensors.index.json`; consolidated and sharded layouts cannot be
mixed. Paths are at most 512 bytes and eight
components; absolute, hidden, traversal, percent-encoded, backslash, control,
symlink, and unexpected entries fail closed. Every file has a positive signed
size and lowercase SHA-256. All URLs must identify the same Hugging Face
repository and exact 40- or 64-hex commit, and each URL path must match its
manifest filename.

Bloom derives the normalized entry's top-level `sha256` as an order-independent
package identity. Sort files by their UTF-8 filename bytes and compute:

```text
SHA256(
  "bloom.model_package.v1\0" ||
  u32_be(file_count) ||
  for each file:
    u32_be(filename_byte_count) || filename_bytes ||
    u64_be(size_bytes) || raw_32_byte_sha256
)
```

The server stores package downloads below a hidden staging directory, resumes
only an exact persisted manifest, reserves the aggregate signed size against
the shared storage quota, and verifies every file before publication. The
server then runs the same strict shard resolver used by manifest inference and
Candle loading. For indexed checkpoints it reads the actual Safetensors headers
and rejects incomplete numbering, index/file membership differences, duplicate
tensors, invalid or overlapping offsets, incorrect tensor ownership, and false
`metadata.total_size` values. Only then is the catalog directory renamed into
place as one no-overwrite operation; on Linux this uses `RENAME_NOREPLACE`.
Unix builds sync verified files, package directories, provenance, and rename
parents to reduce power-loss ambiguity. A bad size, checksum, logical shard
layout, symlink, unexpected file, cancellation, or destination race cannot
expose a partial catalog directory. Version 2 provenance retains the canonical
per-file manifest and package digest. Later integrity checks rescan the exact
tree and every file; removal deletes both the directory and its provenance
record.

## Bounds and refresh behavior

- Signed envelope: at most 512 KiB.
- Decoded payload: at most 384 KiB.
- Entries: at most 200.
- Index validity: at most 366 days; generation may be at most one hour ahead of
  the server clock.
- Entry URLs: at most 2,048 bytes and pinned to an immutable commit.
- Package files: 2 to 256, each at most eight safe relative path components.
- Duplicate IDs and destination filenames are rejected.

Within the successful refresh interval, `GET /v1/model-management/index`
returns the verified cache. `POST` forces a refresh. If refresh fails, Bloom may
return only the last verified snapshot while it remains unexpired, marked
`cache_status: "stale"` with a warning. An expired snapshot is never returned.
Both methods require the same API authentication as other `/v1` endpoints and
return `Cache-Control: no-store`. Clients must require `schema_version: 1` or
`2` and `object: "bloom.model_index"` before consuming the normalized response.
See the [response schema](../examples/model-index-response.schema.json),
[version 1 example](../examples/model-index-response.json), and
[version 2 package example](../examples/model-index-response-v2.json).

A successfully verified generation becomes a source-scoped persistent rollback
watermark before the response is exposed. A signed response with an older
`generated_at`, or a different response reusing the same generation time, is
rejected across restarts. Bloom uses immutable atomic records, retains the two
latest generations for each source identity, and bounds the directory to 64
records. Corrupt, unexpected, symlinked, oversized, or unwritable state fails
closed. The newer in-memory snapshot remains available only while unexpired.
Publishers must therefore increase `generated_at` for every content or
signing-key transition.

Entries exceeding the server download-size limit or acquisition-license policy
remain visible but have `downloadable: false` and a bounded blocking reason.
For an allowed entry, the GUI calls
`POST /v1/model-management/index/{id}/download`. The server looks up the ID in
its verified snapshot and supplies the URL, destination, size, hashes, license,
and package manifest itself; the browser cannot echo or replace signed
acquisition fields.

The endpoint uses one server-authoritative installation-state check shared with
Ollama pull. A missing entry starts the verified acquisition, an identical
in-progress acquisition is joined, and a completely matching installed entry
returns success without network traffic. An installed match requires the exact
catalog filename, file-or-directory shape, format, complete size, aggregate
digest, package file count, license, download acquisition kind, persistent
signed-index ID, and no recorded integrity mismatch. One clean prior download
with the same persistent index ID is an upgrade candidate, including when the
signed destination or file/directory shape changed. Occupied destinations,
duplicate aliases, quarantined entries, and other ambiguous local states return
HTTP 409. The Models drawer applies the same fail-closed comparison to display
`Installed`, `Upgrade signed model`, or an actionable local conflict before the
user clicks.

An upgrade keeps the previous entry intact while the replacement is staged, so
quota admission requires enough room for both complete payloads. Resume metadata
binds the exact previous and replacement identities. Commit then rehashes both
identities under the shared storage lock, records a bounded transaction under
`.bloom-upgrade`, backs up the old provenance, moves the old payload aside, and
atomically publishes the replacement and its new provenance. Load, removal, and
integrity admission reject the source while this transaction is active; an
already loaded source must be unloaded before upgrade. On restart Bloom either
restores the previous model or completes a fully verified replacement according
to the durable marker. Corrupt, symlinked, malformed, unexpected, or ambiguous
transaction state fails closed. A newly started or joined upgrade returns
`upgrading: true`; ordinary installation and installed reuse return `false`.

The same verified snapshot powers Bloom's bounded `POST /api/pull` adapter.
That route accepts only an exact entry ID and derives the URL, destination,
signed size, SHA-256, and license from this snapshot; the client cannot replace
any acquisition field. Streaming and non-streaming Ollama clients therefore
reuse the existing downloader without gaining a registry or arbitrary-URL
bypass. A successful pull installs the entry's single file or complete package
directory under `filename` but does not load it. If that ID names one clean
prior signed download, pull uses the same transactional upgrade rather than
overwriting it in place.
The downloader also persists the signed entry ID in acquisition provenance.
Ollama discovery and lifecycle routes therefore keep exposing that ID after a
restart, while the filename remains Bloom's contained storage destination. A
later Ollama inference or empty preload request using the same ID performs the
normal integrity, compatibility, and atomic runtime-switch checks.
See [Ollama API compatibility](ollama-compatibility.md#verified-pull).

## Rotation and incident response

Use an explicit overlap deployment instead of replacing a key and index at the
same instant:

1. Add the new public key through `--model-index-public-keys` while retaining
   the old key, restart Bloom, and confirm that `--doctor` reports two trusted
   keys.
2. Publish an index signed by the new key with a strictly later `generated_at`.
3. Refresh the authenticated index endpoint and confirm that its `key_id`
   matches the new key's printed key ID.
4. Remove the old key from configuration and restart every Bloom instance.

The model catalog capability includes a deterministic `trust_id`; the GUI uses
it to invalidate an older discovery view when the configured trust set changes.
The trust-set order does not affect this fingerprint. While the Models drawer
is available, the GUI also polls the authenticated endpoint at the server's
bounded refresh interval, so a newer generation appears without a browser
reload. A missing snapshot is retried within 30 seconds.

The same capability reports `persistent_rollback_protection: true`; the Models
drawer displays this state beside the verified cache status.

Bloom does not fetch a remote revocation list. Short index validity windows and
prompt removal of retired keys therefore remain important. If a seed may be
compromised, preserve the rollback state, remove the key or disable the index
immediately, publish a newer generation under a new key, review installed
provenance, and reverify model files from a known trusted source. A privileged
local actor that can delete both the state directory and process memory can
reset this protection; local state integrity remains an operator responsibility.
