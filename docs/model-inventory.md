# Model Inventory Format

Bloom can export its authenticated local catalog as a portable JSON inventory:

```bash
curl http://127.0.0.1:3000/v1/model-management/inventory \
  -H "Authorization: Bearer $BLOOM_API_KEY" \
  --output bloom-model-inventory.json
```

The Models drawer exposes the same operation as `Export JSON`. The response
uses the filename `bloom-model-inventory.json` and conforms to
[`examples/model-inventory.schema.json`](../examples/model-inventory.schema.json).

## Stability and privacy

Current schema version `2` has these invariants:

- Entries are sorted by catalog ID so unchanged catalog state serializes
  deterministically.
- The document contains direct-child catalog IDs, never the catalog root or an
  absolute model path.
- Transient runtime fields such as `active`, file modification time, and export
  time are omitted.
- Source URL query strings, fragments, and credentials are excluded.
- A verified signed-index pull includes its optional `model_index_id`, allowing
  the stable Ollama selector to survive export, restart, and explicit restore
  without turning that selector into a filesystem path.
- Invalid private provenance records are represented only as
  `provenance_status: "invalid"`; their contents or validation errors are not
  exported.
- All model entries are included, including manually copied and directory
  models without acquisition provenance.

Consumers must reject unknown schema versions. Bloom continues to accept legacy
version `1` inventories, which have no `model_index_id`; current exports and the
published JSON Schema use version `2`.

## Source locking

`source_locked` is deliberately conservative. It is `true` only when a
recorded Hugging Face URL uses this form:

```text
https://huggingface.co/OWNER/REPOSITORY/resolve/COMMIT/FILE
```

`COMMIT` must contain exactly 40 or 64 hexadecimal characters. Branches, tags,
and names such as `main` are treated as mutable even when a SHA-256 checksum is
recorded. The checksum still detects changed bytes, but a mutable URL might no
longer serve the recorded artifact.

The inventory is an audit and backup artifact. Bloom does not treat it as a
bulk executable restore plan and does not independently verify license claims.
An operator can explicitly restore one narrowly eligible missing model through
the verified download pipeline described below.

## Reconciliation preview

Compare a saved version `1` or `2` inventory with the current catalog:

```bash
curl http://127.0.0.1:3000/v1/model-management/inventory/reconcile \
  -H "Authorization: Bearer $BLOOM_API_KEY" \
  -H "Content-Type: application/json" \
  --data-binary @bloom-model-inventory.json
```

The Models drawer exposes the same operation as `Compare JSON`. Bloom strictly
validates the object identity, fields, summary counts, sorted unique IDs,
provenance relationships, HTTPS source metadata, and source-lock state before
comparison. Requests are limited to 16 MiB and 20,000 model entries.

The deterministic response conforms to
[`examples/model-inventory-reconciliation.schema.json`](../examples/model-inventory-reconciliation.schema.json).
It reports `missing`, `unexpected`, and `changed` entries. Security- or
load-relevant differences such as missing models, checksum changes, format
changes, incomplete size accounting, invalid provenance, and quarantine are
blocking; other metadata drift is a warning. To avoid returning uploaded
secrets, changed entries contain field names rather than expected or current
values. At most 200 detailed entries are returned, while summary counts always
cover the complete comparison.

`restorable_count` and each detail's `restore_available` flag describe metadata
eligibility only. They do not imply that downloads are enabled, that storage is
available, or that the remote artifact still exists.

Reconciliation is preview-only. It never installs, downloads, removes, loads,
or modifies a model or its provenance metadata. See
[`examples/model-inventory-reconciliation.json`](../examples/model-inventory-reconciliation.json)
for a complete response example.

## Explicit single-model restore

Start the server with `--enable-model-downloads`, compare an inventory, and
then restore one eligible missing model from the Models drawer. The API form is:

```bash
curl -X POST \
  http://127.0.0.1:3000/v1/model-management/inventory/restore/tiny-q4.gguf \
  -H "Authorization: Bearer $BLOOM_API_KEY" \
  -H "Content-Type: application/json" \
  --data-binary @bloom-model-inventory.json
```

The catalog ID in the path must be URL-encoded. Bloom accepts a restore only
when all of these conditions hold:

- The complete supported inventory passes the same strict validation as a
  comparison.
- The selected model is currently missing and is a complete, non-empty
  single-file model.
- Its recorded acquisition is `download`, with a lowercase SHA-256 digest.
- Its Hugging Face source URL is pinned to an exact 40- or 64-character commit.
- Verified downloads are enabled and the inventory's expected size is within
  the configured per-download limit.

The endpoint queues Bloom's existing trusted-host, bounded, resumable download
pipeline. That pipeline follows only trusted redirects, enforces storage
policy, verifies SHA-256, writes provenance, and atomically installs without
overwriting an existing catalog entry. For a version `2` signed acquisition it
also restores the validated `model_index_id`, so the same Ollama selector is
available after installation. It never immediately loads the restored model;
a later Ollama inference or preload request may activate it.
Only one model is accepted per explicit request; there is no bulk or automatic
restore mode.

Download progress and errors appear in the normal model catalog status. After
a successful restore, compare the inventory again. The bytes, source, and
checksum can match while installation or later-verification timestamps still
appear as non-blocking historical drift.

## Integrity values

| Value | Meaning |
| --- | --- |
| `untracked` | No valid acquisition provenance is available |
| `verified_at_acquisition` | Bloom verified SHA-256 during installation |
| `verified` | A later on-demand integrity check matched the recorded checksum |
| `quarantined` | A later check mismatched; loading is blocked |

See [`examples/model-inventory.json`](../examples/model-inventory.json) for a
complete version `2` example.
