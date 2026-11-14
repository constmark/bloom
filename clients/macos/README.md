# Bloom Desktop for macOS

Bloom Desktop is a SwiftUI client for a separately running Bloom HTTP server.
It verifies readiness and API access before sending chat requests. Build and
check the native client with:

```sh
swift build --package-path clients/macos
./scripts/test_macos_client.sh
```

**Models…** opens the server's model catalog. Refresh discovers supported files
and model directories under the server's configured model root; the path refers
to the server machine. Select **Load** to switch models, or **Unload active
model** to release the runtime while keeping its files. The panel shows the
active model, loading progress, and failures; load can be retried after failure.
It polls loading for up to 400 checks, then offers refresh to continue. **Stop
monitoring** cancels the client's request and polling; an accepted server load
continues because the API has no load-cancellation endpoint.

Model management uses the optional **Operator API key**. When empty it uses the
sidebar API key, which works when the server has not configured separate operator
credentials. Chat, embeddings and reranking always use the sidebar API key. Both keys stay
in memory. Changing connection settings or credentials cancels outstanding tool
requests and discards their late responses. Loading or unloading stops the active
chat and encoder request, clears readiness, and verifies the runtime again
after the operation. An encoder can connect successfully; chat is enabled only
for a runtime that advertises generation.

**Embeddings…** provides a complete sentence-vector workflow for an active
embedding model: enter one nonblank input per line, optionally request a vector
width, and select **Generate vectors**. Input is limited to 256 lines, 262,144
characters per line, and 768 KiB of UTF-8 total; dimensions must be 1–16,384.
The panel shows a vector preview and token usage and can copy or save the complete
JSON response. It validates the returned model, row indices, vector widths, and
finite values, L2 normalization and matching token usage before displaying results. **Cancel** stops the client request;
late results cannot replace a newer request. Inputs and vectors stay in memory
unless explicitly copied or saved. Reconnecting clears encoder results and
cancels pending encoder requests, including when another operator has changed
the model remotely. Model download/import panels are not included in this native client.

**Reranking…** ranks documents against a query using an active model that
advertises `rerank`, including both embedding-based and classifier-head
cross-encoders. Enter a query and one document per line; optionally limit the
result count, or leave it empty to return every document. Limits are 65,536
Unicode scalars per query, 256 documents, 262,144 scalars per document, and
768 KiB of combined UTF-8 input. Returned documents, unique indices, descending
scores, stable tie order, model identity and usage are validated before display.
Cancel, connection changes, reconnect and model changes discard late results.
Copy/save exports the original query and ranked documents using the same
versioned `bloom.rerank_result` schema as the browser UI. Editing the query after
completion cannot change the exported request identity. Query, documents and
scores remain outside conversation history and are saved only on explicit export.

Use the sidebar or **Command-N** to create a conversation, select a conversation
to restore its messages and draft, and use **Delete conversation** (or its
context menu) to remove it. Deleting requires confirmation. Deleting the last
conversation creates a fresh empty conversation. **Clear** clears the current
conversation's messages.

Switching, creating, or deleting the active conversation stops its response.
Partial text remains with its original conversation. Late chunks, errors, and
metrics from the cancelled request cannot alter another conversation. Changing
connection settings still requires a fresh readiness check, and an already
started request retains the verified client it started with.

**Max tokens** in the Connection panel bounds each response (1–32,768; default
256) and is saved in user defaults; the next message uses the new value. When
the server reports `finish_reason: "length"`, a notice below the conversation
says the response was truncated at the token limit. Temperature (0.7) and
top-p (0.9) remain fixed.

History is stored at:

```text
~/Library/Application Support/Bloom/conversations.json
```

The versioned archive contains conversation IDs, creation/update times,
messages, drafts, and the selected conversation. API keys, server URLs,
readiness state, stream tasks, and metrics are not stored in it. API keys remain
in memory. As before, the last successfully verified server URL is saved
separately in application preferences. Conversations are local plain text;
the history file is created with owner-only permissions (`0600`), inside a
new owner-only directory (`0700`).

Writes use a same-directory temporary file, flush it, and atomically replace
the archive. Drafts and partial responses are checkpointed at most every
500 ms while changing, and immediately at conversation and application
lifecycle boundaries. An abrupt process or machine failure may lose changes
since the last checkpoint; the existing complete checkpoint remains readable.
An interrupted response restores as text, without restarting generation.

Limits are 128 conversations, 2,048 messages per conversation, 1 MiB of UTF-8
per message or draft, and 16 MiB for the complete serialized archive. A response
that reaches its message limit is stopped with an explanation. Reads are
bounded before JSON decoding and reject nonregular files, symlinks, malformed
archives, unsupported versions, and invalid selections or duplicate IDs.

If history cannot be read, the client preserves it and pauses writes. Use
**Retry loading history** after correcting the problem, or **Back up damaged
history and start fresh** to move the original to a uniquely named
`conversations-recovery-*.json` file before creating a new archive. A failed
save keeps the current conversations in memory and offers **Retry saving**;
its error stays visible until saving succeeds.

The regression harness covers atomic-write failure, bounded reads, permissions,
recovery, restart and draft restoration, credential exclusion, completed and
partial responses, deletion, and delayed callbacks after conversation changes,
in addition to the HTTP/readiness/SSE checks. Model and encoder checks cover the
published catalog fixtures, authenticated routes, load/error/retry/unload,
readiness invalidation, cancelled and superseded operations, response bounds,
invalid vectors and usage, reranking-only task support, ranking validation,
request bounds, cancelled/obsolete ranking results, and JSON export. They use simulated HTTP/model responses and
temporary conversation storage; they do not load, unload, or alter real models.

A separate [live acceptance run](../../docs/capability-logprobs-metal-2026-11-13.md)
used a dedicated local Metal server and generated model fixtures. The packaged
app connected, listed models, switched from text to embedding, and displayed two
four-dimensional vectors. Screenshots and accessibility state verified the
layout/result; this does not replace screen-reader or trained-quality testing.
