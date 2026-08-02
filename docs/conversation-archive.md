# Conversation Archive

Bloom UI stores conversations in the current browser profile. The conversation
sidebar can export that state to `bloom-conversations.json` and restore it in a
different browser or deployment.

The archive is intentionally narrow. It contains conversation titles, user and
assistant message text, conversation order, the active-conversation index, and
optional assistant execution-model and structured-output rendering metadata.
It does not contain the Bloom server address, API key, generation settings,
system prompt, model inventory, request IDs, timing, token counts, or image
bytes. A displayed image attachment remains only as the text marker already
present in its user message.

## Version 2

The top-level object contains:

- `version`: the integer `2`
- `object`: the fixed string `bloom.conversation_archive`
- `active_conversation`: a zero-based index into `conversations`
- `conversations`: one or more ordered conversation objects

Each conversation has a non-empty `title` and a `messages` array. Each message
has a `role` of either `user` or `assistant` and a string `content` value. Local
conversation IDs are never exported; import assigns fresh sequential IDs. A
user message can include `attachment_unavailable: true` when its original image
bytes were intentionally not persisted. The field prevents an imported image
request from being incorrectly regenerated as text; false values are omitted.
An assistant message can include `response_format: "json_object"` or
`response_format: "json_schema"`. This preserves escaped code rendering without
exporting the schema or other generation settings.
An assistant message can also include the exact bounded `model` identity
confirmed by its stream. Bloom uses the latest recorded identity to stop Send,
Retry, and Edit after an active-model change until the user starts a new chat or
explicitly acknowledges sending existing history to that exact model. The
archive does not include timing, token usage, request IDs, completion outcomes,
or generation settings.

Bloom continues to import strict version 1 archives. Version 1 has the same
shape except that it cannot contain `model`; imported responses therefore have
unknown model provenance until a new confirmed response is recorded. New
exports always use version 2.

See the [Draft-07 JSON Schema](../examples/conversation-archive.schema.json) and
[example archive](../examples/conversation-archive.json).

## Import behavior and limits

Bloom parses and validates the file, shows the conversation and message counts,
and requires the user to choose **Merge** or **Replace all**. Merge appends each
archived conversation in archive order, assigns fresh local IDs above the
current ID space, and preserves every existing conversation plus the current
active selection. It intentionally does not deduplicate identical
conversations. Replace all restores the archive's ordering and active selection
without retaining current history. Connection settings, generation settings,
and other excluded state are unchanged in either mode.

Both modes construct and validate a complete candidate store and write it to
`localStorage` before changing visible history. Capacity, ID-space, validation,
or storage failures therefore leave the current state untouched. Merge checks
the limits across existing and imported history together. When saved history is
unreadable and recovery-locked, Merge is disabled: users can download the raw
recovery copy, cancel, or explicitly choose Replace all, but a valid archive
cannot accidentally merge into the temporary empty recovery view and overwrite
the unreadable bytes.

The UI rejects files larger than 8 MiB, unknown fields or versions other than 1
and 2, version 1 model fields, invalid or non-assistant model provenance, an invalid
active index, more than 1,000 conversations, more than 50,000 messages in total,
titles longer than 200 characters, message content longer than 1,000,000
characters, control characters in titles, and unsupported roles. The same
1,000-conversation and 50,000-message ceilings apply to the combined store when
merging. These limits bound browser memory use and make future format changes
explicit.

Loading a large valid archive does not mount every message into the document at
once. Bloom initially renders the latest 100 messages and reveals earlier
history in 100-message pages. This is only a rendering window: export, branching,
search, validation, and request admission continue to use the full stored
conversation.

Conversation text remains untrusted after import. Normal assistant messages pass
through the constrained Markdown renderer; structured assistant messages render
as escaped code. User messages remain plain text.

## Local branches and continuations

Branching is a browser-local conversation operation, not a distinct archive
type. A branch contains cloned history through one selected message and has a
fresh local ID plus a unique bounded title; the source remains unchanged. On
export, branches are ordinary conversations and import assigns them fresh IDs
like every other entry. Runtime-only generation diagnostics are omitted from the
archive even when they exist in a local branch; bounded assistant model
provenance is retained. Attachment availability markers are retained, but image
bytes are never added to the archive.

A continuation is also an ordinary browser-local conversation. It begins at an
explicitly selected user message after the first and retains every later
message. Earlier messages remain only in the unchanged source conversation.
This user-controlled suffix operation helps fit a long discussion into a model
context window without silent truncation, invented summaries, or loss of the
original history. It uses the same bounded title, capacity, model-provenance,
structured-output, and attachment-marker behavior as a branch.
