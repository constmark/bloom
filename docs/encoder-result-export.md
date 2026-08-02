# Encoder Result Exports

Bloom's browser encoder workspace can copy one complete result or download the
complete validated batch. Copy and export are explicit user actions; results
are not written to browser persistence automatically.

## Files

Embedding downloads use `bloom-embeddings.json` with object identity
`bloom.embedding_result`. Every vector retains its zero-based response index,
exact submitted input, and complete normalized float array. Rerank downloads
use `bloom-rerank.json` with object identity `bloom.rerank_result`. They retain
the response request ID, active model, exact query, prompt-token usage, stable
rank order, original document indices, scores, and complete documents.

Both formats use `schema_version: 1`. Their Draft-07
[JSON Schema](../examples/encoder-result.schema.json) and
[embedding](../examples/embedding-result.json) and
[rerank](../examples/rerank-result.json) examples are included in application
archives.

## Validation and limits

The UI revalidates a result immediately before copying or exporting it. An
embedding artifact must contain 1 through 256 contiguous indexed vectors with
one consistent width of 1 through 16,384 values, at most 1,048,576 values in
aggregate, finite components, an L2 norm within 0.001 of one, and at most
768 KiB of associated input. Its encoded download is limited to 40 MiB.

A rerank artifact must contain a bounded request ID and query plus 1 through
256 unique document results. Scores must be finite and within `[-1, 1]`, sorted
descending with the original document index breaking ties. Query and returned
document content is limited to 768 KiB in aggregate, and the encoded download
is limited to 4 MiB.

Copying a vector writes its complete JSON float array. Copying a rerank result
writes one JSON object containing `index`, `relevance_score`, and `document`.
Each clipboard payload has an independent 1 MiB limit and uses the browser's
secure-context clipboard permission boundary.

The schema describes structural interoperability. Bloom's runtime checks the
cross-field index, ordering, aggregate-size, consistent-dimension, finite-value,
and vector-normalization invariants that Draft-07 cannot express by itself.

## Privacy

Exports intentionally contain the submitted text because a vector without its
input association is difficult to audit or reuse safely. Treat these files as
private application data. API keys, server URLs, filesystem paths, connection
settings, and browser-local conversation history are never included.
