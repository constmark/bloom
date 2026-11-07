# Numeric tensor inference HTTP API

`POST /v1/tensors/infer` exposes the existing CoreML and TensorRT numeric
execution contracts through an authenticated Bloom extension. It is not an
OpenAI generation or embedding operation. The loaded model must advertise the
`tensor` task in `/ready`; text and encoder models receive HTTP 422
`unsupported_operation` on this endpoint.

The request selects the active runtime by its exact published `model` ID,
`"default"`, or an omitted `model`. It never loads an inactive catalog model.
An unmatched selector returns 404, an invalid selector returns 400, an unloaded
runtime returns 503, and a revoked signed model version returns 410.

## CoreML input

Use a flat JSON number array matching the model's fixed input element count:

```sh
curl http://127.0.0.1:3000/v1/tensors/infer \
  -H "Authorization: Bearer $BLOOM_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"model":"default","input":[1,2,3]}'
```

No tokenizer, chat template, generation context limit, or embedding
normalization is applied. The CoreML adapter validates the exact fixed input
shape and executes its existing macOS CPU runtime. See the
[CoreML contract](coreml-tensor.md).

## TensorRT input

Pass the adapter's named input object inside `input`:

```json
{
  "model": "default",
  "input": {
    "inputs": {
      "features": {"shape": [1, 3], "data": [1, 2, 4]}
    },
    "output": "prediction"
  }
}
```

Names must match the engine inputs. `output` selects an output when required
by a multi-output engine; omission delegates to the existing single-output
adapter rule. Each shape must exactly match its flat data count. Rank-zero
shapes (`[]`) represent one scalar element. Integer JSON numbers are preserved
when forwarding the payload, including integers beyond float64's exact integer
range; the worker then validates the engine's actual dtype and shape bounds.
The HTTP adapter does not override dtype, reshape data, or supply missing
inputs. See the [TensorRT contract](tensorrt-tensor.md).

## Response

A successful request returns HTTP 200 and a versioned JSON result:

```json
{
  "schema_version": 1,
  "object": "bloom.tensor_inference",
  "id": "tensor-EXAMPLE",
  "model": "the-exact-loaded-model-id",
  "output": {"element_count": 1, "data": [4.5]}
}
```

`data` contains the selected numeric output as finite float32 values, flattened
in the order produced by the model adapter. `element_count` equals its length.
The existing engine output contract does not publish output dimensions or all
named outputs; this response therefore does not invent a multidimensional
shape. Values are neither probabilities nor L2-normalized embeddings.

The request counts toward inference concurrency and request metrics. Numeric
values are not counted as prompt or generated language tokens. A successful
response reports the exact leased model identity even when `default` was used.

## Bounds and lifecycle

- The JSON body and serialized model input are each limited to 8 MiB.
- Flat input and aggregate named input contain 1 to 1,048,576 finite numbers.
  Values must be representable within float32's finite range; model-specific
  integer/floating dtype checks remain authoritative.
- Named requests contain 1 to 32 inputs, with at most eight dimensions per
  tensor. Dimensions are positive; products must match data lengths and stay
  within the aggregate value limit. Input/output names contain 1 to 256
  characters without surrounding whitespace or controls.
- Output contains 1 to 1,048,576 finite float32 values. Missing numeric output,
  mixed text/image/audio/video output, and non-finite or oversized results fail.
- Unknown fields, duplicate tensor names, query parameters, booleans in numeric
  arrays, generation settings and streaming controls are rejected. The endpoint
  uses the numeric adapters' required default internal generation parameters;
  it has no public sampling controls.

Execution runs on a blocking worker and retains its exact runtime lease and
concurrency permit until the worker settles. Disconnect or HTTP timeout cancels
the client side without making the occupied worker capacity available early.
A vendor call already in progress may finish before its resources can be
released; its late result is not published to the disconnected client. Vendor
runtime deadlines still apply.

Errors use Bloom's `/v1` envelope and `Cache-Control: no-store`, request
correlation and authentication policy. HTTP 429 reports exhausted concurrency;
422 `tensor_inference_failed` reports invalid model-specific input, execution
failure or invalid numeric output; worker panic/join failure returns 500.
Execution errors are fixed and do not echo model paths, inputs or vendor error
text. This bounded extension does not provide streaming, multi-model batching,
remote model loading or arbitrary filesystem inputs.

Unit and HTTP handler fixtures cover exact input forwarding, default/exact model
selection, malformed/unsupported requests, numeric output validation, error
redaction, concurrency and disconnect ownership. Mock execution proves the
transport boundary. Actual CoreML or TensorRT execution still requires the
model and runtime documented for that backend; HTTP availability does not
promote either backend's support status.

A known numeric model can be checked through a real server with:

```sh
python3 scripts/native_operation_smoke.py --model /path/to/affine.mlmodel \
  --backend coreml --input '[1,2,3]' --expected-output '[4.5]'
```

Choose input and expected output for the supplied model. The script uses an
isolated configuration and checks authentication, query/field rejection, task
isolation and model identity in addition to numeric output.

The 2026-11-07 macOS CPU verification used an actual CoreML GLM affine fixture
(weights `[2,-1,0.5]`, bias `3`): HTTP input `[1,2,3]` returned `[4.5]` through
the strict server load/admission path. This is numeric execution evidence, not
a trained-model quality result or an NVIDIA hardware validation.
