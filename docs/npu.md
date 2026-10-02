# Intel NPU Guide

Bloom currently provides lightweight Intel NPU detection through the
`intel-npu` backend. Actual OpenVINO execution is handled by the OpenVINO
engine path and requires an OpenVINO-exported model.

## Quick Check

```bash
cargo run --bin bloom_infer -- --list-engines
```

On Linux, Bloom checks common `/dev/accel*` nodes, `ivpu`/`intel_vpu` driver
locations, and OpenVINO runtime hints. On Windows, Bloom checks common driver,
DLL, OpenVINO install, and environment variable locations.

## OpenVINO Runtime

Install OpenVINO using one of the official distribution paths:

```bash
pip install openvino
```

When Bloom needs a Python-backed NPU/TTS helper, set `BLOOM_PYTHON` to the
interpreter that contains the OpenVINO packages. ModelScope downloads use
`BLOOM_MODEL_ROOT` when set; otherwise they go to the current user's
`models` directory on both Unix and Windows.

To run through the OpenVINO engine:

```bash
cargo run --bin bloom_infer -- \
  --backend intel-npu \
  --engine openvino \
  --model /path/to/openvino-ir-model \
  --prompt "hello"
```

If you have a raw AWQ Hugging Face model, export it to OpenVINO IR first. For
experimentation, Bloom can run the export command when explicitly enabled:

```bash
BLOOM_OPENVINO_AUTO_EXPORT=1 cargo run --bin bloom_infer -- \
  --backend intel-npu \
  --engine openvino \
  --model /path/to/awq-model \
  --prompt "hello"
```

That path requires:

```bash
pip install "optimum-intel[openvino]" openvino nncf
```
