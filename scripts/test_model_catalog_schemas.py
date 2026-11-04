"""TensorRT acquisition contracts; use requirements/schema-validation.txt."""
import copy
import json
from pathlib import Path
import unittest

try:
    import jsonschema
except ImportError:
    jsonschema = None

ROOT = Path(__file__).resolve().parents[1]


def example(name):
    return json.loads((ROOT / "examples" / name).read_text())


@unittest.skipUnless(jsonschema, "install requirements/schema-validation.txt")
class TensorRtCatalogSchemas(unittest.TestCase):
    def test_catalog_accepts_normalized_tensor_format_and_rejects_unknown_formats(self):
        schema = example("model-catalog.schema.json")
        for extension in ("plan", "engine"):
            document = example("model-catalog.json")
            filename = f"numeric.{extension}"
            document["data"][0].update(id=filename, format="tensorrt")
            document["active_model"]["catalog_id"] = filename
            jsonschema.validate(document, schema)
            document["data"][0]["format"] = "unknown_runtime"
            with self.assertRaises(jsonschema.ValidationError):
                jsonschema.validate(document, schema)

    def test_signed_payload_accepts_weights_and_rejects_traversal_or_other_extensions(self):
        schema = example("model-index-payload.schema.json")
        for extension in ("plan", "engine"):
            document = example("model-index-payload.json")
            document["models"][0]["filename"] = f"numeric.{extension}"
            jsonschema.validate(document, schema)
            for filename in (f"../numeric.{extension}", ".hidden.engine", "numeric.spv"):
                invalid = copy.deepcopy(document)
                invalid["models"][0]["filename"] = filename
                with self.assertRaises(jsonschema.ValidationError):
                    jsonschema.validate(invalid, schema)

    def test_signed_response_preserves_suffix_format(self):
        schema = example("model-index-response.schema.json")
        for extension in ("plan", "engine"):
            document = example("model-index-response.json")
            document["data"][0].update(filename=f"numeric.{extension}", format=extension)
            jsonschema.validate(document, schema)
            document["data"][0]["format"] = "tensorrt"
            with self.assertRaises(jsonschema.ValidationError):
                jsonschema.validate(document, schema)


if __name__ == "__main__":
    unittest.main()
