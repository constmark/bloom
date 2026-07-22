# World Model Schema

Bloom's world model path targets edge agents, robots, environment
perception, and closed-loop control tasks. It is not ordinary
prompt-response, but:

```text
observation -> world state -> predicted futures -> action
```

The stable schema lives at:

- [`examples/world-model-schema.json`](../examples/world-model-schema.json)
- [`examples/world-model-example.json`](../examples/world-model-example.json)

## Observation constraints

`WorldStateSchema` constrains input observations:

| Field | Meaning |
| :--- | :--- |
| `scalar_ranges` | Allowed ranges for scalar observations, e.g. temperature, battery, speed. |
| `allowed_image_mimes` | Image MIME allowlist; empty means unrestricted. |
| `tensor_shapes` | Allowed tensor shapes; empty means unrestricted. |
| `allow_text` | Whether text observations are allowed. |
| `allow_audio` | Whether audio observations are allowed. |

Example:

```json
{
  "scalar_ranges": {
    "battery_percent": [0.0, 100.0]
  },
  "allowed_image_mimes": ["image/png", "image/jpeg"],
  "tensor_shapes": [[1, 3, 224, 224]],
  "allow_text": true,
  "allow_audio": false
}
```

## Action constraints

`ActionSchema` constrains policy output:

| Field | Meaning |
| :--- | :--- |
| `allowed_action_spaces` | Names of allowed action spaces. |
| `action_dimensions` | Vector dimension per action space. |
| `value_range` | Global range of action values. |

Example:

```json
{
  "allowed_action_spaces": ["robot_velocity"],
  "action_dimensions": {
    "robot_velocity": 2
  },
  "value_range": [-1.0, 1.0]
}
```

## Acceptance

Default tests cover schema validation, observation rejection, action
dimension and value-range rejection, environment degradation, state cache
expiry, and compression.

```bash
cargo test -p bloomai-engine world::tests
./scripts/validate_json_artifacts.py
```

When adding a new world model or policy engine, declare the schema
first, then plug in real model execution. This lets the service layer
reject invalid observation/action before entering the model, instead of
pushing invalid inputs into long-running inference paths.
