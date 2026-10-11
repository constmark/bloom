import Foundation

private func require(_ condition: @autoclosure () -> Bool, _ message: String) {
    guard condition() else {
        fatalError(message)
    }
}

let expectedPayload = Data("{\"choices\":[]}".utf8)
require(
    SSEParser.parse(line: "data: {\"choices\":[]}") == .data(expectedPayload),
    "SSE data payload was not parsed"
)
require(
    SSEParser.parse(line: "data: [DONE]\r") == .done,
    "SSE completion marker was not parsed"
)
require(SSEParser.parse(line: ": keepalive") == nil, "SSE comment was not ignored")
require(SSEParser.parse(line: "event: message") == nil, "Unknown SSE field was not ignored")
require(SSEParser.parse(line: "") == nil, "Empty SSE line was not ignored")

private func decodeReadiness(_ fixture: String) -> BloomReadiness {
    do {
        return try JSONDecoder().decode(BloomReadiness.self, from: Data(fixture.utf8))
    } catch {
        fatalError("Readiness fixture failed to decode: \(error)")
    }
}

let ready = decodeReadiness(
    """
    {
      "schema_version": 3,
      "object": "bloom.readiness",
      "protocol_version": 3,
      "minimum_ui_protocol_version": 3,
      "maximum_ui_protocol_version": 3,
      "server_version": "0.1.0",
      "status": "ready",
      "progress": 100,
      "model": "tiny",
      "loading": false,
      "load_error": null,
      "input_modalities": ["text"],
      "model_tasks": ["generation"],
      "context_window": 1024,
      "in_flight_requests": 0,
      "available_permits": 4,
      "memory_pressure_high": false,
      "ram_utilization": 0.25
    }
    """
)
require(ready.isCompatible, "Current readiness contract should be accepted")
require(ready.supportsChat, "Generation readiness should be accepted")

let futureCompatible = decodeReadiness(
    """
    {
      "schema_version": 3,
      "object": "bloom.readiness",
      "protocol_version": 4,
      "minimum_ui_protocol_version": 3,
      "maximum_ui_protocol_version": 4,
      "server_version": "0.2.0",
      "status": "ready",
      "progress": 100,
      "model": "tiny",
      "loading": false,
      "load_error": null,
      "input_modalities": ["text"],
      "model_tasks": ["generation"],
      "context_window": 1024,
      "in_flight_requests": 0,
      "available_permits": 4,
      "memory_pressure_high": false,
      "ram_utilization": 0.25,
      "future_additive_field": true
    }
    """
)
require(futureCompatible.isCompatible, "A future server retaining UI protocol 3 should be accepted")
require(futureCompatible.supportsChat, "A compatible future server should support chat")

let futureIncompatible = decodeReadiness(
    """
    {
      "schema_version": 3,
      "object": "bloom.readiness",
      "protocol_version": 4,
      "minimum_ui_protocol_version": 4,
      "maximum_ui_protocol_version": 4,
      "server_version": "0.3.0",
      "status": "ready",
      "progress": 100,
      "model": "tiny",
      "loading": false,
      "load_error": null,
      "input_modalities": ["text"],
      "model_tasks": ["generation"],
      "context_window": 1024,
      "in_flight_requests": 0,
      "available_permits": 4,
      "memory_pressure_high": false,
      "ram_utilization": 0.25
    }
    """
)
require(!futureIncompatible.isCompatible, "A future-only protocol must be rejected")
require(!futureIncompatible.supportsChat, "An incompatible future server must not enable chat")

let invalidRange = decodeReadiness(
    """
    {
      "schema_version": 3,
      "object": "bloom.readiness",
      "protocol_version": 3,
      "minimum_ui_protocol_version": 4,
      "maximum_ui_protocol_version": 3,
      "server_version": "0.1.0",
      "status": "ready",
      "progress": 100,
      "model": "tiny",
      "loading": false,
      "load_error": null,
      "input_modalities": ["text"],
      "model_tasks": ["generation"],
      "context_window": 1024,
      "in_flight_requests": 0,
      "available_permits": 4,
      "memory_pressure_high": false,
      "ram_utilization": 0.25
    }
    """
)
require(!invalidRange.isCompatible, "An inverted protocol range must be rejected")

let loading = decodeReadiness(
    """
    {
      "schema_version": 3,
      "object": "bloom.readiness",
      "protocol_version": 3,
      "minimum_ui_protocol_version": 3,
      "maximum_ui_protocol_version": 3,
      "server_version": "0.1.0",
      "status": "not_ready",
      "progress": 42,
      "model": "tiny",
      "loading": true,
      "load_error": null,
      "input_modalities": [],
      "model_tasks": [],
      "context_window": null,
      "in_flight_requests": 0,
      "available_permits": 4,
      "memory_pressure_high": false,
      "ram_utilization": 0.25
    }
    """
)
require(loading.isCompatible, "A loading readiness response should be accepted")
require(loading.loading, "Loading readiness should preserve the loading flag")
require(!loading.supportsChat, "Loading readiness must not enable chat")

let encoderOnly = decodeReadiness(
    """
    {
      "schema_version": 3,
      "object": "bloom.readiness",
      "protocol_version": 3,
      "minimum_ui_protocol_version": 3,
      "maximum_ui_protocol_version": 3,
      "server_version": "0.1.0",
      "status": "ready",
      "progress": 100,
      "model": "encoder",
      "loading": false,
      "load_error": null,
      "input_modalities": ["text"],
      "model_tasks": ["embedding"],
      "context_window": 256,
      "in_flight_requests": 0,
      "available_permits": 4,
      "memory_pressure_high": false,
      "ram_utilization": 0.25
    }
    """
)
require(!encoderOnly.supportsChat, "Encoder-only readiness should not enable chat")

let emptyModel = decodeReadiness(
    """
    {
      "schema_version": 3,
      "object": "bloom.readiness",
      "protocol_version": 3,
      "minimum_ui_protocol_version": 3,
      "maximum_ui_protocol_version": 3,
      "server_version": "0.1.0",
      "status": "ready",
      "progress": 100,
      "model": "   ",
      "loading": false,
      "load_error": null,
      "input_modalities": ["text"],
      "model_tasks": ["generation"],
      "context_window": 1024,
      "in_flight_requests": 0,
      "available_permits": 4,
      "memory_pressure_high": false,
      "ram_utilization": 0.25
    }
    """
)
require(!emptyModel.isCompatible, "A blank model name must make readiness incompatible")
require(!emptyModel.supportsChat, "A blank model name must not enable chat")

_ = try BloomAPIClient(serverURL: " https://localhost:3000/bloom/ ", apiKey: "")
for invalidURL in [
    "localhost:3000",
    "ftp://localhost:3000",
    "https://user:password@localhost:3000",
    "https://localhost:3000?token=secret",
    "https://localhost:3000/#fragment"
] {
    do {
        _ = try BloomAPIClient(serverURL: invalidURL, apiKey: "")
        fatalError("URL should have been rejected: \(invalidURL)")
    } catch BloomClientError.invalidServerURL {
        // Expected.
    }
}
require(
    BloomClientError.streamError("model failed").errorDescription == "Generation failed: model failed",
    "Streaming API errors should be readable"
)

print("Bloom Desktop checks passed")
