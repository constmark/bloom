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

let ready = try JSONDecoder().decode(
    BloomReadiness.self,
    from: Data(
        """
        {
          "schema_version": 3,
          "object": "bloom.readiness",
          "status": "ready",
          "progress": 100,
          "model": "tiny",
          "model_tasks": ["generation"],
          "context_window": 1024,
          "load_error": null
        }
        """.utf8
    )
)
require(ready.supportsChat, "Generation readiness should be accepted")

let encoderOnly = try JSONDecoder().decode(
    BloomReadiness.self,
    from: Data(
        """
        {
          "schema_version": 3,
          "object": "bloom.readiness",
          "status": "ready",
          "progress": 100,
          "model": "encoder",
          "model_tasks": ["embedding"],
          "context_window": 256,
          "load_error": null
        }
        """.utf8
    )
)
require(!encoderOnly.supportsChat, "Encoder-only readiness should not enable chat")

print("Bloom Desktop checks passed")
