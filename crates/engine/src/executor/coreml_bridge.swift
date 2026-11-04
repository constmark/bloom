// Embedded, dependency-free bridge to Apple's Core ML framework.
// This source is compiled into a private per-model workspace by the Rust adapter.
import Foundation
import CoreML

let maxValues = 1_048_576
let maxRequestBytes = 32 * 1024 * 1024

struct BridgeError: Error, CustomStringConvertible {
    let description: String
    init(_ message: String) { description = message }
}

func checkedShape(_ shape: [NSNumber]) throws -> [Int] {
    let dimensions = shape.map { $0.intValue }
    guard !dimensions.isEmpty && dimensions.count <= 16 else {
        throw BridgeError("CoreML arrays require 1 through 16 dimensions")
    }
    var count = 1
    for dimension in dimensions {
        guard dimension > 0 && dimension <= maxValues / count else {
            throw BridgeError("CoreML tensor shape is empty, overflowing, or oversized")
        }
        count *= dimension
    }
    return dimensions
}

func fixedShape(_ constraint: MLMultiArrayConstraint) throws -> [Int] {
    let shape = try checkedShape(constraint.shape)
    switch constraint.shapeConstraint.type {
    case .unspecified: return shape
    case .enumerated:
        guard constraint.shapeConstraint.enumeratedShapes.count == 1,
              try checkedShape(constraint.shapeConstraint.enumeratedShapes[0]) == shape else {
            throw BridgeError("Flexible CoreML tensor shapes are unsupported")
        }
    case .range:
        let ranges = constraint.shapeConstraint.sizeRangeForDimension.map { $0.rangeValue }
        guard ranges.count == shape.count && zip(ranges, shape).allSatisfy({ $0.0.location == $0.1 && $0.0.length == 1 }) else {
            throw BridgeError("Flexible CoreML tensor shapes are unsupported")
        }
    @unknown default: throw BridgeError("Unknown CoreML tensor shape constraint")
    }
    return shape
}

func checkedName(_ name: String) throws {
    guard !name.isEmpty && name.utf8.count <= 512 && !name.unicodeScalars.contains(where: { CharacterSet.controlCharacters.contains($0) }) else {
        throw BridgeError("CoreML feature names must be bounded and contain no control characters")
    }
}

func dtypeName(_ dtype: MLMultiArrayDataType) throws -> String {
    switch dtype {
    case .float16: return "float16"
    case .float32: return "float32"
    case .double: return "float64"
    default: throw BridgeError("CoreML tensor adapter supports floating-point arrays only")
    }
}

struct TensorContract {
    let input: MLFeatureDescription
    let output: MLFeatureDescription
    let shape: [Int]
    let dtype: MLMultiArrayDataType
    let outputShape: [Int]?

    init(_ model: MLModel) throws {
        let description = model.modelDescription
        guard description.inputDescriptionsByName.count == 1 && description.outputDescriptionsByName.count == 1 else {
            throw BridgeError("CoreML tensor adapter requires exactly one input and one output")
        }
        if #available(macOS 15.0, *), !description.stateDescriptionsByName.isEmpty {
            throw BridgeError("Stateful CoreML models require a separate stateful adapter")
        }
        input = description.inputDescriptionsByName.values.first!
        output = description.outputDescriptionsByName.values.first!
        try checkedName(input.name)
        try checkedName(output.name)
        guard input.type == .multiArray, !input.isOptional, let constraint = input.multiArrayConstraint else {
            throw BridgeError("CoreML tensor input must be one required floating-point multiarray")
        }
        shape = try fixedShape(constraint)
        dtype = constraint.dataType
        _ = try dtypeName(dtype)
        if output.type == .multiArray, let constraint = output.multiArrayConstraint {
            outputShape = try fixedShape(constraint)
            _ = try dtypeName(constraint.dataType)
        } else if output.type == .double {
            outputShape = nil
        } else {
            throw BridgeError("CoreML output must be one floating-point array or double scalar")
        }
    }

    func probe() throws -> [String: Any] {
        var result: [String: Any] = [
            "schema_version": 1, "compute_units": "cpu_only",
            "input_name": input.name, "input_shape": shape, "input_dtype": try dtypeName(dtype),
            "output_name": output.name, "output_kind": outputShape == nil ? "double" : "multiarray"
        ]
        if let shape = outputShape { result["output_shape"] = shape }
        return result
    }
}

func configuration() -> MLModelConfiguration {
    let configuration = MLModelConfiguration()
    configuration.computeUnits = .cpuOnly
    return configuration
}

func emit(_ value: [String: Any]) throws {
    let data = try JSONSerialization.data(withJSONObject: value, options: [.sortedKeys])
    try FileHandle.standardOutput.write(contentsOf: data)
    try FileHandle.standardOutput.write(contentsOf: Data([10]))
}

func readRequest() throws -> [Float] {
    var data = Data()
    while let chunk = try FileHandle.standardInput.read(upToCount: min(65536, maxRequestBytes + 1 - data.count)), !chunk.isEmpty {
        data.append(chunk)
        guard data.count <= maxRequestBytes else { throw BridgeError("CoreML input exceeds the JSON byte limit") }
    }
    guard let object = try JSONSerialization.jsonObject(with: data) as? [String: Any], Set(object.keys) == ["values"] else {
        throw BridgeError("CoreML request requires exactly one values field")
    }
    struct Request: Decodable { let values: [Float] }
    let values = try JSONDecoder().decode(Request.self, from: data).values
    guard !values.isEmpty && values.count <= maxValues && values.allSatisfy({ $0.isFinite }) else {
        throw BridgeError("CoreML values must be nonempty, finite, and bounded")
    }
    return values
}

func main() throws {
    let arguments = CommandLine.arguments
    guard arguments.count >= 3 else { throw BridgeError("Expected prepare MODEL TARGET or infer COMPILED_MODEL") }
    let operation = arguments[1]
    let source = URL(fileURLWithPath: arguments[2])
    if operation == "prepare" && arguments.count == 4 {
        let compiled = try MLModel.compileModel(at: source)
        defer { try? FileManager.default.removeItem(at: compiled) }
        let target = URL(fileURLWithPath: arguments[3])
        guard !FileManager.default.fileExists(atPath: target.path) else { throw BridgeError("CoreML compiled target already exists") }
        try FileManager.default.copyItem(at: compiled, to: target)
        let model = try MLModel(contentsOf: target, configuration: configuration())
        try emit(TensorContract(model).probe())
    } else if operation == "infer" && arguments.count == 3 {
        let values = try readRequest()
        let model = try MLModel(contentsOf: source, configuration: configuration())
        let contract = try TensorContract(model)
        guard values.count == contract.shape.reduce(1, *) else { throw BridgeError("CoreML input does not match the model's tensor shape") }
        let array = try MLMultiArray(shape: contract.shape.map { NSNumber(value: $0) }, dataType: contract.dtype)
        for (index, value) in values.enumerated() {
            array[index] = NSNumber(value: value)
            guard array[index].doubleValue.isFinite else { throw BridgeError("CoreML input overflows its model dtype") }
        }
        let input = try MLDictionaryFeatureProvider(dictionary: [contract.input.name: MLFeatureValue(multiArray: array)])
        let output = try model.prediction(from: input)
        guard let value = output.featureValue(for: contract.output.name) else { throw BridgeError("CoreML output feature is missing") }
        var result: [Float] = []
        if let shape = contract.outputShape, let array = value.multiArrayValue {
            guard try checkedShape(array.shape) == shape && array.count <= maxValues else { throw BridgeError("CoreML output shape changed") }
            result.reserveCapacity(array.count)
            for index in 0..<array.count { result.append(Float(array[index].doubleValue)) }
        } else if contract.outputShape == nil && value.type == .double {
            result = [Float(value.doubleValue)]
        } else { throw BridgeError("CoreML output type changed") }
        guard !result.isEmpty && result.allSatisfy({ $0.isFinite }) else { throw BridgeError("CoreML returned non-finite or empty output") }
        try emit(["schema_version": 1, "compute_units": "cpu_only", "values": result])
    } else { throw BridgeError("Invalid CoreML bridge operation or arguments") }
}

do { try main() }
catch {
    let message = "CoreML bridge: \(error)\n"
    try? FileHandle.standardError.write(contentsOf: Data(message.prefix(8192).utf8))
    exit(1)
}
