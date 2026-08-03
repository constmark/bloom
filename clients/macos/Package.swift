// swift-tools-version: 5.10

import PackageDescription

let package = Package(
    name: "BloomDesktop",
    platforms: [
        .macOS(.v13),
    ],
    products: [
        .executable(name: "BloomDesktop", targets: ["BloomDesktop"]),
    ],
    targets: [
        .executableTarget(
            name: "BloomDesktop",
            path: "Sources/BloomDesktop"
        ),
    ]
)
