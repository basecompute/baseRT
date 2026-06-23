// swift-tools-version: 5.9

import Foundation
import PackageDescription

// The Swift wrapper links the prebuilt single-file engine, libbaseRT.dylib.
// Point at the directory containing it with BASERT_LIB_DIR; defaults to the
// repo's build/ directory (where `make shared` writes it). The dylib carries
// the Metal kernels embedded, so no sidecar metallib file is needed at runtime.
let libDir =
    ProcessInfo.processInfo.environment["BASERT_LIB_DIR"]
    ?? "\(FileManager.default.currentDirectoryPath)/../../build"

let package = Package(
    name: "BaseRT",
    platforms: [
        .macOS(.v13),
        .iOS(.v16),
    ],
    products: [
        .library(name: "BaseRT", targets: ["BaseRT"]),
    ],
    targets: [
        // C shim exposing the public baseRT.h / types.h as a clang module.
        .target(
            name: "CBaseRT",
            path: "Sources/CBaseRT",
            publicHeadersPath: "include"
        ),
        .target(
            name: "BaseRT",
            dependencies: ["CBaseRT"],
            path: "Sources/BaseRT",
            linkerSettings: [
                // unsafeFlags keeps this package local-only (SwiftPM forbids
                // them in remotely-resolved deps); fine for a prebuilt engine
                // at a known path. Distributors who want a resolvable package
                // should swap this for an XCFramework binary target.
                .unsafeFlags([
                    "-L\(libDir)",
                    "-lbaseRT",
                    "-Xlinker", "-rpath", "-Xlinker", libDir,
                ])
            ]
        ),
        .testTarget(
            name: "BaseRTTests",
            dependencies: ["BaseRT"],
            path: "Tests/BaseRTTests"
        ),
    ]
)
