import CBaseRT
import XCTest

@testable import BaseRT

/// Pure unit tests for the Swift wrapper. They exercise struct mapping and
/// config conversion without calling into the engine, so no model file is
/// needed (the engine dylib is linked but never invoked here). An optional
/// end-to-end test runs when BASERT_TEST_MODEL points at a `.base` file.
final class BaseRTTests: XCTestCase {

    // MARK: - SamplingConfig

    func testSamplingConfigDefaults() {
        let config = SamplingConfig()
        XCTAssertEqual(config.temperature, 0.0)
        XCTAssertEqual(config.topK, 40)
        XCTAssertEqual(config.topP, 0.9, accuracy: 1e-6)
        XCTAssertEqual(config.minP, 0.0)
        XCTAssertEqual(config.repeatPenalty, 1.0)
        XCTAssertEqual(config.seed, 0)
        XCTAssertTrue(config.logitBias.isEmpty)
    }

    func testSamplingConfigWithCValueDefaults() {
        let config = SamplingConfig()
        config.withCValue { c in
            XCTAssertEqual(c.temperature, 0.0)
            XCTAssertEqual(c.top_k, 40)
            XCTAssertEqual(c.top_p, 0.9, accuracy: 1e-6)
            XCTAssertEqual(c.repeat_penalty, 1.0)
            XCTAssertEqual(c.n_logit_bias, 0)
            XCTAssertNil(c.logit_bias_tokens)
        }
    }

    func testSamplingConfigWithCValuePenaltiesAndBias() {
        let config = SamplingConfig(
            temperature: 0.8,
            presencePenalty: 0.5,
            frequencyPenalty: 0.3,
            seed: 42,
            logitBias: [10: 1.5, 20: -2.0]
        )
        config.withCValue { c in
            XCTAssertEqual(c.temperature, 0.8, accuracy: 1e-6)
            XCTAssertEqual(c.presence_penalty, 0.5, accuracy: 1e-6)
            XCTAssertEqual(c.frequency_penalty, 0.3, accuracy: 1e-6)
            XCTAssertEqual(c.seed, 42)
            XCTAssertEqual(c.n_logit_bias, 2)
            XCTAssertNotNil(c.logit_bias_tokens)
            XCTAssertNotNil(c.logit_bias_values)
        }
    }

    // MARK: - Error descriptions

    func testErrorDescriptions() {
        XCTAssertTrue(
            (BaseRTError.loadFailed("nope").errorDescription ?? "").contains("nope"))
        XCTAssertTrue(
            (BaseRTError.encodeFailed("bad").errorDescription ?? "").contains("Encoding failed"))
        XCTAssertTrue(
            (BaseRTError.transcribeFailed("x").errorDescription ?? "").contains("Transcription"))
    }

    // MARK: - ModelConfig mapping from the real C struct

    func testModelConfigFromZeroedCStruct() {
        var c = BaseRTModelConfig()
        memset(&c, 0, MemoryLayout<BaseRTModelConfig>.size)
        let config = ModelConfig(c)
        XCTAssertEqual(config.dim, 0)
        XCTAssertEqual(config.architecture, "")
        XCTAssertFalse(config.hasVision)
        XCTAssertFalse(config.hasAudio)
        XCTAssertEqual(config.nExperts, 0)
    }

    func testModelConfigReadsPopulatedFieldsIncludingSlidingWindow() {
        var c = BaseRTModelConfig()
        memset(&c, 0, MemoryLayout<BaseRTModelConfig>.size)
        c.dim = 1024
        c.n_layers = 28
        c.sliding_window = 4096
        c.head_dim_global = 128
        c.n_experts = 128
        c.n_experts_used = 8
        c.vision_n_layers = 27
        // architecture is a C fixed char array; write "qwen".
        withUnsafeMutableBytes(of: &c.architecture) { raw in
            let bytes: [UInt8] = Array("qwen".utf8)
            for (i, b) in bytes.enumerated() { raw[i] = b }
        }
        let config = ModelConfig(c)
        XCTAssertEqual(config.dim, 1024)
        XCTAssertEqual(config.nLayers, 28)
        XCTAssertEqual(config.slidingWindow, 4096)
        XCTAssertEqual(config.headDimGlobal, 128)
        XCTAssertEqual(config.nExperts, 128)
        XCTAssertEqual(config.nExpertsUsed, 8)
        XCTAssertTrue(config.hasVision)
        XCTAssertEqual(config.architecture, "qwen")
    }

    // MARK: - Stats mapping

    func testGenerationStatsMapping() {
        var c = BaseRTGenerationStats()
        memset(&c, 0, MemoryLayout<BaseRTGenerationStats>.size)
        c.prompt_tokens = 5
        c.generated_tokens = 20
        let stats = GenerationStats(c)
        XCTAssertEqual(stats.promptTokens, 5)
        XCTAssertEqual(stats.generatedTokens, 20)
    }

    func testGeneratedTokenConstruction() {
        let token = GeneratedToken(tokenID: 42, text: "hi")
        XCTAssertEqual(token.tokenID, 42)
        XCTAssertEqual(token.text, "hi")
    }

    // MARK: - Optional end-to-end (needs a model)

    func testEndToEndGenerationIfModelAvailable() throws {
        guard let modelPath = ProcessInfo.processInfo.environment["BASERT_TEST_MODEL"] else {
            throw XCTSkip("Set BASERT_TEST_MODEL to a .base file to run the e2e test.")
        }
        // nil metallib path → load the metallib embedded in the dylib.
        let model = try BaseRTModel(modelPath: modelPath, kernelLibraryPath: nil)
        let tokens = try model.encode(text: "The capital of France is")
        XCTAssertFalse(tokens.isEmpty)
        let text = model.generateText(tokens: tokens, maxTokens: 8)
        XCTAssertFalse(text.isEmpty)
    }
}
