import CBaseRT
import Foundation

// MARK: - Swift-native types

/// Model configuration mirroring the C BaseRTModelConfig struct.
public struct ModelConfig: Sendable {
    public let dim: UInt32
    public let nLayers: UInt32
    public let nHeads: UInt32
    public let nKVHeads: UInt32
    public let headDim: UInt32
    public let qDim: UInt32
    public let kvDim: UInt32
    public let ffnDim: UInt32
    public let vocabSize: UInt32
    public let maxSeqLen: UInt32
    public let normEps: Float
    public let ropeTheta: Float
    public let slidingWindowPattern: UInt32
    public let slidingWindow: UInt32
    public let ropeLocalTheta: Float
    public let architecture: String

    // Encoder parameters (zero for decoder-only models)
    public let encNLayers: UInt32
    public let encNHeads: UInt32
    public let encDim: UInt32
    public let encFFNDim: UInt32
    public let nMels: UInt32
    public let encMaxSeqLen: UInt32

    // Capability summary (derived from the Gemma4 / MoE / tower fields).
    public let headDimSwa: UInt32
    public let headDimGlobal: UInt32
    public let nExperts: UInt32
    public let nExpertsUsed: UInt32
    /// True when the bundle carries a vision tower.
    public let hasVision: Bool
    /// True when the bundle carries an audio tower.
    public let hasAudio: Bool

    init(_ c: BaseRTModelConfig) {
        self.dim = c.dim
        self.nLayers = c.n_layers
        self.nHeads = c.n_heads
        self.nKVHeads = c.n_kv_heads
        self.headDim = c.head_dim
        self.qDim = c.q_dim
        self.kvDim = c.kv_dim
        self.ffnDim = c.ffn_dim
        self.vocabSize = c.vocab_size
        self.maxSeqLen = c.max_seq_len
        self.normEps = c.norm_eps
        self.ropeTheta = c.rope_theta
        self.slidingWindowPattern = c.sliding_window_pattern
        self.slidingWindow = c.sliding_window
        self.ropeLocalTheta = c.rope_local_theta
        var arch = c.architecture
        self.architecture = withUnsafePointer(to: &arch) {
            $0.withMemoryRebound(to: CChar.self, capacity: 32) {
                String(cString: $0)
            }
        }
        self.encNLayers = c.enc_n_layers
        self.encNHeads = c.enc_n_heads
        self.encDim = c.enc_dim
        self.encFFNDim = c.enc_ffn_dim
        self.nMels = c.n_mels
        self.encMaxSeqLen = c.enc_max_seq_len
        self.headDimSwa = c.head_dim_swa
        self.headDimGlobal = c.head_dim_global
        self.nExperts = c.n_experts
        self.nExpertsUsed = c.n_experts_used
        self.hasVision = c.vision_n_layers > 0
        self.hasAudio = c.audio_n_layers > 0
    }
}

/// Sampling parameters for text generation.
///
/// Extended in baseRT 0.2 with OpenAI-compat penalties (presence,
/// frequency), a deterministic-sample `seed`, and a per-token
/// `logitBias` map. New fields default to "disabled" so existing
/// callers that only pass the first five keep working.
public struct SamplingConfig: Sendable {
    public var temperature: Float
    public var topK: Int32
    public var topP: Float
    public var minP: Float
    public var repeatPenalty: Float
    public var presencePenalty: Float
    public var frequencyPenalty: Float
    /// 0 = wall-clock-seeded (non-deterministic). Non-zero re-seeds the
    /// thread-local sampling RNG so the run is reproducible.
    public var seed: UInt32
    /// Additive per-token bias map: `[tokenId: bias]` where bias ∈ [-100, 100].
    public var logitBias: [Int32: Float]

    public init(
        temperature: Float = 0.0,
        topK: Int32 = 40,
        topP: Float = 0.9,
        minP: Float = 0.0,
        repeatPenalty: Float = 1.0,
        presencePenalty: Float = 0.0,
        frequencyPenalty: Float = 0.0,
        seed: UInt32 = 0,
        logitBias: [Int32: Float] = [:]
    ) {
        self.temperature = temperature
        self.topK = topK
        self.topP = topP
        self.minP = minP
        self.repeatPenalty = repeatPenalty
        self.presencePenalty = presencePenalty
        self.frequencyPenalty = frequencyPenalty
        self.seed = seed
        self.logitBias = logitBias
    }

    /// Build the C-side struct. The closure form (`withCValue`) is required
    /// rather than a plain getter because `logitBias` is stored on the heap
    /// here and the engine reads the bias arrays via raw pointers — they
    /// must stay alive across the FFI call. Using a closure pins the buffers
    /// to the call's stack frame for guaranteed lifetime.
    func withCValue<R>(_ body: (BaseRTSamplingConfig) -> R) -> R {
        var tokens: [Int32] = []
        var values: [Float] = []
        tokens.reserveCapacity(logitBias.count)
        values.reserveCapacity(logitBias.count)
        for (k, v) in logitBias {
            tokens.append(k)
            values.append(v)
        }
        return tokens.withUnsafeBufferPointer { tokBuf in
            values.withUnsafeBufferPointer { valBuf in
                let cfg = BaseRTSamplingConfig(
                    temperature: temperature,
                    top_k: topK,
                    top_p: topP,
                    min_p: minP,
                    repeat_penalty: repeatPenalty,
                    presence_penalty: presencePenalty,
                    frequency_penalty: frequencyPenalty,
                    seed: seed,
                    n_logit_bias: Int32(tokens.count),
                    logit_bias_tokens: tokens.isEmpty ? nil : tokBuf.baseAddress,
                    logit_bias_values: values.isEmpty ? nil : valBuf.baseAddress
                )
                return body(cfg)
            }
        }
    }
}

/// Statistics from a generation run.
public struct GenerationStats: Sendable {
    public let promptTokens: Int32
    public let generatedTokens: Int32
    public let prefillTimeMs: Float
    public let decodeTimeMs: Float
    public let prefillTokensPerSec: Float
    public let decodeTokensPerSec: Float

    init(_ c: BaseRTGenerationStats) {
        self.promptTokens = c.prompt_tokens
        self.generatedTokens = c.generated_tokens
        self.prefillTimeMs = c.prefill_time_ms
        self.decodeTimeMs = c.decode_time_ms
        self.prefillTokensPerSec = c.prefill_tokens_per_sec
        self.decodeTokensPerSec = c.decode_tokens_per_sec
    }
}

/// Statistics from a transcription run.
public struct TranscribeStats: Sendable {
    public let nTokens: Int32
    public let audioMs: Float
    public let encodeMs: Float
    public let decodeMs: Float
    public let totalMs: Float

    init(_ c: BaseRTTranscribeStats) {
        self.nTokens = c.n_tokens
        self.audioMs = c.audio_ms
        self.encodeMs = c.encode_ms
        self.decodeMs = c.decode_ms
        self.totalMs = c.total_ms
    }
}

// MARK: - Error type

/// Errors thrown by BaseRTModel operations.
public enum BaseRTError: Error, LocalizedError, Sendable {
    case loadFailed(String)
    case encodeFailed(String)
    case transcribeFailed(String)

    public var errorDescription: String? {
        switch self {
        case .loadFailed(let msg): return "Failed to load model: \(msg)"
        case .encodeFailed(let msg): return "Encoding failed: \(msg)"
        case .transcribeFailed(let msg): return "Transcription failed: \(msg)"
        }
    }
}

// MARK: - Generated token for streaming

/// A single generated token emitted during streaming.
public struct GeneratedToken: Sendable {
    public let tokenID: UInt32
    public let text: String
}

/// A transcribed audio segment with timestamps.
public struct TranscriptSegment: Sendable {
    /// Start time in milliseconds.
    public let startMs: Int
    /// End time in milliseconds.
    public let endMs: Int
    /// Transcribed text for this segment.
    public let text: String
}

// MARK: - BaseRTModel

/// Swift wrapper around the BaseRT C inference engine.
///
/// Thread safety: instances are not thread-safe. Use one model per thread/actor.
public final class BaseRTModel {
    let handle: baseRT_model_t  // internal: used by extensions in this module

    // MARK: Lifecycle

    /// Load a model from a `.base` bundle.
    ///
    /// - Parameters:
    ///   - modelPath: Path to the `.base` model file.
    ///   - kernelLibraryPath: Path to the compiled GPU kernel library (on Metal,
    ///     `baseRT.metallib`). Pass `nil` to auto-detect — including the copy
    ///     embedded in the single-file libbaseRT dylib.
    ///   - maxContext: Maximum context window size. Pass 0 for the model default.
    /// - Throws: `BaseRTError.loadFailed` if the model cannot be loaded.
    public init(modelPath: String, kernelLibraryPath: String? = nil, maxContext: Int = 0) throws {
        guard let model = baseRT_load_model(modelPath, kernelLibraryPath, Int32(maxContext)) else {
            let err = baseRT_get_error().flatMap { String(cString: $0) } ?? "unknown error"
            throw BaseRTError.loadFailed(err)
        }
        self.handle = model
    }

    deinit {
        baseRT_free_model(handle)
    }

    // MARK: Model info

    /// The model configuration (dimensions, layers, architecture, etc.).
    public var config: ModelConfig {
        ModelConfig(baseRT_get_config(handle))
    }

    /// Total GPU memory used by the model, in bytes.
    public var memoryUsage: Int {
        baseRT_model_memory(handle)
    }

    /// Whether this is a Whisper audio model.
    public var isWhisper: Bool {
        baseRT_is_whisper(handle)
    }

    /// Current KV cache position (number of tokens processed so far).
    public var position: Int {
        Int(baseRT_get_position(handle))
    }

    // MARK: Tokenization

    /// Encode text into token IDs.
    ///
    /// - Parameter text: The input text to tokenize.
    /// - Returns: Array of token IDs.
    /// - Throws: `BaseRTError.encodeFailed` if encoding fails.
    public func encode(text: String) throws -> [UInt32] {
        let maxTokens = max(text.utf8.count * 2, 1024)
        var tokens = [UInt32](repeating: 0, count: maxTokens)
        let count = baseRT_encode(handle, text, &tokens, Int32(maxTokens))
        if count < 0 {
            let err = baseRT_get_error().flatMap { String(cString: $0) } ?? "encoding failed"
            throw BaseRTError.encodeFailed(err)
        }
        return Array(tokens.prefix(Int(count)))
    }

    /// Decode a single token ID back to its text representation.
    ///
    /// - Parameter tokenID: The token ID to decode.
    /// - Returns: The text for this token, or an empty string if invalid.
    public func decodeToken(_ tokenID: UInt32) -> String {
        guard let ptr = baseRT_decode_token(handle, tokenID) else {
            return ""
        }
        return String(cString: ptr)
    }

    // MARK: Generation

    /// Generate tokens from a prompt.
    ///
    /// - Parameters:
    ///   - tokens: Prompt token IDs (from `encode`).
    ///   - maxTokens: Maximum number of tokens to generate.
    ///   - sampling: Sampling configuration. Defaults to greedy decoding.
    ///   - onToken: Optional closure called for each generated token.
    ///              Return `false` to stop generation early.
    /// - Returns: Generation statistics.
    @discardableResult
    public func generate(
        tokens: [UInt32],
        maxTokens: Int,
        sampling: SamplingConfig = SamplingConfig(),
        onToken: ((GeneratedToken) -> Bool)? = nil
    ) -> GenerationStats {
        let stats = tokens.withUnsafeBufferPointer { buf in
            sampling.withCValue { cfg in
                if let callback = onToken {
                    let ctx = CallbackContext(callback: callback)
                    let unmanaged = Unmanaged.passRetained(ctx)
                    defer { unmanaged.release() }
                    return baseRT_generate(
                        handle,
                        buf.baseAddress,
                        Int32(tokens.count),
                        Int32(maxTokens),
                        cfg,
                        cTokenCallback,
                        unmanaged.toOpaque()
                    )
                } else {
                    return baseRT_generate(
                        handle,
                        buf.baseAddress,
                        Int32(tokens.count),
                        Int32(maxTokens),
                        cfg,
                        nil,
                        nil
                    )
                }
            }
        }
        return GenerationStats(stats)
    }

    /// Continue generating from the current KV cache state (for multi-turn chat).
    ///
    /// - Parameters:
    ///   - tokens: New token IDs to prefill before continuing generation.
    ///   - maxTokens: Maximum number of tokens to generate.
    ///   - sampling: Sampling configuration.
    ///   - onToken: Optional closure called for each generated token.
    ///              Return `false` to stop generation early.
    /// - Returns: Generation statistics.
    @discardableResult
    public func generateContinue(
        tokens: [UInt32],
        maxTokens: Int,
        sampling: SamplingConfig = SamplingConfig(),
        onToken: ((GeneratedToken) -> Bool)? = nil
    ) -> GenerationStats {
        let stats = tokens.withUnsafeBufferPointer { buf in
            sampling.withCValue { cfg in
                if let callback = onToken {
                    let ctx = CallbackContext(callback: callback)
                    let unmanaged = Unmanaged.passRetained(ctx)
                    defer { unmanaged.release() }
                    return baseRT_generate_continue(
                        handle,
                        buf.baseAddress,
                        Int32(tokens.count),
                        Int32(maxTokens),
                        cfg,
                        cTokenCallback,
                        unmanaged.toOpaque()
                    )
                } else {
                    return baseRT_generate_continue(
                        handle,
                        buf.baseAddress,
                        Int32(tokens.count),
                        Int32(maxTokens),
                        cfg,
                        nil,
                        nil
                    )
                }
            }
        }
        return GenerationStats(stats)
    }

    // MARK: Whisper transcription

    /// Transcribe audio from a WAV file.
    ///
    /// - Parameters:
    ///   - wavPath: Path to the WAV file.
    ///   - language: Language code (e.g. "en", "auto"). Defaults to "en".
    /// - Returns: Tuple of transcribed text and timing statistics.
    /// - Throws: `BaseRTError.transcribeFailed` if transcription fails.
    public func transcribe(wavPath: String, language: String = "en") throws -> (String, TranscribeStats) {
        var stats = BaseRTTranscribeStats()
        guard let result = baseRT_transcribe(handle, wavPath, language, &stats) else {
            let err = baseRT_get_error().flatMap { String(cString: $0) } ?? "transcription failed"
            throw BaseRTError.transcribeFailed(err)
        }
        return (String(cString: result), TranscribeStats(stats))
    }

    /// Transcribe audio from a WAV file with per-segment streaming.
    ///
    /// - Parameters:
    ///   - wavPath: Path to the WAV file.
    ///   - language: Language code (e.g. "en", "auto"). Defaults to "en".
    ///   - onSegment: Closure called for each decoded segment.
    ///                Receives start/end timestamps in milliseconds and segment text.
    ///                Return `false` to stop transcription early.
    /// - Returns: Tuple of full transcribed text and timing statistics.
    /// - Throws: `BaseRTError.transcribeFailed` if transcription fails.
    public func transcribe(
        wavPath: String,
        language: String = "en",
        onSegment: @escaping (TranscriptSegment) -> Bool
    ) throws -> (String, TranscribeStats) {
        var stats = BaseRTTranscribeStats()
        let ctx = SegmentCallbackContext(callback: onSegment)
        let unmanaged = Unmanaged.passRetained(ctx)
        defer { unmanaged.release() }
        guard let result = baseRT_transcribe_stream(
            handle, wavPath, language, &stats, cSegmentCallback, unmanaged.toOpaque()
        ) else {
            let err = baseRT_get_error().flatMap { String(cString: $0) } ?? "transcription failed"
            throw BaseRTError.transcribeFailed(err)
        }
        return (String(cString: result), TranscribeStats(stats))
    }

    /// Transcribe audio from raw PCM samples (16kHz, mono, Float32).
    ///
    /// - Parameters:
    ///   - samples: Array of Float32 audio samples at 16kHz.
    ///   - language: Language code (e.g. "en", "auto"). Defaults to "en".
    /// - Returns: Tuple of transcribed text and timing statistics.
    /// - Throws: `BaseRTError.transcribeFailed` if transcription fails.
    public func transcribePCM(samples: [Float], language: String = "en") throws -> (String, TranscribeStats) {
        var stats = BaseRTTranscribeStats()
        let result = samples.withUnsafeBufferPointer { buf in
            baseRT_transcribe_pcm(handle, buf.baseAddress, Int32(samples.count), language, &stats)
        }
        guard let result else {
            let err = baseRT_get_error().flatMap { String(cString: $0) } ?? "transcription failed"
            throw BaseRTError.transcribeFailed(err)
        }
        return (String(cString: result), TranscribeStats(stats))
    }

    /// Transcribe raw PCM samples with per-segment streaming.
    public func transcribePCM(
        samples: [Float],
        language: String = "en",
        onSegment: @escaping (TranscriptSegment) -> Bool
    ) throws -> (String, TranscribeStats) {
        var stats = BaseRTTranscribeStats()
        let ctx = SegmentCallbackContext(callback: onSegment)
        let unmanaged = Unmanaged.passRetained(ctx)
        defer { unmanaged.release() }
        let result = samples.withUnsafeBufferPointer { buf in
            baseRT_transcribe_pcm_stream(
                handle, buf.baseAddress, Int32(samples.count), language, &stats,
                cSegmentCallback, unmanaged.toOpaque()
            )
        }
        guard let result else {
            let err = baseRT_get_error().flatMap { String(cString: $0) } ?? "transcription failed"
            throw BaseRTError.transcribeFailed(err)
        }
        return (String(cString: result), TranscribeStats(stats))
    }

    // MARK: Embeddings

    /// Compute embeddings from token IDs using the model's hidden states.
    ///
    /// - Parameter tokens: Array of token IDs.
    /// - Returns: Array of float embedding values.
    /// - Throws: `BaseRTError.encodeFailed` if embedding fails.
    public func embed(tokens: [UInt32]) throws -> [Float] {
        let dim = Int(baseRT_embedding_dim(handle))
        guard dim > 0 else {
            let err = baseRT_get_error().flatMap { String(cString: $0) } ?? "embedding failed"
            throw BaseRTError.encodeFailed(err)
        }
        var out = [Float](repeating: 0, count: dim)
        let n = tokens.withUnsafeBufferPointer { buf in
            baseRT_embed(handle, buf.baseAddress, Int32(tokens.count), &out, Int32(dim))
        }
        guard n > 0 else {
            let err = baseRT_get_error().flatMap { String(cString: $0) } ?? "embedding failed"
            throw BaseRTError.encodeFailed(err)
        }
        return Array(out.prefix(Int(n)))
    }

    /// Compute embeddings from text directly (tokenizes internally).
    ///
    /// - Parameter text: Input text to embed.
    /// - Returns: Array of float embedding values.
    /// - Throws: `BaseRTError.encodeFailed` if embedding fails.
    public func embedText(_ text: String) throws -> [Float] {
        let dim = Int(baseRT_embedding_dim(handle))
        guard dim > 0 else {
            let err = baseRT_get_error().flatMap { String(cString: $0) } ?? "embedding failed"
            throw BaseRTError.encodeFailed(err)
        }
        var out = [Float](repeating: 0, count: dim)
        let n = baseRT_embed_text(handle, text, &out, Int32(dim))
        guard n > 0 else {
            let err = baseRT_get_error().flatMap { String(cString: $0) } ?? "embedding failed"
            throw BaseRTError.encodeFailed(err)
        }
        return Array(out.prefix(Int(n)))
    }

    /// The embedding dimension for this model.
    public var embeddingDim: Int {
        Int(baseRT_embedding_dim(handle))
    }

    // MARK: Chat templates

    /// Format a chat prompt using the model's native template.
    ///
    /// - Parameters:
    ///   - system: System prompt.
    ///   - user: User message.
    /// - Returns: Formatted chat string.
    public func formatChat(system: String, user: String) -> String {
        guard let ptr = baseRT_format_chat(handle, system, user) else {
            return ""
        }
        return String(cString: ptr)
    }

    /// The chat template name for the loaded model (e.g. "chatml", "llama3", "gemma").
    public var chatTemplate: String {
        guard let ptr = baseRT_chat_template(handle) else {
            return ""
        }
        return String(cString: ptr)
    }

    // MARK: Token counting

    /// Count tokens in text without allocating an output buffer.
    ///
    /// - Parameter text: Input text.
    /// - Returns: Number of tokens.
    public func tokenCount(_ text: String) -> Int {
        Int(baseRT_token_count(handle, text))
    }

    // MARK: Whisper settings

    /// Enable or disable timestamp generation for Whisper transcription.
    ///
    /// When enabled (default), output includes `[start --> end] text` segments.
    /// When disabled, faster greedy decode produces plain text only.
    public func setTimestamps(enabled: Bool) {
        baseRT_set_timestamps(handle, enabled)
    }

    // MARK: State management

    /// Enable or disable speculative decoding (n-gram prediction).
    /// Only affects greedy (temperature=0) mode.
    public func setSpeculation(enabled: Bool) {
        baseRT_set_speculation(handle, enabled)
    }

    /// Reset KV cache and internal state.
    public func reset() {
        baseRT_reset(handle)
    }

    // MARK: Low-level API

    /// Run prefill on tokens, populating the KV cache.
    /// - Returns: The first generated token (argmax of prefill logits).
    public func prefill(tokens: [UInt32]) -> UInt32 {
        tokens.withUnsafeBufferPointer { buf in
            baseRT_prefill(handle, buf.baseAddress, Int32(tokens.count))
        }
    }

    /// Run one decode step.
    /// - Returns: The sampled token ID.
    public func decodeStep(tokenID: UInt32, position: Int) -> UInt32 {
        baseRT_decode_step(handle, tokenID, Int32(position))
    }

    /// Chain decode: generate multiple tokens in one GPU submission.
    /// - Returns: Array of generated token IDs.
    public func chainDecode(firstToken: UInt32, startPosition: Int, count: Int) -> [UInt32] {
        var out = [UInt32](repeating: 0, count: count)
        let n = baseRT_chain_decode(handle, firstToken, Int32(startPosition), Int32(count), &out)
        return Array(out.prefix(Int(n)))
    }

    // MARK: Model inspection

    /// Number of tensors in the model.
    public var tensorCount: Int {
        Int(baseRT_tensor_count(handle))
    }

    /// Get the name of a tensor by index.
    public func tensorName(at index: Int) -> String? {
        guard let ptr = baseRT_tensor_name(handle, Int32(index)) else { return nil }
        return String(cString: ptr)
    }

    /// Get the dtype code of a tensor by index.
    public func tensorDtype(at index: Int) -> UInt32 {
        baseRT_tensor_dtype(handle, Int32(index))
    }
}

// MARK: - Callback bridging

/// Internal class to bridge Swift closures to C callbacks.
private final class CallbackContext {
    let callback: (GeneratedToken) -> Bool
    init(callback: @escaping (GeneratedToken) -> Bool) {
        self.callback = callback
    }
}

/// C-compatible callback that bridges to the Swift closure stored in user_data.
private func cTokenCallback(
    tokenID: UInt32,
    text: UnsafePointer<CChar>?,
    userData: UnsafeMutableRawPointer?
) -> Bool {
    guard let userData else { return false }
    let ctx = Unmanaged<CallbackContext>.fromOpaque(userData).takeUnretainedValue()
    let str = text.map { String(cString: $0) } ?? ""
    let token = GeneratedToken(tokenID: tokenID, text: str)
    return ctx.callback(token)
}

// MARK: - Segment callback bridging

/// Internal class to bridge Swift closures to C segment callbacks.
private final class SegmentCallbackContext {
    let callback: (TranscriptSegment) -> Bool
    init(callback: @escaping (TranscriptSegment) -> Bool) {
        self.callback = callback
    }
}

/// C-compatible callback that bridges to the Swift closure for segment streaming.
private func cSegmentCallback(
    startMs: Int32,
    endMs: Int32,
    text: UnsafePointer<CChar>?,
    userData: UnsafeMutableRawPointer?
) -> Bool {
    guard let userData else { return false }
    let ctx = Unmanaged<SegmentCallbackContext>.fromOpaque(userData).takeUnretainedValue()
    let str = text.map { String(cString: $0) } ?? ""
    let segment = TranscriptSegment(startMs: Int(startMs), endMs: Int(endMs), text: str)
    return ctx.callback(segment)
}

// MARK: - AsyncSequence support for streaming generation

/// An asynchronous sequence that yields tokens as they are generated.
@available(macOS 10.15, iOS 13.0, *)
public struct TokenStream: AsyncSequence {
    public typealias Element = GeneratedToken

    private let model: BaseRTModel
    private let tokens: [UInt32]
    private let maxTokens: Int
    private let sampling: SamplingConfig
    private let isContinuation: Bool

    init(model: BaseRTModel, tokens: [UInt32], maxTokens: Int, sampling: SamplingConfig, isContinuation: Bool) {
        self.model = model
        self.tokens = tokens
        self.maxTokens = maxTokens
        self.sampling = sampling
        self.isContinuation = isContinuation
    }

    public func makeAsyncIterator() -> AsyncIterator {
        AsyncIterator(stream: self)
    }

    public struct AsyncIterator: AsyncIteratorProtocol {
        private let stream: TokenStream
        private var started = false
        private var continuation: AsyncStream<GeneratedToken>.Iterator
        private let asyncStream: AsyncStream<GeneratedToken>

        init(stream: TokenStream) {
            self.stream = stream
            var capturedContinuation: AsyncStream<GeneratedToken>.Continuation!
            self.asyncStream = AsyncStream { continuation in
                capturedContinuation = continuation
            }
            self.continuation = asyncStream.makeAsyncIterator()

            let cont = capturedContinuation!
            let model = stream.model
            let tokens = stream.tokens
            let maxTokens = stream.maxTokens
            let sampling = stream.sampling
            let isContinuation = stream.isContinuation

            // Run generation on a background thread to avoid blocking the caller.
            DispatchQueue.global(qos: .userInitiated).async {
                let callback: (GeneratedToken) -> Bool = { token in
                    cont.yield(token)
                    return true
                }
                if isContinuation {
                    model.generateContinue(tokens: tokens, maxTokens: maxTokens, sampling: sampling, onToken: callback)
                } else {
                    model.generate(tokens: tokens, maxTokens: maxTokens, sampling: sampling, onToken: callback)
                }
                cont.finish()
            }
        }

        public mutating func next() async -> GeneratedToken? {
            await continuation.next()
        }
    }
}

@available(macOS 10.15, iOS 13.0, *)
extension BaseRTModel {

    /// Stream generated tokens as an AsyncSequence.
    ///
    /// Usage:
    /// ```swift
    /// for await token in model.stream(tokens: promptTokens, maxTokens: 256) {
    ///     print(token.text, terminator: "")
    /// }
    /// ```
    public func stream(
        tokens: [UInt32],
        maxTokens: Int,
        sampling: SamplingConfig = SamplingConfig()
    ) -> TokenStream {
        TokenStream(model: self, tokens: tokens, maxTokens: maxTokens, sampling: sampling, isContinuation: false)
    }

    /// Stream continued generation as an AsyncSequence.
    public func streamContinue(
        tokens: [UInt32],
        maxTokens: Int,
        sampling: SamplingConfig = SamplingConfig()
    ) -> TokenStream {
        TokenStream(model: self, tokens: tokens, maxTokens: maxTokens, sampling: sampling, isContinuation: true)
    }
}
