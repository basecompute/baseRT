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
    /// Thrown when a model API is called from within a generation/transcription
    /// callback (`onToken` / `onSegment`) **on the same thread** while the native
    /// handle is still busy running that generation. The handle is exclusively in
    /// use for the callback's duration, so a same-thread re-entrant call is
    /// rejected fast (this error) instead of corrupting KV/GPU state.
    ///
    /// The guard is thread-local, so it only covers same-thread re-entry. If a
    /// callback instead hands work **synchronously to another thread** that then
    /// calls the model, that call blocks on the busy handle and can **deadlock**
    /// — it is *not* rejected. That pattern violates the single-owner contract
    /// (do not call model APIs from a different thread a callback is
    /// synchronously waiting on).
    case reentrantModelCall
    /// Thrown internally by `generateCore` when the caller's cancellation check
    /// fires AFTER the handle lock is acquired but BEFORE the native generate
    /// call starts. Used only by the stream worker to skip prompt prefill and
    /// generation entirely for an already-cancelled stream; it is swallowed by
    /// the worker (`try?`) and surfaces to callers as a cleanly finished,
    /// empty stream. The public `generate` / `generateContinue` paths never pass
    /// a cancellation check, so they never see this.
    case cancelledBeforeGeneration

    public var errorDescription: String? {
        switch self {
        case .loadFailed(let msg): return "Failed to load model: \(msg)"
        case .encodeFailed(let msg): return "Encoding failed: \(msg)"
        case .transcribeFailed(let msg): return "Transcription failed: \(msg)"
        case .reentrantModelCall:
            return "Cannot call a model API from within a generation callback: "
                + "the native handle is busy for the duration of the callback."
        case .cancelledBeforeGeneration:
            return "Generation was cancelled before it started."
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

/// One segment of the last transcription (verbose_json surface), with the
/// text copied into owned memory.
///
/// `avgLogprob` / `noSpeechProb` / `compressionRatio` / `temperature` are
/// window-level values applied to every segment decoded in that 30 s window
/// (the engine's documented approximation).
public struct TranscribeSegment: Sendable {
    /// Start time in milliseconds.
    public let startMs: Int
    /// End time in milliseconds.
    public let endMs: Int
    /// Transcribed text for this segment.
    public let text: String
    /// Average token log-probability of the window's decode.
    public let avgLogprob: Float
    /// p(<|nospeech|>) at the window's SOT position.
    public let noSpeechProb: Float
    /// zlib compression ratio of the window's text.
    public let compressionRatio: Float
    /// Temperature of the accepted fallback-ladder attempt.
    public let temperature: Float
}

// MARK: - BaseRTModel

/// Swift wrapper around the BaseRT C inference engine.
///
/// Thread safety: instances are not thread-safe. Use one model per thread/actor.
public final class BaseRTModel {
    let handle: baseRT_model_t  // internal: used by extensions in this module

    /// Serializes every native call that touches the non-thread-safe `handle`.
    ///
    /// The streaming worker runs `baseRT_generate` on a background thread via
    /// `generate`/`generateContinue`, so it holds this lock for the FULL
    /// duration of that dispatched call — not just until the cancellation flag
    /// is observed. EVERY public method that touches `handle` — generation,
    /// transcription, tokenization, embeddings, chat templating, the low-level
    /// prefill/decode/chain-decode API, the multimodal/state/LoRA extension
    /// APIs (see `BaseRTEngine.swift`), state mutators (`reset`,
    /// `setSpeculation`, `setTimestamps`), and even the read-only info
    /// accessors — acquires this lock before entering C. Because a stream's
    /// background worker reuses `generate`/`generateContinue`, any subsequent
    /// call acquires this lock before touching the handle and therefore waits
    /// for the previous worker to fully return. This is the await-before-reuse
    /// guarantee: even if a consumer `break`s (cancels) and immediately reuses
    /// the model — via reset, prefill, another stream, anything — the next
    /// call BLOCKS on this lock until the in-flight worker's `baseRT_generate`
    /// returns and releases it, so two callers never overlap on the handle and
    /// cannot corrupt KV/GPU state. Correctness (no overlap) is guaranteed;
    /// only the promptness of the stop is best-effort (an in-flight C call
    /// cannot be force-killed — see `IteratorCancellation`). The deinit path
    /// never contends here: the worker closure retains `self` for its lifetime,
    /// so deallocation (and `baseRT_free_model`) cannot begin while a worker is
    /// in flight, and therefore never needs — and never takes — this lock.
    ///
    /// This is a NON-recursive lock. `baseRT_generate` / `baseRT_transcribe*`
    /// invoke the user's `onToken` / segment callbacks WHILE this lock is held
    /// on the worker thread AND while the handle is genuinely, exclusively busy.
    /// SAME-THREAD re-entry from inside such a callback cannot be admitted here:
    /// running a native call mid-generation corrupts KV/GPU state, and blocking
    /// on this non-recursive lock (a same-thread re-entrant call) would deadlock.
    /// So it is rejected fast, one level up, by the THREAD-LOCAL callback-depth
    /// marker checked in `withHandleLock`, which throws
    /// `BaseRTError.reentrantModelCall` before ever touching this lock — but ONLY
    /// for a call made ON THE WORKER THREAD that is currently inside this model's
    /// callback (see `callbackDepthKey`). Any OTHER thread that calls a model API
    /// while a callback happens to be executing has a marker of 0, so it does NOT
    /// throw: it falls through to this lock and block-and-waits until the worker's
    /// native call returns, exactly as an ordinary concurrent caller does. That is
    /// the deliberate difference from the former model-wide flag, which could not
    /// tell callback-originated re-entry apart from unrelated concurrency and so
    /// wrongly rejected legitimate concurrent callers.
    ///
    /// - Warning: The one case thread-local detection cannot rescue is a callback
    ///   that synchronously hands work to a *different* thread which then calls
    ///   this model — e.g. `onToken = { DispatchQueue.main.sync { try model.reset() } }`.
    ///   That other thread's marker is 0, so it does not throw; it blocks on this
    ///   lock while the worker (parked inside `.sync`) still holds it, and the two
    ///   deadlock. This is NOT eliminated — it is a single-owner-contract
    ///   violation: do NOT call model APIs from a *different* thread that a
    ///   callback is synchronously waiting on. The supported reentry case — a
    ///   same-thread call from within the callback — throws `reentrantModelCall`
    ///   cleanly instead.
    let handleLock = NSLock()

    /// Per-instance key under which this model records, in the CURRENT thread's
    /// `threadDictionary`, the depth of user callbacks (`onToken` / `onSegment`)
    /// currently executing FOR THIS MODEL on that thread. It replaces the former
    /// model-wide `os_unfair_lock`-guarded boolean, which could not distinguish a
    /// callback-originated re-entrant call from an ordinary concurrent call made
    /// on an unrelated thread while a callback happened to be running.
    ///
    /// Because the token/segment trampolines run on the worker thread, the marker
    /// lives on the worker thread: `withHandleLock` throws only when it sees a
    /// non-zero depth on the CALLING thread, i.e. a genuine same-thread re-entry.
    /// The key is per instance (a unique string) so that being inside model A's
    /// callback never rejects a call to a DIFFERENT model B from the same thread.
    /// A depth counter (not a bool) makes nested callbacks safe.
    private let callbackDepthKey = "com.baseRT.callbackDepth.\(UUID().uuidString)"

    /// Increment this thread's callback depth for this model. Called by the
    /// trampolines immediately before invoking the USER's closure. NOT called for
    /// the internal AsyncStream yield path, which never runs user model-reentrant
    /// code (marking it would spuriously reject consumers that legitimately call
    /// the model between yielded tokens).
    func enterCallbackScope() {
        let dict = Thread.current.threadDictionary
        let depth = (dict[callbackDepthKey] as? Int) ?? 0
        dict[callbackDepthKey] = depth + 1
    }

    /// Decrement this thread's callback depth for this model. Called in a `defer`
    /// by the trampolines right after the user's closure returns.
    func exitCallbackScope() {
        let dict = Thread.current.threadDictionary
        let depth = (dict[callbackDepthKey] as? Int) ?? 0
        if depth <= 1 {
            dict.removeObject(forKey: callbackDepthKey)
        } else {
            dict[callbackDepthKey] = depth - 1
        }
    }

    /// Whether THE CURRENT THREAD is presently inside one of this model's user
    /// callbacks. True only on the worker thread while its `onToken` / `onSegment`
    /// closure runs; false on every other thread, including threads that call the
    /// model concurrently during that window.
    private var isInCallbackOnCurrentThread: Bool {
        ((Thread.current.threadDictionary[callbackDepthKey] as? Int) ?? 0) > 0
    }

    /// Reject same-thread re-entrant calls, then run `body` while holding
    /// `handleLock` so the native handle is touched by at most one worker at a
    /// time.
    ///
    /// If the CURRENT thread is already inside one of this model's callbacks
    /// (`isInCallbackOnCurrentThread`), the handle is busy on this very thread and
    /// a re-entrant native call would corrupt KV/GPU state (or deadlock the
    /// non-recursive lock): throw `BaseRTError.reentrantModelCall` immediately,
    /// BEFORE acquiring the lock. Every OTHER thread — including one that calls a
    /// model API while a callback happens to be executing on the worker thread —
    /// has a zero marker, so it does NOT throw: it falls through to `handleLock`
    /// and block-and-waits until the in-flight native call returns, exactly as an
    /// ordinary concurrent caller does. This is the behavior the former model-wide
    /// flag got wrong (it rejected such unrelated callers too).
    @inline(__always)
    func withHandleLock<T>(_ body: () throws -> T) throws -> T {
        if isInCallbackOnCurrentThread { throw BaseRTError.reentrantModelCall }
        handleLock.lock()
        defer { handleLock.unlock() }
        return try body()
    }

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
        // The thread-local callback-depth marker needs no teardown here: it lives
        // in each worker thread's `threadDictionary` and is always balanced by the
        // trampoline's `defer` before the native call returns, so nothing survives
        // past generation. deinit takes no lock and simply frees the handle.
        baseRT_free_model(handle)
    }

    // MARK: Model info

    /// The model configuration (dimensions, layers, architecture, etc.).
    /// - Throws: `BaseRTError.reentrantModelCall` if read from within a callback.
    public var config: ModelConfig {
        get throws { try withHandleLock { ModelConfig(baseRT_get_config(handle)) } }
    }

    /// Total GPU memory used by the model, in bytes.
    /// - Throws: `BaseRTError.reentrantModelCall` if read from within a callback.
    public var memoryUsage: Int {
        get throws { try withHandleLock { baseRT_model_memory(handle) } }
    }

    /// Whether this is a Whisper audio model.
    /// - Throws: `BaseRTError.reentrantModelCall` if read from within a callback.
    public var isWhisper: Bool {
        get throws { try withHandleLock { baseRT_is_whisper(handle) } }
    }

    /// Current KV cache position (number of tokens processed so far).
    /// - Throws: `BaseRTError.reentrantModelCall` if read from within a callback.
    public var position: Int {
        get throws { try withHandleLock { Int(baseRT_get_position(handle)) } }
    }

    // MARK: Tokenization

    /// Encode text into token IDs.
    ///
    /// - Parameter text: The input text to tokenize.
    /// - Returns: Array of token IDs.
    /// - Throws: `BaseRTError.encodeFailed` if encoding fails.
    public func encode(text: String) throws -> [UInt32] {
        // Serialized against every other handle-touching call (see handleLock);
        // rejects re-entry from a callback.
        try withHandleLock {
            let maxTokens = max(text.utf8.count * 2, 1024)
            var tokens = [UInt32](repeating: 0, count: maxTokens)
            let count = baseRT_encode(handle, text, &tokens, Int32(maxTokens))
            if count < 0 {
                let err = baseRT_get_error().flatMap { String(cString: $0) } ?? "encoding failed"
                throw BaseRTError.encodeFailed(err)
            }
            return Array(tokens.prefix(Int(count)))
        }
    }

    /// Decode a single token ID back to its text representation.
    ///
    /// - Parameter tokenID: The token ID to decode.
    /// - Returns: The text for this token, or an empty string if invalid.
    /// - Throws: `BaseRTError.reentrantModelCall` if called from within a callback.
    public func decodeToken(_ tokenID: UInt32) throws -> String {
        // Serialized against every other handle-touching call (see handleLock).
        try withHandleLock {
            guard let ptr = baseRT_decode_token(handle, tokenID) else {
                return ""
            }
            return String(cString: ptr)
        }
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
    ///
    /// - Warning: `onToken` runs synchronously on the worker thread while the
    ///   native handle is busy. Calling ANY method of this same model from inside
    ///   `onToken` ON THE SAME (worker) THREAD THROWS
    ///   `BaseRTError.reentrantModelCall` rather than corrupting the busy handle
    ///   or self-deadlocking the non-recursive handle lock; generation continues
    ///   past the rejected call. Callbacks that do not re-enter the model (return
    ///   a value, append to a buffer, hand the token to another queue/stream) are
    ///   unaffected.
    ///
    ///   The reentrancy guard is THREAD-LOCAL, so this does NOT cover a callback
    ///   that synchronously hands work to a *different* thread which then calls
    ///   this model — e.g. `onToken = { DispatchQueue.main.sync { try model.reset() } }`.
    ///   That other thread is not marked as inside a callback, so it does not
    ///   throw; it blocks on the handle lock the worker still holds and the two
    ///   deadlock. This is not eliminated — it is a single-owner-contract
    ///   violation: do NOT call model APIs from a different thread that a callback
    ///   is synchronously waiting on.
    @discardableResult
    public func generate(
        tokens: [UInt32],
        maxTokens: Int,
        sampling: SamplingConfig = SamplingConfig(),
        onToken: ((GeneratedToken) -> Bool)? = nil
    ) throws -> GenerationStats {
        try generateCore(
            tokens: tokens, maxTokens: maxTokens, sampling: sampling,
            onToken: onToken, isContinuation: false, guardCallback: true)
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
    ///
    /// - Warning: `onToken` runs synchronously while the native handle is busy.
    ///   Calling this model from inside `onToken` ON THE SAME (worker) THREAD
    ///   THROWS `BaseRTError.reentrantModelCall`. A callback that synchronously
    ///   hops to a *different* thread which then calls this model is a
    ///   single-owner-contract violation that can deadlock and is NOT caught. See
    ///   `generate(tokens:maxTokens:...)`.
    @discardableResult
    public func generateContinue(
        tokens: [UInt32],
        maxTokens: Int,
        sampling: SamplingConfig = SamplingConfig(),
        onToken: ((GeneratedToken) -> Bool)? = nil
    ) throws -> GenerationStats {
        try generateCore(
            tokens: tokens, maxTokens: maxTokens, sampling: sampling,
            onToken: onToken, isContinuation: true, guardCallback: true)
    }

    /// Shared implementation for `generate` / `generateContinue` and the internal
    /// AsyncStream worker.
    ///
    /// - Parameter guardCallback: when true (the direct user-facing paths), the
    ///   token trampoline increments the worker thread's callback-depth marker
    ///   around the user's `onToken` so a same-thread re-entrant model call throws
    ///   instead of corrupting/deadlocking. The AsyncStream worker passes `false`:
    ///   its `onToken` only does `continuation.yield` and never runs user
    ///   model-reentrant code, so marking it would spuriously reject consumers
    ///   that legitimately call the model between tokens.
    @discardableResult
    func generateCore(
        tokens: [UInt32],
        maxTokens: Int,
        sampling: SamplingConfig,
        onToken: ((GeneratedToken) -> Bool)?,
        isContinuation: Bool,
        guardCallback: Bool,
        shouldCancelBeforeGenerate: (() -> Bool)? = nil
    ) throws -> GenerationStats {
        // Hold the handle lock across the whole native call so a concurrent
        // reuse (including a freshly started stream worker) waits for us here;
        // reject re-entry from an active callback before locking.
        let stats = try withHandleLock { () throws -> BaseRTGenerationStats in
            // We now own serialization of the handle. If the stream that
            // dispatched us was cancelled while we were queued behind another
            // in-flight worker for this lock, honor it HERE — before the native
            // call — so we skip prompt prefill and generation entirely instead
            // of burning GPU time only to have the first token callback return
            // false. The lock is still released by withHandleLock's defer on
            // this throw, so it is not leaked.
            if shouldCancelBeforeGenerate?() == true {
                throw BaseRTError.cancelledBeforeGeneration
            }
            return tokens.withUnsafeBufferPointer { buf in
                sampling.withCValue { cfg in
                    let cGen = isContinuation ? baseRT_generate_continue : baseRT_generate
                    if let callback = onToken {
                        // guardModel non-nil => trampoline increments the thread-
                        // local callback-depth marker around the user closure; nil
                        // for the stream yield path.
                        let ctx = CallbackContext(
                            callback: callback, guardModel: guardCallback ? self : nil)
                        let unmanaged = Unmanaged.passRetained(ctx)
                        defer { unmanaged.release() }
                        return cGen(
                            handle,
                            buf.baseAddress,
                            Int32(tokens.count),
                            Int32(maxTokens),
                            cfg,
                            cTokenCallback,
                            unmanaged.toOpaque()
                        )
                    } else {
                        return cGen(
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
        // Serialized against every other handle-touching call (see handleLock);
        // rejects re-entry from a callback.
        try withHandleLock {
            var stats = BaseRTTranscribeStats()
            guard let result = baseRT_transcribe(handle, wavPath, language, &stats) else {
                let err = baseRT_get_error().flatMap { String(cString: $0) } ?? "transcription failed"
                throw BaseRTError.transcribeFailed(err)
            }
            return (String(cString: result), TranscribeStats(stats))
        }
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
    ///
    /// - Warning: `onSegment` runs synchronously while the native handle is busy.
    ///   Calling this model from inside `onSegment` ON THE SAME (worker) THREAD
    ///   THROWS `BaseRTError.reentrantModelCall` rather than corrupting the busy
    ///   handle or self-deadlocking. A callback that synchronously hops to a
    ///   *different* thread which then calls this model is a
    ///   single-owner-contract violation that can deadlock and is NOT caught. See
    ///   `generate(tokens:maxTokens:...)`.
    public func transcribe(
        wavPath: String,
        language: String = "en",
        onSegment: @escaping (TranscriptSegment) -> Bool
    ) throws -> (String, TranscribeStats) {
        // Serialized against every other handle-touching call (see handleLock).
        try withHandleLock {
            var stats = BaseRTTranscribeStats()
            let ctx = SegmentCallbackContext(callback: onSegment, guardModel: self)
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
    }

    /// Transcribe audio from raw PCM samples (16kHz, mono, Float32).
    ///
    /// - Parameters:
    ///   - samples: Array of Float32 audio samples at 16kHz.
    ///   - language: Language code (e.g. "en", "auto"). Defaults to "en".
    /// - Returns: Tuple of transcribed text and timing statistics.
    /// - Throws: `BaseRTError.transcribeFailed` if transcription fails.
    public func transcribePCM(samples: [Float], language: String = "en") throws -> (String, TranscribeStats) {
        // Serialized against every other handle-touching call (see handleLock);
        // rejects re-entry from a callback.
        try withHandleLock {
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
    }

    /// Transcribe raw PCM samples with per-segment streaming.
    ///
    /// - Warning: `onSegment` runs synchronously while the native handle is busy.
    ///   Calling this model from inside `onSegment` ON THE SAME (worker) THREAD
    ///   THROWS `BaseRTError.reentrantModelCall` rather than corrupting the busy
    ///   handle or self-deadlocking. A callback that synchronously hops to a
    ///   *different* thread which then calls this model is a
    ///   single-owner-contract violation that can deadlock and is NOT caught. See
    ///   `generate(tokens:maxTokens:...)`.
    public func transcribePCM(
        samples: [Float],
        language: String = "en",
        onSegment: @escaping (TranscriptSegment) -> Bool
    ) throws -> (String, TranscribeStats) {
        // Serialized against every other handle-touching call (see handleLock).
        try withHandleLock {
            var stats = BaseRTTranscribeStats()
            let ctx = SegmentCallbackContext(callback: onSegment, guardModel: self)
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
    }

    // MARK: Embeddings

    /// Compute embeddings from token IDs using the model's hidden states.
    ///
    /// - Parameter tokens: Array of token IDs.
    /// - Returns: Array of float embedding values.
    /// - Throws: `BaseRTError.encodeFailed` if embedding fails.
    public func embed(tokens: [UInt32]) throws -> [Float] {
        // Serialized against every other handle-touching call (see handleLock);
        // rejects re-entry from a callback. Calls the C `baseRT_embedding_dim`
        // directly (not the `embeddingDim` property) to avoid a redundant
        // re-lock — the non-recursive lock cannot be nested on one thread.
        try withHandleLock {
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
    }

    /// Compute embeddings from text directly (tokenizes internally).
    ///
    /// - Parameter text: Input text to embed.
    /// - Returns: Array of float embedding values.
    /// - Throws: `BaseRTError.encodeFailed` if embedding fails.
    public func embedText(_ text: String) throws -> [Float] {
        // Serialized against every other handle-touching call (see handleLock);
        // rejects re-entry from a callback. Calls the C `baseRT_embedding_dim`
        // directly (not the `embeddingDim` property) to avoid a redundant
        // re-lock — the non-recursive lock cannot be nested on one thread.
        try withHandleLock {
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
    }

    /// The embedding dimension for this model.
    /// - Throws: `BaseRTError.reentrantModelCall` if read from within a callback.
    public var embeddingDim: Int {
        get throws { try withHandleLock { Int(baseRT_embedding_dim(handle)) } }
    }

    // MARK: Chat templates

    /// Format a chat prompt using the model's native template.
    ///
    /// - Parameters:
    ///   - system: System prompt.
    ///   - user: User message.
    /// - Returns: Formatted chat string.
    /// - Throws: `BaseRTError.reentrantModelCall` if called from within a callback.
    public func formatChat(system: String, user: String) throws -> String {
        // Serialized against every other handle-touching call (see handleLock).
        try withHandleLock {
            guard let ptr = baseRT_format_chat(handle, system, user) else {
                return ""
            }
            return String(cString: ptr)
        }
    }

    /// The chat template name for the loaded model (e.g. "chatml", "llama3", "gemma").
    /// - Throws: `BaseRTError.reentrantModelCall` if read from within a callback.
    public var chatTemplate: String {
        get throws {
            try withHandleLock {
                guard let ptr = baseRT_chat_template(handle) else {
                    return ""
                }
                return String(cString: ptr)
            }
        }
    }

    // MARK: Token counting

    /// Count tokens in text without allocating an output buffer.
    ///
    /// - Parameter text: Input text.
    /// - Returns: Number of tokens.
    /// - Throws: `BaseRTError.reentrantModelCall` if called from within a callback.
    public func tokenCount(_ text: String) throws -> Int {
        try withHandleLock { Int(baseRT_token_count(handle, text)) }
    }

    // MARK: Whisper settings

    /// Enable or disable timestamp generation for Whisper transcription.
    ///
    /// When enabled (default), output includes `[start --> end] text` segments.
    /// When disabled, faster greedy decode produces plain text only.
    /// - Throws: `BaseRTError.reentrantModelCall` if called from within a callback.
    public func setTimestamps(enabled: Bool) throws {
        try withHandleLock { baseRT_set_timestamps(handle, enabled) }
    }

    /// Set the Whisper task: `"transcribe"` (default) or `"translate"`.
    /// - Throws: `BaseRTError.reentrantModelCall` if called from within a callback,
    ///   or `BaseRTError.transcribeFailed` on an unknown task / translate with an
    ///   English-only model.
    public func setTask(_ task: String) throws {
        try withHandleLock {
            if !baseRT_set_task(handle, task) {
                let err = baseRT_get_error().flatMap { String(cString: $0) } ?? "baseRT_set_task failed"
                throw BaseRTError.transcribeFailed(err)
            }
        }
    }

    /// Set an initial prompt to bias Whisper decoding (`nil` clears it).
    /// - Throws: `BaseRTError.reentrantModelCall` if called from within a callback.
    public func setInitialPrompt(_ text: String?) throws {
        try withHandleLock { baseRT_set_initial_prompt(handle, text) }
    }

    /// Condition each 30s window on previous decoded text (default true).
    /// - Throws: `BaseRTError.reentrantModelCall` if called from within a callback.
    public func setConditionOnPreviousText(enabled: Bool) throws {
        try withHandleLock { baseRT_set_condition_on_previous_text(handle, enabled) }
    }

    /// Language code of the last transcription (detected or requested).
    /// - Throws: `BaseRTError.reentrantModelCall` if called from within a callback.
    public func transcribeLanguage() throws -> String {
        try withHandleLock {
            guard let cstr = baseRT_transcribe_language(handle) else { return "" }
            return String(cString: cstr)
        }
    }

    /// Duration of the last transcription's source audio in milliseconds
    /// (the OpenAI verbose_json `duration` field is this value in fractional
    /// seconds). 0 if no transcription has run.
    /// - Throws: `BaseRTError.reentrantModelCall` if called from within a callback.
    public func transcribeAudioDurationMs() throws -> Int {
        try withHandleLock { Int(baseRT_transcribe_audio_duration_ms(handle)) }
    }

    /// Per-segment metadata for the last transcription (verbose_json
    /// surface). Empty if no transcription has run. Segment text is copied
    /// out, so the returned array stays valid after later transcriptions.
    /// - Throws: `BaseRTError.reentrantModelCall` if called from within a callback.
    public func transcribeSegments() throws -> [TranscribeSegment] {
        try withHandleLock {
            let n = Int(baseRT_transcribe_segment_count(handle))
            var segments: [TranscribeSegment] = []
            segments.reserveCapacity(max(n, 0))
            var i = 0
            while i < n {
                var seg = BaseRTTranscribeSegment()
                guard baseRT_transcribe_segment(handle, Int32(i), &seg) else { break }
                let text = seg.text.map { String(cString: $0) } ?? ""
                segments.append(
                    TranscribeSegment(
                        startMs: Int(seg.start_ms),
                        endMs: Int(seg.end_ms),
                        text: text,
                        avgLogprob: seg.avg_logprob,
                        noSpeechProb: seg.no_speech_prob,
                        compressionRatio: seg.compression_ratio,
                        temperature: seg.temperature
                    ))
                i += 1
            }
            return segments
        }
    }

    // MARK: State management

    /// Enable or disable speculative decoding (n-gram prediction).
    /// Only affects greedy (temperature=0) mode.
    /// - Throws: `BaseRTError.reentrantModelCall` if called from within a callback.
    public func setSpeculation(enabled: Bool) throws {
        try withHandleLock { baseRT_set_speculation(handle, enabled) }
    }

    /// Reset KV cache and internal state.
    /// - Throws: `BaseRTError.reentrantModelCall` if called from within a callback.
    public func reset() throws {
        try withHandleLock { baseRT_reset(handle) }
    }

    // MARK: Low-level API

    /// Run prefill on tokens, populating the KV cache.
    /// - Returns: The first generated token (argmax of prefill logits).
    /// - Throws: `BaseRTError.reentrantModelCall` if called from within a callback.
    public func prefill(tokens: [UInt32]) throws -> UInt32 {
        // Serialized against every other handle-touching call (see handleLock).
        try withHandleLock {
            tokens.withUnsafeBufferPointer { buf in
                baseRT_prefill(handle, buf.baseAddress, Int32(tokens.count))
            }
        }
    }

    /// Run one decode step.
    /// - Returns: The sampled token ID.
    /// - Throws: `BaseRTError.reentrantModelCall` if called from within a callback.
    public func decodeStep(tokenID: UInt32, position: Int) throws -> UInt32 {
        try withHandleLock { baseRT_decode_step(handle, tokenID, Int32(position)) }
    }

    /// Chain decode: generate multiple tokens in one GPU submission.
    /// - Returns: Array of generated token IDs.
    /// - Throws: `BaseRTError.reentrantModelCall` if called from within a callback.
    public func chainDecode(firstToken: UInt32, startPosition: Int, count: Int) throws -> [UInt32] {
        // Serialized against every other handle-touching call (see handleLock).
        try withHandleLock {
            var out = [UInt32](repeating: 0, count: count)
            let n = baseRT_chain_decode(handle, firstToken, Int32(startPosition), Int32(count), &out)
            return Array(out.prefix(Int(n)))
        }
    }

    // MARK: Model inspection

    /// Number of tensors in the model.
    /// - Throws: `BaseRTError.reentrantModelCall` if read from within a callback.
    public var tensorCount: Int {
        get throws { try withHandleLock { Int(baseRT_tensor_count(handle)) } }
    }

    /// Get the name of a tensor by index.
    /// - Throws: `BaseRTError.reentrantModelCall` if called from within a callback.
    public func tensorName(at index: Int) throws -> String? {
        try withHandleLock {
            guard let ptr = baseRT_tensor_name(handle, Int32(index)) else { return nil }
            return String(cString: ptr)
        }
    }

    /// Get the dtype code of a tensor by index.
    /// - Throws: `BaseRTError.reentrantModelCall` if called from within a callback.
    public func tensorDtype(at index: Int) throws -> UInt32 {
        try withHandleLock { baseRT_tensor_dtype(handle, Int32(index)) }
    }
}

// MARK: - Callback bridging

/// Internal class to bridge Swift closures to C callbacks.
private final class CallbackContext {
    let callback: (GeneratedToken) -> Bool
    /// When non-nil, the trampoline increments this model's thread-local
    /// callback-depth marker around the USER's closure so a same-thread re-entrant
    /// model API call throws instead of corrupting/deadlocking the busy handle.
    /// It is nil for the internal
    /// AsyncStream yield path, whose closure only does `continuation.yield` and
    /// never runs user model-reentrant code — flagging it would spuriously reject
    /// consumers that legitimately call the model between tokens.
    weak var guardModel: BaseRTModel?
    init(callback: @escaping (GeneratedToken) -> Bool, guardModel: BaseRTModel?) {
        self.callback = callback
        self.guardModel = guardModel
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
    // Only the direct user-callback paths supply a guardModel; the AsyncStream
    // yield path leaves it nil and runs unguarded.
    guard let model = ctx.guardModel else {
        return ctx.callback(token)
    }
    // Mark THIS (worker) thread as inside the model's callback so a same-thread
    // re-entrant model call throws; unrelated threads are unaffected.
    model.enterCallbackScope()
    defer { model.exitCallbackScope() }
    return ctx.callback(token)
}

// MARK: - Segment callback bridging

/// Internal class to bridge Swift closures to C segment callbacks.
private final class SegmentCallbackContext {
    let callback: (TranscriptSegment) -> Bool
    /// Always non-nil for the transcription paths — the user's `onSegment` runs
    /// synchronously inside the native call, so the trampoline increments the
    /// model's thread-local callback-depth marker around it to reject same-thread
    /// re-entrant model calls.
    weak var guardModel: BaseRTModel?
    init(callback: @escaping (TranscriptSegment) -> Bool, guardModel: BaseRTModel?) {
        self.callback = callback
        self.guardModel = guardModel
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
    guard let model = ctx.guardModel else {
        return ctx.callback(segment)
    }
    // Mark THIS (worker) thread as inside the model's callback so a same-thread
    // re-entrant model call throws; unrelated threads are unaffected.
    model.enterCallbackScope()
    defer { model.exitCallbackScope() }
    return ctx.callback(segment)
}

// MARK: - AsyncSequence support for streaming generation

/// Thread-safe cancellation flag shared between the async iterator and the
/// background generation thread.
private final class CancellationFlag {
    private let lock = NSLock()
    private var cancelled = false

    func cancel() {
        lock.lock()
        cancelled = true
        lock.unlock()
    }

    var isCancelled: Bool {
        lock.lock()
        defer { lock.unlock() }
        return cancelled
    }
}

/// Ties cancellation to the lifetime of a single `AsyncIterator`.
///
/// `AsyncIterator` is a value type and therefore has no `deinit`, while
/// `AsyncStream.onTermination` is NOT invoked when a consumer leaves a
/// `for await` loop via a plain `break` (the generation closure keeps the
/// continuation alive, so the stream never "terminates"). Holding the
/// cancellation flag inside this reference type — strongly referenced by the
/// iterator (and shared across its value copies) but deliberately NOT captured
/// by the background generation closure — makes cancellation fire from the
/// iterator's own deallocation: once the last iterator copy is dropped (break,
/// task cancellation, or normal exhaustion), this object deinitializes and
/// flips the flag, halting generation instead of letting it run to `maxTokens`
/// and overlap a subsequent call on the non-thread-safe model.
private final class IteratorCancellation {
    let flag: CancellationFlag
    init(_ flag: CancellationFlag) { self.flag = flag }
    deinit { flag.cancel() }
}

/// Shared, thread-safe once-gate for the deferred generation start.
///
/// `AsyncIterator` is a value type, so copying it before the first `next()`
/// would duplicate a plain `started` flag and let two copies each start
/// generation, launching overlapping `baseRT_generate` calls on the
/// non-thread-safe model. Holding the flag in this reference type, captured by
/// all iterator copies, guarantees generation starts exactly once.
private final class StartGate {
    private let lock = NSLock()
    private var started = false

    /// Returns true exactly once (for the first caller); false thereafter.
    func tryStart() -> Bool {
        lock.lock()
        defer { lock.unlock() }
        if started { return false }
        started = true
        return true
    }
}

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
        // Shared across all copies of this value-type iterator so generation
        // starts exactly once, even if the iterator is copied before next().
        private let startGate = StartGate()
        private var continuation: AsyncStream<GeneratedToken>.Iterator
        private let asyncStream: AsyncStream<GeneratedToken>
        private let startGeneration: () -> Void
        // Strongly held by every copy of this value-type iterator; its deinit
        // (fired when the last copy drops) flips the cancellation flag. This is
        // the primary cancellation path, since onTermination does not run on a
        // plain `break`. It is NOT captured by `startGeneration`, so the running
        // worker cannot keep it alive and prevent that deinit.
        private let lifetime: IteratorCancellation

        init(stream: TokenStream) {
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

            // Flipped when the consumer stops iterating (break out of
            // `for await`, task cancellation, iterator deallocation) so the
            // token callback halts generation instead of running to maxTokens.
            let cancelled = CancellationFlag()
            // Own the flag from the iterator's lifetime: its deinit cancels on a
            // plain `break`, which onTermination below does not observe. The
            // onTermination hook remains as a secondary path (e.g. finish()).
            self.lifetime = IteratorCancellation(cancelled)
            cont.onTermination = { _ in cancelled.cancel() }

            // Deferred until the first next() call so that merely creating an
            // iterator does not kick off GPU generation.
            self.startGeneration = {
                // Run generation on a background thread to avoid blocking the caller.
                DispatchQueue.global(qos: .userInitiated).async {
                    let callback: (GeneratedToken) -> Bool = { token in
                        if cancelled.isCancelled { return false }
                        cont.yield(token)
                        return !cancelled.isCancelled
                    }
                    // generateCore acquires the model's handleLock and holds it
                    // for the ENTIRE native call. If a prior worker is still
                    // winding down (e.g. the consumer just broke and immediately
                    // started this stream), this call blocks here on the
                    // background thread until that worker's baseRT_generate
                    // returns and releases the lock — the consumer meanwhile
                    // simply awaits the first yielded token. Two workers thus
                    // never overlap on the non-thread-safe handle.
                    //
                    // guardCallback: false — this internal callback only does
                    // `cont.yield` and never runs user model-reentrant code, so
                    // it must NOT mark the thread-local callback-depth (that would
                    // spuriously reject consumers that legitimately call the model
                    // between tokens).
                    //
                    // shouldCancelBeforeGenerate — checked by generateCore AFTER
                    // it has acquired the handle lock (serialization ownership)
                    // and IMMEDIATELY BEFORE the native call. If the consumer's
                    // task was already cancelled while this worker sat queued
                    // behind another in-flight stream for the lock, generateCore
                    // throws `.cancelledBeforeGeneration` instead of running
                    // native prompt prefill + generation — so a long prompt does
                    // NOT burn GPU time after cancellation.
                    //
                    // `try?`: generateCore throws only `.reentrantModelCall`
                    // (impossible here — this top-level worker sets no callback
                    // flag) or `.cancelledBeforeGeneration` (the skip above);
                    // both simply end the stream cleanly. Either way `cont.finish()`
                    // completes the continuation exactly once, yielding an empty /
                    // cancelled stream when generation was skipped.
                    _ = try? model.generateCore(
                        tokens: tokens, maxTokens: maxTokens, sampling: sampling,
                        onToken: callback, isContinuation: isContinuation, guardCallback: false,
                        shouldCancelBeforeGenerate: { cancelled.isCancelled })
                    cont.finish()
                }
            }
        }

        public mutating func next() async -> GeneratedToken? {
            if startGate.tryStart() {
                startGeneration()
            }
            return await continuation.next()
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
    ///
    /// Consuming the stream (the `for await` body above) is safe: tokens are
    /// delivered through an `AsyncStream` continuation, so the consumer runs on
    /// its own task and never re-enters the model from inside the generation
    /// worker's callback. Calling this same model from the consumer between
    /// tokens is likewise fine — the stream's internal yield callback does NOT
    /// arm the re-entrancy guard, so such calls simply serialize on the handle
    /// lock. (The guard only fires for a USER `onToken` / `onSegment` closure in
    /// the direct `generate` / `transcribe` paths; see
    /// `generate(tokens:maxTokens:...)`.)
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
