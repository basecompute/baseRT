import CBaseRT
import Foundation

// MARK: - Process-wide engine configuration

/// Process-wide engine settings. These map to `baseRT_set_*` globals and MUST
/// be set BEFORE constructing any `BaseRTModel`.
public enum BaseRTEngine {
    /// Runtime engine version string (e.g. "0.2.0").
    public static var version: String {
        guard let p = baseRT_version_string() else { return "" }
        return String(cString: p)
    }

    /// KV-cache element width: 0 = auto, 8 = Q8_0, 16 = F16.
    public static func setKVBits(_ bits: Int32) {
        baseRT_set_kv_bits(bits)
    }

    /// Toggle paged-KV mode for subsequent loads (required for multi-sequence
    /// batching and prefix caching).
    public static func setPagedKV(_ enable: Bool) {
        baseRT_set_paged_kv(enable ? 1 : 0)
    }

    /// Max in-flight batch size for batched decode (sizes the logits scratch).
    public static func setMaxBatchSize(_ n: Int32) {
        baseRT_set_max_batch_size(n)
    }

    /// Last thread-local error message, or nil.
    public static var lastError: String? {
        guard let p = baseRT_get_error() else { return nil }
        let s = String(cString: p)
        return s.isEmpty ? nil : s
    }
}

// MARK: - Additional model surface
//
// Every method below touches the non-thread-safe C `handle`, so — exactly like
// the core surface in `BaseRT.swift` — each goes through `withHandleLock`, which
// rejects re-entry from a token/segment callback (throwing
// `BaseRTError.reentrantModelCall`) before acquiring the non-recursive
// `handleLock`. Ordinary concurrent callers block-and-wait on the lock as usual.
// Because the guard can throw, these APIs are `throws` too.

extension BaseRTModel {

    /// Total tokens currently in the KV cache.
    /// - Throws: `BaseRTError.reentrantModelCall` if read from within a callback.
    public var positionTokens: Int {
        get throws { try withHandleLock { Int(baseRT_get_position(handle)) } }
    }

    /// Primary end-of-sequence token id.
    /// - Throws: `BaseRTError.reentrantModelCall` if read from within a callback.
    public var eosTokenID: UInt32 {
        get throws { try withHandleLock { baseRT_eos_token_id(handle) } }
    }

    /// BOS / EOS token strings (as substituted in HF chat templates).
    /// - Throws: `BaseRTError.reentrantModelCall` if read from within a callback.
    public var bosToken: String {
        get throws { try withHandleLock { cString(baseRT_bos_token(handle)) } }
    }
    /// - Throws: `BaseRTError.reentrantModelCall` if read from within a callback.
    public var eosToken: String {
        get throws { try withHandleLock { cString(baseRT_eos_token(handle)) } }
    }

    /// Raw Jinja chat template folded in from the `.base` bundle (empty if none).
    /// - Throws: `BaseRTError.reentrantModelCall` if read from within a callback.
    public var chatTemplateJinja: String {
        get throws { try withHandleLock { cString(baseRT_chat_template_jinja(handle)) } }
    }

    /// Stateless single-token decode (does not advance the incremental decoder).
    /// - Throws: `BaseRTError.reentrantModelCall` if called from within a callback.
    public func decodeTokenStatic(_ tokenID: UInt32) throws -> String {
        try withHandleLock { cString(baseRT_decode_token_static(handle, tokenID)) }
    }

    /// Generate and return the full decoded text in one call.
    /// - Throws: `BaseRTError.reentrantModelCall` if called from within a callback.
    @discardableResult
    public func generateText(
        tokens: [UInt32],
        maxTokens: Int,
        sampling: SamplingConfig = SamplingConfig()
    ) throws -> String {
        var out = ""
        _ = try generate(tokens: tokens, maxTokens: maxTokens, sampling: sampling) { tok in
            out += tok.text
            return true
        }
        return out
    }

    // MARK: Multimodal

    /// Number of image placeholder tokens the vision tower emits for an image.
    /// - Throws: `BaseRTError.reentrantModelCall` if called from within a callback.
    public func imageTokenCount(imagePath: String) throws -> Int {
        try withHandleLock { Int(baseRT_image_num_tokens(handle, imagePath)) }
    }

    /// Multimodal prefill: run the vision tower on `imagePath`, splice features
    /// at image-token positions, then prefill. Returns the first generated token
    /// (0 on error — check `BaseRTEngine.lastError`).
    /// - Throws: `BaseRTError.reentrantModelCall` if called from within a callback.
    public func prefillImage(tokens: [UInt32], imagePath: String) throws -> UInt32 {
        try withHandleLock {
            tokens.withUnsafeBufferPointer { buf in
                baseRT_prefill_image(handle, buf.baseAddress, Int32(tokens.count), imagePath)
            }
        }
    }

    /// Number of audio placeholder tokens for `nSamples` of 16 kHz mono PCM.
    /// - Throws: `BaseRTError.reentrantModelCall` if called from within a callback.
    public func audioTokenCount(nSamples: Int) throws -> Int {
        try withHandleLock { Int(baseRT_audio_num_tokens(handle, Int32(nSamples))) }
    }

    /// Audio prefill: run the Conformer encoder on PCM (16 kHz mono Float32),
    /// splice features at audio-token positions, then prefill.
    /// - Throws: `BaseRTError.reentrantModelCall` if called from within a callback.
    public func prefillAudio(tokens: [UInt32], pcm: [Float]) throws -> UInt32 {
        try withHandleLock {
            tokens.withUnsafeBufferPointer { tBuf in
                pcm.withUnsafeBufferPointer { pBuf in
                    baseRT_prefill_audio(
                        handle, tBuf.baseAddress, Int32(tokens.count),
                        pBuf.baseAddress, Int32(pcm.count))
                }
            }
        }
    }

    // MARK: KV-cache state

    /// Truncate the KV cache to `toPosition` tokens (drop everything after).
    /// - Throws: `BaseRTError.reentrantModelCall` if called from within a callback.
    public func rollback(toPosition: Int) throws {
        try withHandleLock { baseRT_rollback(handle, Int32(toPosition)) }
    }

    /// Persist the current KV-cache state to `path`. Returns 0 on success.
    /// - Throws: `BaseRTError.reentrantModelCall` if called from within a callback.
    @discardableResult
    public func saveState(path: String) throws -> Int32 {
        try withHandleLock { baseRT_save_state(handle, path) }
    }

    /// Restore KV-cache state previously written by `saveState`. Returns 0 on success.
    /// - Throws: `BaseRTError.reentrantModelCall` if called from within a callback.
    @discardableResult
    public func loadState(path: String) throws -> Int32 {
        try withHandleLock { baseRT_load_state(handle, path) }
    }

    // MARK: LoRA

    /// Install a LoRA adapter (`.base` bundle). Replaces any active adapter.
    /// Returns 0 on success, negative on failure.
    /// - Throws: `BaseRTError.reentrantModelCall` if called from within a callback.
    @discardableResult
    public func loadLoRA(path: String) throws -> Int32 {
        try withHandleLock { baseRT_lora_load(handle, path) }
    }

    /// Detach the active LoRA adapter, if any.
    /// - Throws: `BaseRTError.reentrantModelCall` if called from within a callback.
    public func unloadLoRA() throws {
        try withHandleLock { baseRT_lora_unload(handle) }
    }

    /// Active adapter id (the path it was loaded from), or "" when none.
    /// - Throws: `BaseRTError.reentrantModelCall` if read from within a callback.
    public var loraID: String {
        get throws { try withHandleLock { cString(baseRT_lora_id(handle)) } }
    }

    // MARK: helpers

    private func cString(_ p: UnsafePointer<CChar>?) -> String {
        guard let p else { return "" }
        return String(cString: p)
    }
}
