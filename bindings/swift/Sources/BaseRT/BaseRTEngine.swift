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

extension BaseRTModel {

    /// Total tokens currently in the KV cache.
    public var positionTokens: Int { Int(baseRT_get_position(handle)) }

    /// Primary end-of-sequence token id.
    public var eosTokenID: UInt32 { baseRT_eos_token_id(handle) }

    /// BOS / EOS token strings (as substituted in HF chat templates).
    public var bosToken: String { cString(baseRT_bos_token(handle)) }
    public var eosToken: String { cString(baseRT_eos_token(handle)) }

    /// Raw Jinja chat template folded in from the `.base` bundle (empty if none).
    public var chatTemplateJinja: String { cString(baseRT_chat_template_jinja(handle)) }

    /// Stateless single-token decode (does not advance the incremental decoder).
    public func decodeTokenStatic(_ tokenID: UInt32) -> String {
        cString(baseRT_decode_token_static(handle, tokenID))
    }

    /// Generate and return the full decoded text in one call.
    @discardableResult
    public func generateText(
        tokens: [UInt32],
        maxTokens: Int,
        sampling: SamplingConfig = SamplingConfig()
    ) -> String {
        var out = ""
        _ = generate(tokens: tokens, maxTokens: maxTokens, sampling: sampling) { tok in
            out += tok.text
            return true
        }
        return out
    }

    // MARK: Multimodal

    /// Number of image placeholder tokens the vision tower emits for an image.
    public func imageTokenCount(imagePath: String) -> Int {
        Int(baseRT_image_num_tokens(handle, imagePath))
    }

    /// Multimodal prefill: run the vision tower on `imagePath`, splice features
    /// at image-token positions, then prefill. Returns the first generated token
    /// (0 on error — check `BaseRTEngine.lastError`).
    public func prefillImage(tokens: [UInt32], imagePath: String) -> UInt32 {
        tokens.withUnsafeBufferPointer { buf in
            baseRT_prefill_image(handle, buf.baseAddress, Int32(tokens.count), imagePath)
        }
    }

    /// Number of audio placeholder tokens for `nSamples` of 16 kHz mono PCM.
    public func audioTokenCount(nSamples: Int) -> Int {
        Int(baseRT_audio_num_tokens(handle, Int32(nSamples)))
    }

    /// Audio prefill: run the Conformer encoder on PCM (16 kHz mono Float32),
    /// splice features at audio-token positions, then prefill.
    public func prefillAudio(tokens: [UInt32], pcm: [Float]) -> UInt32 {
        tokens.withUnsafeBufferPointer { tBuf in
            pcm.withUnsafeBufferPointer { pBuf in
                baseRT_prefill_audio(
                    handle, tBuf.baseAddress, Int32(tokens.count),
                    pBuf.baseAddress, Int32(pcm.count))
            }
        }
    }

    // MARK: KV-cache state

    /// Truncate the KV cache to `toPosition` tokens (drop everything after).
    public func rollback(toPosition: Int) {
        baseRT_rollback(handle, Int32(toPosition))
    }

    /// Persist the current KV-cache state to `path`. Returns 0 on success.
    @discardableResult
    public func saveState(path: String) -> Int32 {
        baseRT_save_state(handle, path)
    }

    /// Restore KV-cache state previously written by `saveState`. Returns 0 on success.
    @discardableResult
    public func loadState(path: String) -> Int32 {
        baseRT_load_state(handle, path)
    }

    // MARK: LoRA

    /// Install a LoRA adapter (`.base` bundle). Replaces any active adapter.
    /// Returns 0 on success, negative on failure.
    @discardableResult
    public func loadLoRA(path: String) -> Int32 {
        baseRT_lora_load(handle, path)
    }

    /// Detach the active LoRA adapter, if any.
    public func unloadLoRA() {
        baseRT_lora_unload(handle)
    }

    /// Active adapter id (the path it was loaded from), or "" when none.
    public var loraID: String { cString(baseRT_lora_id(handle)) }

    // MARK: helpers

    private func cString(_ p: UnsafePointer<CChar>?) -> String {
        guard let p else { return "" }
        return String(cString: p)
    }
}
