//! Safe Rust bindings for the BaseRT LLM inference engine.
//!
//! BaseRT is a C++17 inference engine optimized for Apple Silicon via Metal compute
//! shaders. This crate provides a safe, idiomatic Rust wrapper around the C API.
//!
//! # Example
//!
//! ```no_run
//! use baseRT::{Model, SamplingConfig};
//!
//! let model = Model::load("model.base", None, 0).unwrap();
//! let tokens = model.encode("Hello, world!").unwrap();
//!
//! let stats = model.generate(&tokens, 256, SamplingConfig::greedy(), |_id, text| {
//!     print!("{text}");
//!     true
//! }).unwrap();
//!
//! println!("\n({:.0} tok/s)", stats.decode_tokens_per_sec);
//! ```

use std::ffi::{CStr, CString};
use std::fmt;
use std::os::raw::{c_char, c_void};

pub use baseRT_sys::{
    BaseRTGenerationStats as GenerationStats, BaseRTModelConfig as ModelConfig,
    BaseRTTranscribeStats as TranscribeStats,
};

// ---------------------------------------------------------------------------
// Error types
// ---------------------------------------------------------------------------

/// Error type for BaseRT operations.
#[derive(Debug, Clone)]
pub enum Error {
    /// Model failed to load or a C API call returned an error.
    Api(String),
    /// A Rust string could not be converted to a C string (interior NUL byte).
    InvalidString(std::ffi::NulError),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Api(msg) => write!(f, "baseRT error: {msg}"),
            Error::InvalidString(e) => write!(f, "invalid string: {e}"),
        }
    }
}

impl std::error::Error for Error {}

impl From<std::ffi::NulError> for Error {
    fn from(e: std::ffi::NulError) -> Self {
        Error::InvalidString(e)
    }
}

pub type Result<T> = std::result::Result<T, Error>;

// ---------------------------------------------------------------------------
// Sampling config
// ---------------------------------------------------------------------------

/// Sampling parameters for text generation.
///
/// `logit_bias` is owned by the config — pass either an empty `Vec` or
/// `(token_id, bias)` pairs. The wrapper pins the backing buffers to the
/// config's lifetime so the C side never sees a dangling pointer; callers
/// don't need to manage that themselves.
#[derive(Debug, Clone)]
pub struct SamplingConfig {
    pub temperature: f32,
    pub top_k: i32,
    pub top_p: f32,
    pub min_p: f32,
    pub repeat_penalty: f32,
    pub presence_penalty: f32,
    pub frequency_penalty: f32,
    /// 0 = wall-clock-seeded (non-deterministic). Non-zero re-seeds the
    /// thread-local sampling RNG so the run is reproducible.
    pub seed: u32,
    pub logit_bias: Vec<(i32, f32)>,
}

impl SamplingConfig {
    /// Greedy decoding (temperature = 0).
    pub fn greedy() -> Self {
        Self {
            temperature: 0.0,
            top_k: 40,
            top_p: 0.9,
            min_p: 0.0,
            repeat_penalty: 1.0,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            seed: 0,
            logit_bias: Vec::new(),
        }
    }

    /// Sampling with the given temperature.
    pub fn with_temperature(temperature: f32) -> Self {
        Self {
            temperature,
            ..Self::greedy()
        }
    }

    /// Build the FFI struct. Returns alongside the unzipped logit_bias
    /// vectors — the caller must keep both alive for the duration of the
    /// generate call so the C-side pointers stay valid.
    fn to_ffi(&self) -> (baseRT_sys::BaseRTSamplingConfig, Vec<i32>, Vec<f32>) {
        let toks: Vec<i32> = self.logit_bias.iter().map(|&(t, _)| t).collect();
        let vals: Vec<f32> = self.logit_bias.iter().map(|&(_, v)| v).collect();
        let cfg = baseRT_sys::BaseRTSamplingConfig {
            temperature: self.temperature,
            top_k: self.top_k,
            top_p: self.top_p,
            min_p: self.min_p,
            repeat_penalty: self.repeat_penalty,
            presence_penalty: self.presence_penalty,
            frequency_penalty: self.frequency_penalty,
            seed: self.seed,
            n_logit_bias: toks.len() as i32,
            logit_bias_tokens: if toks.is_empty() { std::ptr::null() } else { toks.as_ptr() },
            logit_bias_values: if vals.is_empty() { std::ptr::null() } else { vals.as_ptr() },
        };
        (cfg, toks, vals)
    }
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self::greedy()
    }
}

// ---------------------------------------------------------------------------
// Helper: read last error from C API
// ---------------------------------------------------------------------------

fn last_error() -> String {
    unsafe {
        let ptr = baseRT_sys::baseRT_get_error();
        if ptr.is_null() {
            "unknown error".to_string()
        } else {
            CStr::from_ptr(ptr).to_string_lossy().into_owned()
        }
    }
}

// ---------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------

/// An BaseRT model loaded on the GPU.
///
/// The model is freed automatically when dropped. This type is `Send` but not
/// `Sync` -- the underlying C API is not thread-safe for concurrent calls on
/// the same model handle.
pub struct Model {
    handle: baseRT_sys::baseRT_model_t,
}

// The C API is safe to move across threads; concurrent access is not allowed.
unsafe impl Send for Model {}

impl Drop for Model {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe {
                baseRT_sys::baseRT_free_model(self.handle);
            }
        }
    }
}

impl Model {
    /// Load a model from a `.base` bundle.
    ///
    /// - `model_path`: path to the `.base` bundle.
    /// - `kernel_library_path`: optional path to the compiled GPU kernel library
    ///   (on Metal, `baseRT.metallib`). Pass `None` to auto-detect — including
    ///   the copy embedded in the single-file `libbaseRT` dylib.
    /// - `max_context`: maximum context window. Pass `0` for the model default
    ///   (capped at 4096).
    pub fn load(
        model_path: &str,
        kernel_library_path: Option<&str>,
        max_context: i32,
    ) -> Result<Self> {
        let c_model_path = CString::new(model_path)?;
        let c_kernel_library_path = kernel_library_path.map(CString::new).transpose()?;

        let kernel_library_ptr = c_kernel_library_path
            .as_ref()
            .map_or(std::ptr::null(), |s| s.as_ptr());

        let handle = unsafe {
            baseRT_sys::baseRT_load_model(c_model_path.as_ptr(), kernel_library_ptr, max_context)
        };

        if handle.is_null() {
            return Err(Error::Api(last_error()));
        }

        Ok(Self { handle })
    }

    /// Get model configuration.
    pub fn config(&self) -> ModelConfig {
        unsafe { baseRT_sys::baseRT_get_config(self.handle) }
    }

    /// Get total GPU memory used by the model in bytes.
    pub fn memory(&self) -> usize {
        unsafe { baseRT_sys::baseRT_model_memory(self.handle) }
    }

    /// Get the model architecture as a string (e.g. "llama", "whisper").
    pub fn architecture(&self) -> String {
        let cfg = self.config();
        let bytes: Vec<u8> = cfg
            .architecture
            .iter()
            .take_while(|&&b| b != 0)
            .map(|&b| b as u8)
            .collect();
        String::from_utf8_lossy(&bytes).into_owned()
    }

    /// Check if this is a Whisper model.
    pub fn is_whisper(&self) -> bool {
        unsafe { baseRT_sys::baseRT_is_whisper(self.handle) }
    }

    // === Tokenization ===

    /// Encode text into token IDs.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let c_text = CString::new(text)?;
        let mut tokens = vec![0u32; 8192];
        let n = unsafe {
            baseRT_sys::baseRT_encode(
                self.handle,
                c_text.as_ptr(),
                tokens.as_mut_ptr(),
                tokens.len() as i32,
            )
        };
        if n < 0 {
            return Err(Error::Api(last_error()));
        }
        tokens.truncate(n as usize);
        Ok(tokens)
    }

    /// Decode a single token ID to its text representation.
    pub fn decode_token(&self, token_id: u32) -> Option<String> {
        let ptr = unsafe { baseRT_sys::baseRT_decode_token(self.handle, token_id) };
        if ptr.is_null() {
            return None;
        }
        Some(unsafe { CStr::from_ptr(ptr) }.to_string_lossy().into_owned())
    }

    // === Generation ===

    /// Generate tokens from a prompt with a streaming callback.
    ///
    /// The callback receives `(token_id, text)` for each generated token and
    /// should return `true` to continue or `false` to stop.
    pub fn generate<F>(
        &self,
        prompt_tokens: &[u32],
        max_tokens: i32,
        sampling: SamplingConfig,
        callback: F,
    ) -> Result<GenerationStats>
    where
        F: FnMut(u32, &str) -> bool,
    {
        let mut cb = CallbackWrapper { func: callback };
        let user_data = &mut cb as *mut CallbackWrapper<F> as *mut c_void;

        // Hold `_toks` / `_vals` alive across the FFI call — `ffi` carries
        // raw pointers into them.
        let (ffi, _toks, _vals) = sampling.to_ffi();
        let stats = unsafe {
            baseRT_sys::baseRT_generate(
                self.handle,
                prompt_tokens.as_ptr(),
                prompt_tokens.len() as i32,
                max_tokens,
                ffi,
                Some(trampoline::<F>),
                user_data,
            )
        };

        Ok(stats)
    }

    /// Continue generation from current KV cache state (no reset).
    ///
    /// Used for multi-turn chat: prefill new tokens only, then decode.
    pub fn generate_continue<F>(
        &self,
        new_tokens: &[u32],
        max_tokens: i32,
        sampling: SamplingConfig,
        callback: F,
    ) -> Result<GenerationStats>
    where
        F: FnMut(u32, &str) -> bool,
    {
        let mut cb = CallbackWrapper { func: callback };
        let user_data = &mut cb as *mut CallbackWrapper<F> as *mut c_void;

        let (ffi, _toks, _vals) = sampling.to_ffi();
        let stats = unsafe {
            baseRT_sys::baseRT_generate_continue(
                self.handle,
                new_tokens.as_ptr(),
                new_tokens.len() as i32,
                max_tokens,
                ffi,
                Some(trampoline::<F>),
                user_data,
            )
        };

        Ok(stats)
    }

    /// Collect all generated tokens into a `String` (non-streaming).
    pub fn generate_text(
        &self,
        prompt_tokens: &[u32],
        max_tokens: i32,
        sampling: SamplingConfig,
    ) -> Result<(String, GenerationStats)> {
        let mut output = String::new();
        let stats = self.generate(prompt_tokens, max_tokens, sampling, |_id, text| {
            output.push_str(text);
            true
        })?;
        Ok((output, stats))
    }

    // === Low-level API ===

    /// Run prefill on tokens, populating the KV cache.
    /// Returns the first generated token (argmax of prefill logits).
    pub fn prefill(&self, tokens: &[u32]) -> u32 {
        unsafe { baseRT_sys::baseRT_prefill(self.handle, tokens.as_ptr(), tokens.len() as i32) }
    }

    /// Run one decode step. Returns the sampled token ID.
    pub fn decode_step(&self, token_id: u32, position: i32) -> u32 {
        unsafe { baseRT_sys::baseRT_decode_step(self.handle, token_id, position) }
    }

    /// Chain decode: generate multiple tokens in one GPU submission.
    pub fn chain_decode(
        &self,
        first_token: u32,
        start_position: i32,
        count: i32,
    ) -> Result<Vec<u32>> {
        let mut out = vec![0u32; count as usize];
        let n = unsafe {
            baseRT_sys::baseRT_chain_decode(
                self.handle,
                first_token,
                start_position,
                count,
                out.as_mut_ptr(),
            )
        };
        if n < 0 {
            return Err(Error::Api(last_error()));
        }
        out.truncate(n as usize);
        Ok(out)
    }

    /// Get current KV cache position (number of tokens processed).
    pub fn position(&self) -> i32 {
        unsafe { baseRT_sys::baseRT_get_position(self.handle) }
    }

    /// Enable or disable speculative decoding (n-gram prediction).
    pub fn set_speculation(&self, enabled: bool) {
        unsafe { baseRT_sys::baseRT_set_speculation(self.handle, enabled) }
    }

    /// Reset KV cache and internal state.
    pub fn reset(&self) {
        unsafe { baseRT_sys::baseRT_reset(self.handle) }
    }

    // === Whisper transcription ===

    /// Transcribe audio from a WAV file.
    ///
    /// Returns `(text, stats)`. Language can be `"en"`, `"auto"`, etc.
    /// Pass `None` for English.
    pub fn transcribe(
        &self,
        wav_path: &str,
        language: Option<&str>,
    ) -> Result<(String, TranscribeStats)> {
        let c_path = CString::new(wav_path)?;
        let c_lang = language.map(CString::new).transpose()?;
        let lang_ptr = c_lang.as_ref().map_or(std::ptr::null(), |s| s.as_ptr());

        let mut stats = baseRT_sys::BaseRTTranscribeStats::default();
        let ptr = unsafe {
            baseRT_sys::baseRT_transcribe(self.handle, c_path.as_ptr(), lang_ptr, &mut stats)
        };

        if ptr.is_null() {
            return Err(Error::Api(last_error()));
        }

        let text = unsafe { CStr::from_ptr(ptr) }.to_string_lossy().into_owned();
        Ok((text, stats))
    }

    /// Transcribe from raw f32 PCM samples (16 kHz, mono).
    pub fn transcribe_pcm(
        &self,
        samples: &[f32],
        language: Option<&str>,
    ) -> Result<(String, TranscribeStats)> {
        let c_lang = language.map(CString::new).transpose()?;
        let lang_ptr = c_lang.as_ref().map_or(std::ptr::null(), |s| s.as_ptr());

        let mut stats = baseRT_sys::BaseRTTranscribeStats::default();
        let ptr = unsafe {
            baseRT_sys::baseRT_transcribe_pcm(
                self.handle,
                samples.as_ptr(),
                samples.len() as i32,
                lang_ptr,
                &mut stats,
            )
        };

        if ptr.is_null() {
            return Err(Error::Api(last_error()));
        }

        let text = unsafe { CStr::from_ptr(ptr) }.to_string_lossy().into_owned();
        Ok((text, stats))
    }

    /// Transcribe a WAV file with per-segment streaming callback.
    ///
    /// The callback receives `(start_ms, end_ms, text)` for each segment and
    /// should return `true` to continue or `false` to stop.
    pub fn transcribe_stream<F>(
        &self,
        wav_path: &str,
        language: Option<&str>,
        mut callback: F,
    ) -> Result<(String, TranscribeStats)>
    where
        F: FnMut(i32, i32, &str) -> bool,
    {
        let c_path = CString::new(wav_path)?;
        let c_lang = language.map(CString::new).transpose()?;
        let lang_ptr = c_lang.as_ref().map_or(std::ptr::null(), |s| s.as_ptr());

        let mut stats = baseRT_sys::BaseRTTranscribeStats::default();
        let mut wrapper = SegmentCallbackWrapper { func: &mut callback };
        let user_data = &mut wrapper as *mut SegmentCallbackWrapper<F> as *mut c_void;

        let ptr = unsafe {
            baseRT_sys::baseRT_transcribe_stream(
                self.handle,
                c_path.as_ptr(),
                lang_ptr,
                &mut stats,
                Some(segment_trampoline::<F>),
                user_data,
            )
        };

        if ptr.is_null() {
            return Err(Error::Api(last_error()));
        }

        let text = unsafe { CStr::from_ptr(ptr) }.to_string_lossy().into_owned();
        Ok((text, stats))
    }

    /// Transcribe raw f32 PCM samples with per-segment streaming callback.
    pub fn transcribe_pcm_stream<F>(
        &self,
        samples: &[f32],
        language: Option<&str>,
        mut callback: F,
    ) -> Result<(String, TranscribeStats)>
    where
        F: FnMut(i32, i32, &str) -> bool,
    {
        let c_lang = language.map(CString::new).transpose()?;
        let lang_ptr = c_lang.as_ref().map_or(std::ptr::null(), |s| s.as_ptr());

        let mut stats = baseRT_sys::BaseRTTranscribeStats::default();
        let mut wrapper = SegmentCallbackWrapper { func: &mut callback };
        let user_data = &mut wrapper as *mut SegmentCallbackWrapper<F> as *mut c_void;

        let ptr = unsafe {
            baseRT_sys::baseRT_transcribe_pcm_stream(
                self.handle,
                samples.as_ptr(),
                samples.len() as i32,
                lang_ptr,
                &mut stats,
                Some(segment_trampoline::<F>),
                user_data,
            )
        };

        if ptr.is_null() {
            return Err(Error::Api(last_error()));
        }

        let text = unsafe { CStr::from_ptr(ptr) }.to_string_lossy().into_owned();
        Ok((text, stats))
    }

    /// Enable or disable timestamp generation for Whisper transcription.
    pub fn set_timestamps(&self, enabled: bool) {
        unsafe { baseRT_sys::baseRT_set_timestamps(self.handle, enabled) }
    }

    // === Embeddings ===

    /// Compute embeddings from token IDs.
    ///
    /// Returns a vector of floats representing the embedding.
    pub fn embed(&self, tokens: &[u32]) -> Result<Vec<f32>> {
        let dim = self.embedding_dim();
        if dim <= 0 {
            return Err(Error::Api(last_error()));
        }
        let mut out = vec![0.0f32; dim as usize];
        let n = unsafe {
            baseRT_sys::baseRT_embed(
                self.handle,
                tokens.as_ptr(),
                tokens.len() as i32,
                out.as_mut_ptr(),
                dim,
            )
        };
        if n <= 0 {
            return Err(Error::Api(last_error()));
        }
        out.truncate(n as usize);
        Ok(out)
    }

    /// Compute embeddings from text directly (tokenizes internally).
    pub fn embed_text(&self, text: &str) -> Result<Vec<f32>> {
        let c_text = CString::new(text)?;
        let dim = self.embedding_dim();
        if dim <= 0 {
            return Err(Error::Api(last_error()));
        }
        let mut out = vec![0.0f32; dim as usize];
        let n = unsafe {
            baseRT_sys::baseRT_embed_text(self.handle, c_text.as_ptr(), out.as_mut_ptr(), dim)
        };
        if n <= 0 {
            return Err(Error::Api(last_error()));
        }
        out.truncate(n as usize);
        Ok(out)
    }

    /// Get the embedding dimension for this model.
    pub fn embedding_dim(&self) -> i32 {
        unsafe { baseRT_sys::baseRT_embedding_dim(self.handle) }
    }

    // === Chat templates ===

    /// Format a chat prompt using the model's native template.
    pub fn format_chat(&self, system: &str, user: &str) -> Result<String> {
        let c_system = CString::new(system)?;
        let c_user = CString::new(user)?;
        let ptr = unsafe {
            baseRT_sys::baseRT_format_chat(self.handle, c_system.as_ptr(), c_user.as_ptr())
        };
        if ptr.is_null() {
            return Err(Error::Api(last_error()));
        }
        Ok(unsafe { CStr::from_ptr(ptr) }.to_string_lossy().into_owned())
    }

    /// Get the chat template name for the loaded model (e.g. "chatml", "llama3").
    pub fn chat_template(&self) -> Option<String> {
        let ptr = unsafe { baseRT_sys::baseRT_chat_template(self.handle) };
        if ptr.is_null() {
            return None;
        }
        Some(unsafe { CStr::from_ptr(ptr) }.to_string_lossy().into_owned())
    }

    // === Token counting ===

    /// Count tokens in text without allocating an output buffer.
    pub fn token_count(&self, text: &str) -> Result<i32> {
        let c_text = CString::new(text)?;
        Ok(unsafe { baseRT_sys::baseRT_token_count(self.handle, c_text.as_ptr()) })
    }

    // === Model inspection ===

    /// Get the number of tensors in the model.
    pub fn tensor_count(&self) -> i32 {
        unsafe { baseRT_sys::baseRT_tensor_count(self.handle) }
    }

    /// Get tensor name by index.
    pub fn tensor_name(&self, index: i32) -> Option<String> {
        let ptr = unsafe { baseRT_sys::baseRT_tensor_name(self.handle, index) };
        if ptr.is_null() {
            return None;
        }
        Some(unsafe { CStr::from_ptr(ptr) }.to_string_lossy().into_owned())
    }

    /// Get tensor dtype code by index.
    pub fn tensor_dtype(&self, index: i32) -> u32 {
        unsafe { baseRT_sys::baseRT_tensor_dtype(self.handle, index) }
    }

    /// Get raw tensor dtype string by index (e.g. "F16", "BF16").
    /// Returns `None` if the tensor has no recorded raw dtype string.
    pub fn tensor_raw_dtype(&self, index: i32) -> Option<String> {
        let ptr = unsafe { baseRT_sys::baseRT_tensor_raw_dtype(self.handle, index) };
        if ptr.is_null() {
            return None;
        }
        let s = unsafe { CStr::from_ptr(ptr) }.to_string_lossy().into_owned();
        if s.is_empty() {
            None
        } else {
            Some(s)
        }
    }

    /// Iterate over all tensors as `(name, dtype_code)` pairs.
    pub fn tensors(&self) -> TensorIter<'_> {
        TensorIter {
            model: self,
            index: 0,
            count: self.tensor_count(),
        }
    }
}

// ---------------------------------------------------------------------------
// Tensor iterator
// ---------------------------------------------------------------------------

/// Iterator over model tensors.
pub struct TensorIter<'a> {
    model: &'a Model,
    index: i32,
    count: i32,
}

impl<'a> Iterator for TensorIter<'a> {
    type Item = (String, u32);

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.count {
            return None;
        }
        let name = self.model.tensor_name(self.index).unwrap_or_default();
        let dtype = self.model.tensor_dtype(self.index);
        self.index += 1;
        Some((name, dtype))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = (self.count - self.index).max(0) as usize;
        (remaining, Some(remaining))
    }
}

impl<'a> ExactSizeIterator for TensorIter<'a> {}

// ---------------------------------------------------------------------------
// Callback trampoline
// ---------------------------------------------------------------------------

struct CallbackWrapper<F> {
    func: F,
}

struct SegmentCallbackWrapper<'a, F> {
    func: &'a mut F,
}

unsafe extern "C" fn trampoline<F>(
    token_id: u32,
    text: *const c_char,
    user_data: *mut c_void,
) -> bool
where
    F: FnMut(u32, &str) -> bool,
{
    let wrapper = &mut *(user_data as *mut CallbackWrapper<F>);
    let text_str = if text.is_null() {
        ""
    } else {
        CStr::from_ptr(text).to_str().unwrap_or("")
    };
    (wrapper.func)(token_id, text_str)
}

unsafe extern "C" fn segment_trampoline<F>(
    start_ms: i32,
    end_ms: i32,
    text: *const c_char,
    user_data: *mut c_void,
) -> bool
where
    F: FnMut(i32, i32, &str) -> bool,
{
    let wrapper = &mut *(user_data as *mut SegmentCallbackWrapper<F>);
    let text_str = if text.is_null() {
        ""
    } else {
        CStr::from_ptr(text).to_str().unwrap_or("")
    };
    (wrapper.func)(start_ms, end_ms, text_str)
}

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // SamplingConfig builder / defaults
    // -----------------------------------------------------------------------

    #[test]
    fn sampling_config_greedy_defaults() {
        let cfg = SamplingConfig::greedy();
        assert_eq!(cfg.temperature, 0.0);
        assert_eq!(cfg.top_k, 40);
        assert!((cfg.top_p - 0.9).abs() < f32::EPSILON);
        assert_eq!(cfg.min_p, 0.0);
        assert_eq!(cfg.repeat_penalty, 1.0);
    }

    #[test]
    fn sampling_config_default_is_greedy() {
        let def = SamplingConfig::default();
        let greedy = SamplingConfig::greedy();
        assert_eq!(def.temperature, greedy.temperature);
        assert_eq!(def.top_k, greedy.top_k);
        assert_eq!(def.top_p, greedy.top_p);
        assert_eq!(def.min_p, greedy.min_p);
        assert_eq!(def.repeat_penalty, greedy.repeat_penalty);
    }

    #[test]
    fn sampling_config_with_temperature() {
        let cfg = SamplingConfig::with_temperature(0.7);
        assert_eq!(cfg.temperature, 0.7);
        // Other fields should inherit from greedy()
        assert_eq!(cfg.top_k, 40);
        assert!((cfg.top_p - 0.9).abs() < f32::EPSILON);
        assert_eq!(cfg.min_p, 0.0);
        assert_eq!(cfg.repeat_penalty, 1.0);
    }

    #[test]
    fn sampling_config_to_ffi_roundtrip() {
        let cfg = SamplingConfig {
            temperature: 1.2,
            top_k: 50,
            top_p: 0.95,
            min_p: 0.05,
            repeat_penalty: 1.1,
            presence_penalty: 0.4,
            frequency_penalty: -0.1,
            seed: 7,
            logit_bias: vec![(42, 5.0), (100, -3.5)],
        };
        let (ffi, toks, vals) = cfg.to_ffi();
        assert_eq!(ffi.temperature, 1.2);
        assert_eq!(ffi.top_k, 50);
        assert!((ffi.top_p - 0.95).abs() < f32::EPSILON);
        assert!((ffi.min_p - 0.05).abs() < f32::EPSILON);
        assert!((ffi.repeat_penalty - 1.1).abs() < f32::EPSILON);
        assert!((ffi.presence_penalty - 0.4).abs() < f32::EPSILON);
        assert!((ffi.frequency_penalty - (-0.1)).abs() < f32::EPSILON);
        assert_eq!(ffi.seed, 7);
        assert_eq!(ffi.n_logit_bias, 2);
        assert_eq!(toks, vec![42, 100]);
        assert_eq!(vals, vec![5.0, -3.5]);
        // Pointers should resolve into the returned backing vectors so the
        // caller-of-record can keep them alive across the FFI call.
        assert_eq!(ffi.logit_bias_tokens, toks.as_ptr());
        assert_eq!(ffi.logit_bias_values, vals.as_ptr());
    }

    // -----------------------------------------------------------------------
    // Error type formatting
    // -----------------------------------------------------------------------

    #[test]
    fn error_api_display() {
        let err = Error::Api("model not found".to_string());
        assert_eq!(format!("{err}"), "baseRT error: model not found");
    }

    #[test]
    fn error_api_debug() {
        let err = Error::Api("oops".to_string());
        let debug = format!("{err:?}");
        assert!(debug.contains("Api"));
        assert!(debug.contains("oops"));
    }

    #[test]
    fn error_invalid_string_display() {
        let nul_err = CString::new("hello\0world").unwrap_err();
        let err = Error::InvalidString(nul_err);
        let msg = format!("{err}");
        assert!(msg.starts_with("invalid string:"));
    }

    #[test]
    fn error_from_nul_error() {
        let nul_err = CString::new("a\0b").unwrap_err();
        let err: Error = nul_err.into();
        assert!(matches!(err, Error::InvalidString(_)));
    }

    #[test]
    fn error_implements_std_error() {
        let err = Error::Api("test".to_string());
        let _: &dyn std::error::Error = &err;
    }

    // -----------------------------------------------------------------------
    // Model::load with nonexistent path
    // -----------------------------------------------------------------------

    #[test]
    fn model_load_nonexistent_path_returns_err() {
        let result = Model::load("/nonexistent/path/to/model.base", None, 0);
        assert!(result.is_err());
        if let Err(Error::Api(msg)) = &result {
            // The error message should contain something meaningful
            assert!(!msg.is_empty(), "error message should not be empty");
        }
        // It could also be an Api error — either way, it must be Err
    }

    #[test]
    fn model_load_with_interior_nul_returns_err() {
        let result = Model::load("path\0with\0nuls", None, 0);
        assert!(result.is_err());
        assert!(matches!(result, Err(Error::InvalidString(_))));
    }

    #[test]
    fn model_load_metallib_with_interior_nul_returns_err() {
        let result = Model::load("model.base", Some("bad\0path"), 0);
        assert!(result.is_err());
        assert!(matches!(result, Err(Error::InvalidString(_))));
    }

    // -----------------------------------------------------------------------
    // SamplingConfig is Clone + Debug
    //
    // Note: `Copy` was dropped when `logit_bias: Vec<(i32, f32)>` was added.
    // Callers that previously relied on implicit copies now clone explicitly.
    // -----------------------------------------------------------------------

    #[test]
    fn sampling_config_clone() {
        let a = SamplingConfig::with_temperature(0.5);
        let b = a.clone();
        assert_eq!(a.temperature, b.temperature);
        assert_eq!(a.top_k, b.top_k);
    }

    #[test]
    fn sampling_config_debug() {
        let cfg = SamplingConfig::greedy();
        let debug = format!("{cfg:?}");
        assert!(debug.contains("SamplingConfig"));
        assert!(debug.contains("temperature"));
    }
}
