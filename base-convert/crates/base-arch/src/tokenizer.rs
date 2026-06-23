//! Tokenizer extraction from GGUF metadata into the `.base`
//! `TokenizerBlob` field map.
//!
//! GGUF stores the tokenizer as a set of `tokenizer.ggml.*` kv entries:
//! model type, vocab list, BPE merges, special token IDs, chat template.
//! This module copies the well-known keys into the blob verbatim so a
//! runtime can reconstruct a usable tokenizer without falling back to
//! the source GGUF.
//!
//! We do NOT currently emit a full HF-format `tokenizer.json` — that's a
//! follow-up. The field map carries `tokenizer_type: "gguf"` so the
//! loader knows to interpret the fields as GGUF-style.

use base_readers::gguf::KvValue;
use serde_json::{json, Value};
use std::collections::BTreeMap;

/// Copy `tokenizer.ggml.*` keys + `tokenizer.chat_template` out of a
/// GGUF metadata map into a JSON field map suitable for TokenizerBlob.
pub fn extract_from_gguf(meta: &BTreeMap<String, KvValue>) -> BTreeMap<String, Value> {
    let mut out = BTreeMap::new();
    out.insert("tokenizer_type".into(), json!("gguf"));

    for (k, v) in meta.iter() {
        if !k.starts_with("tokenizer.") {
            continue;
        }
        if let Some(json_v) = kv_to_json(v) {
            out.insert(k.clone(), json_v);
        }
    }
    out
}

fn kv_to_json(v: &KvValue) -> Option<Value> {
    match v {
        KvValue::U8(n) => Some(json!(*n)),
        KvValue::I8(n) => Some(json!(*n)),
        KvValue::U16(n) => Some(json!(*n)),
        KvValue::I16(n) => Some(json!(*n)),
        KvValue::U32(n) => Some(json!(*n)),
        KvValue::I32(n) => Some(json!(*n)),
        KvValue::U64(n) => Some(json!(*n)),
        KvValue::I64(n) => Some(json!(*n)),
        KvValue::F32(f) => Some(json!(*f)),
        KvValue::F64(f) => Some(json!(*f)),
        KvValue::Bool(b) => Some(json!(*b)),
        KvValue::String(s) => Some(json!(s)),
        KvValue::Array(arr) => {
            // Arrays of strings or arrays of scalars are both common.
            let items: Option<Vec<Value>> = arr.iter().map(kv_to_json).collect();
            items.map(Value::Array)
        }
    }
}
