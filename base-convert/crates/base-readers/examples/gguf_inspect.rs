//! Minimal GGUF dumper used for manual validation against real models.
//!
//! Run: `cargo run -p base-readers --example gguf_inspect -- <path.gguf>`

use base_readers::gguf::{ggml_type_name, GgufFile};

fn main() -> anyhow::Result<()> {
    let path = std::env::args().nth(1).expect("usage: gguf_inspect <path.gguf>");
    let f = GgufFile::open(&path)?;
    println!("gguf v{}", f.version);
    println!("arch:        {:?}", f.arch());
    println!("n_tensors:   {}", f.tensors.len());
    println!("n_metadata:  {}", f.metadata.len());
    if std::env::args().any(|a| a == "--all-kv") {
        for (k, v) in f.metadata.iter() {
            let short = match v {
                base_readers::gguf::KvValue::Array(a) => format!("Array[{}]", a.len()),
                other => format!("{:?}", other),
            };
            let short = if short.len() > 80 { format!("{}...", &short[..80]) } else { short };
            println!("  {:60} {}", k, short);
        }
        return Ok(());
    }
    println!("--- select metadata ---");
    for key in [
        "general.architecture",
        "general.name",
        "general.quantization_version",
        "llama.embedding_length",
        "llama.block_count",
        "llama.attention.head_count",
        "llama.attention.head_count_kv",
        "llama.rope.freq_base",
        "qwen2.embedding_length",
        "qwen3.embedding_length",
        "gemma3.embedding_length",
    ] {
        if let Some(v) = f.metadata.get(key) {
            println!("  {:50} {:?}", key, v);
        }
    }
    println!("--- first 15 tensors ---");
    for t in f.tensors.iter().take(15) {
        println!(
            "  {:50} {:>6}  {:?}",
            t.name,
            ggml_type_name(t.ggml_type),
            t.shape
        );
    }
    println!("--- ggml_type histogram ---");
    let mut hist = std::collections::BTreeMap::<&'static str, usize>::new();
    for t in f.tensors.iter() {
        *hist.entry(ggml_type_name(t.ggml_type)).or_default() += 1;
    }
    for (k, v) in hist {
        println!("  {:>8}  {}", v, k);
    }
    Ok(())
}
