// End-to-end smoke test: load via the single dylib (embedded metallib) and generate.
use baseRT::Model;

fn main() {
    let model_path = std::env::args().nth(1).expect("usage: smoke <model.base>");
    // metallib = None -> engine loads the metallib embedded in libbaseRT.dylib.
    let model = Model::load(&model_path, None, 0).expect("load");
    let cfg = model.config();
    let arch = unsafe { std::ffi::CStr::from_ptr(cfg.architecture.as_ptr()) }
        .to_string_lossy();
    println!("arch={arch} layers={} vocab={} dim={} experts={}",
        cfg.n_layers, cfg.vocab_size, cfg.dim, cfg.n_experts);
    let toks = model.encode("The capital of France is").expect("encode");
    println!("encoded {} tokens", toks.len());
    let (text, stats) = model
        .generate_text(&toks, 20, Default::default())
        .expect("generate");
    println!("generated: {text:?}");
    println!("decode tok/s: {:.1}", stats.decode_tokens_per_sec);
    println!("OK");
}
