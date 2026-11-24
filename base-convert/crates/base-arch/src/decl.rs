//! What the architecture's mapper declares about the bundle's tensors
//! (PIPELINE_V3 §4.8): each canonical tensor's CLASS — how the engine
//! reads it — and the AWQ FOLD GROUPS. `basert inventory` reports these
//! and `build_awq_fold_plan` applies them; nothing downstream derives a
//! class, a role or a fold from a tensor name.
//!
//! A fold group is one tensor that absorbs `1/s` — a norm gain (`gamma`)
//! or an upstream linear's rows (`rows`) — plus the complete list of
//! consumers whose input columns take `s`. Every consumer must be
//! `linear` or `expert_stack` (§14.7 c): a scaled `other` consumer would
//! round the scaled columns at the stored precision, so an activation
//! that also feeds an `other` tensor declares no fold at all. "No fold"
//! is a valid declaration; a group that exists structurally but cannot
//! fold (Gemma 4's per-head V norm) is declared with `foldable: false`
//! and the reason, which the fold plan reports as its RTN fallback.

use crate::ArchConfig;
use std::collections::BTreeMap;

/// How the engine reads a tensor (§4.8).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum TensorClass {
    /// Read through `dispatch_gemm` (dense quant cells).
    Linear,
    /// The fused MoE expert GEMMs (`emit_moe_ffn`; moe cells).
    ExpertStack,
    /// The embedding lookup (embedding cells).
    Embedding,
    /// Model-specific kernels: norms, biases, conv weights, routers,
    /// small gates, recurrence parameters. No quant cells.
    Other,
}

impl TensorClass {
    pub fn as_str(self) -> &'static str {
        match self {
            TensorClass::Linear => "linear",
            TensorClass::ExpertStack => "expert_stack",
            TensorClass::Embedding => "embedding",
            TensorClass::Other => "other",
        }
    }
    /// A class quant prices (has cells); `other` has none.
    pub fn quantizable(self) -> bool {
        self != TensorClass::Other
    }
    /// May take `s` on its input columns as a fold consumer.
    pub fn may_consume_fold(self) -> bool {
        matches!(self, TensorClass::Linear | TensorClass::ExpertStack)
    }
    /// The shared helper that reads the class and logs it under `shared`
    /// (§4.8, §6.3); `other` has none.
    pub fn helper(self) -> Option<&'static str> {
        match self {
            TensorClass::Linear => Some("dispatch_gemm"),
            TensorClass::ExpertStack => Some("emit_moe_ffn"),
            TensorClass::Embedding => Some("embedding_lookup"),
            TensorClass::Other => None,
        }
    }
}

/// An extension request (§4.8): an `other` tensor the mapper would declare
/// quantizable if the shared helper could serve it — a dtype, a shape, a
/// size limit or a routing mode the helper lacks. The engine reads it with
/// a model kernel meanwhile; `basert inventory` lists the request, the add
/// record carries it, the `other` bound does not price it, and finish
/// writes the `TODO_dispatch` hand-off for whoever extends the helper. It
/// is not a way to keep a GEMM private: the request names what is missing,
/// and the hand-off names the bytes it leaves on the table.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DispatchExtension {
    /// The class the tensor takes once the helper serves it (never `other`).
    pub class: TensorClass,
    /// What the helper lacks, and the evidence: shapes, dtypes, the mode.
    pub needs: String,
}

/// Where `1/s` goes.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum FoldKind {
    /// Into a norm gain (`upstream` is 1-D).
    Gamma,
    /// Into the rows of the upstream tensor whose outputs the consumer reads.
    Rows,
}

impl FoldKind {
    pub fn as_str(self) -> &'static str {
        match self {
            FoldKind::Gamma => "gamma",
            FoldKind::Rows => "rows",
        }
    }
}

/// Which rows of a `rows` fold's upstream absorb `1/s`.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum UpstreamRows {
    /// Every row (a plain `up_proj`, `v_proj`, or a split `up` expert stack).
    All,
    /// A fused `gate_up` expert stack `[E, 2F, K]`: the up rows are the
    /// second half `[F, 2F)` of each expert's slab (gate rows first).
    SecondHalf,
}

impl UpstreamRows {
    pub fn as_str(self) -> &'static str {
        match self {
            UpstreamRows::All => "all",
            UpstreamRows::SecondHalf => "second_half",
        }
    }
}

/// One declared fold group.
#[derive(Debug, Clone, PartialEq)]
pub struct FoldGroup {
    /// The group's kind — one of the names autoquant's table rows derive
    /// from: `attn`, `mlp`, `down`, `o`, `head`, `moe`, `moe_down`, and
    /// `gdn_out` (the Gated-DeltaNet output fold). The fold plan keys on
    /// `upstream` / `consumers`; only `fold_tile`'s GQA case keys on `o`.
    pub kind: &'static str,
    /// The block, or `None` for the head.
    pub layer: Option<usize>,
    /// The tensor that absorbs `1/s` (canonical name).
    pub upstream: String,
    pub fold: FoldKind,
    /// Every tensor whose input columns take `s` (canonical names, sorted).
    pub consumers: Vec<String>,
    pub foldable: bool,
    /// Why not, when `!foldable` (the fold plan's RTN-fallback reason).
    pub refusal: Option<String>,
    /// `gamma` folds: the model uses the stored gain `g` as `(1 + g)` (the
    /// converter's `norm_shift`), so the folded gain is `(1 + g)/s − 1`
    /// on the source side; `false` when it is used as `g` (stores `g/s`).
    pub gamma_shift: bool,
    /// `rows` folds: which rows of `upstream` absorb `1/s`.
    pub upstream_rows: UpstreamRows,
}

/// How a fold's `s` — one entry per upstream channel — lands on the
/// consumers' input columns. The mapper declares the group; the tiling
/// follows from the shapes (and, for `o`, the attention config) and is
/// derived here once, for the fold plan and for `basert inventory` alike,
/// so the fold probe reads it from the inventory instead of inferring it
/// from the kind (PIPELINE_V3 §4.8: nothing downstream derives a fold).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FoldTile {
    /// One upstream channel per consumer column.
    Identity,
    /// A GQA `o_proj`: input column (h, d) takes v row (h / (heads / kv), d).
    Gqa {
        heads: usize,
        kv: usize,
        head_dim: usize,
    },
    /// A `period`-wide gain in front of a wider input (the Gated-DeltaNet
    /// output norm: one `head_dim` gain for every head): each repeat of the
    /// period takes the whole `s`.
    Period(usize),
}

impl FoldTile {
    /// The inventory's `tile`: `null` | `{"gqa": [heads, kv, head_dim]}` | `{"period": n}`.
    pub fn to_json(self) -> serde_json::Value {
        match self {
            FoldTile::Identity => serde_json::Value::Null,
            FoldTile::Gqa {
                heads,
                kv,
                head_dim,
            } => serde_json::json!({"gqa": [heads, kv, head_dim]}),
            FoldTile::Period(p) => serde_json::json!({"period": p}),
        }
    }
}

/// A declared group's tiling and its consumers' `in_features` (the length
/// of `s`), from the shapes; `Err(why)` when the shapes do not fold as
/// declared — the fold plan's refusal and the inventory's `refusal` alike.
pub fn fold_tile(
    g: &FoldGroup,
    tensors: &TensorSet,
    config: &crate::ArchConfig,
) -> Result<(FoldTile, usize), String> {
    let up = tensors
        .shape(&g.upstream)
        .ok_or_else(|| format!("{} missing; cannot fold 1/s into it", g.upstream))?;
    if g.consumers.is_empty() {
        return Err(format!(
            "{} fold of {} declares no consumer",
            g.kind, g.upstream
        ));
    }
    // every consumer's in_features (its last dim), all equal
    let mut width: Option<usize> = None;
    for c in &g.consumers {
        let s = tensors.shape(c).ok_or_else(|| format!("{c} missing"))?;
        if s.len() != 2 && s.len() != 3 {
            return Err(format!("{c} {s:?} is not a 2-D/3-D consumer"));
        }
        let w = *s.last().unwrap() as usize;
        match width {
            None => width = Some(w),
            Some(x) if x != w => {
                return Err(format!(
                    "consumers of {} disagree on in_features ({x} vs {w} at {c})",
                    g.upstream
                ))
            }
            _ => {}
        }
    }
    let width = width.unwrap();
    match g.fold {
        FoldKind::Gamma => {
            if up.len() != 1 {
                return Err(format!("{} is not 1-D", g.upstream));
            }
            let gain = up[0] as usize;
            if gain == width {
                Ok((FoldTile::Identity, width))
            } else if gain > 0 && width % gain == 0 {
                Ok((FoldTile::Period(gain), width))
            } else {
                Err(format!(
                    "{} [{gain}] does not feed its consumers [.., {width}]",
                    g.upstream
                ))
            }
        }
        FoldKind::Rows => {
            if up.len() != 2 && up.len() != 3 {
                return Err(format!("{} {up:?} is not a 2-D/3-D upstream", g.upstream));
            }
            let n = up[up.len() - 2] as usize;
            let rows = match g.upstream_rows {
                UpstreamRows::All => n,
                UpstreamRows::SecondHalf => n / 2,
            };
            if rows == width {
                return Ok((FoldTile::Identity, width));
            }
            let consumer = &g.consumers[0];
            if g.kind != "o" {
                return Err(format!(
                    "{} {up:?} does not feed {consumer} [.., {width}]",
                    g.upstream
                ));
            }
            let heads = config.num_attention_heads as usize;
            let kv = config.num_kv_heads.max(1) as usize;
            let hd = if config.head_dim > 0 {
                config.head_dim as usize
            } else {
                width / heads.max(1)
            };
            if heads == 0 || width != heads * hd || rows != kv * hd || heads % kv != 0 {
                return Err(format!(
                    "{consumer} [.., {width}] / {} {up:?} do not match heads={heads} kv={kv} head_dim={hd}",
                    g.upstream
                ));
            }
            Ok((
                FoldTile::Gqa {
                    heads,
                    kv,
                    head_dim: hd,
                },
                width,
            ))
        }
    }
}

/// The canonical tensors a declaration ranges over: name → shape.
pub struct TensorSet {
    map: BTreeMap<String, Vec<u64>>,
}

impl TensorSet {
    pub fn new(tensors: &[(String, Vec<u64>)]) -> Self {
        TensorSet {
            map: tensors.iter().cloned().collect(),
        }
    }
    pub fn shape(&self, name: &str) -> Option<&[u64]> {
        self.map.get(name).map(|v| v.as_slice())
    }
    pub fn has(&self, name: &str) -> bool {
        self.map.contains_key(name)
    }
    /// Sorted, distinct block indices of every `layers.N.*` tensor.
    pub fn layers(&self) -> Vec<usize> {
        let mut v: Vec<usize> = self
            .map
            .keys()
            .filter_map(|n| layer_split(n).map(|(l, _)| l))
            .collect();
        v.sort_unstable();
        v.dedup();
        v
    }
    pub fn iter(&self) -> impl Iterator<Item = (&str, &[u64])> {
        self.map.iter().map(|(n, s)| (n.as_str(), s.as_slice()))
    }
}

/// Split `layers.N.rest` → (N, rest).
pub fn layer_split(canonical: &str) -> Option<(usize, &str)> {
    let rest = canonical.strip_prefix("layers.")?;
    let (idx, tail) = rest.split_once('.')?;
    Some((idx.parse().ok()?, tail))
}

/// The part of a canonical name after `layers.N.` (or the whole name).
pub fn tail(canonical: &str) -> &str {
    layer_split(canonical).map(|(_, t)| t).unwrap_or(canonical)
}

/// The default class of a tensor in a multimodal tower (the `mmproj`
/// section — a ViT / audio encoder and its projector), which the LM's
/// walks never fetch (the add record lists them `unread`): a 2-D weight
/// matrix that is neither a norm nor an embedding table is a GEMM, every
/// other tensor (norm gains, biases, conv kernels, position tables) is
/// `other`. A mapper whose tower reads differently overrides
/// `HfMapper::mmproj_class`.
pub fn tower_class_default(canonical: &str, shape: &[u64]) -> TensorClass {
    let leaf = canonical.rsplit('.').nth(1).unwrap_or("");
    if shape.len() == 2
        && canonical.ends_with(".weight")
        && !leaf.contains("norm")
        && !leaf.contains("embed")
        && !leaf.starts_with("ln")
    {
        TensorClass::Linear
    } else {
        TensorClass::Other
    }
}

/// Builds a mapper's fold-group list. Consumers that the checkpoint does
/// not carry are dropped (an architecture family declares the superset
/// of its variants' tensors); a group with no consumer present is not
/// declared; a group whose upstream is missing is declared unfoldable
/// with the reason, as the fold plan would refuse it.
pub struct Declare<'a> {
    tensors: &'a TensorSet,
    norm_shift: &'a dyn Fn(&str) -> f32,
    groups: Vec<FoldGroup>,
}

impl<'a> Declare<'a> {
    pub fn new(tensors: &'a TensorSet, norm_shift: &'a dyn Fn(&str) -> f32) -> Self {
        Declare {
            tensors,
            norm_shift,
            groups: Vec::new(),
        }
    }

    pub fn tensors(&self) -> &TensorSet {
        self.tensors
    }

    /// A `gamma` group: `upstream` (a 1-D norm gain) absorbs `1/s`, every
    /// present `consumer` takes `s` on its columns.
    pub fn gamma(
        &mut self,
        kind: &'static str,
        layer: Option<usize>,
        upstream: String,
        consumers: &[String],
    ) {
        let mut present: Vec<String> = consumers
            .iter()
            .filter(|c| self.tensors.has(c))
            .cloned()
            .collect();
        if present.is_empty() {
            return;
        }
        present.sort();
        present.dedup();
        let (foldable, refusal) = match self.tensors.shape(&upstream) {
            None => (
                false,
                Some(format!("{upstream} missing; cannot fold 1/s into the norm")),
            ),
            Some(s) if s.len() != 1 => (false, Some(format!("{upstream} is not 1-D"))),
            Some(_) => (true, None),
        };
        let gamma_shift = (self.norm_shift)(&upstream) != 0.0;
        self.groups.push(FoldGroup {
            kind,
            layer,
            upstream,
            fold: FoldKind::Gamma,
            consumers: present,
            foldable,
            refusal,
            gamma_shift,
            upstream_rows: UpstreamRows::All,
        });
    }

    /// A `rows` group: the rows of `upstream` absorb `1/s`, `consumer`
    /// takes `s` on its columns.
    pub fn rows(
        &mut self,
        kind: &'static str,
        layer: Option<usize>,
        upstream: String,
        consumer: String,
        upstream_rows: UpstreamRows,
    ) {
        if !self.tensors.has(&consumer) {
            return;
        }
        let (foldable, refusal) = if self.tensors.has(&upstream) {
            (true, None)
        } else {
            (
                false,
                Some(format!("{upstream} missing; cannot fold 1/s into its rows")),
            )
        };
        self.groups.push(FoldGroup {
            kind,
            layer,
            upstream,
            fold: FoldKind::Rows,
            consumers: vec![consumer],
            foldable,
            refusal,
            gamma_shift: false,
            upstream_rows,
        });
    }

    /// Mark the most recently declared group unfoldable, with the reason.
    pub fn refuse_last(&mut self, why: String) {
        if let Some(g) = self.groups.last_mut() {
            g.foldable = false;
            g.refusal = Some(why);
        }
    }

    /// The groups, ordered by block then kind (the head last).
    pub fn finish(mut self) -> Vec<FoldGroup> {
        self.groups.sort_by(|a, b| {
            (a.layer.unwrap_or(usize::MAX), a.kind).cmp(&(b.layer.unwrap_or(usize::MAX), b.kind))
        });
        self.groups
    }
}

/// The Llama-shaped dense block: `input_norm → q/k/v`, `o ← v rows`,
/// `post_attn_norm → gate/up`, `down ← up rows`, and `final_norm →
/// lm_head` when the head is its own tensor. Shared by every mapper
/// whose blocks are built this way; each declares which of its tensors
/// are the linears.
pub mod dense {
    use super::*;

    pub const ATTN_LINEARS: [&str; 4] = [
        "self_attn.q_proj.weight",
        "self_attn.k_proj.weight",
        "self_attn.v_proj.weight",
        "self_attn.o_proj.weight",
    ];
    pub const MLP_LINEARS: [&str; 3] = [
        "mlp.gate_proj.weight",
        "mlp.up_proj.weight",
        "mlp.down_proj.weight",
    ];

    /// `embed_tokens` is the lookup, `lm_head` and the attention / MLP
    /// projections are GEMMs, everything else (norms, biases, …) is `other`.
    pub fn class(canonical: &str, _shape: &[u64]) -> TensorClass {
        match canonical {
            "embed_tokens.weight" => return TensorClass::Embedding,
            "lm_head.weight" | "output.weight" => return TensorClass::Linear,
            _ => {}
        }
        let t = tail(canonical);
        if ATTN_LINEARS.contains(&t) || MLP_LINEARS.contains(&t) {
            TensorClass::Linear
        } else {
            TensorClass::Other
        }
    }

    /// `attn` (input_norm → q, k, v) and `o` (v rows → o) of one block.
    pub fn attn_groups(d: &mut Declare, layer: usize) {
        let p = format!("layers.{layer}.");
        d.gamma(
            "attn",
            Some(layer),
            format!("{p}input_norm.weight"),
            &[
                format!("{p}self_attn.q_proj.weight"),
                format!("{p}self_attn.k_proj.weight"),
                format!("{p}self_attn.v_proj.weight"),
            ],
        );
        d.rows(
            "o",
            Some(layer),
            format!("{p}self_attn.v_proj.weight"),
            format!("{p}self_attn.o_proj.weight"),
            UpstreamRows::All,
        );
    }

    /// `mlp` (post_attn_norm → gate, up) and `down` (up rows → down) of
    /// one block's SwiGLU FFN under `prefix` (`mlp` or `mlp.shared_expert`).
    /// `down` is exact only for a gated FFN: down reads `act(gate·x) ⊙
    /// (up·x)`, linear in up.
    pub fn mlp_groups(d: &mut Declare, layer: usize) {
        let p = format!("layers.{layer}.");
        d.gamma(
            "mlp",
            Some(layer),
            format!("{p}post_attn_norm.weight"),
            &[
                format!("{p}mlp.gate_proj.weight"),
                format!("{p}mlp.up_proj.weight"),
            ],
        );
        d.rows(
            "down",
            Some(layer),
            format!("{p}mlp.up_proj.weight"),
            format!("{p}mlp.down_proj.weight"),
            UpstreamRows::All,
        );
    }

    /// The gated shared expert's `down ← up rows`: the same `down` fold
    /// as a dense FFN's, on the shared expert's tensors (a block has one
    /// or the other, never both).
    pub fn shared_expert_down(d: &mut Declare, layer: usize) {
        let p = format!("layers.{layer}.mlp.shared_expert.");
        d.rows(
            "down",
            Some(layer),
            format!("{p}up_proj.weight"),
            format!("{p}down_proj.weight"),
            UpstreamRows::All,
        );
    }

    /// `final_norm → lm_head`, only when the head is its own tensor: a
    /// tied embedding shares the norm output with the lookup's fallback
    /// path, and the converter's synthesized head copy is not an
    /// independent consumer either.
    pub fn head_group(d: &mut Declare, cfg: &ArchConfig) {
        if cfg.tie_word_embeddings {
            return;
        }
        d.gamma(
            "head",
            None,
            "final_norm.weight".to_string(),
            &["lm_head.weight".to_string()],
        );
    }

    /// Every block's attention + MLP groups and the head.
    pub fn fold_groups(
        tensors: &[(String, Vec<u64>)],
        cfg: &ArchConfig,
        norm_shift: &dyn Fn(&str) -> f32,
    ) -> Vec<FoldGroup> {
        let set = TensorSet::new(tensors);
        let mut d = Declare::new(&set, norm_shift);
        for layer in set.layers() {
            attn_groups(&mut d, layer);
            mlp_groups(&mut d, layer);
        }
        head_group(&mut d, cfg);
        d.finish()
    }
}

/// Fused / split MoE expert stacks as the converter names them.
pub mod moe {
    use super::*;

    /// Fused `[E, 2F, K]` gate|up (HF `experts.gate_up_proj`, MLX
    /// `switch_mlp` on Gemma 4) and the split `[E, F, K]` stacks (HF
    /// per-expert tensors stacked by the converter, Nemotron's `experts.up`).
    pub const STACKS: [&str; 7] = [
        "ffn_gate_up_exps.weight",
        "ffn_gate_exps.weight",
        "ffn_up_exps.weight",
        "ffn_down_exps.weight",
        "mlp.experts.gate_proj.weight",
        "mlp.experts.up_proj.weight",
        "mlp.experts.down_proj.weight",
    ];

    pub fn is_stack(canonical: &str, shape: &[u64]) -> bool {
        shape.len() == 3 && STACKS.contains(&tail(canonical))
    }

    /// `moe_down`: the experts' down stack takes `s_e` per expert on its
    /// columns, the up rows of the same expert absorb `1/s_e` — the second
    /// half of a fused gate|up slab, every row of a split up stack. Exact
    /// for a gated (SwiGLU / GeGLU) expert; a squared-ReLU expert (no gate)
    /// is not linear in up and must not declare it.
    pub fn down_group(d: &mut Declare, layer: usize, fused: &str, up: &str, down: &str) {
        let p = format!("layers.{layer}.");
        let down = format!("{p}{down}");
        if !d.tensors().has(&down) {
            return;
        }
        let fused = format!("{p}{fused}");
        let up = format!("{p}{up}");
        if d.tensors().has(&fused) || !d.tensors().has(&up) {
            d.rows(
                "moe_down",
                Some(layer),
                fused,
                down,
                UpstreamRows::SecondHalf,
            );
        } else {
            d.rows("moe_down", Some(layer), up, down, UpstreamRows::All);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t(names: &[(&str, &[u64])]) -> Vec<(String, Vec<u64>)> {
        names
            .iter()
            .map(|(n, s)| (n.to_string(), s.to_vec()))
            .collect()
    }

    #[test]
    fn helper_per_class() {
        assert_eq!(TensorClass::Linear.helper(), Some("dispatch_gemm"));
        assert_eq!(TensorClass::ExpertStack.helper(), Some("emit_moe_ffn"));
        assert_eq!(TensorClass::Embedding.helper(), Some("embedding_lookup"));
        assert_eq!(TensorClass::Other.helper(), None);
        let e = DispatchExtension {
            class: TensorClass::Linear,
            needs: "f32 inputs".into(),
        };
        assert_eq!(e.class.helper(), Some("dispatch_gemm"));
    }

    #[test]
    fn dense_groups_follow_the_block() {
        let tensors = t(&[
            ("embed_tokens.weight", &[10, 4]),
            ("layers.0.input_norm.weight", &[4]),
            ("layers.0.self_attn.q_proj.weight", &[4, 4]),
            ("layers.0.self_attn.k_proj.weight", &[2, 4]),
            ("layers.0.self_attn.v_proj.weight", &[2, 4]),
            ("layers.0.self_attn.o_proj.weight", &[4, 4]),
            ("layers.0.post_attn_norm.weight", &[4]),
            ("layers.0.mlp.gate_proj.weight", &[8, 4]),
            ("layers.0.mlp.up_proj.weight", &[8, 4]),
            ("layers.0.mlp.down_proj.weight", &[4, 8]),
            ("final_norm.weight", &[4]),
            ("lm_head.weight", &[10, 4]),
        ]);
        let cfg = ArchConfig::default();
        let g = dense::fold_groups(&tensors, &cfg, &|_| 1.0);
        let kinds: Vec<&str> = g.iter().map(|x| x.kind).collect();
        assert_eq!(kinds, ["attn", "down", "mlp", "o", "head"]);
        assert!(g.iter().all(|x| x.foldable));
        assert!(g[0].gamma_shift && g[2].gamma_shift && g[4].gamma_shift);
        assert_eq!(g[0].consumers.len(), 3);
        assert_eq!(g[1].upstream, "layers.0.mlp.up_proj.weight");
        assert_eq!(g[1].upstream_rows, UpstreamRows::All);
        assert_eq!(g[4].layer, None);
        // tied: no head group
        let tied = ArchConfig {
            tie_word_embeddings: true,
            ..Default::default()
        };
        let g = dense::fold_groups(&tensors, &tied, &|_| 0.0);
        assert!(g.iter().all(|x| x.kind != "head"));
        assert!(g.iter().all(|x| !x.gamma_shift));
    }

    #[test]
    fn missing_upstream_is_declared_unfoldable() {
        let tensors = t(&[
            ("layers.3.self_attn.v_proj.weight", &[2, 4]),
            ("layers.3.self_attn.o_proj.weight", &[4, 4]),
            ("layers.3.mlp.down_proj.weight", &[4, 8]),
        ]);
        let g = dense::fold_groups(&tensors, &ArchConfig::default(), &|_| 0.0);
        let kinds: Vec<&str> = g.iter().map(|x| x.kind).collect();
        assert_eq!(kinds, ["attn", "down", "o"]);
        for k in ["attn", "down"] {
            let x = g.iter().find(|x| x.kind == k).unwrap();
            assert!(!x.foldable, "{k}");
            assert!(x.refusal.as_deref().unwrap().contains("missing"), "{k}");
        }
        assert!(g.iter().find(|x| x.kind == "o").unwrap().foldable);
    }

    #[test]
    fn moe_down_picks_fused_or_split_upstream() {
        let fused = t(&[
            ("layers.0.ffn_gate_up_exps.weight", &[2, 8, 4]),
            ("layers.0.ffn_down_exps.weight", &[2, 4, 4]),
        ]);
        let set = TensorSet::new(&fused);
        let mut d = Declare::new(&set, &|_| 0.0);
        moe::down_group(
            &mut d,
            0,
            "ffn_gate_up_exps.weight",
            "ffn_up_exps.weight",
            "ffn_down_exps.weight",
        );
        let g = d.finish();
        assert_eq!(g[0].upstream, "layers.0.ffn_gate_up_exps.weight");
        assert_eq!(g[0].upstream_rows, UpstreamRows::SecondHalf);
        let split = t(&[
            ("layers.0.ffn_gate_exps.weight", &[2, 4, 4]),
            ("layers.0.ffn_up_exps.weight", &[2, 4, 4]),
            ("layers.0.ffn_down_exps.weight", &[2, 4, 4]),
        ]);
        let set = TensorSet::new(&split);
        let mut d = Declare::new(&set, &|_| 0.0);
        moe::down_group(
            &mut d,
            0,
            "ffn_gate_up_exps.weight",
            "ffn_up_exps.weight",
            "ffn_down_exps.weight",
        );
        let g = d.finish();
        assert_eq!(g[0].upstream, "layers.0.ffn_up_exps.weight");
        assert_eq!(g[0].upstream_rows, UpstreamRows::All);
        assert!(g[0].foldable);
    }
}

/// The pipeline's mappers, over the canonical tensor sets their
/// checkpoints produce (one block of each kind, as `basert inventory`
/// lists them): every tensor has a class, the GEMMs / stacks / lookups are
/// not `other`, every fold consumer is `linear` / `expert_stack`, every
/// fold kind is one autoquant knows, and the folds are what §14 says.
#[cfg(test)]
mod mapper_tests {
    use super::*;
    use crate::HfMapper;
    use std::collections::BTreeMap;

    const KINDS: [&str; 8] = [
        "attn", "mlp", "down", "o", "head", "moe", "moe_down", "gdn_out",
    ];

    fn set(names: &[(&str, &[u64])]) -> Vec<(String, Vec<u64>)> {
        names
            .iter()
            .map(|(n, s)| (n.to_string(), s.to_vec()))
            .collect()
    }

    /// Classes + folds of `mapper` over `tensors`, with the invariants.
    fn check(
        mapper: &dyn HfMapper,
        tensors: &[(String, Vec<u64>)],
        cfg: &ArchConfig,
    ) -> (BTreeMap<String, TensorClass>, Vec<FoldGroup>) {
        let classes: BTreeMap<String, TensorClass> = tensors
            .iter()
            .map(|(n, s)| (n.clone(), mapper.tensor_class(n, s)))
            .collect();
        for (n, s) in tensors {
            let c = classes[n];
            if s.len() == 1 {
                assert_eq!(c, TensorClass::Other, "{n}: a 1-D tensor is other");
            }
            if c == TensorClass::ExpertStack {
                assert_eq!(s.len(), 3, "{n}: an expert stack is 3-D");
            }
        }
        let groups = mapper.fold_groups(tensors, cfg);
        for g in &groups {
            assert!(KINDS.contains(&g.kind), "{}: unknown fold kind", g.kind);
            assert!(!g.consumers.is_empty(), "{}: no consumers", g.kind);
            for c in &g.consumers {
                let k = classes
                    .get(c)
                    .unwrap_or_else(|| panic!("{}: consumer {c} not a tensor", g.kind));
                assert!(
                    k.may_consume_fold(),
                    "{}: consumer {c} is {}",
                    g.kind,
                    k.as_str()
                );
            }
            if g.foldable {
                let up = tensors.iter().find(|(n, _)| n == &g.upstream);
                let up = up
                    .unwrap_or_else(|| panic!("{}: upstream {} missing", g.kind, g.upstream))
                    .1
                    .clone();
                match g.fold {
                    FoldKind::Gamma => assert_eq!(up.len(), 1, "{}: gamma upstream 1-D", g.kind),
                    FoldKind::Rows => assert!(up.len() >= 2, "{}: rows upstream 2-D/3-D", g.kind),
                }
            } else {
                assert!(
                    g.refusal.is_some(),
                    "{}: unfoldable without a reason",
                    g.kind
                );
            }
        }
        (classes, groups)
    }

    fn kinds(groups: &[FoldGroup]) -> Vec<(&'static str, Option<usize>)> {
        groups.iter().map(|g| (g.kind, g.layer)).collect()
    }

    fn dense_block(layer: usize, h: u64, qk_norm: bool) -> Vec<(String, Vec<u64>)> {
        let p = format!("layers.{layer}.");
        let mut v = set(&[
            (&format!("{p}input_norm.weight"), &[h]),
            (&format!("{p}post_attn_norm.weight"), &[h]),
            (&format!("{p}self_attn.q_proj.weight"), &[64, h]),
            (&format!("{p}self_attn.k_proj.weight"), &[32, h]),
            (&format!("{p}self_attn.v_proj.weight"), &[32, h]),
            (&format!("{p}self_attn.o_proj.weight"), &[h, 64]),
            (&format!("{p}mlp.gate_proj.weight"), &[128, h]),
            (&format!("{p}mlp.up_proj.weight"), &[128, h]),
            (&format!("{p}mlp.down_proj.weight"), &[h, 128]),
        ]);
        if qk_norm {
            v.extend(set(&[
                (&format!("{p}self_attn.q_norm.weight"), &[32]),
                (&format!("{p}self_attn.k_norm.weight"), &[32]),
            ]));
        }
        v
    }

    fn gdn_block(layer: usize, h: u64) -> Vec<(String, Vec<u64>)> {
        let p = format!("layers.{layer}.");
        set(&[
            (&format!("{p}input_norm.weight"), &[h]),
            (&format!("{p}post_attn_norm.weight"), &[h]),
            (&format!("{p}linear_attn.in_proj_qkv.weight"), &[128, h]),
            (&format!("{p}linear_attn.in_proj_z.weight"), &[64, h]),
            (&format!("{p}linear_attn.in_proj_a.weight"), &[4, h]),
            (&format!("{p}linear_attn.in_proj_b.weight"), &[4, h]),
            (&format!("{p}linear_attn.out_proj.weight"), &[h, 64]),
            (&format!("{p}linear_attn.conv1d.weight"), &[128, 1, 4]),
            (&format!("{p}linear_attn.norm.weight"), &[16]),
            (&format!("{p}linear_attn.A_log"), &[4]),
            (&format!("{p}linear_attn.dt_bias"), &[4]),
            (&format!("{p}mlp.gate_proj.weight"), &[128, h]),
            (&format!("{p}mlp.up_proj.weight"), &[128, h]),
            (&format!("{p}mlp.down_proj.weight"), &[h, 128]),
        ])
    }

    fn qwen_moe_ffn(layer: usize, h: u64, fused: bool) -> Vec<(String, Vec<u64>)> {
        let p = format!("layers.{layer}.");
        let mut v = set(&[
            (&format!("{p}mlp.gate.weight"), &[4, h]),
            (&format!("{p}mlp.shared_expert.gate_proj.weight"), &[128, h]),
            (&format!("{p}mlp.shared_expert.up_proj.weight"), &[128, h]),
            (&format!("{p}mlp.shared_expert.down_proj.weight"), &[h, 128]),
            (&format!("{p}mlp.shared_expert_gate.weight"), &[1, h]),
            (&format!("{p}ffn_down_exps.weight"), &[4, h, 32]),
        ]);
        if fused {
            v.push((format!("{p}ffn_gate_up_exps.weight"), vec![4, 64, h]));
        } else {
            v.push((format!("{p}ffn_gate_exps.weight"), vec![4, 32, h]));
            v.push((format!("{p}ffn_up_exps.weight"), vec![4, 32, h]));
        }
        v
    }

    fn top(tied: bool, h: u64) -> Vec<(String, Vec<u64>)> {
        let mut v = set(&[
            ("embed_tokens.weight", &[160, h]),
            ("final_norm.weight", &[h]),
        ]);
        if !tied {
            v.push(("lm_head.weight".to_string(), vec![160, h]));
        }
        v
    }

    #[test]
    fn llama_and_qwen_dense() {
        for (m, tied) in [
            (&crate::llama::LlamaHfMapper as &dyn HfMapper, true),
            (&crate::llama::Phi3HfMapper, false),
            (&crate::qwen::QwenHfMapper, false),
            (&crate::muse_glimmer::MuseGlimmerHfMapper, false),
        ] {
            let mut t = top(tied, 64);
            t.extend(dense_block(0, 64, true));
            let cfg = ArchConfig {
                tie_word_embeddings: tied,
                ..Default::default()
            };
            let (classes, groups) = check(m, &t, &cfg);
            assert_eq!(classes["embed_tokens.weight"], TensorClass::Embedding);
            assert_eq!(
                classes["layers.0.self_attn.q_proj.weight"],
                TensorClass::Linear
            );
            assert_eq!(
                classes["layers.0.mlp.down_proj.weight"],
                TensorClass::Linear
            );
            assert_eq!(
                classes["layers.0.self_attn.q_norm.weight"],
                TensorClass::Other
            );
            let mut want = vec![
                ("attn", Some(0)),
                ("down", Some(0)),
                ("mlp", Some(0)),
                ("o", Some(0)),
            ];
            if !tied {
                assert_eq!(classes["lm_head.weight"], TensorClass::Linear);
                want.push(("head", None));
            }
            assert_eq!(kinds(&groups), want, "{}", m.canonical_arch());
            assert!(groups.iter().all(|g| g.foldable));
        }
    }

    #[test]
    fn qwen3_moe_split_experts() {
        let mut t = top(false, 64);
        t.extend(dense_block(0, 64, true));
        t.retain(|(n, _)| !n.starts_with("layers.0.mlp."));
        t.extend(qwen_moe_ffn(0, 64, false));
        let (classes, groups) = check(&crate::qwen::QwenMoeHfMapper, &t, &ArchConfig::default());
        assert_eq!(
            classes["layers.0.ffn_up_exps.weight"],
            TensorClass::ExpertStack
        );
        assert_eq!(classes["layers.0.mlp.gate.weight"], TensorClass::Other);
        assert_eq!(
            classes["layers.0.mlp.shared_expert_gate.weight"],
            TensorClass::Other
        );
        assert_eq!(
            classes["layers.0.mlp.shared_expert.up_proj.weight"],
            TensorClass::Linear
        );
        assert_eq!(
            kinds(&groups),
            [
                ("attn", Some(0)),
                ("down", Some(0)),
                ("moe_down", Some(0)),
                ("o", Some(0)),
                ("head", None)
            ]
        );
        let md = groups.iter().find(|g| g.kind == "moe_down").unwrap();
        assert_eq!(md.upstream, "layers.0.ffn_up_exps.weight");
        assert_eq!(md.upstream_rows, UpstreamRows::All);
        assert!(md.foldable);
        let down = groups.iter().find(|g| g.kind == "down").unwrap();
        assert_eq!(down.upstream, "layers.0.mlp.shared_expert.up_proj.weight");
        // no input-side `moe` group: the router reads the same norm output
        assert!(groups.iter().all(|g| g.kind != "moe"));
    }

    #[test]
    fn qwen35_dense_and_moe() {
        // dense, tied (the 2B): three GDN blocks, one attention block
        let mut t = top(true, 64);
        t.extend(gdn_block(0, 64));
        t.extend(dense_block(1, 64, true));
        let cfg = ArchConfig {
            tie_word_embeddings: true,
            ..Default::default()
        };
        let (classes, groups) = check(&crate::qwen::Qwen35HfMapper, &t, &cfg);
        for n in [
            "layers.0.linear_attn.in_proj_qkv.weight",
            "layers.0.linear_attn.in_proj_z.weight",
            "layers.0.linear_attn.in_proj_a.weight",
            "layers.0.linear_attn.in_proj_b.weight",
            "layers.0.linear_attn.out_proj.weight",
        ] {
            assert_eq!(classes[n], TensorClass::Linear, "{n}");
        }
        for n in [
            "layers.0.linear_attn.conv1d.weight",
            "layers.0.linear_attn.norm.weight",
            "layers.0.linear_attn.A_log",
        ] {
            assert_eq!(classes[n], TensorClass::Other, "{n}");
        }
        assert_eq!(
            kinds(&groups),
            [
                ("attn", Some(0)),
                ("down", Some(0)),
                ("gdn_out", Some(0)),
                ("mlp", Some(0)),
                ("attn", Some(1)),
                ("down", Some(1)),
                ("mlp", Some(1)),
                ("o", Some(1)),
            ]
        );
        let attn0 = &groups[0];
        assert_eq!(
            attn0.consumers.len(),
            4,
            "the GDN attn group lists all four projections"
        );
        assert!(attn0.gamma_shift, "Qwen3.5 norms are (1 + g)");
        let gdn = groups.iter().find(|g| g.kind == "gdn_out").unwrap();
        assert_eq!(gdn.upstream, "layers.0.linear_attn.norm.weight");
        assert!(!gdn.gamma_shift, "the gated norm is a plain gain");

        // MoE (the 35B-A3B): untied, fused gate_up, shared expert
        let mut t = top(false, 64);
        t.extend(gdn_block(0, 64));
        t.retain(|(n, _)| !n.starts_with("layers.0.mlp."));
        t.extend(qwen_moe_ffn(0, 64, true));
        let (classes, groups) = check(&crate::qwen::Qwen35MoeHfMapper, &t, &ArchConfig::default());
        assert_eq!(
            classes["layers.0.ffn_gate_up_exps.weight"],
            TensorClass::ExpertStack
        );
        assert_eq!(
            kinds(&groups),
            [
                ("attn", Some(0)),
                ("down", Some(0)),
                ("gdn_out", Some(0)),
                ("moe_down", Some(0)),
                ("head", None)
            ]
        );
        let md = groups.iter().find(|g| g.kind == "moe_down").unwrap();
        assert_eq!(md.upstream_rows, UpstreamRows::SecondHalf);
    }

    #[test]
    fn qwen4_exp_flash_next() {
        let h = 64u64;
        let mut t = set(&[
            ("embed_tokens.weight", &[160, h]),
            ("lm_head.weight", &[160, h]),
            ("hyper_connection_mixer.hc_norm.weight", &[h]),
            (
                "hyper_connection_mixer.input_mix_weight_down.weight",
                &[8, 4 * h],
            ),
            (
                "hyper_connection_mixer.input_mix_weight_up.weight",
                &[4 * h, 8],
            ),
            (
                "layers.1.ple.ple_embedding.ngram_embedding.weight",
                &[192, 160],
            ),
            ("layers.1.ple.key_proj.weight", &[h, 2560]),
            ("layers.1.ple.value_proj.weight", &[h, h]),
            ("layers.1.ple.conv1d.weight", &[h, 1, 4]),
            ("layers.1.ple.norm_key.weight", &[h]),
            ("layers.3.self_attn.indexer.index_qk_proj.weight", &[48, h]),
            ("layers.3.self_attn.indexer.q_layernorm.weight", &[16]),
        ]);
        for layer in [1usize, 3] {
            for hc in ["attn_hyper_connection", "mlp_hyper_connection"] {
                let p = format!("layers.{layer}.{hc}.");
                t.extend(set(&[
                    (&format!("{p}hc_norm.weight"), &[h]),
                    (&format!("{p}block_inject_weight.weight"), &[4, h]),
                    (&format!("{p}input_mix_weight_down.weight"), &[8, 4 * h]),
                    (&format!("{p}input_mix_weight_up.weight"), &[4 * h, 8]),
                ]));
            }
            t.extend(qwen_moe_ffn(layer, h, true));
        }
        let mut g = gdn_block(1, h);
        g.retain(|(n, _)| !n.contains("norm.weight") || n.contains("linear_attn.norm"));
        g.retain(|(n, _)| !n.starts_with("layers.1.mlp."));
        t.extend(g);
        let mut a = dense_block(3, h, true);
        a.retain(|(n, _)| {
            !n.ends_with("input_norm.weight") && !n.ends_with("post_attn_norm.weight")
        });
        a.retain(|(n, _)| !n.starts_with("layers.3.mlp."));
        t.extend(a);
        let (classes, groups) = check(&crate::qwen::Qwen4ExpHfMapper, &t, &ArchConfig::default());
        assert_eq!(
            classes["layers.1.ple.ple_embedding.ngram_embedding.weight"],
            TensorClass::Embedding
        );
        assert_eq!(classes["embed_tokens.weight"], TensorClass::Embedding);
        for n in [
            "lm_head.weight",
            "hyper_connection_mixer.input_mix_weight_down.weight",
            "hyper_connection_mixer.input_mix_weight_up.weight",
            "layers.1.attn_hyper_connection.input_mix_weight_down.weight",
            "layers.1.mlp_hyper_connection.input_mix_weight_up.weight",
            "layers.1.ple.key_proj.weight",
            "layers.1.ple.value_proj.weight",
            "layers.1.linear_attn.in_proj_qkv.weight",
            "layers.1.linear_attn.in_proj_a.weight",
            "layers.1.linear_attn.out_proj.weight",
            "layers.1.mlp.shared_expert.down_proj.weight",
            "layers.3.self_attn.q_proj.weight",
            "layers.3.self_attn.o_proj.weight",
        ] {
            assert_eq!(classes[n], TensorClass::Linear, "{n}");
        }
        for n in [
            "layers.1.attn_hyper_connection.block_inject_weight.weight",
            "layers.1.attn_hyper_connection.hc_norm.weight",
            "layers.1.mlp.gate.weight",
            "layers.1.mlp.shared_expert_gate.weight",
            "layers.1.ple.conv1d.weight",
            "layers.1.linear_attn.conv1d.weight",
            "layers.3.self_attn.indexer.index_qk_proj.weight",
        ] {
            assert_eq!(classes[n], TensorClass::Other, "{n}");
        }
        assert_eq!(
            classes["layers.1.ffn_gate_up_exps.weight"],
            TensorClass::ExpertStack
        );
        // output-side folds only
        assert_eq!(
            kinds(&groups),
            [
                ("down", Some(1)),
                ("gdn_out", Some(1)),
                ("moe_down", Some(1)),
                ("down", Some(3)),
                ("moe_down", Some(3)),
                ("o", Some(3)),
            ]
        );
        assert!(groups.iter().all(|g| g.foldable));
    }

    #[test]
    fn gemma3_and_gemma4() {
        let h = 64u64;
        let mut t = top(true, h);
        t.extend(dense_block(0, h, true));
        let cfg = ArchConfig {
            tie_word_embeddings: true,
            ..Default::default()
        };
        let (classes, groups) = check(&crate::gemma::Gemma3HfMapper, &t, &cfg);
        assert_eq!(classes["embed_tokens.weight"], TensorClass::Embedding);
        assert_eq!(
            kinds(&groups),
            [
                ("attn", Some(0)),
                ("down", Some(0)),
                ("mlp", Some(0)),
                ("o", Some(0))
            ]
        );
        assert!(groups.iter().all(|g| g.foldable));
        assert!(groups
            .iter()
            .filter(|g| g.fold == FoldKind::Gamma)
            .all(|g| g.gamma_shift));

        // Gemma 4 MoE: PLE + fused experts + router beside the dense FFN
        t.extend(set(&[
            ("per_layer_token_embd.weight", &[160, 32]),
            ("per_layer_model_proj.weight", &[32, h]),
            ("per_layer_proj_norm.weight", &[16]),
            ("layers.0.per_layer_inp_gate.weight", &[16, h]),
            ("layers.0.per_layer_proj.weight", &[h, 16]),
            ("layers.0.per_layer_post_norm.weight", &[h]),
            ("layers.0.pre_ffw_norm_2.weight", &[h]),
            ("layers.0.ffn_gate_up_exps.weight", &[4, 64, h]),
            ("layers.0.ffn_down_exps.weight", &[4, h, 32]),
            ("layers.0.mlp.router.weight", &[4, h]),
            ("layers.0.mlp.ffn_gate_inp.scale", &[1]),
            ("layers.0.mlp.ffn_down_exps.scale", &[4]),
        ]));
        let (classes, groups) = check(&crate::gemma::Gemma4HfMapper, &t, &cfg);
        assert_eq!(
            classes["per_layer_token_embd.weight"],
            TensorClass::Embedding
        );
        assert_eq!(classes["per_layer_model_proj.weight"], TensorClass::Linear);
        assert_eq!(
            classes["layers.0.per_layer_inp_gate.weight"],
            TensorClass::Linear
        );
        assert_eq!(
            classes["layers.0.per_layer_proj.weight"],
            TensorClass::Linear
        );
        assert_eq!(
            classes["layers.0.ffn_gate_up_exps.weight"],
            TensorClass::ExpertStack
        );
        assert_eq!(classes["layers.0.mlp.router.weight"], TensorClass::Other);
        assert_eq!(
            kinds(&groups),
            [
                ("attn", Some(0)),
                ("down", Some(0)),
                ("mlp", Some(0)),
                ("moe", Some(0)),
                ("moe_down", Some(0)),
                ("o", Some(0))
            ]
        );
        let o = groups.iter().find(|g| g.kind == "o").unwrap();
        assert!(!o.foldable && o.refusal.as_deref().unwrap().contains("RMS-normalizes V"));
        let moe = groups.iter().find(|g| g.kind == "moe").unwrap();
        assert!(
            moe.foldable && !moe.gamma_shift,
            "Gemma 4 dropped the unit offset"
        );
        assert_eq!(
            groups
                .iter()
                .find(|g| g.kind == "moe_down")
                .unwrap()
                .upstream_rows,
            UpstreamRows::SecondHalf
        );
    }

    #[test]
    fn nemotron_h() {
        let h = 64u64;
        let mut t = top(false, h);
        t.extend(set(&[
            ("layers.0.input_norm.weight", &[h]),
            ("layers.0.ssm.in_proj.weight", &[164, h]),
            ("layers.0.ssm.out_proj.weight", &[h, 64]),
            ("layers.0.ssm.conv1d.weight", &[96, 1, 4]),
            ("layers.0.ssm.conv1d.bias", &[96]),
            ("layers.0.ssm.a_log", &[4]),
            ("layers.0.ssm.d", &[4]),
            ("layers.0.ssm.dt_bias", &[4]),
            ("layers.0.ssm.norm.weight", &[64]),
            ("layers.1.input_norm.weight", &[h]),
            ("layers.1.self_attn.q_proj.weight", &[64, h]),
            ("layers.1.self_attn.k_proj.weight", &[32, h]),
            ("layers.1.self_attn.v_proj.weight", &[32, h]),
            ("layers.1.self_attn.o_proj.weight", &[h, 64]),
            ("layers.2.input_norm.weight", &[h]),
            ("layers.2.mlp.router.weight", &[4, h]),
            ("layers.2.mlp.router.e_score_correction_bias", &[4]),
            ("layers.2.mlp.experts.up_proj.weight", &[4, 32, h]),
            ("layers.2.mlp.experts.down_proj.weight", &[4, h, 32]),
            ("layers.2.mlp.shared_expert.up_proj.weight", &[128, h]),
            ("layers.2.mlp.shared_expert.down_proj.weight", &[h, 128]),
        ]));
        let (classes, groups) = check(
            &crate::nemotron::NemotronHHfMapper,
            &t,
            &ArchConfig::default(),
        );
        assert_eq!(classes["layers.0.ssm.in_proj.weight"], TensorClass::Linear);
        assert_eq!(classes["layers.0.ssm.out_proj.weight"], TensorClass::Linear);
        assert_eq!(classes["layers.0.ssm.conv1d.weight"], TensorClass::Other);
        assert_eq!(
            classes["layers.2.mlp.experts.up_proj.weight"],
            TensorClass::ExpertStack
        );
        assert_eq!(classes["layers.2.mlp.router.weight"], TensorClass::Other);
        assert_eq!(
            classes["layers.2.mlp.shared_expert.up_proj.weight"],
            TensorClass::Linear
        );
        // squared-ReLU experts: no down folds; router: no moe fold
        assert_eq!(
            kinds(&groups),
            [("attn", Some(1)), ("o", Some(1)), ("head", None)]
        );
    }

    #[test]
    fn glm_dsa() {
        let h = 64u64;
        let mut t = top(false, h);
        for layer in [0usize, 1] {
            let p = format!("layers.{layer}.");
            t.extend(set(&[
                (&format!("{p}input_norm.weight"), &[h]),
                (&format!("{p}post_attn_norm.weight"), &[h]),
                (&format!("{p}self_attn.q_a_proj.weight"), &[32, h]),
                (&format!("{p}self_attn.q_a_layernorm.weight"), &[32]),
                (&format!("{p}self_attn.q_b_proj.weight"), &[48, 32]),
                (&format!("{p}self_attn.kv_a_proj_with_mqa.weight"), &[24, h]),
                (&format!("{p}self_attn.kv_a_layernorm.weight"), &[16]),
                (&format!("{p}self_attn.kv_b_proj.weight"), &[64, 16]),
                (&format!("{p}self_attn.k_b_proj.weight"), &[2, 16, 16]),
                (&format!("{p}self_attn.v_b_proj.weight"), &[2, 16, 16]),
                (&format!("{p}self_attn.o_proj.weight"), &[h, 32]),
                (&format!("{p}self_attn.indexer.q_b_proj.weight"), &[32, 32]),
                (&format!("{p}self_attn.indexer.k_proj.weight"), &[16, h]),
                (
                    &format!("{p}self_attn.indexer.weights_proj.weight"),
                    &[2, h],
                ),
                (&format!("{p}self_attn.indexer.k_norm.weight"), &[16]),
                (&format!("{p}self_attn.indexer.k_norm.bias"), &[16]),
            ]));
        }
        t.extend(set(&[
            ("layers.0.mlp.gate_proj.weight", &[128, h]),
            ("layers.0.mlp.up_proj.weight", &[128, h]),
            ("layers.0.mlp.down_proj.weight", &[h, 128]),
            ("layers.1.mlp.router.weight", &[4, h]),
            ("layers.1.mlp.gate.e_score_correction_bias", &[4]),
            ("layers.1.mlp.experts.gate_proj.weight", &[4, 32, h]),
            ("layers.1.mlp.experts.up_proj.weight", &[4, 32, h]),
            ("layers.1.mlp.experts.down_proj.weight", &[4, h, 32]),
            ("layers.1.mlp.shared_expert.gate_proj.weight", &[128, h]),
            ("layers.1.mlp.shared_expert.up_proj.weight", &[128, h]),
            ("layers.1.mlp.shared_expert.down_proj.weight", &[h, 128]),
        ]));
        let (classes, groups) = check(&crate::glm::GlmDsaMapper, &t, &ArchConfig::default());
        for n in [
            "layers.0.self_attn.q_a_proj.weight",
            "layers.0.self_attn.q_b_proj.weight",
            "layers.0.self_attn.kv_a_proj_with_mqa.weight",
            "layers.0.self_attn.o_proj.weight",
            "layers.0.self_attn.indexer.k_proj.weight",
            "layers.0.self_attn.indexer.q_b_proj.weight",
            "layers.0.self_attn.indexer.weights_proj.weight",
            "layers.1.mlp.shared_expert.down_proj.weight",
            "layers.0.mlp.down_proj.weight",
        ] {
            assert_eq!(classes[n], TensorClass::Linear, "{n}");
        }
        for n in [
            "layers.0.self_attn.kv_b_proj.weight",
            "layers.0.self_attn.k_b_proj.weight",
            "layers.0.self_attn.v_b_proj.weight",
            "layers.1.mlp.router.weight",
        ] {
            assert_eq!(classes[n], TensorClass::Other, "{n}");
        }
        assert_eq!(
            classes["layers.1.mlp.experts.up_proj.weight"],
            TensorClass::ExpertStack
        );
        assert_eq!(
            kinds(&groups),
            [
                ("attn", Some(0)),
                ("down", Some(0)),
                ("mlp", Some(0)),
                ("attn", Some(1)),
                ("down", Some(1)),
                ("moe_down", Some(1)),
                ("head", None),
            ]
        );
        let attn = &groups[0];
        assert_eq!(attn.consumers.len(), 4);
        let md = groups.iter().find(|g| g.kind == "moe_down").unwrap();
        assert_eq!(md.upstream, "layers.1.mlp.experts.up_proj.weight");
        assert_eq!(md.upstream_rows, UpstreamRows::All);
        // no `o`: MLA has no v_proj
        assert!(groups.iter().all(|g| g.kind != "o"));
    }

    #[test]
    fn sections_and_declaring_mapper() {
        let m = crate::declaring_mapper("qwen35").unwrap();
        assert_eq!(
            m.speculator_class("mtp.layers.0.self_attn.q_proj.weight", &[64, 64]),
            TensorClass::Linear
        );
        assert_eq!(
            m.speculator_class("mtp.fc.weight", &[64, 128]),
            TensorClass::Linear
        );
        assert_eq!(
            m.speculator_class("mtp.pre_fc_norm_hidden.weight", &[64]),
            TensorClass::Other
        );
        assert_eq!(
            m.mmproj_class("visual.blocks.0.attn.qkv.weight", &[96, 32]),
            TensorClass::Linear
        );
        assert_eq!(
            m.mmproj_class("visual.blocks.0.norm1.weight", &[32]),
            TensorClass::Other
        );
        assert_eq!(
            m.mmproj_class("visual.pos_embed.weight", &[2304, 32]),
            TensorClass::Other
        );
        assert_eq!(
            m.mmproj_class("visual.patch_embed.proj.weight", &[32, 3, 2, 16, 16]),
            TensorClass::Other
        );
        for arch in [
            "llama",
            "qwen",
            "qwen3_moe",
            "qwen35",
            "qwen35moe",
            "qwen4_exp",
            "gemma3",
            "gemma4",
            "nemotron_h_moe",
            "glm_dsa",
        ] {
            assert_eq!(
                crate::declaring_mapper(arch).unwrap().canonical_arch(),
                arch
            );
        }
        assert!(crate::declaring_mapper("synthetic").is_none());
    }
}
