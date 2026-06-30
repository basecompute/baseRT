//! base-hub — model hub for baseRT: resolve, download, and catalog `.base`
//! models from HuggingFace.
//!
//! Two source kinds resolve behind one [`registry::MergedRegistry`]:
//! pre-converted `.base` artifacts hosted in the basecompute HF org (fast path,
//! no local conversion), and arbitrary HF repos of source safetensors that
//! are downloaded and converted locally by `base-convert`.

pub mod cache;
pub mod catalog;
pub mod fetch;
pub mod registry;

pub use cache::{models_dir, HubSidecar};
pub use registry::{ModelEntry, ModelRef, Registry, SourceKind};
