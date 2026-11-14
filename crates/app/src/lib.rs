//! Process-facing configuration shared by Bloom's CLI and HTTP applications.
//!
//! The inference engine and FFI do not depend on this crate. Configuration-file
//! paths, server deployment settings, and CLI parsing belong at this boundary.

#![cfg_attr(not(test), warn(clippy::unwrap_used))]

pub mod backend;
pub mod config;
pub mod text_metrics;

pub use backend::{
    SUPPORTED_ENGINES_HINT, build_engine_registry, parse_cli_dtype, select_backend_name,
    validate_cli_model_task,
};
pub use config::{
    BenchConfig, BloomConfig, InferConfig, ServerConfig, default_config_dir, default_config_path,
    load_config, resolve_config_path, write_default_config,
};
