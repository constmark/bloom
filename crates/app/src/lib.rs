//! Process-facing configuration shared by Bloom's CLI and HTTP applications.
//!
//! The inference engine and FFI do not depend on this crate. Configuration-file
//! paths, server deployment settings, and CLI parsing belong at this boundary.

#![cfg_attr(not(test), warn(clippy::unwrap_used))]

pub mod config;

pub use config::{
    BenchConfig, BloomConfig, InferConfig, ServerConfig, default_config_dir, default_config_path,
    load_config, resolve_config_path, write_default_config,
};
