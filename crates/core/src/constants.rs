//! Shared byte-unit constants used across crates.
//!
//! Centralising these here avoids duplicate local definitions and raw
//! `1024 * 1024 * 1024` literals scattered through backend probes, memory
//! budgets, and engine defaults.

/// 1 kibibyte in bytes.
pub const KIB: u64 = 1024;
/// 1 mebibyte in bytes.
pub const MIB: u64 = 1024 * 1024;
/// 1 gibibyte in bytes.
pub const GIB: u64 = 1024 * 1024 * 1024;

/// 1 kibibyte as `f64`.
pub const KIB_F64: f64 = 1024.0;
/// 1 mebibyte as `f64`.
pub const MIB_F64: f64 = 1024.0 * 1024.0;
/// 1 gibibyte as `f64`.
pub const GIB_F64: f64 = 1024.0 * 1024.0 * 1024.0;
