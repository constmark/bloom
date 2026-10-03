//! Application services shared by HTTP adapters and runtime workers.
//! Dependencies point to engine contracts and model infrastructure, never to
//! the server composition root, CLI parsing, or HTTP protocol modules.
pub(crate) mod backend_registry;
pub(crate) mod config;
pub(crate) mod embedding;
pub(crate) mod inference;
pub(crate) mod loader;
pub(crate) mod memory;
pub(crate) mod model_selector;
pub(crate) mod pool;
pub(crate) mod runtime;
pub(crate) mod runtime_service;
pub(crate) mod scheduling;
