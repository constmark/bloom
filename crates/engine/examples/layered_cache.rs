//! Small end-to-end example for the independent three-tier block cache.
//!
//! Run with:
//!
//! ```text
//! cargo run -p bloomai-engine --example layered_cache
//! ```

use std::sync::Arc;

use anyhow::Result;
use bloomai_engine::{
    FileSystemLayeredCache, LayeredBlockCache, LayeredCacheBackend, LayeredCacheConfig,
    LayeredCacheTier,
};

fn main() -> Result<()> {
    let root = std::env::temp_dir().join(format!(
        "bloom-layered-cache-example-{}",
        std::process::id()
    ));
    let backend: Arc<dyn LayeredCacheBackend> = Arc::new(FileSystemLayeredCache::new(&root)?);
    let cache = LayeredBlockCache::with_backend(
        LayeredCacheConfig {
            l1_capacity_bytes: 64,
            l2_capacity_bytes: 128,
            max_entry_bytes: 1024,
            write_through: true,
        },
        Some(Arc::clone(&backend)),
    )?;

    let key = "demo/model-revision/layer-0/block-0";
    let tier = cache.put(key, b"hello from Bloom".to_vec())?;
    assert_eq!(tier, LayeredCacheTier::L1);

    // Simulate process pressure or a model switch. The value remains in L3
    // because write_through was enabled, and get promotes it back into memory.
    cache.clear_memory();
    let restored = cache.get(key)?.expect("value should be in the L3 backend");
    assert_eq!(restored, b"hello from Bloom");

    println!("restored {} bytes from the layered cache", restored.len());
    println!("{}", serde_json::to_string_pretty(&cache.metrics())?);

    cache.remove(key)?;
    let _ = std::fs::remove_dir_all(root);
    Ok(())
}
