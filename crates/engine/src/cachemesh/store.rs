use std::collections::{HashMap, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{anyhow, bail, Context, Result};
use bloomai_core::constants::GIB;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum CacheMeshTier {
    L1,
    L2,
    L3,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CacheMeshKey {
    pub namespace: String,
    pub layer_idx: usize,
    pub digest: String,
}

impl CacheMeshKey {
    pub fn from_tokens(namespace: impl Into<String>, layer_idx: usize, tokens: &[u32]) -> Self {
        Self::from_tokens_multimodal(namespace, layer_idx, tokens, None)
    }

    pub fn from_tokens_multimodal(
        namespace: impl Into<String>,
        layer_idx: usize,
        tokens: &[u32],
        multimodal_hash: Option<&str>,
    ) -> Self {
        let mut hasher = Sha256::new();
        hasher.update(b"bloom-cachemesh-v1/tokens");
        hasher.update(layer_idx.to_le_bytes());
        for token in tokens {
            hasher.update(token.to_le_bytes());
        }
        if let Some(h) = multimodal_hash {
            hasher.update(b"multimodal");
            hasher.update(h.as_bytes());
        }
        Self {
            namespace: namespace.into(),
            layer_idx,
            digest: hex_digest(hasher.finalize().as_slice()),
        }
    }

    pub fn from_request_block(
        namespace: impl Into<String>,
        layer_idx: usize,
        request_id: &str,
        block_idx: usize,
    ) -> Self {
        let mut hasher = Sha256::new();
        hasher.update(b"bloom-cachemesh-v1/request-block");
        hasher.update(layer_idx.to_le_bytes());
        hasher.update(request_id.as_bytes());
        hasher.update(block_idx.to_le_bytes());
        Self {
            namespace: namespace.into(),
            layer_idx,
            digest: hex_digest(hasher.finalize().as_slice()),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheMeshBlock {
    pub key: CacheMeshKey,
    pub block_size: usize,
    pub kv_dim: usize,
    pub keys: Vec<f32>,
    pub values: Vec<f32>,
    pub num_tokens: usize,
}

impl CacheMeshBlock {
    pub fn bytes(&self) -> usize {
        (self.keys.len() + self.values.len()) * std::mem::size_of::<f32>()
    }

    pub fn validate(&self) -> Result<()> {
        if self.block_size == 0 {
            bail!("cachemesh block_size must be greater than zero");
        }
        if self.kv_dim == 0 {
            bail!("cachemesh kv_dim must be greater than zero");
        }
        if self.num_tokens > self.block_size {
            bail!(
                "cachemesh block has {} tokens but block_size is {}",
                self.num_tokens,
                self.block_size
            );
        }
        if self.keys.len() != self.values.len() {
            bail!(
                "cachemesh key/value length mismatch: keys={}, values={}",
                self.keys.len(),
                self.values.len()
            );
        }
        let expected = self.num_tokens.saturating_mul(self.kv_dim);
        if self.keys.len() != expected {
            bail!(
                "cachemesh block shape mismatch: expected {} elements, got {}",
                expected,
                self.keys.len()
            );
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheMeshConfig {
    pub enabled: bool,
    pub namespace: String,
    pub l2_capacity_bytes: usize,
    pub l3_enabled: bool,
    pub write_through_l3: bool,
}

impl Default for CacheMeshConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            namespace: "default".to_string(),
            l2_capacity_bytes: 2 * GIB as usize,
            l3_enabled: false,
            write_through_l3: false,
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TierMetrics {
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    pub offloads: u64,
    pub restores: u64,
    pub failed_offloads: u64,
    pub dropped: u64,
    pub bytes: usize,
    pub items: usize,
    pub hit_rate: f64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CacheMeshMetrics {
    pub enabled: bool,
    pub l1: TierMetrics,
    pub l2: TierMetrics,
    pub l3: TierMetrics,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CacheMeshSnapshot {
    pub metrics: CacheMeshMetrics,
    pub l2_keys: usize,
}

pub trait RemoteCacheBackend: Send + Sync {
    fn get(&self, key: &CacheMeshKey) -> Result<Option<CacheMeshBlock>>;
    fn put(&self, block: CacheMeshBlock) -> Result<()>;
    fn remove(&self, key: &CacheMeshKey) -> Result<()>;
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    fn bytes(&self) -> Option<usize> {
        None
    }
}

#[derive(Default)]
pub struct InMemoryRemoteCache {
    blocks: Mutex<HashMap<CacheMeshKey, CacheMeshBlock>>,
}

impl InMemoryRemoteCache {
    pub fn new() -> Self {
        Self::default()
    }
}

impl RemoteCacheBackend for InMemoryRemoteCache {
    fn get(&self, key: &CacheMeshKey) -> Result<Option<CacheMeshBlock>> {
        let block = self.blocks.lock().unwrap().get(key).cloned();
        if let Some(block) = &block {
            block.validate()?;
        }
        Ok(block)
    }

    fn put(&self, block: CacheMeshBlock) -> Result<()> {
        block.validate()?;
        self.blocks.lock().unwrap().insert(block.key.clone(), block);
        Ok(())
    }

    fn remove(&self, key: &CacheMeshKey) -> Result<()> {
        self.blocks.lock().unwrap().remove(key);
        Ok(())
    }

    fn len(&self) -> usize {
        self.blocks.lock().unwrap().len()
    }

    fn bytes(&self) -> Option<usize> {
        Some(
            self.blocks
                .lock()
                .unwrap()
                .values()
                .map(|b| b.bytes())
                .sum(),
        )
    }
}

pub struct FileSystemRemoteCache {
    root: PathBuf,
}

impl FileSystemRemoteCache {
    pub fn new(root: impl Into<PathBuf>) -> Result<Self> {
        let root = root.into();
        std::fs::create_dir_all(&root)
            .with_context(|| format!("failed to create cachemesh L3 root '{}'", root.display()))?;
        Ok(Self { root })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    fn path_for(&self, key: &CacheMeshKey) -> PathBuf {
        self.root
            .join(sanitize_component(&key.namespace))
            .join(format!("layer-{}", key.layer_idx))
            .join(format!("{}.json", key.digest))
    }
}

impl RemoteCacheBackend for FileSystemRemoteCache {
    fn get(&self, key: &CacheMeshKey) -> Result<Option<CacheMeshBlock>> {
        let path = self.path_for(key);
        if !path.exists() {
            return Ok(None);
        }
        let content = std::fs::read_to_string(&path)
            .with_context(|| format!("failed to read cachemesh block '{}'", path.display()))?;
        let block: CacheMeshBlock = serde_json::from_str(&content)
            .with_context(|| format!("failed to parse cachemesh block '{}'", path.display()))?;
        block.validate()?;
        Ok(Some(block))
    }

    fn put(&self, block: CacheMeshBlock) -> Result<()> {
        block.validate()?;
        let path = self.path_for(&block.key);
        let parent = path
            .parent()
            .ok_or_else(|| anyhow!("cachemesh block path has no parent: {}", path.display()))?;
        std::fs::create_dir_all(parent).with_context(|| {
            format!(
                "failed to create cachemesh L3 parent directory '{}'",
                parent.display()
            )
        })?;

        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or_default();
        let tmp = path.with_extension(format!("json.tmp-{now}"));
        let content = serde_json::to_vec(&block)?;
        std::fs::write(&tmp, content)
            .with_context(|| format!("failed to write cachemesh temp file '{}'", tmp.display()))?;
        std::fs::rename(&tmp, &path).with_context(|| {
            format!(
                "failed to atomically install cachemesh block '{}' from '{}'",
                path.display(),
                tmp.display()
            )
        })?;
        Ok(())
    }

    fn remove(&self, key: &CacheMeshKey) -> Result<()> {
        let path = self.path_for(key);
        match std::fs::remove_file(&path) {
            Ok(()) => Ok(()),
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(err) => Err(err)
                .with_context(|| format!("failed to remove cachemesh block '{}'", path.display())),
        }
    }

    fn len(&self) -> usize {
        count_files_with_extension(&self.root, "json").unwrap_or(0)
    }

    fn bytes(&self) -> Option<usize> {
        sum_file_bytes_with_extension(&self.root, "json").ok()
    }
}

struct L2State {
    blocks: HashMap<CacheMeshKey, CacheMeshBlock>,
    lru: VecDeque<CacheMeshKey>,
    bytes: usize,
}

impl L2State {
    fn new() -> Self {
        Self {
            blocks: HashMap::new(),
            lru: VecDeque::new(),
            bytes: 0,
        }
    }

    fn touch(&mut self, key: &CacheMeshKey) {
        if let Some(pos) = self.lru.iter().position(|k| k == key) {
            self.lru.remove(pos);
        }
        self.lru.push_back(key.clone());
    }
}

#[derive(Default)]
struct AtomicTierMetrics {
    hits: AtomicU64,
    misses: AtomicU64,
    evictions: AtomicU64,
    offloads: AtomicU64,
    restores: AtomicU64,
    failed_offloads: AtomicU64,
    dropped: AtomicU64,
}

impl AtomicTierMetrics {
    fn snapshot(&self, bytes: usize, items: usize) -> TierMetrics {
        let hits = self.hits.load(Ordering::Relaxed);
        let misses = self.misses.load(Ordering::Relaxed);
        let lookups = hits + misses;
        TierMetrics {
            hits,
            misses,
            evictions: self.evictions.load(Ordering::Relaxed),
            offloads: self.offloads.load(Ordering::Relaxed),
            restores: self.restores.load(Ordering::Relaxed),
            failed_offloads: self.failed_offloads.load(Ordering::Relaxed),
            dropped: self.dropped.load(Ordering::Relaxed),
            bytes,
            items,
            hit_rate: if lookups == 0 {
                0.0
            } else {
                hits as f64 / lookups as f64
            },
        }
    }
}

pub struct CacheMesh {
    config: CacheMeshConfig,
    l2: Mutex<L2State>,
    remote: Option<Arc<dyn RemoteCacheBackend>>,
    l1_metrics: AtomicTierMetrics,
    l2_metrics: AtomicTierMetrics,
    l3_metrics: AtomicTierMetrics,
}

impl CacheMesh {
    pub fn new(config: CacheMeshConfig) -> Self {
        let config = normalize_config(config);
        Self {
            config,
            l2: Mutex::new(L2State::new()),
            remote: None,
            l1_metrics: AtomicTierMetrics::default(),
            l2_metrics: AtomicTierMetrics::default(),
            l3_metrics: AtomicTierMetrics::default(),
        }
    }

    pub fn with_remote(config: CacheMeshConfig, remote: Arc<dyn RemoteCacheBackend>) -> Self {
        let config = normalize_config(config);
        Self {
            config,
            l2: Mutex::new(L2State::new()),
            remote: Some(remote),
            l1_metrics: AtomicTierMetrics::default(),
            l2_metrics: AtomicTierMetrics::default(),
            l3_metrics: AtomicTierMetrics::default(),
        }
    }

    pub fn config(&self) -> &CacheMeshConfig {
        &self.config
    }

    pub fn record_l1_lookup(&self, hit: bool) {
        if hit {
            self.l1_metrics.hits.fetch_add(1, Ordering::Relaxed);
        } else {
            self.l1_metrics.misses.fetch_add(1, Ordering::Relaxed);
        }
    }

    pub fn offload_to_l2(&self, block: CacheMeshBlock) -> Result<()> {
        if !self.config.enabled {
            return Ok(());
        }
        block.validate()?;

        let mut state = self.l2.lock().unwrap();
        let block_bytes = block.bytes();
        if block_bytes > self.config.l2_capacity_bytes {
            drop(state);
            if self.config.l3_enabled {
                if let Err(err) = self.offload_to_l3(block) {
                    self.l3_metrics
                        .failed_offloads
                        .fetch_add(1, Ordering::Relaxed);
                    return Err(err);
                }
            } else {
                self.l2_metrics.dropped.fetch_add(1, Ordering::Relaxed);
            }
            return Ok(());
        }

        if self.config.write_through_l3 {
            if let Err(err) = self.offload_to_l3(block.clone()) {
                self.l3_metrics
                    .failed_offloads
                    .fetch_add(1, Ordering::Relaxed);
                return Err(err);
            }
        }

        if let Some(old) = state.blocks.remove(&block.key) {
            state.bytes = state.bytes.saturating_sub(old.bytes());
        }
        state.bytes = state.bytes.saturating_add(block_bytes);
        state.touch(&block.key);
        state.blocks.insert(block.key.clone(), block);
        self.l2_metrics.offloads.fetch_add(1, Ordering::Relaxed);

        while state.bytes > self.config.l2_capacity_bytes {
            let Some(victim_key) = state.lru.pop_front() else {
                break;
            };
            let Some(victim) = state.blocks.remove(&victim_key) else {
                continue;
            };
            state.bytes = state.bytes.saturating_sub(victim.bytes());
            self.l2_metrics.evictions.fetch_add(1, Ordering::Relaxed);
            drop(state);
            if self.config.l3_enabled {
                if let Err(err) = self.offload_to_l3(victim) {
                    self.l3_metrics
                        .failed_offloads
                        .fetch_add(1, Ordering::Relaxed);
                    return Err(err);
                }
            } else {
                self.l2_metrics.dropped.fetch_add(1, Ordering::Relaxed);
            }
            state = self.l2.lock().unwrap();
        }

        Ok(())
    }

    pub fn restore(&self, key: &CacheMeshKey) -> Result<Option<(CacheMeshTier, CacheMeshBlock)>> {
        if !self.config.enabled {
            return Ok(None);
        }

        {
            let mut state = self.l2.lock().unwrap();
            if let Some(block) = state.blocks.get(key).cloned() {
                state.touch(key);
                self.l2_metrics.hits.fetch_add(1, Ordering::Relaxed);
                self.l2_metrics.restores.fetch_add(1, Ordering::Relaxed);
                return Ok(Some((CacheMeshTier::L2, block)));
            }
        }
        self.l2_metrics.misses.fetch_add(1, Ordering::Relaxed);

        if self.config.l3_enabled {
            let remote = self.remote.as_ref().ok_or_else(|| {
                anyhow!("cachemesh L3 is enabled but no remote backend is configured")
            })?;
            if let Some(block) = remote.get(key)? {
                self.l3_metrics.hits.fetch_add(1, Ordering::Relaxed);
                self.l3_metrics.restores.fetch_add(1, Ordering::Relaxed);
                self.offload_to_l2(block.clone())?;
                return Ok(Some((CacheMeshTier::L3, block)));
            }
            self.l3_metrics.misses.fetch_add(1, Ordering::Relaxed);
        }

        Ok(None)
    }

    pub fn metrics(&self) -> CacheMeshMetrics {
        let state = self.l2.lock().unwrap();
        let l3_items = self.remote.as_ref().map(|remote| remote.len()).unwrap_or(0);
        let l3_bytes = self
            .remote
            .as_ref()
            .and_then(|remote| remote.bytes())
            .unwrap_or(0);
        CacheMeshMetrics {
            enabled: self.config.enabled,
            l1: self.l1_metrics.snapshot(0, 0),
            l2: self.l2_metrics.snapshot(state.bytes, state.blocks.len()),
            l3: self.l3_metrics.snapshot(l3_bytes, l3_items),
        }
    }

    pub fn snapshot(&self) -> CacheMeshSnapshot {
        let l2_keys = self.l2.lock().unwrap().blocks.len();
        CacheMeshSnapshot {
            metrics: self.metrics(),
            l2_keys,
        }
    }

    fn offload_to_l3(&self, block: CacheMeshBlock) -> Result<()> {
        if !self.config.l3_enabled {
            return Ok(());
        }
        let remote = self.remote.as_ref().ok_or_else(|| {
            anyhow!("cachemesh L3 is enabled but no remote backend is configured")
        })?;
        block.validate()?;
        remote.put(block)?;
        self.l3_metrics.offloads.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }
}

fn normalize_config(mut config: CacheMeshConfig) -> CacheMeshConfig {
    if config.namespace.trim().is_empty() {
        config.namespace = "default".to_string();
    }
    config
}

fn hex_digest(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        out.push(HEX[(byte >> 4) as usize] as char);
        out.push(HEX[(byte & 0x0f) as usize] as char);
    }
    out
}

fn sanitize_component(value: &str) -> String {
    let mut out = String::with_capacity(value.len().max(1));
    for ch in value.chars() {
        if ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_' | '.') {
            out.push(ch);
        } else {
            out.push('_');
        }
    }
    if out.is_empty() {
        "default".to_string()
    } else {
        out
    }
}

fn count_files_with_extension(root: &Path, extension: &str) -> Result<usize> {
    let mut count = 0usize;
    if !root.exists() {
        return Ok(count);
    }
    for entry in std::fs::read_dir(root)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            count += count_files_with_extension(&path, extension)?;
        } else if path.extension().and_then(|e| e.to_str()) == Some(extension) {
            count += 1;
        }
    }
    Ok(count)
}

fn sum_file_bytes_with_extension(root: &Path, extension: &str) -> Result<usize> {
    let mut bytes = 0usize;
    if !root.exists() {
        return Ok(bytes);
    }
    for entry in std::fs::read_dir(root)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            bytes += sum_file_bytes_with_extension(&path, extension)?;
        } else if path.extension().and_then(|e| e.to_str()) == Some(extension) {
            bytes += entry.metadata()?.len() as usize;
        }
    }
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn block(key: CacheMeshKey, fill: f32) -> CacheMeshBlock {
        CacheMeshBlock {
            key,
            block_size: 4,
            kv_dim: 2,
            keys: vec![fill; 8],
            values: vec![fill + 1.0; 8],
            num_tokens: 4,
        }
    }

    #[test]
    fn l2_restore_tracks_hit_rate() {
        let mesh = CacheMesh::new(CacheMeshConfig {
            enabled: true,
            l2_capacity_bytes: 1024,
            ..Default::default()
        });
        let key = CacheMeshKey::from_tokens("m", 0, &[1, 2, 3, 4]);
        mesh.offload_to_l2(block(key.clone(), 1.0)).unwrap();

        let restored = mesh.restore(&key).unwrap();
        assert!(matches!(restored, Some((CacheMeshTier::L2, _))));
        let metrics = mesh.metrics();
        assert_eq!(metrics.l2.hits, 1);
        assert_eq!(metrics.l2.misses, 0);
        assert_eq!(metrics.l2.hit_rate, 1.0);
    }

    #[test]
    fn l2_eviction_spills_to_l3() {
        let remote = Arc::new(InMemoryRemoteCache::new());
        let mesh = CacheMesh::with_remote(
            CacheMeshConfig {
                enabled: true,
                l2_capacity_bytes: 64,
                l3_enabled: true,
                ..Default::default()
            },
            remote,
        );
        let key1 = CacheMeshKey::from_tokens("m", 0, &[1, 2, 3, 4]);
        let key2 = CacheMeshKey::from_tokens("m", 0, &[5, 6, 7, 8]);
        mesh.offload_to_l2(block(key1.clone(), 1.0)).unwrap();
        mesh.offload_to_l2(block(key2, 2.0)).unwrap();

        let restored = mesh.restore(&key1).unwrap();
        assert!(matches!(restored, Some((CacheMeshTier::L3, _))));
        let metrics = mesh.metrics();
        assert!(metrics.l3.hits >= 1);
        assert!(metrics.l2.evictions >= 1);
    }

    #[test]
    fn oversized_block_without_l3_is_dropped() {
        let mesh = CacheMesh::new(CacheMeshConfig {
            enabled: true,
            l2_capacity_bytes: 1,
            l3_enabled: false,
            ..Default::default()
        });
        let key = CacheMeshKey::from_tokens("m", 0, &[1, 2, 3, 4]);
        mesh.offload_to_l2(block(key, 1.0)).unwrap();

        let metrics = mesh.metrics();
        assert_eq!(metrics.l2.items, 0);
        assert_eq!(metrics.l2.dropped, 1);
    }

    #[test]
    fn rejects_malformed_block_shapes() {
        let mesh = CacheMesh::new(CacheMeshConfig {
            enabled: true,
            ..Default::default()
        });
        let key = CacheMeshKey::from_tokens("m", 0, &[1, 2, 3, 4]);
        let mut malformed = block(key, 1.0);
        malformed.values.pop();

        assert!(mesh.offload_to_l2(malformed).is_err());
    }

    #[test]
    fn filesystem_remote_cache_round_trips_between_instances() {
        let tmp = tempfile::tempdir().unwrap();
        let remote_a = Arc::new(FileSystemRemoteCache::new(tmp.path()).unwrap());
        let mesh_a = CacheMesh::with_remote(
            CacheMeshConfig {
                enabled: true,
                l2_capacity_bytes: 1,
                l3_enabled: true,
                ..Default::default()
            },
            remote_a,
        );
        let key = CacheMeshKey::from_tokens("model/a", 0, &[1, 2, 3, 4]);
        mesh_a.offload_to_l2(block(key.clone(), 7.0)).unwrap();

        let remote_b = Arc::new(FileSystemRemoteCache::new(tmp.path()).unwrap());
        let mesh_b = CacheMesh::with_remote(
            CacheMeshConfig {
                enabled: true,
                l2_capacity_bytes: 1024,
                l3_enabled: true,
                ..Default::default()
            },
            remote_b,
        );
        let restored = mesh_b.restore(&key).unwrap();
        assert!(matches!(restored, Some((CacheMeshTier::L3, _))));
        let metrics = mesh_b.metrics();
        assert_eq!(metrics.l3.hits, 1);
        assert!(metrics.l3.bytes > 0);
    }
}
