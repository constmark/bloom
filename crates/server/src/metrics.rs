//! Prometheus-compatible metrics collection for Bloom server.
//!
//! Provides counters, gauges, and histograms for monitoring inference
//! performance, KV cache utilization, and request lifecycle.

use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

const MAX_OBSERVATIONS: usize = 4096;

/// Global server metrics collector.
pub struct ServerMetrics {
    /// Total requests received.
    pub requests_total: AtomicU64,
    /// Total requests completed successfully.
    pub requests_completed: AtomicU64,
    /// Total requests that failed.
    pub requests_failed: AtomicU64,
    /// Total tokens generated across all requests.
    pub tokens_generated_total: AtomicU64,
    /// Total prompt tokens processed.
    pub prompt_tokens_total: AtomicU64,
    /// Current number of in-flight requests.
    pub in_flight_requests: AtomicU64,
    /// Request duration observations (for computing avg/p50/p99).
    durations: Mutex<Vec<f64>>,
    /// Inference latency observations (time in model forward only).
    inference_latencies: Mutex<Vec<f64>>,
    /// First token latency observations (TTFT in milliseconds).
    first_token_latencies: Mutex<Vec<f64>>,
    /// Inter-token latency observations (time between consecutive tokens in seconds).
    inter_token_latencies: Mutex<Vec<f64>>,
    /// Server start time.
    start_time: Instant,
}

impl ServerMetrics {
    pub fn new() -> Self {
        Self {
            requests_total: AtomicU64::new(0),
            requests_completed: AtomicU64::new(0),
            requests_failed: AtomicU64::new(0),
            tokens_generated_total: AtomicU64::new(0),
            prompt_tokens_total: AtomicU64::new(0),
            in_flight_requests: AtomicU64::new(0),
            durations: Mutex::new(Vec::new()),
            inference_latencies: Mutex::new(Vec::new()),
            first_token_latencies: Mutex::new(Vec::new()),
            inter_token_latencies: Mutex::new(Vec::new()),
            start_time: Instant::now(),
        }
    }

    /// Record a new request start.
    pub fn record_request_start(&self) {
        self.requests_total.fetch_add(1, Ordering::Relaxed);
        self.in_flight_requests.fetch_add(1, Ordering::AcqRel);
    }

    /// Whole seconds elapsed since this metrics collector was created.
    pub fn uptime_seconds(&self) -> u64 {
        self.start_time.elapsed().as_secs()
    }

    /// Record a request completion.
    pub fn record_request_end(
        &self,
        success: bool,
        duration_secs: f64,
        tokens_generated: u64,
        prompt_tokens: u64,
    ) {
        self.in_flight_requests.fetch_sub(1, Ordering::Release);
        if success {
            self.requests_completed.fetch_add(1, Ordering::Relaxed);
        } else {
            self.requests_failed.fetch_add(1, Ordering::Relaxed);
        }
        self.tokens_generated_total
            .fetch_add(tokens_generated, Ordering::Relaxed);
        self.prompt_tokens_total
            .fetch_add(prompt_tokens, Ordering::Relaxed);
        if let Ok(mut durations) = self.durations.lock() {
            Self::push_observation(&mut durations, duration_secs);
        }
    }

    /// Record an inference latency observation.
    pub fn record_inference_latency(&self, secs: f64) {
        if let Ok(mut latencies) = self.inference_latencies.lock() {
            Self::push_observation(&mut latencies, secs);
        }
    }

    /// Record a first-token latency (TTFT) observation in milliseconds.
    pub fn record_first_token_latency(&self, ms: f64) {
        if let Ok(mut latencies) = self.first_token_latencies.lock() {
            Self::push_observation(&mut latencies, ms);
        }
    }

    /// Record an inter-token latency observation.
    pub fn record_inter_token_latency(&self, secs: f64) {
        if let Ok(mut latencies) = self.inter_token_latencies.lock() {
            Self::push_observation(&mut latencies, secs);
        }
    }

    fn push_observation(observations: &mut Vec<f64>, value: f64) {
        if observations.len() >= MAX_OBSERVATIONS {
            let overflow = observations.len() + 1 - MAX_OBSERVATIONS;
            observations.drain(0..overflow);
        }
        observations.push(value);
    }

    /// Render metrics in Prometheus text format.
    pub fn render_prometheus(
        &self,
        kv_cache_metrics: &bloomai_engine::KvCacheMetrics,
        cachemesh_metrics: Option<&bloomai_engine::CacheMeshMetrics>,
        queue_stats: (usize, usize, usize),
    ) -> String {
        let mut out = String::new();
        let uptime = self.start_time.elapsed().as_secs_f64();

        // Counters
        out.push_str("# HELP bloom_requests_total Total number of requests received.\n");
        out.push_str("# TYPE bloom_requests_total counter\n");
        out.push_str(&format!(
            "bloom_requests_total {}\n",
            self.requests_total.load(Ordering::Relaxed)
        ));

        out.push_str("# HELP bloom_requests_completed_total Total completed requests.\n");
        out.push_str("# TYPE bloom_requests_completed_total counter\n");
        out.push_str(&format!(
            "bloom_requests_completed_total {}\n",
            self.requests_completed.load(Ordering::Relaxed)
        ));

        out.push_str("# HELP bloom_requests_failed_total Total failed requests.\n");
        out.push_str("# TYPE bloom_requests_failed_total counter\n");
        out.push_str(&format!(
            "bloom_requests_failed_total {}\n",
            self.requests_failed.load(Ordering::Relaxed)
        ));

        out.push_str("# HELP bloom_tokens_generated_total Total tokens generated.\n");
        out.push_str("# TYPE bloom_tokens_generated_total counter\n");
        out.push_str(&format!(
            "bloom_tokens_generated_total {}\n",
            self.tokens_generated_total.load(Ordering::Relaxed)
        ));

        out.push_str("# HELP bloom_prompt_tokens_total Total prompt tokens processed.\n");
        out.push_str("# TYPE bloom_prompt_tokens_total counter\n");
        out.push_str(&format!(
            "bloom_prompt_tokens_total {}\n",
            self.prompt_tokens_total.load(Ordering::Relaxed)
        ));

        // Gauges
        out.push_str("# HELP bloom_in_flight_requests Current in-flight requests.\n");
        out.push_str("# TYPE bloom_in_flight_requests gauge\n");
        out.push_str(&format!(
            "bloom_in_flight_requests {}\n",
            self.in_flight_requests.load(Ordering::Relaxed)
        ));

        out.push_str("# HELP bloom_uptime_seconds Server uptime in seconds.\n");
        out.push_str("# TYPE bloom_uptime_seconds gauge\n");
        out.push_str(&format!("bloom_uptime_seconds {:.1}\n", uptime));

        // KV cache metrics
        out.push_str("# HELP bloom_kv_cache_total_blocks Total KV cache blocks.\n");
        out.push_str("# TYPE bloom_kv_cache_total_blocks gauge\n");
        out.push_str(&format!(
            "bloom_kv_cache_total_blocks {}\n",
            kv_cache_metrics.total_blocks
        ));

        out.push_str("# HELP bloom_kv_cache_free_blocks Free KV cache blocks.\n");
        out.push_str("# TYPE bloom_kv_cache_free_blocks gauge\n");
        out.push_str(&format!(
            "bloom_kv_cache_free_blocks {}\n",
            kv_cache_metrics.free_blocks
        ));

        out.push_str("# HELP bloom_kv_cache_active_blocks Active KV cache blocks.\n");
        out.push_str("# TYPE bloom_kv_cache_active_blocks gauge\n");
        out.push_str(&format!(
            "bloom_kv_cache_active_blocks {}\n",
            kv_cache_metrics.active_blocks
        ));

        out.push_str("# HELP bloom_kv_cache_hits_total KV cache prefix hits.\n");
        out.push_str("# TYPE bloom_kv_cache_hits_total counter\n");
        out.push_str(&format!(
            "bloom_kv_cache_hits_total {}\n",
            kv_cache_metrics.hits
        ));

        out.push_str("# HELP bloom_kv_cache_misses_total KV cache prefix misses.\n");
        out.push_str("# TYPE bloom_kv_cache_misses_total counter\n");
        out.push_str(&format!(
            "bloom_kv_cache_misses_total {}\n",
            kv_cache_metrics.misses
        ));

        out.push_str("# HELP bloom_kv_cache_evictions_total KV cache evictions.\n");
        out.push_str("# TYPE bloom_kv_cache_evictions_total counter\n");
        out.push_str(&format!(
            "bloom_kv_cache_evictions_total {}\n",
            kv_cache_metrics.evictions
        ));

        // Utilization gauge
        let utilization = if kv_cache_metrics.total_blocks > 0 {
            (kv_cache_metrics.total_blocks - kv_cache_metrics.free_blocks) as f64
                / kv_cache_metrics.total_blocks as f64
        } else {
            0.0
        };
        out.push_str("# HELP bloom_kv_cache_utilization KV cache utilization ratio.\n");
        out.push_str("# TYPE bloom_kv_cache_utilization gauge\n");
        out.push_str(&format!("bloom_kv_cache_utilization {:.4}\n", utilization));

        if let Some(mesh) = cachemesh_metrics {
            for (tier, metrics) in [("l1", &mesh.l1), ("l2", &mesh.l2), ("l3", &mesh.l3)] {
                out.push_str("# HELP bloom_cachemesh_hits_total CacheMesh hits by tier.\n");
                out.push_str("# TYPE bloom_cachemesh_hits_total counter\n");
                out.push_str(&format!(
                    "bloom_cachemesh_hits_total{{tier=\"{}\"}} {}\n",
                    tier, metrics.hits
                ));
                out.push_str("# HELP bloom_cachemesh_misses_total CacheMesh misses by tier.\n");
                out.push_str("# TYPE bloom_cachemesh_misses_total counter\n");
                out.push_str(&format!(
                    "bloom_cachemesh_misses_total{{tier=\"{}\"}} {}\n",
                    tier, metrics.misses
                ));
                out.push_str("# HELP bloom_cachemesh_hit_rate CacheMesh hit rate by tier.\n");
                out.push_str("# TYPE bloom_cachemesh_hit_rate gauge\n");
                out.push_str(&format!(
                    "bloom_cachemesh_hit_rate{{tier=\"{}\"}} {:.4}\n",
                    tier, metrics.hit_rate
                ));
                out.push_str("# HELP bloom_cachemesh_items CacheMesh items by tier.\n");
                out.push_str("# TYPE bloom_cachemesh_items gauge\n");
                out.push_str(&format!(
                    "bloom_cachemesh_items{{tier=\"{}\"}} {}\n",
                    tier, metrics.items
                ));
                out.push_str(
                    "# HELP bloom_cachemesh_failed_offloads_total CacheMesh failed offloads by tier.\n",
                );
                out.push_str("# TYPE bloom_cachemesh_failed_offloads_total counter\n");
                out.push_str(&format!(
                    "bloom_cachemesh_failed_offloads_total{{tier=\"{}\"}} {}\n",
                    tier, metrics.failed_offloads
                ));
                out.push_str(
                    "# HELP bloom_cachemesh_dropped_total CacheMesh dropped blocks by tier.\n",
                );
                out.push_str("# TYPE bloom_cachemesh_dropped_total counter\n");
                out.push_str(&format!(
                    "bloom_cachemesh_dropped_total{{tier=\"{}\"}} {}\n",
                    tier, metrics.dropped
                ));
            }
        }

        // Scheduler queue depths
        out.push_str("# HELP bloom_scheduler_prefill_queue Prefill queue depth.\n");
        out.push_str("# TYPE bloom_scheduler_prefill_queue gauge\n");
        out.push_str(&format!(
            "bloom_scheduler_prefill_queue {}\n",
            queue_stats.0
        ));

        out.push_str("# HELP bloom_scheduler_decoding_queue Decoding queue depth.\n");
        out.push_str("# TYPE bloom_scheduler_decoding_queue gauge\n");
        out.push_str(&format!(
            "bloom_scheduler_decoding_queue {}\n",
            queue_stats.1
        ));

        out.push_str("# HELP bloom_scheduler_active_requests Active scheduler requests.\n");
        out.push_str("# TYPE bloom_scheduler_active_requests gauge\n");
        out.push_str(&format!(
            "bloom_scheduler_active_requests {}\n",
            queue_stats.2
        ));

        // Duration statistics
        if let Ok(durations) = self.durations.lock() {
            render_observation_stats(
                &mut out,
                &durations,
                "bloom_request_duration_seconds",
                "request duration",
                "{:.6}",
            );
        }

        // Inference latency statistics
        if let Ok(latencies) = self.inference_latencies.lock() {
            render_observation_stats(
                &mut out,
                &latencies,
                "bloom_inference_latency_seconds",
                "inference latency",
                "{:.6}",
            );
        }

        // First token latency (TTFT) statistics
        if let Ok(ftl) = self.first_token_latencies.lock() {
            render_observation_stats(
                &mut out,
                &ftl,
                "bloom_first_token_latency_ms",
                "time to first token",
                "{:.3}",
            );
        }

        // Inter-token latency statistics
        if let Ok(itl) = self.inter_token_latencies.lock() {
            render_observation_stats(
                &mut out,
                &itl,
                "bloom_inter_token_latency_seconds",
                "inter-token latency",
                "{:.6}",
            );
        }

        // System / Process memory metrics
        let mut memory = bloomai_engine::MemoryTelemetry::new();
        memory.refresh_ram();
        let memory_pressure = if memory.is_high_pressure() { 1.0 } else { 0.0 };

        out.push_str("# HELP bloom_system_memory_total_bytes Total system memory in bytes.\n");
        out.push_str("# TYPE bloom_system_memory_total_bytes gauge\n");
        out.push_str(&format!(
            "bloom_system_memory_total_bytes {}\n",
            memory.total_ram
        ));

        out.push_str("# HELP bloom_system_memory_used_bytes Used process memory (RSS) in bytes.\n");
        out.push_str("# TYPE bloom_system_memory_used_bytes gauge\n");
        out.push_str(&format!(
            "bloom_system_memory_used_bytes {}\n",
            memory.used_ram
        ));

        out.push_str("# HELP bloom_system_memory_peak_bytes Peak process memory (RSS) in bytes.\n");
        out.push_str("# TYPE bloom_system_memory_peak_bytes gauge\n");
        out.push_str(&format!(
            "bloom_system_memory_peak_bytes {}\n",
            memory.peak_ram
        ));

        out.push_str("# HELP bloom_system_memory_pressure_high Whether memory pressure is high (>85% utilization on either RAM or VRAM).\n");
        out.push_str("# TYPE bloom_system_memory_pressure_high gauge\n");
        out.push_str(&format!(
            "bloom_system_memory_pressure_high {}\n",
            memory_pressure
        ));

        // Resource Coordinator metrics
        let res_snapshot = bloomai_core::global_resource_coordinator().snapshot();

        out.push_str("# HELP bloom_resource_ram_budget_bytes RAM budget allocated in resource coordinator.\n");
        out.push_str("# TYPE bloom_resource_ram_budget_bytes gauge\n");
        out.push_str(&format!(
            "bloom_resource_ram_budget_bytes {}\n",
            res_snapshot.ram_budget
        ));

        out.push_str("# HELP bloom_resource_vram_budget_bytes VRAM budget allocated in resource coordinator.\n");
        out.push_str("# TYPE bloom_resource_vram_budget_bytes gauge\n");
        out.push_str(&format!(
            "bloom_resource_vram_budget_bytes {}\n",
            res_snapshot.vram_budget
        ));

        out.push_str(
            "# HELP bloom_resource_ram_allocated_bytes RAM allocated in resource coordinator.\n",
        );
        out.push_str("# TYPE bloom_resource_ram_allocated_bytes gauge\n");
        out.push_str(&format!(
            "bloom_resource_ram_allocated_bytes {}\n",
            res_snapshot.ram_allocated
        ));

        out.push_str(
            "# HELP bloom_resource_vram_allocated_bytes VRAM allocated in resource coordinator.\n",
        );
        out.push_str("# TYPE bloom_resource_vram_allocated_bytes gauge\n");
        out.push_str(&format!(
            "bloom_resource_vram_allocated_bytes {}\n",
            res_snapshot.vram_allocated
        ));

        out.push_str("# HELP bloom_resource_model_count Number of resident models in resource coordinator.\n");
        out.push_str("# TYPE bloom_resource_model_count gauge\n");
        out.push_str(&format!(
            "bloom_resource_model_count {}\n",
            res_snapshot.model_count
        ));

        out.push_str("# HELP bloom_resource_cache_count Number of active cache entries in resource coordinator.\n");
        out.push_str("# TYPE bloom_resource_cache_count gauge\n");
        out.push_str(&format!(
            "bloom_resource_cache_count {}\n",
            res_snapshot.cache_count
        ));

        out.push_str(
            "# HELP bloom_resource_lease_count Number of active leases in resource coordinator.\n",
        );
        out.push_str("# TYPE bloom_resource_lease_count gauge\n");
        out.push_str(&format!(
            "bloom_resource_lease_count {}\n",
            res_snapshot.lease_count
        ));

        out
    }
}

fn render_observation_stats(
    out: &mut String,
    observations: &[f64],
    metric: &str,
    description: &str,
    _format: &str,
) {
    if observations.is_empty() {
        return;
    }
    let sum: f64 = observations.iter().sum();
    let avg = sum / observations.len() as f64;
    let mut sorted = observations.to_vec();
    sorted.sort_by(f64::total_cmp);
    let p50 = sorted[sorted.len() / 2];
    let p99_idx = ((sorted.len() - 1) as f64 * 0.99).round() as usize;
    let p99 = sorted[p99_idx.min(sorted.len() - 1)];

    out.push_str(&format!(
        "# HELP {metric}_avg Average {description}.\n# TYPE {metric}_avg gauge\n{metric}_avg {avg:.6}\n"
    ));
    out.push_str(&format!(
        "# HELP {metric}_p50 P50 {description}.\n# TYPE {metric}_p50 gauge\n{metric}_p50 {p50:.6}\n"
    ));
    out.push_str(&format!(
        "# HELP {metric}_p99 P99 {description}.\n# TYPE {metric}_p99 gauge\n{metric}_p99 {p99:.6}\n"
    ));
}

impl Default for ServerMetrics {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bloomai_engine::KvCacheMetrics;

    #[test]
    fn test_metrics_request_lifecycle() {
        let m = ServerMetrics::new();
        m.record_request_start();
        m.record_request_start();
        assert_eq!(m.requests_total.load(Ordering::Relaxed), 2);
        assert_eq!(m.in_flight_requests.load(Ordering::Relaxed), 2);

        m.record_request_end(true, 1.5, 100, 50);
        assert_eq!(m.requests_completed.load(Ordering::Relaxed), 1);
        assert_eq!(m.in_flight_requests.load(Ordering::Relaxed), 1);
        assert_eq!(m.tokens_generated_total.load(Ordering::Relaxed), 100);

        m.record_request_end(false, 0.5, 0, 10);
        assert_eq!(m.requests_failed.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn test_prometheus_render() {
        let m = ServerMetrics::new();
        m.record_request_start();
        m.record_request_end(true, 0.5, 50, 20);

        let kv = KvCacheMetrics {
            total_blocks: 100,
            free_blocks: 80,
            active_blocks: 10,
            cached_blocks: 10,
            hits: 5,
            misses: 3,
            evictions: 1,
            reuses: 10,
        };

        let output = m.render_prometheus(&kv, None, (0, 1, 2));
        assert!(output.contains("bloom_requests_total 1"));
        assert!(output.contains("bloom_kv_cache_total_blocks 100"));
        assert!(output.contains("bloom_kv_cache_utilization 0.2000"));
        assert!(output.contains("bloom_scheduler_decoding_queue 1"));
        assert!(output.contains("bloom_request_duration_seconds_avg"));
        assert!(output.contains("bloom_system_memory_used_bytes"));
        assert!(output.contains("bloom_resource_ram_budget_bytes"));
    }

    #[test]
    fn test_inference_latency() {
        let m = ServerMetrics::new();
        m.record_inference_latency(0.01);
        m.record_inference_latency(0.02);
        let latencies = m.inference_latencies.lock().unwrap();
        assert_eq!(latencies.len(), 2);
    }

    #[test]
    fn test_observation_window_is_bounded() {
        let m = ServerMetrics::new();
        for idx in 0..(MAX_OBSERVATIONS + 10) {
            m.record_inference_latency(idx as f64);
        }
        let latencies = m.inference_latencies.lock().unwrap();
        assert_eq!(latencies.len(), MAX_OBSERVATIONS);
        assert_eq!(latencies[0], 10.0);
    }

    #[test]
    fn test_inter_token_latency() {
        let m = ServerMetrics::new();
        m.record_inter_token_latency(0.05);
        m.record_inter_token_latency(0.08);
        let latencies = m.inter_token_latencies.lock().unwrap();
        assert_eq!(latencies.len(), 2);
        assert_eq!(latencies[0], 0.05);
        assert_eq!(latencies[1], 0.08);
    }
}
