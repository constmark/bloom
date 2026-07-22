//! Token Bucket Rate Limiter — Aegaeon §3
//!
//! Aegaeon uses per-model token buckets to prevent noisy neighbors:
//! - each model has an independent token bucket;
//! - `burst` is the maximum immediately consumable token count;
//! - `rate` is the number of tokens replenished per second; and
//! - requests beyond the burst capacity are queued instead of dropped.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};

fn default_burst() -> usize {
    32768
}

fn default_rate_per_second() -> f64 {
    200.0
}

/// Configuration for one token bucket.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TokenBucketConfig {
    /// Token-bucket capacity and burst limit.
    #[serde(default = "default_burst")]
    pub burst: usize,
    /// Tokens replenished per second.
    #[serde(default = "default_rate_per_second")]
    pub rate_per_second: f64,
}

impl Default for TokenBucketConfig {
    fn default() -> Self {
        Self {
            burst: default_burst(),
            rate_per_second: default_rate_per_second(),
        }
    }
}

/// Rate-limiter configuration.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct RateLimiterConfig {
    /// Whether rate limiting is enabled.
    #[serde(default)]
    pub enabled: bool,
    /// Default token-bucket configuration for all models.
    #[serde(default)]
    pub default_bucket: TokenBucketConfig,
    /// Per-model token-bucket overrides.
    #[serde(default)]
    pub per_model_overrides: HashMap<String, TokenBucketConfig>,
}

/// Runtime state for one token bucket.
#[derive(Debug, Clone)]
pub struct TokenBucket {
    /// Currently available tokens.
    tokens: f64,
    /// Token-bucket capacity.
    capacity: f64,
    /// Refill rate per second.
    rate_per_second: f64,
    /// Time of the most recent refill.
    last_refill: Instant,
}

impl TokenBucket {
    fn new(config: &TokenBucketConfig) -> Self {
        Self {
            tokens: config.burst as f64,
            capacity: config.burst as f64,
            rate_per_second: config.rate_per_second,
            last_refill: Instant::now(),
        }
    }

    /// Refill tokens according to elapsed time.
    fn refill(&mut self, now: Instant) {
        let elapsed = now.duration_since(self.last_refill);
        let new_tokens = elapsed.as_secs_f64() * self.rate_per_second;
        self.tokens = (self.tokens + new_tokens).min(self.capacity);
        self.last_refill = now;
    }

    /// Try to consume `tokens`; returns `true` on success.
    fn try_consume(&mut self, tokens: usize, now: Instant) -> bool {
        self.refill(now);
        let needed = tokens as f64;
        if self.tokens >= needed {
            self.tokens -= needed;
            true
        } else {
            false
        }
    }

    /// Return currently available tokens without triggering a refill.
    pub fn available_tokens(&self) -> f64 {
        self.tokens
    }

    /// Return the wait required for `tokens` to become available.
    pub fn wait_time_for(&self, tokens: usize) -> Duration {
        let deficit = tokens as f64 - self.tokens;
        if deficit <= 0.0 {
            Duration::ZERO
        } else {
            Duration::from_secs_f64(deficit / self.rate_per_second)
        }
    }
}

/// Rate-limit decision.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RateLimitDecision {
    /// Allow the request.
    Allowed,
    /// Rate-limit the request for the specified duration.
    Throttled { wait: Duration },
}

/// Per-model token-bucket rate limiter.
#[derive(Debug)]
pub struct TokenBucketRateLimiter {
    config: RateLimiterConfig,
    buckets: HashMap<String, TokenBucket>,
    /// Number of rate-limited requests.
    total_throttled: usize,
    /// Total number of requests.
    total_requests: usize,
}

impl TokenBucketRateLimiter {
    pub fn new(config: RateLimiterConfig) -> Self {
        Self {
            config,
            buckets: HashMap::new(),
            total_throttled: 0,
            total_requests: 0,
        }
    }

    pub fn config(&self) -> &RateLimiterConfig {
        &self.config
    }

    pub fn total_throttled(&self) -> usize {
        self.total_throttled
    }

    pub fn total_requests(&self) -> usize {
        self.total_requests
    }

    /// Get or create the token bucket for a model.
    fn get_or_create_bucket(&mut self, model_id: &str) -> &mut TokenBucket {
        let bucket_config = self
            .config
            .per_model_overrides
            .get(model_id)
            .cloned()
            .unwrap_or_else(|| self.config.default_bucket.clone());

        self.buckets
            .entry(model_id.to_string())
            .or_insert_with(|| TokenBucket::new(&bucket_config))
    }

    /// Try to consume `tokens` from a model's bucket.
    pub fn try_acquire(&mut self, model_id: &str, tokens: usize) -> RateLimitDecision {
        self.total_requests += 1;

        if !self.config.enabled {
            return RateLimitDecision::Allowed;
        }

        let now = Instant::now();
        let bucket_config = self
            .config
            .per_model_overrides
            .get(model_id)
            .cloned()
            .unwrap_or_else(|| self.config.default_bucket.clone());

        let bucket = self
            .buckets
            .entry(model_id.to_string())
            .or_insert_with(|| TokenBucket::new(&bucket_config));

        if bucket.try_consume(tokens, now) {
            RateLimitDecision::Allowed
        } else {
            let wait = bucket.wait_time_for(tokens);
            self.total_throttled += 1;
            RateLimitDecision::Throttled { wait }
        }
    }

    /// Query the currently available tokens for a model.
    pub fn available_tokens(&mut self, model_id: &str) -> f64 {
        if !self.config.enabled {
            return f64::MAX;
        }
        let now = Instant::now();
        let bucket = self.get_or_create_bucket(model_id);
        bucket.refill(now);
        bucket.available_tokens()
    }

    /// Reset a model's token bucket to full capacity.
    pub fn reset_bucket(&mut self, model_id: &str) {
        let bucket_config = self
            .config
            .per_model_overrides
            .get(model_id)
            .cloned()
            .unwrap_or_else(|| self.config.default_bucket.clone());
        self.buckets
            .insert(model_id.to_string(), TokenBucket::new(&bucket_config));
    }

    /// Snapshot available tokens for all registered models.
    pub fn snapshot(&self) -> HashMap<String, f64> {
        self.buckets
            .iter()
            .map(|(k, v)| (k.clone(), v.available_tokens()))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn rate_limiter_disabled_always_allows() {
        let config = RateLimiterConfig {
            enabled: false,
            ..Default::default()
        };
        let mut limiter = TokenBucketRateLimiter::new(config);
        for _ in 0..100 {
            assert_eq!(limiter.try_acquire("m1", 10000), RateLimitDecision::Allowed);
        }
    }

    #[test]
    fn rate_limiter_allows_within_burst() {
        let config = RateLimiterConfig {
            enabled: true,
            default_bucket: TokenBucketConfig {
                burst: 100,
                rate_per_second: 10.0,
            },
            ..Default::default()
        };
        let mut limiter = TokenBucketRateLimiter::new(config);

        // The first 100 tokens fit the burst capacity.
        assert_eq!(limiter.try_acquire("m1", 100), RateLimitDecision::Allowed);

        // Additional requests should be rate-limited.
        match limiter.try_acquire("m1", 10) {
            RateLimitDecision::Throttled { wait } => {
                assert!(wait > Duration::ZERO);
            }
            RateLimitDecision::Allowed => panic!("Expected throttled"),
        }
    }

    #[test]
    fn rate_limiter_refills_over_time() {
        let config = RateLimiterConfig {
            enabled: true,
            default_bucket: TokenBucketConfig {
                burst: 100,
                rate_per_second: 1000.0, // 1000 tokens/sec for fast test
            },
            ..Default::default()
        };
        let mut limiter = TokenBucketRateLimiter::new(config);

        // Consume all 100 tokens.
        assert_eq!(limiter.try_acquire("m1", 100), RateLimitDecision::Allowed);
        assert!(matches!(
            limiter.try_acquire("m1", 10),
            RateLimitDecision::Throttled { .. }
        ));

        // Wait for a refill.
        thread::sleep(Duration::from_millis(50)); // ~50 tokens refilled

        // A small consumption should now succeed.
        assert_eq!(limiter.try_acquire("m1", 20), RateLimitDecision::Allowed);
    }

    #[test]
    fn per_model_override() {
        let mut overrides = HashMap::new();
        overrides.insert(
            "premium-model".to_string(),
            TokenBucketConfig {
                burst: 1000,
                rate_per_second: 500.0,
            },
        );
        let config = RateLimiterConfig {
            enabled: true,
            default_bucket: TokenBucketConfig {
                burst: 10,
                rate_per_second: 5.0,
            },
            per_model_overrides: overrides,
        };
        let mut limiter = TokenBucketRateLimiter::new(config);

        // The default model has a burst capacity of 10.
        assert_eq!(limiter.try_acquire("basic", 10), RateLimitDecision::Allowed);
        assert!(matches!(
            limiter.try_acquire("basic", 1),
            RateLimitDecision::Throttled { .. }
        ));

        // The premium model has a burst capacity of 1,000.
        assert_eq!(
            limiter.try_acquire("premium-model", 500),
            RateLimitDecision::Allowed
        );
    }

    #[test]
    fn token_bucket_wait_time() {
        let bucket = TokenBucket {
            tokens: 50.0,
            capacity: 100.0,
            rate_per_second: 100.0,
            last_refill: Instant::now(),
        };

        // Requesting 100 with 50 available requires 50 / 100 = 0.5 seconds.
        let wait = bucket.wait_time_for(100);
        assert!((wait.as_secs_f64() - 0.5).abs() < 0.1);

        // Requesting 30 with 50 available requires no wait.
        let wait2 = bucket.wait_time_for(30);
        assert_eq!(wait2, Duration::ZERO);
    }

    #[test]
    fn rate_limiter_throttle_stats() {
        let config = RateLimiterConfig {
            enabled: true,
            default_bucket: TokenBucketConfig {
                burst: 5,
                rate_per_second: 1.0,
            },
            ..Default::default()
        };
        let mut limiter = TokenBucketRateLimiter::new(config);

        // 5 allowed + 3 throttled
        for _ in 0..5 {
            limiter.try_acquire("m1", 1);
        }
        for _ in 0..3 {
            limiter.try_acquire("m1", 1);
        }

        assert_eq!(limiter.total_requests(), 8);
        assert_eq!(limiter.total_throttled(), 3);
    }
}
