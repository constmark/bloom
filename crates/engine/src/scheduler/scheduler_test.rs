#[cfg(test)]
mod tests {
    use crate::io::DataBlock;
    use crate::scheduler::*;
    use anyhow::Result;
    use bloomai_core::{
        BloomError, DeviceCapability, DeviceClass, GenerationParams, MemoryTopology, PowerState,
        ResourcePriority, ResourceTicket, ThermalState,
    };
    use std::sync::Arc;

    struct MockExecutor;
    impl EngineExecutor for MockExecutor {
        fn execute(&self, batch: ExecutionBatch) -> Result<BatchResult> {
            Ok(BatchResult {
                next_tokens: vec![42; batch.request_ids.len()],
                speculative_tokens: None,
            })
        }
        fn max_batch_size(&self, _phase: ExecutionPhase) -> usize {
            4
        }
    }

    struct MockKvPool {
        free_slots: Mutex<VecDeque<usize>>,
        capacity: usize,
    }
    impl MockKvPool {
        fn new(size: usize) -> Self {
            let mut free_slots = VecDeque::new();
            for i in 0..size {
                free_slots.push_back(i);
            }
            Self {
                free_slots: Mutex::new(free_slots),
                capacity: size,
            }
        }
    }
    impl KvCachePool for MockKvPool {
        fn allocate(&self, _tokens: usize) -> Result<usize> {
            self.free_slots
                .lock()
                .unwrap()
                .pop_front()
                .ok_or_else(|| BloomError::SchedulingFailed("KV cache pool full".into()).into())
        }
        fn free(&self, handle: usize) {
            self.free_slots.lock().unwrap_or_else(|e| e.into_inner()).push_back(handle);
        }
        fn allocate_paged(
            &self,
            _request_id: &str,
            _prompt_tokens: &[u32],
            max_new_tokens: usize,
            _multimodal_hash: Option<&str>,
        ) -> Result<KvCacheAllocation> {
            let handle = self.allocate(max_new_tokens)?;
            Ok(KvCacheAllocation {
                handle,
                matched_tokens: 0,
                allocated_blocks: vec![handle],
            })
        }
        fn free_paged(&self, _request_id: &str) {
            // no-op for mock
        }
        fn get_metrics(&self) -> KvCacheMetrics {
            let free = self.free_slots.lock().unwrap_or_else(|e| e.into_inner()).len();
            KvCacheMetrics {
                total_blocks: self.capacity,
                free_blocks: free,
                active_blocks: self.capacity.saturating_sub(free),
                ..Default::default()
            }
        }
    }

    #[test]
    fn test_inference_scheduler_basic() {
        let executor = Arc::new(MockExecutor);
        let kv_pool = Arc::new(MockKvPool::new(10));
        let scheduler = InferenceScheduler::new(executor, kv_pool);

        let req1 = Request {
            id: "req1".to_string(),
            model_id: "m1".to_string(),
            prompt_tokens: vec![1, 2, 3],
            generated_tokens: Vec::new(),
            params: GenerationParams {
                max_tokens: 5,
                ..Default::default()
            },
            state: RequestState::Pending,
            priority: 1,
            kv_handle: None,
            created_at: std::time::Instant::now(),
            last_accessed: std::time::Instant::now(),
            preemption_count: 0,
            decode_started_at: None,
            last_scheduled_at: None,
            multimodal_hash: None,
        };

        scheduler.submit(req1).unwrap();

        // First step runs prefill and emits the first sampled token.
        scheduler.step().unwrap();

        {
            let decoding = scheduler.decoding_queue.lock().unwrap_or_else(|e| e.into_inner());
            assert_eq!(decoding.len(), 1);
            assert_eq!(decoding[0].id, "req1");
            assert_eq!(decoding[0].generated_tokens, vec![42]);
        }

        // Second step should do next decoding for req1
        scheduler.step().unwrap();
        {
            let decoding = scheduler.decoding_queue.lock().unwrap_or_else(|e| e.into_inner());
            assert_eq!(decoding[0].generated_tokens, vec![42, 42]);
        }
    }

    #[test]
    fn test_inference_scheduler_continuous_batching_admits_new_prefill_during_decode() {
        let executor = Arc::new(MockExecutor);
        let kv_pool = Arc::new(MockKvPool::new(10));
        let scheduler = InferenceScheduler::new(executor, kv_pool);

        let req1 = Request {
            id: "req1".to_string(),
            model_id: "m1".to_string(),
            prompt_tokens: vec![1, 2, 3],
            generated_tokens: Vec::new(),
            params: GenerationParams {
                max_tokens: 5,
                ..Default::default()
            },
            state: RequestState::Pending,
            priority: 1,
            kv_handle: None,
            created_at: std::time::Instant::now(),
            last_accessed: std::time::Instant::now(),
            preemption_count: 0,
            decode_started_at: None,
            last_scheduled_at: None,
            multimodal_hash: None,
        };

        scheduler.submit(req1).unwrap();
        scheduler.step().unwrap();

        let req2 = Request {
            id: "req2".to_string(),
            model_id: "m1".to_string(),
            prompt_tokens: vec![4, 5, 6],
            generated_tokens: Vec::new(),
            params: GenerationParams {
                max_tokens: 5,
                ..Default::default()
            },
            state: RequestState::Pending,
            priority: 1,
            kv_handle: None,
            created_at: std::time::Instant::now(),
            last_accessed: std::time::Instant::now(),
            preemption_count: 0,
            decode_started_at: None,
            last_scheduled_at: None,
            multimodal_hash: None,
        };

        scheduler.submit(req2).unwrap();
        scheduler.step().unwrap();

        let decoding = scheduler.decoding_queue.lock().unwrap_or_else(|e| e.into_inner());
        assert_eq!(decoding.len(), 2);
        let req1 = decoding.iter().find(|r| r.id == "req1").unwrap();
        let req2 = decoding.iter().find(|r| r.id == "req2").unwrap();
        assert_eq!(req1.generated_tokens.len(), 2);
        assert_eq!(req2.generated_tokens.len(), 1);
    }

    #[test]
    fn test_inference_scheduler_prioritizes_decode_when_budget_is_tight() {
        let executor = Arc::new(MockExecutor);
        let kv_pool = Arc::new(MockKvPool::new(10));
        let mut config = bloomai_core::TokenSchedulingConfig::default();
        config.max_total_tokens_per_step = 1;
        config.max_prefill_tokens_per_step = 4;
        config.max_decode_tokens_per_step = 1;

        let scheduler = InferenceScheduler::with_config(executor, kv_pool, config);

        let req1 = Request {
            id: "req1".to_string(),
            model_id: "m1".to_string(),
            prompt_tokens: vec![1],
            generated_tokens: Vec::new(),
            params: GenerationParams {
                max_tokens: 5,
                ..Default::default()
            },
            state: RequestState::Pending,
            priority: 1,
            kv_handle: None,
            created_at: std::time::Instant::now(),
            last_accessed: std::time::Instant::now(),
            preemption_count: 0,
            decode_started_at: None,
            last_scheduled_at: None,
            multimodal_hash: None,
        };
        scheduler.submit(req1).unwrap();
        scheduler.step().unwrap();

        let req2 = Request {
            id: "req2".to_string(),
            model_id: "m1".to_string(),
            prompt_tokens: vec![2],
            generated_tokens: Vec::new(),
            params: GenerationParams {
                max_tokens: 5,
                ..Default::default()
            },
            state: RequestState::Pending,
            priority: 1,
            kv_handle: None,
            created_at: std::time::Instant::now(),
            last_accessed: std::time::Instant::now(),
            preemption_count: 0,
            decode_started_at: None,
            last_scheduled_at: None,
            multimodal_hash: None,
        };
        scheduler.submit(req2).unwrap();

        scheduler.step().unwrap();

        {
            let decoding = scheduler.decoding_queue.lock().unwrap_or_else(|e| e.into_inner());
            assert_eq!(decoding.len(), 1);
            assert_eq!(decoding[0].id, "req1");
            assert_eq!(decoding[0].generated_tokens.len(), 2);
        }
        {
            let prefill = scheduler.prefill_queue.lock().unwrap_or_else(|e| e.into_inner());
            assert_eq!(prefill.len(), 1);
            assert_eq!(prefill[0].id, "req2");
        }
    }

    #[test]
    fn test_inference_scheduler_continuous_decode_quantum() {
        let executor = Arc::new(MockExecutor);
        let kv_pool = Arc::new(MockKvPool::new(10));
        let mut config = bloomai_core::TokenSchedulingConfig::default();
        config.decode_quantum_tokens = 2;
        config.max_decode_tokens_per_step = 2;
        config.max_total_tokens_per_step = 4;

        let scheduler = InferenceScheduler::with_config(executor, kv_pool, config);

        let req = Request {
            id: "req1".to_string(),
            model_id: "m1".to_string(),
            prompt_tokens: vec![1, 2],
            generated_tokens: Vec::new(),
            params: GenerationParams {
                max_tokens: 5,
                ..Default::default()
            },
            state: RequestState::Pending,
            priority: 1,
            kv_handle: None,
            created_at: std::time::Instant::now(),
            last_accessed: std::time::Instant::now(),
            preemption_count: 0,
            decode_started_at: None,
            last_scheduled_at: None,
            multimodal_hash: None,
        };

        scheduler.submit(req).unwrap();
        scheduler.step().unwrap();
        scheduler.step().unwrap();

        let decoding = scheduler.decoding_queue.lock().unwrap_or_else(|e| e.into_inner());
        assert_eq!(decoding.len(), 1);
        assert_eq!(decoding[0].generated_tokens.len(), 3);
    }

    #[test]
    fn test_inference_scheduler_rate_limiter() {
        let executor = Arc::new(MockExecutor);
        let kv_pool = Arc::new(MockKvPool::new(10));
        let mut config = bloomai_core::TokenSchedulingConfig::default();
        config.rate_limiter.enabled = true;
        config.rate_limiter.default_bucket.burst = 5; // only 5 tokens max burst
        config.rate_limiter.default_bucket.rate_per_second = 1.0;

        let scheduler = InferenceScheduler::with_config(executor, kv_pool, config);

        // req1 prompt size = 10 > burst. It will be throttled.
        let req1 = Request {
            id: "req1".to_string(),
            model_id: "m1".to_string(),
            prompt_tokens: vec![0; 10],
            generated_tokens: Vec::new(),
            params: GenerationParams {
                max_tokens: 5,
                ..Default::default()
            },
            state: RequestState::Pending,
            priority: 1,
            kv_handle: None,
            created_at: std::time::Instant::now(),
            last_accessed: std::time::Instant::now(),
            preemption_count: 0,
            decode_started_at: None,
            last_scheduled_at: None,
            multimodal_hash: None,
        };

        // req2 prompt size = 3 <= burst. It will be allowed.
        let req2 = Request {
            id: "req2".to_string(),
            model_id: "m2".to_string(),
            prompt_tokens: vec![0; 3],
            generated_tokens: Vec::new(),
            params: GenerationParams {
                max_tokens: 5,
                ..Default::default()
            },
            state: RequestState::Pending,
            priority: 1,
            kv_handle: None,
            created_at: std::time::Instant::now(),
            last_accessed: std::time::Instant::now(),
            preemption_count: 0,
            decode_started_at: None,
            last_scheduled_at: None,
            multimodal_hash: None,
        };

        scheduler.submit(req1).unwrap();
        scheduler.submit(req2).unwrap();

        // Step once. Since req1 is throttled, the scheduler should skip it (avoid HOL blocking)
        // and schedule req2!
        scheduler.step().unwrap();

        {
            let decoding = scheduler.decoding_queue.lock().unwrap_or_else(|e| e.into_inner());
            assert_eq!(decoding.len(), 1);
            assert_eq!(decoding[0].id, "req2"); // req2 was scheduled successfully!
        }
        {
            let prefill = scheduler.prefill_queue.lock().unwrap_or_else(|e| e.into_inner());
            assert_eq!(prefill.len(), 1);
            assert_eq!(prefill[0].id, "req1"); // req1 is still pending in prefill queue
        }
    }

    #[test]
    fn test_inference_scheduler_chunked_prefill() {
        let executor = Arc::new(MockExecutor);
        let kv_pool = Arc::new(MockKvPool::new(10));
        let mut config = bloomai_core::TokenSchedulingConfig::default();
        config.chunked_prefill.enabled = true;
        config.chunked_prefill.chunk_size = 4;
        config.max_total_tokens_per_step = 4; // limit step to 4 tokens

        let scheduler = InferenceScheduler::with_config(executor, kv_pool, config);

        let req = Request {
            id: "req1".to_string(),
            model_id: "m1".to_string(),
            prompt_tokens: vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10], // 10 tokens
            generated_tokens: Vec::new(),
            params: GenerationParams {
                max_tokens: 5,
                ..Default::default()
            },
            state: RequestState::Pending,
            priority: 1,
            kv_handle: None,
            created_at: std::time::Instant::now(),
            last_accessed: std::time::Instant::now(),
            preemption_count: 0,
            decode_started_at: None,
            last_scheduled_at: None,
            multimodal_hash: None,
        };

        scheduler.submit(req).unwrap();

        // Step 1: schedules first chunk of 4 tokens.
        scheduler.step().unwrap();
        {
            let cp_queue = scheduler.chunked_prefill_queue.lock().unwrap_or_else(|e| e.into_inner());
            assert_eq!(cp_queue.queue[0].filled_tokens, 4);
            let active = scheduler.active_requests.lock().unwrap_or_else(|e| e.into_inner());
            assert_eq!(active["req1"].state, RequestState::Prefill);
        }

        // Step 2: schedules second chunk of 4 tokens.
        scheduler.step().unwrap();
        {
            let cp_queue = scheduler.chunked_prefill_queue.lock().unwrap_or_else(|e| e.into_inner());
            assert_eq!(cp_queue.queue[0].filled_tokens, 8);
        }

        // Step 3: schedules final chunk of 2 tokens -> transitions to decoding.
        scheduler.step().unwrap();
        {
            let decoding = scheduler.decoding_queue.lock().unwrap_or_else(|e| e.into_inner());
            assert_eq!(decoding.len(), 1);
            assert_eq!(decoding[0].id, "req1");
            assert_eq!(decoding[0].generated_tokens.len(), 1);
        }
    }

    #[test]
    fn test_inference_scheduler_chunked_prefill_splits_to_token_budget() {
        let executor = Arc::new(MockExecutor);
        let kv_pool = Arc::new(MockKvPool::new(10));
        let mut config = bloomai_core::TokenSchedulingConfig::default();
        config.chunked_prefill.enabled = true;
        config.chunked_prefill.chunk_size = 8;
        config.max_total_tokens_per_step = 4;

        let scheduler = InferenceScheduler::with_config(executor, kv_pool, config);

        let req = Request {
            id: "req1".to_string(),
            model_id: "m1".to_string(),
            prompt_tokens: vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            generated_tokens: Vec::new(),
            params: GenerationParams {
                max_tokens: 5,
                ..Default::default()
            },
            state: RequestState::Pending,
            priority: 1,
            kv_handle: None,
            created_at: std::time::Instant::now(),
            last_accessed: std::time::Instant::now(),
            preemption_count: 0,
            decode_started_at: None,
            last_scheduled_at: None,
            multimodal_hash: None,
        };

        scheduler.submit(req).unwrap();

        scheduler.step().unwrap();
        {
            let cp_queue = scheduler.chunked_prefill_queue.lock().unwrap_or_else(|e| e.into_inner());
            assert_eq!(cp_queue.queue[0].filled_tokens, 4);
        }

        scheduler.step().unwrap();
        {
            let cp_queue = scheduler.chunked_prefill_queue.lock().unwrap_or_else(|e| e.into_inner());
            assert_eq!(cp_queue.queue[0].filled_tokens, 8);
        }

        scheduler.step().unwrap();
        {
            let decoding = scheduler.decoding_queue.lock().unwrap_or_else(|e| e.into_inner());
            assert_eq!(decoding.len(), 1);
            assert_eq!(decoding[0].id, "req1");
            assert_eq!(decoding[0].generated_tokens.len(), 1);
        }
    }

    #[test]
    fn test_inference_scheduler_chunked_prefill_can_disable_decode_interleave() {
        let executor = Arc::new(MockExecutor);
        let kv_pool = Arc::new(MockKvPool::new(10));
        let mut config = bloomai_core::TokenSchedulingConfig::default();
        config.chunked_prefill.enabled = true;
        config.chunked_prefill.chunk_size = 4;
        config.chunked_prefill.interleave_with_decode = false;
        config.max_total_tokens_per_step = 4;

        let scheduler = InferenceScheduler::with_config(executor, kv_pool, config);

        let req = Request {
            id: "req1".to_string(),
            model_id: "m1".to_string(),
            prompt_tokens: vec![1, 2, 3, 4, 5, 6, 7, 8],
            generated_tokens: Vec::new(),
            params: GenerationParams {
                max_tokens: 5,
                ..Default::default()
            },
            state: RequestState::Pending,
            priority: 1,
            kv_handle: None,
            created_at: std::time::Instant::now(),
            last_accessed: std::time::Instant::now(),
            preemption_count: 0,
            decode_started_at: None,
            last_scheduled_at: None,
            multimodal_hash: None,
        };

        scheduler.submit(req).unwrap();

        scheduler.step().unwrap();
        {
            let active = scheduler.active_requests.lock().unwrap_or_else(|e| e.into_inner());
            assert_eq!(active["req1"].generated_tokens.len(), 0);
        }

        scheduler.step().unwrap();
        {
            let decoding = scheduler.decoding_queue.lock().unwrap_or_else(|e| e.into_inner());
            assert_eq!(decoding.len(), 1);
            assert_eq!(decoding[0].generated_tokens.len(), 1);
        }
    }

    #[test]
    fn test_inference_scheduler_preemption() {
        let executor = Arc::new(MockExecutor);
        let kv_pool = Arc::new(MockKvPool::new(10));
        let mut config = bloomai_core::TokenSchedulingConfig::default();
        config.preemption.enabled = true;
        config.preemption.preemption_threshold_ms = 1; // trigger instantly

        let scheduler = InferenceScheduler::with_config(executor, kv_pool, config);

        // Low priority request
        let req_low = Request {
            id: "req-low".to_string(),
            model_id: "m1".to_string(),
            prompt_tokens: vec![1, 2],
            generated_tokens: Vec::new(),
            params: GenerationParams {
                max_tokens: 10,
                ..Default::default()
            },
            state: RequestState::Pending,
            priority: 1,
            kv_handle: None,
            created_at: std::time::Instant::now() - std::time::Duration::from_millis(100),
            last_accessed: std::time::Instant::now(),
            preemption_count: 0,
            decode_started_at: None,
            last_scheduled_at: None,
            multimodal_hash: None,
        };

        scheduler.submit(req_low).unwrap();
        scheduler.step().unwrap(); // prefill and start decoding for req-low

        // Ensure req-low is in decoding
        {
            let decoding = scheduler.decoding_queue.lock().unwrap_or_else(|e| e.into_inner());
            assert_eq!(decoding[0].id, "req-low");
        }

        // High priority request enters waiting queue
        let req_high = Request {
            id: "req-high".to_string(),
            model_id: "m1".to_string(),
            prompt_tokens: vec![3, 4],
            generated_tokens: Vec::new(),
            params: GenerationParams {
                max_tokens: 10,
                ..Default::default()
            },
            state: RequestState::Pending,
            priority: 10, // high priority
            kv_handle: None,
            created_at: std::time::Instant::now() - std::time::Duration::from_millis(50),
            last_accessed: std::time::Instant::now(),
            preemption_count: 0,
            decode_started_at: None,
            last_scheduled_at: None,
            multimodal_hash: None,
        };

        scheduler.submit(req_high).unwrap();

        // Sleep 2ms to ensure wait time > threshold
        std::thread::sleep(std::time::Duration::from_millis(2));

        // Step. Preemption should trigger, preempting req-low and scheduling req-high!
        scheduler.step().unwrap();

        {
            let decoding = scheduler.decoding_queue.lock().unwrap_or_else(|e| e.into_inner());
            assert_eq!(decoding.len(), 1);
            assert_eq!(decoding[0].id, "req-high"); // req-high scheduled!
        }
        {
            let prefill = scheduler.prefill_queue.lock().unwrap_or_else(|e| e.into_inner());
            assert_eq!(prefill.len(), 1);
            assert_eq!(prefill[0].id, "req-low"); // req-low preempted and sent back to prefill
            assert_eq!(prefill[0].preemption_count, 1);
        }
    }

    #[test]
    fn test_bloom_kv_cache_pool_priority_eviction() {
        let pool = BloomKvCachePool::new(4, 4); // total 4 blocks

        // Request 1: prompt length 8 -> 2 blocks. Priority = 10, value = 10.0 (high)
        let prompt1 = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let _alloc1 = pool.allocate_paged("req1", &prompt1, 0, None).unwrap();
        pool.update_request_metadata("req1", 10, 0, 10.0);
        pool.free_paged("req1"); // cached

        // Request 2: prompt length 8 -> 2 blocks. Priority = 1, value = 0.1 (low) -> low score
        let prompt2 = vec![10, 11, 12, 13, 14, 15, 16, 17];
        let _alloc2 = pool.allocate_paged("req2", &prompt2, 0, None).unwrap();
        pool.update_request_metadata("req2", 1, 0, 0.1);
        pool.free_paged("req2"); // cached

        // Request 3 requires 2 blocks. Need to evict.
        // priority score: req2 has lower score and should be evicted, despite req1 being older (LRU candidate).
        let prompt3 = vec![20, 21, 22, 23, 24, 25, 26, 27];
        let _alloc3 = pool.allocate_paged("req3", &prompt3, 0, None).unwrap();

        let state = pool.state.lock().unwrap_or_else(|e| e.into_inner());
        assert!(state.active_requests.contains_key("req1"));
        assert!(!state.active_requests.contains_key("req2")); // req2 was evicted!
    }

    // -----------------------------------------------------------------------
    // F05: BloomScheduler generic segment scheduling tests
    // -----------------------------------------------------------------------

    fn make_ticket(model_id: &str, backend: &str) -> ResourceTicket {
        ResourceTicket {
            model_id: model_id.to_string(),
            ram_bytes: 1024,
            vram_bytes: 2048,
            cache_bytes: 0,
            priority: ResourcePriority::Normal,
            strategy: bloomai_core::ResidencyStrategy::OnDemand,
            preferred_backend: Some(backend.to_string()),
            fallback_backends: vec![],
        }
    }

    fn make_device(thermal: ThermalState, power: PowerState) -> DeviceCapability {
        DeviceCapability {
            backend_name: "cpu".to_string(),
            vendor: None,
            device_class: DeviceClass::Cpu,
            memory_topology: MemoryTopology::Unified,
            max_memory: 16 * 1024 * 1024 * 1024,
            available_memory: 8 * 1024 * 1024 * 1024,
            supported_dtypes: vec![],
            supported_formats: vec![],
            supports_mmap: true,
            has_quantization_kernels: false,
            supports_streaming: true,
            thermal_state: thermal,
            power_state: power,
            max_batch_tokens: None,
            available_parallelism: Some(8),
        }
    }

    fn submit_multi_class(scheduler: &BloomScheduler) {
        // Submit background batch first
        scheduler
            .submit_segment(
                "bg-1".into(),
                "model-a".into(),
                ExecutionPhase::Encode,
                vec![DataBlock::Text("bg input".into())],
                RequestClass::BackgroundBatch,
                make_ticket("model-a", "cpu"),
            )
            .unwrap();

        // Submit maintenance
        scheduler
            .submit_segment(
                "mt-1".into(),
                "model-a".into(),
                ExecutionPhase::Postprocess,
                vec![],
                RequestClass::Maintenance,
                make_ticket("model-a", "cpu"),
            )
            .unwrap();

        // Submit realtime stream
        scheduler
            .submit_segment(
                "rt-1".into(),
                "model-a".into(),
                ExecutionPhase::Decode,
                vec![DataBlock::Tokens(vec![1, 2, 3])],
                RequestClass::RealtimeStream,
                make_ticket("model-a", "cpu"),
            )
            .unwrap();

        // Submit foreground interactive (highest priority)
        scheduler
            .submit_segment(
                "fg-1".into(),
                "model-a".into(),
                ExecutionPhase::Prefill,
                vec![DataBlock::Text("hello".into())],
                RequestClass::ForegroundInteractive,
                make_ticket("model-a", "cpu"),
            )
            .unwrap();
    }

    #[test]
    fn test_bloom_scheduler_strict_priority() {
        let scheduler = BloomScheduler::new(4);
        submit_multi_class(&scheduler);

        // Queue depths should reflect all 4 submissions
        let depths = scheduler.queue_depths();
        assert_eq!(depths[&RequestClass::ForegroundInteractive], 1);
        assert_eq!(depths[&RequestClass::RealtimeStream], 1);
        assert_eq!(depths[&RequestClass::BackgroundBatch], 1);
        assert_eq!(depths[&RequestClass::Maintenance], 1);

        let devices = vec![make_device(ThermalState::Nominal, PowerState::PluggedIn)];
        let segments = scheduler.next_segments(&devices).unwrap();
        assert_eq!(segments.len(), 4);

        // Strict priority: FG first, then RT, then BG, then MT
        assert_eq!(segments[0].class, RequestClass::ForegroundInteractive);
        assert_eq!(segments[1].class, RequestClass::RealtimeStream);
        assert_eq!(segments[2].class, RequestClass::BackgroundBatch);
        assert_eq!(segments[3].class, RequestClass::Maintenance);

        // All active now
        assert_eq!(scheduler.active_count(), 4);
        assert_eq!(
            scheduler.queue_depths()[&RequestClass::ForegroundInteractive],
            0
        );
    }

    #[test]
    fn test_bloom_scheduler_token_budget_limits_prefill() {
        let scheduler = BloomScheduler::with_token_config(
            4,
            TokenSchedulingConfig {
                max_prefill_tokens_per_step: 2,
                max_decode_tokens_per_step: 4,
                max_total_tokens_per_step: 2,
                ..Default::default()
            },
        );

        for i in 0..2 {
            scheduler
                .submit_segment(
                    format!("fg-{i}"),
                    "model-a".into(),
                    ExecutionPhase::Prefill,
                    vec![DataBlock::Tokens(vec![1, 2])],
                    RequestClass::ForegroundInteractive,
                    make_ticket("model-a", "cpu"),
                )
                .unwrap();
        }

        let devices = vec![make_device(ThermalState::Nominal, PowerState::PluggedIn)];
        let segments = scheduler.next_segments(&devices).unwrap();
        assert_eq!(segments.len(), 1);
        assert_eq!(
            scheduler.queue_depths()[&RequestClass::ForegroundInteractive],
            1
        );
    }

    #[test]
    fn test_bloom_scheduler_thermal_degradation() {
        let scheduler = BloomScheduler::new(8);
        submit_multi_class(&scheduler);

        // Submit 4 more to have 8 total
        for i in 0..4 {
            scheduler
                .submit_segment(
                    format!("bg-extra-{}", i),
                    "model-a".into(),
                    ExecutionPhase::Encode,
                    vec![],
                    RequestClass::BackgroundBatch,
                    make_ticket("model-a", "cpu"),
                )
                .unwrap();
        }

        let devices = vec![make_device(ThermalState::Serious, PowerState::PluggedIn)];
        let segments = scheduler.next_segments(&devices).unwrap();

        // Thermal serious limits to thermal_batch_limit (2)
        assert!(
            segments.len() <= 2,
            "Expected thermal limit, got {}",
            segments.len()
        );

        // All returned segments should be marked degraded
        for seg in &segments {
            assert!(seg.route.degraded);
            assert_eq!(
                seg.route.switch_reason,
                Some(ModelSwitchReason::ThermalLimit)
            );
        }
    }

    #[test]
    fn test_bloom_scheduler_power_degradation() {
        let scheduler = BloomScheduler::new(8);
        submit_multi_class(&scheduler);

        let devices = vec![make_device(ThermalState::Nominal, PowerState::Battery)];
        let segments = scheduler.next_segments(&devices).unwrap();

        // Battery limits to power_batch_limit (4)
        assert!(
            segments.len() <= 4,
            "Expected power limit, got {}",
            segments.len()
        );
        assert!(!segments.is_empty(), "Should still schedule at least 1");

        // Degraded due to battery
        for seg in &segments {
            assert!(seg.route.degraded);
        }
    }

    #[test]
    fn test_bloom_scheduler_critical_thermal() {
        let scheduler = BloomScheduler::new(16);
        submit_multi_class(&scheduler);

        let devices = vec![make_device(ThermalState::Critical, PowerState::PluggedIn)];
        let segments = scheduler.next_segments(&devices).unwrap();

        // Critical thermal -> max 1 segment
        assert_eq!(segments.len(), 1);
        // Should be highest priority class (foreground)
        assert_eq!(segments[0].class, RequestClass::ForegroundInteractive);
        assert!(segments[0].route.degraded);
    }

    #[test]
    fn test_bloom_scheduler_complete_segment() {
        let scheduler = BloomScheduler::new(4);
        scheduler
            .submit_segment(
                "req-done".into(),
                "model-a".into(),
                ExecutionPhase::Prefill,
                vec![DataBlock::Text("hi".into())],
                RequestClass::ForegroundInteractive,
                make_ticket("model-a", "cpu"),
            )
            .unwrap();

        let devices = vec![make_device(ThermalState::Nominal, PowerState::PluggedIn)];
        let _ = scheduler.next_segments(&devices).unwrap();
        assert_eq!(scheduler.active_count(), 1);

        // Complete with success
        let follow_up = scheduler
            .complete_segment("req-done", SegmentResult::Success { outputs: vec![] })
            .unwrap();
        assert!(follow_up.is_none()); // Success = no continuation
        assert_eq!(scheduler.active_count(), 0);
    }

    #[test]
    fn test_bloom_scheduler_segment_continuation() {
        let scheduler = BloomScheduler::new(4);
        scheduler
            .submit_segment(
                "req-cont".into(),
                "model-a".into(),
                ExecutionPhase::Prefill,
                vec![DataBlock::Tokens(vec![1, 2])],
                RequestClass::ForegroundInteractive,
                make_ticket("model-a", "cpu"),
            )
            .unwrap();

        let devices = vec![make_device(ThermalState::Nominal, PowerState::PluggedIn)];
        let _ = scheduler.next_segments(&devices).unwrap();

        // Complete with Continue -> Decode
        let follow_up = scheduler
            .complete_segment(
                "req-cont",
                SegmentResult::Continue {
                    next_phase: ExecutionPhase::Decode,
                },
            )
            .unwrap();
        // follow_up is Some (the new segment info) but the actual segment is re-queued
        assert!(follow_up.is_some());

        // The segment should be back in the queue for Decode phase
        let depths = scheduler.queue_depths();
        assert_eq!(depths[&RequestClass::ForegroundInteractive], 1);

        let next = scheduler.next_segments(&devices).unwrap();
        assert_eq!(next.len(), 1);
        assert_eq!(next[0].phase, ExecutionPhase::Decode);
    }

    #[test]
    fn test_bloom_scheduler_wfq_fairness() {
        let mut scheduler = BloomScheduler::new(4);
        scheduler.set_fairness(FairnessStrategy::WeightedFairQueue);

        // Submit 6 BG and 2 FG requests
        for i in 0..6 {
            scheduler
                .submit_segment(
                    format!("bg-{}", i),
                    "m".into(),
                    ExecutionPhase::Encode,
                    vec![],
                    RequestClass::BackgroundBatch,
                    make_ticket("m", "cpu"),
                )
                .unwrap();
        }
        for i in 0..2 {
            scheduler
                .submit_segment(
                    format!("fg-{}", i),
                    "m".into(),
                    ExecutionPhase::Prefill,
                    vec![],
                    RequestClass::ForegroundInteractive,
                    make_ticket("m", "cpu"),
                )
                .unwrap();
        }

        let devices = vec![make_device(ThermalState::Nominal, PowerState::PluggedIn)];
        let segments = scheduler.next_segments(&devices).unwrap();
        assert_eq!(segments.len(), 4);

        // WFQ should schedule some FG and some BG (not all FG first)
        let fg_count = segments
            .iter()
            .filter(|s| s.class == RequestClass::ForegroundInteractive)
            .count();
        let bg_count = segments
            .iter()
            .filter(|s| s.class == RequestClass::BackgroundBatch)
            .count();
        // FG weight=4, so gets 4 slots, but only 2 FG requests, remaining 2 from BG
        assert_eq!(fg_count, 2);
        assert_eq!(bg_count, 2);
    }

    #[test]
    fn test_bloom_scheduler_deadline_and_classes() {
        let scheduler = BloomScheduler::new(2);
        scheduler
            .submit_segment(
                "dl-req".into(),
                "model-a".into(),
                ExecutionPhase::Simulate,
                vec![DataBlock::WorldState {
                    state_id: "obs1".into(),
                    latent: None,
                    step: 0,
                }],
                RequestClass::RealtimeStream,
                make_ticket("model-a", "cpu"),
            )
            .unwrap();

        let devices = vec![make_device(ThermalState::Nominal, PowerState::PluggedIn)];
        let segments = scheduler.next_segments(&devices).unwrap();
        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0].phase, ExecutionPhase::Simulate);
        assert_eq!(segments[0].class, RequestClass::RealtimeStream);
        assert_eq!(segments[0].route.model_id, "model-a");
    }

    #[test]
    fn test_execution_phase_extended() {
        // Verify all phase variants exist
        let phases = [
            ExecutionPhase::Load,
            ExecutionPhase::Prefill,
            ExecutionPhase::Decode,
            ExecutionPhase::Encode,
            ExecutionPhase::Generate,
            ExecutionPhase::Simulate,
            ExecutionPhase::Postprocess,
        ];
        assert_eq!(phases.len(), 7);
    }

    #[test]
    fn test_request_class_priority_order() {
        assert!(RequestClass::ForegroundInteractive > RequestClass::RealtimeStream);
        assert!(RequestClass::RealtimeStream > RequestClass::BackgroundBatch);
        assert!(RequestClass::BackgroundBatch > RequestClass::Maintenance);
    }

    #[test]
    fn test_environment_constraints() {
        let normal = EnvironmentConstraints::default();
        assert_eq!(normal.effective_max_batch(16), 16);
        assert!(!normal.is_degraded());

        let thermal_serious = EnvironmentConstraints {
            thermal_state: ThermalState::Serious,
            ..Default::default()
        };
        assert_eq!(thermal_serious.effective_max_batch(16), 2);
        assert!(thermal_serious.is_degraded());

        let battery = EnvironmentConstraints {
            power_state: PowerState::Battery,
            ..Default::default()
        };
        assert_eq!(battery.effective_max_batch(16), 4);
        assert!(battery.is_degraded());

        let critical_battery = EnvironmentConstraints {
            thermal_state: ThermalState::Critical,
            power_state: PowerState::Battery,
            ..Default::default()
        };
        // Critical thermal limits to 1, battery to 4 -> min = 1
        assert_eq!(critical_battery.effective_max_batch(16), 1);
    }

    #[test]
    fn test_bloom_kv_cache_pool_paged_allocation() {
        let pool = BloomKvCachePool::new(4, 8); // block_size = 4, total_blocks = 8

        // Allocate for request 1: prompt length 6, max new tokens 2
        // total tokens = 8, needed blocks = 2
        let alloc1 = pool
            .allocate_paged("req1", &[1, 2, 3, 4, 5, 6], 2, None)
            .unwrap();
        assert_eq!(alloc1.allocated_blocks.len(), 2);
        assert_eq!(alloc1.matched_tokens, 0); // first time, no match

        let metrics = pool.get_metrics();
        assert_eq!(metrics.free_blocks, 6);
        assert_eq!(metrics.active_blocks, 2);
        assert_eq!(metrics.cached_blocks, 0);
        assert_eq!(metrics.misses, 1);
        assert_eq!(metrics.hits, 0);
    }

    #[test]
    fn test_bloom_kv_cache_pool_prefix_reuse() {
        let pool = BloomKvCachePool::new(4, 8); // block_size = 4, total_blocks = 8

        // Request 1: prompt [1, 2, 3, 4, 5, 6, 7, 8], max new tokens 0
        // Needed blocks = 2 (both are full prompt blocks, since len = 8)
        let prompt1 = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let alloc1 = pool.allocate_paged("req1", &prompt1, 0, None).unwrap();
        assert_eq!(alloc1.allocated_blocks.len(), 2);
        assert_eq!(alloc1.matched_tokens, 0);

        // Free req1, marking it inactive
        pool.free_paged("req1");

        let metrics = pool.get_metrics();
        assert_eq!(metrics.free_blocks, 6);
        assert_eq!(metrics.active_blocks, 0);
        assert_eq!(metrics.cached_blocks, 2); // 2 blocks cached in LRU

        // Request 2: prompt [1, 2, 3, 4, 10, 11], max new tokens 2
        // Needed blocks = 2.
        // Prompt block 0 prefix: [1, 2, 3, 4] -> matches prompt1's block 0!
        // Prompt block 1 prefix: [1, 2, 3, 4, 10, 11] -> doesn't match.
        // Reuses block 0 (1 block reused), allocates 1 new block.
        let prompt2 = vec![1, 2, 3, 4, 10, 11];
        let alloc2 = pool.allocate_paged("req2", &prompt2, 2, None).unwrap();
        assert_eq!(alloc2.allocated_blocks.len(), 2);
        assert_eq!(alloc2.matched_tokens, 4); // 4 tokens matched
        assert_eq!(alloc2.allocated_blocks[0], alloc1.allocated_blocks[0]); // Reused first block!

        let metrics = pool.get_metrics();
        assert_eq!(metrics.hits, 1);
        assert_eq!(metrics.reuses, 1);
        assert_eq!(metrics.misses, 1);
        assert_eq!(metrics.free_blocks, 5); // 8 - (1 shared + 1 unique for req1 + 1 unique for req2)
        assert_eq!(metrics.active_blocks, 2); // req2's 2 blocks are active
        assert_eq!(metrics.cached_blocks, 1); // req1's block 1 is still cached but not active
    }

    #[test]
    fn test_bloom_kv_cache_pool_lru_eviction() {
        let pool = BloomKvCachePool::new(4, 4); // total 4 blocks

        // Request 1: prompt [1, 2, 3, 4, 5, 6, 7, 8], max new tokens 0 -> 2 blocks (cached)
        let prompt1 = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let _alloc1 = pool.allocate_paged("req1", &prompt1, 0, None).unwrap();
        pool.free_paged("req1");

        // Request 2: prompt [10, 11, 12, 13, 14, 15, 16, 17], max new tokens 0 -> 2 blocks (cached)
        let prompt2 = vec![10, 11, 12, 13, 14, 15, 16, 17];
        let _alloc2 = pool.allocate_paged("req2", &prompt2, 0, None).unwrap();
        pool.free_paged("req2");

        let metrics = pool.get_metrics();
        assert_eq!(metrics.free_blocks, 0);
        assert_eq!(metrics.cached_blocks, 4);

        // LRU order: req1 (oldest), req2 (newest)
        // Request 3: prompt [20, 21, 22, 23, 24, 25, 26, 27], max new tokens 0 -> 2 blocks
        // Requires 2 blocks. Since free_blocks = 0, we evict req1.
        let prompt3 = vec![20, 21, 22, 23, 24, 25, 26, 27];
        let alloc3 = pool.allocate_paged("req3", &prompt3, 0, None).unwrap();
        assert_eq!(alloc3.allocated_blocks.len(), 2);

        let metrics = pool.get_metrics();
        assert_eq!(metrics.evictions, 1);
        assert_eq!(metrics.free_blocks, 0);
        assert_eq!(metrics.active_blocks, 2);
        assert_eq!(metrics.cached_blocks, 2); // req2 is still cached
    }

    #[test]
    fn test_bloom_kv_cache_pool_concurrent_access() {
        use std::thread;
        let pool = std::sync::Arc::new(BloomKvCachePool::new(4, 100));

        let mut handles = Vec::new();
        for i in 0..10 {
            let pool_clone = pool.clone();
            let handle = thread::spawn(move || {
                let req_id = format!("thread-req-{}", i);
                // Unique prompt for each thread to avoid prefix caching reuse in this test
                let alloc = pool_clone
                    .allocate_paged(&req_id, &[i as u32, 2, 3, 4], 4, None)
                    .unwrap();
                pool_clone.free_paged(&req_id);
                alloc.allocated_blocks.len()
            });
            handles.push(handle);
        }

        for h in handles {
            assert_eq!(h.join().unwrap(), 2);
        }

        let metrics = pool.get_metrics();
        assert_eq!(metrics.free_blocks, 80); // 100 - 10 * 2 (all cached, none active)
        assert_eq!(metrics.cached_blocks, 20);
        assert_eq!(metrics.active_blocks, 0);
    }

    /// Regression for the scheduler → paged-cache disconnect.
    ///
    /// The scheduler used to call `KvCachePool::allocate(num_tokens)` which
    /// only pops a handle from the free list — it does NOT register the
    /// request in the pool's `active_requests` map. As a result
    /// `block_for_handle(handle, pos)` returned `None` for every position,
    /// and the batch executor's `write_request_kv` / `restore_request_kv`
    /// silently skipped all KV writes. The paged cache was effectively
    /// metadata-only even when a `KvHook` was attached.
    ///
    /// The fix switches the scheduler to `allocate_paged`, which sets up
    /// `active_requests[request_id].blocks` so `block_for_handle` resolves.
    #[test]
    fn test_scheduler_paged_cache_allocation_registers_blocks() {
        let pool = Arc::new(BloomKvCachePool::new(4, 8));
        let executor = Arc::new(MockExecutor);
        let scheduler =
            InferenceScheduler::new(executor, Arc::clone(&pool) as Arc<dyn KvCachePool>);

        let request = Request {
            id: "req-paged-1".to_string(),
            model_id: "m1".to_string(),
            prompt_tokens: vec![1u32, 2, 3, 4],
            generated_tokens: Vec::new(),
            params: GenerationParams {
                max_tokens: 2,
                ..Default::default()
            },
            state: RequestState::Pending,
            priority: 1,
            kv_handle: None,
            created_at: std::time::Instant::now(),
            last_accessed: std::time::Instant::now(),
            preemption_count: 0,
            decode_started_at: None,
            last_scheduled_at: None,
            multimodal_hash: None,
        };

        scheduler.submit(request).unwrap();
        // After submit, the scheduler must have allocated a handle via
        // `allocate_paged`, which also registers the request in the pool's
        // `active_requests` map. `block_for_handle` is the executor's
        // lookup path — it must resolve to a real block id.
        let handle = {
            let prefill = scheduler.prefill_queue.lock().unwrap_or_else(|e| e.into_inner());
            prefill[0]
                .kv_handle
                .expect("submit must allocate a kv_handle")
        };
        assert!(
            pool.block_for_handle(handle, 0).is_some(),
            "block_for_handle returned None after submit — allocate_paged was not used"
        );

        // Metrics must reflect the active allocation.
        let metrics = pool.get_metrics();
        assert_eq!(metrics.active_blocks, 2); // 4 prompt + 2 max_new = 6 tokens, ceil(6/4) = 2 blocks
        assert_eq!(metrics.free_blocks, 6);
        assert_eq!(metrics.misses, 1);
    }

    /// Regression for the scheduler → prefix-cache disconnect.
    ///
    /// `BloomKvCachePool::allocate(num_tokens)` delegates to
    /// `allocate_paged` with `prompt_tokens = &[]`. That still registers
    /// `active_requests` (so `block_for_handle` resolves), but it never
    /// populates `block_table` (the prefix → block_id map). As a result,
    /// a second request sharing a prompt prefix with a cached inactive
    /// request would NOT get a prefix cache hit — `metrics.hits` stays 0.
    ///
    /// The fix passes the real `request.prompt_tokens` to `allocate_paged`
    /// so the pool's prefix index is populated.
    #[test]
    fn test_scheduler_submit_populates_prefix_for_cache_hits() {
        let pool = Arc::new(BloomKvCachePool::new(4, 8));
        let executor = Arc::new(MockExecutor);
        let scheduler =
            InferenceScheduler::new(executor, Arc::clone(&pool) as Arc<dyn KvCachePool>);

        // Request 1: prompt [1,2,3,4,5,6,7,8], max_tokens 0.
        // `allocate_paged` should record prefix [1,2,3,4] -> block_a and
        // prefix [1,2,3,4,5,6,7,8] -> block_b in `block_table`.
        let req1 = Request {
            id: "req-prefix-1".to_string(),
            model_id: "m1".to_string(),
            prompt_tokens: vec![1u32, 2, 3, 4, 5, 6, 7, 8],
            generated_tokens: Vec::new(),
            params: GenerationParams {
                max_tokens: 0,
                ..Default::default()
            },
            state: RequestState::Pending,
            priority: 1,
            kv_handle: None,
            created_at: std::time::Instant::now(),
            last_accessed: std::time::Instant::now(),
            preemption_count: 0,
            decode_started_at: None,
            last_scheduled_at: None,
            multimodal_hash: None,
        };
        scheduler.submit(req1).unwrap();

        // Mark req1 inactive so its blocks become cached prefix entries.
        {
            let prefill = scheduler.prefill_queue.lock().unwrap_or_else(|e| e.into_inner());
            let handle = prefill[0].kv_handle.unwrap();
            pool.free_paged("req-prefix-1");
            // free_paged marks the request inactive but keeps blocks cached.
            let _ = handle;
        }

        // Request 2: prompt [1,2,3,4,10,11] shares the first 4-token block
        // with req1. With prefix caching working, allocate_paged must reuse
        // the cached block and increment `metrics.hits`.
        let req2 = Request {
            id: "req-prefix-2".to_string(),
            model_id: "m1".to_string(),
            prompt_tokens: vec![1u32, 2, 3, 4, 10, 11],
            generated_tokens: Vec::new(),
            params: GenerationParams {
                max_tokens: 2,
                ..Default::default()
            },
            state: RequestState::Pending,
            priority: 1,
            kv_handle: None,
            created_at: std::time::Instant::now(),
            last_accessed: std::time::Instant::now(),
            preemption_count: 0,
            decode_started_at: None,
            last_scheduled_at: None,
            multimodal_hash: None,
        };
        scheduler.submit(req2).unwrap();

        let metrics = pool.get_metrics();
        assert!(
            metrics.hits >= 1,
            "prefix cache hit not recorded — submit did not pass real prompt_tokens to allocate_paged. metrics = {:?}",
            metrics
        );
        assert!(
            metrics.reuses >= 1,
            "prefix block reuse not recorded. metrics = {:?}",
            metrics
        );
    }

    #[test]
    fn test_multimodal_prefix_caching() {
        let pool = Arc::new(BloomKvCachePool::new(4, 10)); // block size = 4

        // Request 1: prompt [1,2,3,4] with image-1 hash
        let alloc1 = pool
            .allocate_paged("req-m-1", &[1u32, 2, 3, 4], 0, Some("image-hash-1"))
            .unwrap();
        assert_eq!(alloc1.matched_tokens, 0);
        pool.free_paged("req-m-1");

        // Request 2: same tokens, but different multimodal hash (image-hash-2)
        // Should NOT hit prefix cache because the multimodal hash differs.
        let alloc2 = pool
            .allocate_paged("req-m-2", &[1u32, 2, 3, 4], 0, Some("image-hash-2"))
            .unwrap();
        assert_eq!(
            alloc2.matched_tokens, 0,
            "Expected cache miss due to different multimodal hash"
        );
        pool.free_paged("req-m-2");

        // Request 3: same tokens AND same multimodal hash (image-hash-1)
        // Should HIT prefix cache.
        let alloc3 = pool
            .allocate_paged("req-m-3", &[1u32, 2, 3, 4], 0, Some("image-hash-1"))
            .unwrap();
        assert_eq!(
            alloc3.matched_tokens, 4,
            "Expected cache hit for matching tokens and multimodal hash"
        );
    }

    #[test]
    fn test_prune_low_attention_blocks() {
        let pool = BloomKvCachePool::new(4, 10);
        let prompt = vec![1u32; 24];
        let alloc = pool.allocate_paged("req-prune", &prompt, 0, None).unwrap();
        assert_eq!(alloc.allocated_blocks.len(), 6);

        {
            let state = pool.state.lock().unwrap_or_else(|e| e.into_inner());
            assert_eq!(state.free_blocks.len(), 4);
        }

        let weights = vec![0.01f32, 0.02, 0.9, 0.9, 0.9, 0.9];
        let pruned = pool.prune_low_attention_blocks(alloc.handle, &weights, 6);
        assert_eq!(pruned, 2);

        {
            let state = pool.state.lock().unwrap_or_else(|e| e.into_inner());
            assert_eq!(state.free_blocks.len(), 6);
            let record = state.active_requests.get("req-prune").unwrap();
            assert_eq!(record.blocks.len(), 4);
        }
    }

    #[test]
    fn test_defragment_and_compact_prefixes() {
        let pool = BloomKvCachePool::new(4, 10);

        let prompt1 = vec![1u32, 2, 3, 4, 5, 6, 7, 8];
        let _alloc1 = pool
            .allocate_paged("req-compact-1", &prompt1, 0, Some("hash1"))
            .unwrap();

        let prompt2 = vec![1u32, 2, 3, 4, 9, 10, 11, 12];
        let _alloc2 = pool
            .allocate_paged("req-compact-2", &prompt2, 0, Some("hash2"))
            .unwrap();

        {
            let state = pool.state.lock().unwrap_or_else(|e| e.into_inner());
            let rec1 = &state.active_requests["req-compact-1"];
            let rec2 = &state.active_requests["req-compact-2"];
            assert_ne!(rec1.blocks[0], rec2.blocks[0]);
        }

        let merged = pool.defragment_and_compact_prefixes();
        assert_eq!(merged, 1);

        {
            let state = pool.state.lock().unwrap_or_else(|e| e.into_inner());
            let rec1 = &state.active_requests["req-compact-1"];
            let rec2 = &state.active_requests["req-compact-2"];
            assert_eq!(rec1.blocks[0], rec2.blocks[0]);
            assert_eq!(state.block_ref_counts[&rec1.blocks[0]], 2);
        }
    }
}
