//! Multi-GPU / multi-node parallelism abstractions.
//!
//! Provides trait definitions for tensor parallel, pipeline parallel,
//! data parallel, and collective communication operations.
//! Current implementations are no-op stubs; real CUDA/NCCL backends
//! can be plugged in via feature flags.

use serde::{Deserialize, Serialize};

#[cfg(feature = "candle-engine")]
use candle_core::Tensor;

/// Parallel execution strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum ParallelStrategy {
    /// No parallelism — single device execution.
    #[default]
    None,
    /// Tensor parallelism — split weight matrices across GPUs.
    TensorParallel,
    /// Pipeline parallelism — split layers across GPUs.
    PipelineParallel,
    /// Data parallelism — replicate model, split batch.
    DataParallel,
    /// Expert parallelism — for MoE models.
    ExpertParallel,
}

impl std::fmt::Display for ParallelStrategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::None => write!(f, "none"),
            Self::TensorParallel => write!(f, "tensor_parallel"),
            Self::PipelineParallel => write!(f, "pipeline_parallel"),
            Self::DataParallel => write!(f, "data_parallel"),
            Self::ExpertParallel => write!(f, "expert_parallel"),
        }
    }
}

/// Configuration for Mixture-of-Experts (MoE) expert placement.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct MoeParallelConfig {
    /// Total number of experts.
    pub num_experts: usize,
    /// Number of experts processed by the current rank.
    pub experts_per_rank: usize,
    /// Rank index for each expert index.
    pub expert_placement: Vec<usize>,
}

/// Configuration for parallel execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParallelConfig {
    /// Parallelism strategy.
    pub strategy: ParallelStrategy,
    /// Total number of participating devices (world size).
    pub world_size: usize,
    /// Rank of the current device (0-based).
    pub rank: usize,
    /// Mixture of Experts (MoE) configuration.
    pub moe: Option<MoeParallelConfig>,
}

impl Default for ParallelConfig {
    fn default() -> Self {
        Self {
            strategy: ParallelStrategy::None,
            world_size: 1,
            rank: 0,
            moe: None,
        }
    }
}

impl ParallelConfig {
    /// Initialize automatic MoE expert placement based on world size.
    pub fn with_moe_placement(mut self, num_experts: usize) -> Self {
        if self.world_size == 0 {
            return self;
        }
        let experts_per_rank = num_experts.div_ceil(self.world_size);
        let mut expert_placement = Vec::with_capacity(num_experts);
        for i in 0..num_experts {
            expert_placement.push(i / experts_per_rank);
        }
        self.moe = Some(MoeParallelConfig {
            num_experts,
            experts_per_rank,
            expert_placement,
        });
        self
    }

    /// Whether parallelism is active (more than one device).
    pub fn is_active(&self) -> bool {
        self.world_size > 1 && self.strategy != ParallelStrategy::None
    }
}

/// Collective communication operations for multi-GPU/multi-node.
///
/// Implementations can wrap NCCL, Gloo, or MPI backends.
/// The default stub implementation performs no-ops (single device).
pub trait CollectiveOps: Send + Sync {
    /// All-reduce: sum tensors across all ranks.
    fn all_reduce(&self, data: &mut [f32]) -> anyhow::Result<()>;

    /// Broadcast: send data from root rank to all others.
    fn broadcast(&self, data: &mut [f32], root: usize) -> anyhow::Result<()>;

    /// All-gather: concatenate tensors from all ranks.
    fn all_gather(&self, local: &[f32], world_size: usize) -> anyhow::Result<Vec<f32>>;

    /// Barrier synchronization across all ranks.
    fn barrier(&self) -> anyhow::Result<()>;
}

/// No-op collective operations for single-device execution.
pub struct NoOpCollective;

impl CollectiveOps for NoOpCollective {
    fn all_reduce(&self, _data: &mut [f32]) -> anyhow::Result<()> {
        Ok(())
    }

    fn broadcast(&self, _data: &mut [f32], _root: usize) -> anyhow::Result<()> {
        Ok(())
    }

    fn all_gather(&self, local: &[f32], _world_size: usize) -> anyhow::Result<Vec<f32>> {
        Ok(local.to_vec())
    }

    fn barrier(&self) -> anyhow::Result<()> {
        Ok(())
    }
}

#[cfg(feature = "candle-engine")]
/// Light-weight collective communication operations directly on Candle Tensors.
/// Enables tensor parallel and pipeline parallel weight-splitting across devices.
pub trait TensorCollectiveOps: Send + Sync {
    /// All-reduce: sum tensors across all ranks.
    fn all_reduce(&self, tensor: &Tensor) -> candle_core::Result<Tensor>;

    /// Broadcast: send tensor from root rank to all other ranks.
    fn broadcast(&self, tensor: &Tensor, root: usize) -> candle_core::Result<Tensor>;

    /// All-gather: concatenate tensors from all ranks along a specified dimension.
    fn all_gather(&self, tensor: &Tensor, dim: usize) -> candle_core::Result<Tensor>;

    /// Scatter: scatter a tensor along a specified dimension across all ranks.
    fn scatter(&self, tensor: &Tensor, dim: usize, rank: usize) -> candle_core::Result<Tensor>;
}

#[cfg(feature = "candle-engine")]
/// No-op collective operations for single-device execution.
pub struct NoOpTensorCollective;

#[cfg(feature = "candle-engine")]
impl TensorCollectiveOps for NoOpTensorCollective {
    fn all_reduce(&self, tensor: &Tensor) -> candle_core::Result<Tensor> {
        Ok(tensor.clone())
    }

    fn broadcast(&self, tensor: &Tensor, _root: usize) -> candle_core::Result<Tensor> {
        Ok(tensor.clone())
    }

    fn all_gather(&self, tensor: &Tensor, _dim: usize) -> candle_core::Result<Tensor> {
        Ok(tensor.clone())
    }

    fn scatter(&self, tensor: &Tensor, _dim: usize, _rank: usize) -> candle_core::Result<Tensor> {
        Ok(tensor.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parallel_strategy_default() {
        assert_eq!(ParallelStrategy::default(), ParallelStrategy::None);
    }

    #[test]
    fn test_parallel_config_default() {
        let cfg = ParallelConfig::default();
        assert!(!cfg.is_active());
        assert_eq!(cfg.world_size, 1);
        assert_eq!(cfg.rank, 0);
    }

    #[test]
    fn test_parallel_config_active() {
        let cfg = ParallelConfig {
            strategy: ParallelStrategy::TensorParallel,
            world_size: 4,
            rank: 0,
            moe: None,
        };
        assert!(cfg.is_active());
    }

    #[test]
    fn test_noop_collective() {
        let coll = NoOpCollective;
        let mut data = vec![1.0, 2.0, 3.0];
        coll.all_reduce(&mut data).unwrap();
        assert_eq!(data, vec![1.0, 2.0, 3.0]);

        let gathered = coll.all_gather(&[1.0, 2.0], 1).unwrap();
        assert_eq!(gathered, vec![1.0, 2.0]);

        coll.barrier().unwrap();
    }

    #[test]
    fn test_parallel_strategy_display() {
        assert_eq!(
            ParallelStrategy::TensorParallel.to_string(),
            "tensor_parallel"
        );
        assert_eq!(ParallelStrategy::None.to_string(), "none");
    }

    #[test]
    fn test_moe_expert_placement() {
        let cfg = ParallelConfig {
            strategy: ParallelStrategy::ExpertParallel,
            world_size: 4,
            rank: 1,
            moe: None,
        }
        .with_moe_placement(8);

        let moe = cfg.moe.unwrap();
        assert_eq!(moe.num_experts, 8);
        assert_eq!(moe.experts_per_rank, 2);
        assert_eq!(moe.expert_placement, vec![0, 0, 1, 1, 2, 2, 3, 3]);
    }

    #[test]
    #[cfg(feature = "candle-engine")]
    fn test_noop_tensor_collective() {
        use candle_core::Device;
        let coll = NoOpTensorCollective;
        let device = Device::Cpu;
        let t = Tensor::new(&[1.0f32, 2.0, 3.0], &device).unwrap();

        let reduced = coll.all_reduce(&t).unwrap();
        assert_eq!(reduced.to_vec1::<f32>().unwrap(), vec![1.0, 2.0, 3.0]);

        let broadcasted = coll.broadcast(&t, 0).unwrap();
        assert_eq!(broadcasted.to_vec1::<f32>().unwrap(), vec![1.0, 2.0, 3.0]);

        let gathered = coll.all_gather(&t, 0).unwrap();
        assert_eq!(gathered.to_vec1::<f32>().unwrap(), vec![1.0, 2.0, 3.0]);

        let scattered = coll.scatter(&t, 0, 0).unwrap();
        assert_eq!(scattered.to_vec1::<f32>().unwrap(), vec![1.0, 2.0, 3.0]);
    }
}
