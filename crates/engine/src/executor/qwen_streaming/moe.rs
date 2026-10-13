//! Sparse Qwen2-MoE feed-forward blocks for the extractable Qwen KV runtime.
//! Checkpoint semantics follow Qwen2-MoE and Apollo's inference-only upcycling
//! variant. No remote Python model code is executed.

use super::{StreamingLinear, StreamingQwenMLP};
use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{Activation, VarBuilder};
use serde::Deserialize;
use std::sync::Arc;

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct MoeConfig {
    pub num_experts: usize,
    pub num_experts_per_tok: usize,
    pub decoder_sparse_step: usize,
    pub moe_intermediate_size: usize,
    pub shared_expert_intermediate_size: usize,
    pub norm_topk_prob: bool,
    #[serde(default)]
    pub mlp_only_layers: Vec<usize>,
    #[serde(skip)]
    shared_expert: bool,
}

impl MoeConfig {
    pub(crate) fn from_json(value: &serde_json::Value) -> anyhow::Result<Self> {
        let mut config: Self = serde_json::from_value(value.clone())?;
        anyhow::ensure!(config.num_experts > 0, "MoE num_experts must be positive");
        anyhow::ensure!(
            config.num_experts_per_tok > 0 && config.num_experts_per_tok <= config.num_experts,
            "MoE num_experts_per_tok must be in 1..=num_experts"
        );
        anyhow::ensure!(
            config.decoder_sparse_step > 0,
            "MoE decoder_sparse_step must be positive"
        );
        anyhow::ensure!(
            config.moe_intermediate_size > 0,
            "MoE intermediate size must be positive"
        );
        anyhow::ensure!(
            value.get("use_sliding_window").and_then(|v| v.as_bool()) != Some(true),
            "Qwen2-MoE sliding-window attention is not supported by the extractable KV runtime"
        );
        let upcycling =
            value.get("model_type").and_then(|v| v.as_str()) == Some("upcycling-qwen2-moe");
        // Apollo's shared variant changes the expert counts during Python
        // construction; reject it until that layout has pinned evidence.
        anyhow::ensure!(
            !upcycling || value.get("share_flag").and_then(|v| v.as_bool()) == Some(false),
            "Apollo shared-expert upcycling is not supported; expected share_flag=false"
        );
        config.shared_expert = !upcycling;
        anyhow::ensure!(
            !config.shared_expert || config.shared_expert_intermediate_size > 0,
            "Qwen2-MoE shared expert size must be positive"
        );
        Ok(config)
    }

    pub(super) fn sparse_layer(&self, layer: usize) -> bool {
        !self.mlp_only_layers.contains(&layer)
            && (layer + 1).is_multiple_of(self.decoder_sparse_step)
    }
}

#[derive(Debug, Clone)]
pub(super) struct SparseMoe {
    gate: StreamingLinear,
    experts: Vec<StreamingQwenMLP>,
    shared: Option<(StreamingQwenMLP, StreamingLinear)>,
    top_k: usize,
    norm_topk_prob: bool,
}

impl SparseMoe {
    pub(super) fn new(
        config: &MoeConfig,
        hidden: usize,
        activation: Activation,
        vb: VarBuilder,
        quantizer: Option<&Arc<crate::executor::metal_quant::MetalQuantizer>>,
    ) -> Result<Self> {
        let gate = StreamingLinear::new(
            vb.pp("gate").get((config.num_experts, hidden), "weight")?,
            None,
        );
        let mut experts = Vec::with_capacity(config.num_experts);
        let expert_vb = vb.pp("experts");
        if expert_vb.contains_tensor("gate_up_proj") {
            let gate_up = expert_vb.get(
                (config.num_experts, 2 * config.moe_intermediate_size, hidden),
                "gate_up_proj",
            )?;
            let down = expert_vb.get(
                (config.num_experts, hidden, config.moe_intermediate_size),
                "down_proj",
            )?;
            for index in 0..config.num_experts {
                let gate_up = gate_up.get(index)?;
                experts.push(StreamingQwenMLP {
                    gate_proj: StreamingLinear::new(
                        gate_up
                            .narrow(0, 0, config.moe_intermediate_size)?
                            .contiguous()?,
                        None,
                    ),
                    up_proj: StreamingLinear::new(
                        gate_up
                            .narrow(
                                0,
                                config.moe_intermediate_size,
                                config.moe_intermediate_size,
                            )?
                            .contiguous()?,
                        None,
                    ),
                    down_proj: StreamingLinear::new(down.get(index)?.contiguous()?, None),
                    act_fn: activation,
                });
            }
        } else {
            for index in 0..config.num_experts {
                experts.push(StreamingQwenMLP::new(
                    hidden,
                    config.moe_intermediate_size,
                    activation,
                    expert_vb.pp(index),
                    quantizer,
                )?);
            }
        }
        let shared = if config.shared_expert {
            Some((
                StreamingQwenMLP::new(
                    hidden,
                    config.shared_expert_intermediate_size,
                    activation,
                    vb.pp("shared_expert"),
                    quantizer,
                )?,
                StreamingLinear::new(
                    vb.pp("shared_expert_gate").get((1, hidden), "weight")?,
                    None,
                ),
            ))
        } else {
            None
        };
        Ok(Self {
            gate,
            experts,
            shared,
            top_k: config.num_experts_per_tok,
            norm_topk_prob: config.norm_topk_prob,
        })
    }

    pub(super) fn forward(&self, input: &Tensor, device: &Device) -> Result<Tensor> {
        let (batch, sequence, hidden) = input.dims3()?;
        let input = input.reshape((batch * sequence, hidden))?;
        // Routing softmax is F32, matching Transformers even for FP16 weights.
        let logits = self.gate.forward(&input, device)?.to_dtype(DType::F32)?;
        let probabilities = candle_nn::ops::softmax_last_dim(&logits)?;
        let indices = probabilities
            .arg_sort_last_dim(false)?
            .narrow(1, 0, self.top_k)?
            .contiguous()?;
        let selected = probabilities.gather(&indices, 1)?.to_vec2::<f32>()?;
        let indices = indices.to_vec2::<u32>()?;
        let mut rows = vec![Vec::new(); self.experts.len()];
        let mut weights = vec![Vec::new(); self.experts.len()];
        for (row, (experts, probabilities)) in indices.iter().zip(&selected).enumerate() {
            let sum: f32 = probabilities.iter().sum();
            for (&expert, &weight) in experts.iter().zip(probabilities) {
                rows[expert as usize].push(row as u32);
                weights[expert as usize].push(if self.norm_topk_prob {
                    weight / sum
                } else {
                    weight
                });
            }
        }
        let mut output = input.zeros_like()?;
        for (index, expert) in self.experts.iter().enumerate() {
            if rows[index].is_empty() {
                continue;
            }
            let row_ids = Tensor::new(rows[index].as_slice(), device)?;
            let weight = Tensor::new(weights[index].as_slice(), device)?
                .unsqueeze(1)?
                .to_dtype(input.dtype())?;
            let expert_input = input.index_select(&row_ids, 0)?;
            let result = expert
                .forward(&expert_input, device)?
                .broadcast_mul(&weight)?;
            output = output.index_add(&row_ids, &result, 0)?;
        }
        if let Some((expert, gate)) = &self.shared {
            let weight = candle_nn::ops::sigmoid(&gate.forward(&input, device)?)?;
            output = (output + expert.forward(&input, device)?.broadcast_mul(&weight)?)?;
        }
        output.reshape((batch, sequence, hidden))
    }
}

#[derive(Debug, Clone)]
pub(super) enum FeedForward {
    Dense(StreamingQwenMLP),
    Sparse(SparseMoe),
}

impl FeedForward {
    pub(super) fn forward(&self, input: &Tensor, device: &Device) -> Result<Tensor> {
        match self {
            Self::Dense(model) => model.forward(input, device),
            Self::Sparse(model) => model.forward(input, device),
        }
    }
}
