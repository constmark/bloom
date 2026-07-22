//! Conv3d stubs for Candle (which lacks native 3D convolution).
//!
//! Provides placeholder types and functions for future TileLang integration.
//! Currently, Wan modules use Conv2d (applied per temporal frame) instead.

use candle_core::{Result, Tensor};
use candle_nn as nn;

/// Configuration for 3D convolution.
#[derive(Debug, Clone, Copy)]
pub struct Conv3dConfig {
    pub stride: usize,
    pub padding: usize,
}

impl Default for Conv3dConfig {
    fn default() -> Self {
        Self {
            stride: 1,
            padding: 0,
        }
    }
}

/// 3D convolution layer placeholder.
///
/// This will be replaced by TileLang kernels for GPU execution.
/// Currently unused — modules apply Conv2d per temporal frame instead.
pub struct Conv3d {
    #[allow(dead_code)]
    weight: Tensor,
    #[allow(dead_code)]
    bias: Option<Tensor>,
    #[allow(dead_code)]
    in_channels: usize,
    #[allow(dead_code)]
    out_channels: usize,
    #[allow(dead_code)]
    kernel_size: usize,
    #[allow(dead_code)]
    config: Conv3dConfig,
}

impl Conv3d {
    /// Forward pass placeholder — not used in current implementation.
    pub fn forward(&self, _x: &Tensor) -> Result<Tensor> {
        unimplemented!("Conv3d forward not yet implemented; use Conv2d per-frame fallback")
    }
}

/// Build a Conv3d layer from a VarBuilder.
#[allow(dead_code)]
pub fn conv3d(
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    config: Conv3dConfig,
    vb: nn::VarBuilder,
) -> Result<Conv3d> {
    let weight = vb.get(
        (
            out_channels,
            in_channels,
            kernel_size,
            kernel_size,
            kernel_size,
        ),
        "weight",
    )?;
    let bias = vb.get((out_channels,), "bias").ok();

    Ok(Conv3d {
        weight,
        bias,
        in_channels,
        out_channels,
        kernel_size,
        config,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conv3d_config_default() {
        let cfg = Conv3dConfig::default();
        assert_eq!(cfg.stride, 1);
        assert_eq!(cfg.padding, 0);
    }
}
