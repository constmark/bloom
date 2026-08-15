// Kernel reference loops use explicit indices to match Metal thread coordinates.
#![allow(clippy::needless_range_loop)]

use candle_core::{Device, Result, Tensor};

#[cfg(feature = "metal")]
use std::sync::Arc;

pub const AWQ_GPTQ_KERNEL: &str = include_str!("metal_kernels/awq_gptq.metal");

/// Direct Metal implementation for AWQ and GPTQ Dequantization.
/// This prevents falling back to CPU or Python and runs natively on Apple Silicon GPUs.
pub struct MetalQuantizer {
    #[cfg(feature = "metal")]
    pipeline_awq: Arc<candle_metal_kernels::metal::compute_pipeline::ComputePipeline>,
    #[cfg(feature = "metal")]
    pipeline_gptq: Arc<candle_metal_kernels::metal::compute_pipeline::ComputePipeline>,
}

impl std::fmt::Debug for MetalQuantizer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MetalQuantizer").finish()
    }
}

impl MetalQuantizer {
    #[cfg(feature = "metal")]
    pub fn new(device: &Device) -> Result<Self> {
        if let Device::Metal(metal_device) = device {
            let device_ptr = metal_device.device();
            let library = device_ptr
                .new_library_with_source(AWQ_GPTQ_KERNEL, None)
                .map_err(|e| {
                    candle_core::Error::Msg(format!("Metal library compilation failed: {}", e))
                })?;

            let func_awq = library
                .get_function("dequantize_awq_int4", None)
                .map_err(|e| {
                    candle_core::Error::Msg(format!("Failed to load dequantize_awq_int4: {}", e))
                })?;
            let pipeline_awq = device_ptr
                .new_compute_pipeline_state_with_function(&func_awq)
                .map_err(|e| candle_core::Error::Msg(format!("Pipeline AWQ failed: {}", e)))?;

            let func_gptq = library
                .get_function("dequantize_gptq_int4", None)
                .map_err(|e| {
                    candle_core::Error::Msg(format!("Failed to load dequantize_gptq_int4: {}", e))
                })?;
            let pipeline_gptq = device_ptr
                .new_compute_pipeline_state_with_function(&func_gptq)
                .map_err(|e| candle_core::Error::Msg(format!("Pipeline GPTQ failed: {}", e)))?;

            Ok(Self {
                pipeline_awq: Arc::new(pipeline_awq),
                pipeline_gptq: Arc::new(pipeline_gptq),
            })
        } else {
            Err(candle_core::Error::Msg(
                "MetalQuantizer requires a Metal device".into(),
            ))
        }
    }

    #[cfg(not(feature = "metal"))]
    pub fn new(_device: &Device) -> Result<Self> {
        Err(candle_core::Error::Msg(
            "bloom-engine was not compiled with 'metal' feature".into(),
        ))
    }

    /// Dequantize an AWQ packed Int4 tensor to Float16 natively on the Metal GPU
    #[cfg(feature = "metal")]
    pub fn dequantize_awq(
        &self,
        qweight: &Tensor,
        scales: &Tensor,
        qzeros: &Tensor,
        total_elements: usize,
    ) -> Result<Tensor> {
        tracing::info!("Running native Metal AWQ dequantization...");
        let out = Tensor::zeros(total_elements, candle_core::DType::F16, qweight.device())?;

        if let Device::Metal(metal_device) = qweight.device() {
            let (qw_storage, qw_layout) = qweight.storage_and_layout();
            let (scales_storage, scales_layout) = scales.storage_and_layout();
            let (qz_storage, qz_layout) = qzeros.storage_and_layout();
            let (out_storage, out_layout) = out.storage_and_layout();

            if let (
                candle_core::Storage::Metal(qw_ms),
                candle_core::Storage::Metal(scales_ms),
                candle_core::Storage::Metal(qz_ms),
                candle_core::Storage::Metal(out_ms),
            ) = (&*qw_storage, &*scales_storage, &*qz_storage, &*out_storage)
            {
                let encoder = metal_device.command_encoder()?;
                encoder.set_label("dequantize_awq");
                encoder.set_compute_pipeline_state(&self.pipeline_awq);

                encoder.set_buffer(
                    0,
                    Some(qw_ms.buffer()),
                    qw_layout.start_offset() * qweight.dtype().size_in_bytes(),
                );
                encoder.set_buffer(
                    1,
                    Some(scales_ms.buffer()),
                    scales_layout.start_offset() * scales.dtype().size_in_bytes(),
                );
                encoder.set_buffer(
                    2,
                    Some(qz_ms.buffer()),
                    qz_layout.start_offset() * qzeros.dtype().size_in_bytes(),
                );
                encoder.set_buffer(
                    3,
                    Some(out_ms.buffer()),
                    out_layout.start_offset() * out.dtype().size_in_bytes(),
                );

                let total_elements_u32 = total_elements as u32;
                encoder.set_bytes(4, &total_elements_u32);

                let threads_per_grid = objc2_metal::MTLSize {
                    width: total_elements,
                    height: 1,
                    depth: 1,
                };
                let threads_per_threadgroup = objc2_metal::MTLSize {
                    width: std::cmp::min(
                        self.pipeline_awq.max_total_threads_per_threadgroup(),
                        total_elements,
                    ),
                    height: 1,
                    depth: 1,
                };
                encoder.dispatch_threads(threads_per_grid, threads_per_threadgroup);
            }
        }

        Ok(out)
    }

    #[cfg(not(feature = "metal"))]
    pub fn dequantize_awq(
        &self,
        _q: &Tensor,
        _s: &Tensor,
        _z: &Tensor,
        _tot: usize,
    ) -> Result<Tensor> {
        unreachable!()
    }

    /// Dequantize a GPTQ packed Int4 tensor to Float16 natively on the Metal GPU
    #[cfg(feature = "metal")]
    pub fn dequantize_gptq(
        &self,
        qweight: &Tensor,
        scales: &Tensor,
        qzeros: &Tensor,
        g_idx: &Tensor,
        total_elements: usize,
    ) -> Result<Tensor> {
        tracing::info!("Running native Metal GPTQ dequantization...");
        let out = Tensor::zeros(total_elements, candle_core::DType::F16, qweight.device())?;

        if let Device::Metal(metal_device) = qweight.device() {
            let (qw_storage, qw_layout) = qweight.storage_and_layout();
            let (scales_storage, scales_layout) = scales.storage_and_layout();
            let (qz_storage, qz_layout) = qzeros.storage_and_layout();
            let (g_storage, g_layout) = g_idx.storage_and_layout();
            let (out_storage, out_layout) = out.storage_and_layout();

            if let (
                candle_core::Storage::Metal(qw_ms),
                candle_core::Storage::Metal(scales_ms),
                candle_core::Storage::Metal(qz_ms),
                candle_core::Storage::Metal(g_ms),
                candle_core::Storage::Metal(out_ms),
            ) = (
                &*qw_storage,
                &*scales_storage,
                &*qz_storage,
                &*g_storage,
                &*out_storage,
            ) {
                let encoder = metal_device.command_encoder()?;
                encoder.set_label("dequantize_gptq");
                encoder.set_compute_pipeline_state(&self.pipeline_gptq);

                encoder.set_buffer(
                    0,
                    Some(qw_ms.buffer()),
                    qw_layout.start_offset() * qweight.dtype().size_in_bytes(),
                );
                encoder.set_buffer(
                    1,
                    Some(scales_ms.buffer()),
                    scales_layout.start_offset() * scales.dtype().size_in_bytes(),
                );
                encoder.set_buffer(
                    2,
                    Some(qz_ms.buffer()),
                    qz_layout.start_offset() * qzeros.dtype().size_in_bytes(),
                );
                encoder.set_buffer(
                    3,
                    Some(g_ms.buffer()),
                    g_layout.start_offset() * g_idx.dtype().size_in_bytes(),
                );
                encoder.set_buffer(
                    4,
                    Some(out_ms.buffer()),
                    out_layout.start_offset() * out.dtype().size_in_bytes(),
                );

                let total_elements_u32 = total_elements as u32;
                encoder.set_bytes(5, &total_elements_u32);

                let threads_per_grid = objc2_metal::MTLSize {
                    width: total_elements,
                    height: 1,
                    depth: 1,
                };
                let threads_per_threadgroup = objc2_metal::MTLSize {
                    width: std::cmp::min(
                        self.pipeline_gptq.max_total_threads_per_threadgroup(),
                        total_elements,
                    ),
                    height: 1,
                    depth: 1,
                };
                encoder.dispatch_threads(threads_per_grid, threads_per_threadgroup);
            }
        }

        Ok(out)
    }

    #[cfg(not(feature = "metal"))]
    pub fn dequantize_gptq(
        &self,
        _q: &Tensor,
        _s: &Tensor,
        _z: &Tensor,
        _g: &Tensor,
        _tot: usize,
    ) -> Result<Tensor> {
        unreachable!()
    }

    /// CPU fallback dequantization for AWQ Int4
    pub fn dequantize_awq_cpu(
        qweight: &Tensor,
        scales: &Tensor,
        qzeros: &Tensor,
        total_elements: usize,
    ) -> Result<Tensor> {
        let dev = qweight.device();
        let qw = qweight.to_device(&Device::Cpu)?;
        let sc = scales.to_device(&Device::Cpu)?;
        let qz = qzeros.to_device(&Device::Cpu)?;

        let qw_data = qw
            .to_dtype(candle_core::DType::U32)?
            .flatten_all()?
            .to_vec1::<u32>()?;
        let sc_data = sc
            .to_dtype(candle_core::DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let qz_data = qz
            .to_dtype(candle_core::DType::U32)?
            .flatten_all()?
            .to_vec1::<u32>()?;

        let mut output = vec![0.0f32; total_elements];

        for id in 0..total_elements {
            let in_idx = id / 8;
            let shift = (id % 8) * 4;
            if in_idx >= qw_data.len() {
                continue;
            }
            let packed_w = qw_data[in_idx];
            let w_quant = (packed_w >> shift) & 0xF;

            let block_idx = id / 128;
            if block_idx >= sc_data.len() {
                continue;
            }
            let scale = sc_data[block_idx];

            let z_idx = block_idx / 8;
            let z_shift = (block_idx % 8) * 4;
            let z_quant = if z_idx < qz_data.len() {
                let packed_z = qz_data[z_idx];
                (packed_z >> z_shift) & 0xF
            } else {
                0
            };

            output[id] = (w_quant as f32 - z_quant as f32) * scale;
        }

        Tensor::from_vec(output, total_elements, &Device::Cpu)?
            .to_device(dev)?
            .to_dtype(candle_core::DType::F16)
    }

    /// CPU fallback dequantization for GPTQ Int4
    pub fn dequantize_gptq_cpu(
        qweight: &Tensor,
        scales: &Tensor,
        qzeros: &Tensor,
        g_idx: &Tensor,
        total_elements: usize,
    ) -> Result<Tensor> {
        let dev = qweight.device();
        let qw = qweight.to_device(&Device::Cpu)?;
        let sc = scales.to_device(&Device::Cpu)?;
        let qz = qzeros.to_device(&Device::Cpu)?;
        let gi = g_idx.to_device(&Device::Cpu)?;

        let qw_data = qw
            .to_dtype(candle_core::DType::U32)?
            .flatten_all()?
            .to_vec1::<u32>()?;
        let sc_data = sc
            .to_dtype(candle_core::DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let qz_data = qz
            .to_dtype(candle_core::DType::U32)?
            .flatten_all()?
            .to_vec1::<u32>()?;
        let gi_data = gi
            .to_dtype(candle_core::DType::U32)?
            .flatten_all()?
            .to_vec1::<u32>()?;

        let mut output = vec![0.0f32; total_elements];

        for id in 0..total_elements {
            let in_idx = id / 8;
            let shift = (id % 8) * 4;
            if in_idx >= qw_data.len() {
                continue;
            }
            let packed_w = qw_data[in_idx];
            let w_quant = (packed_w >> shift) & 0xF;

            if id >= gi_data.len() {
                continue;
            }
            let group = gi_data[id] as usize;
            if group >= sc_data.len() {
                continue;
            }
            let scale = sc_data[group];

            let z_idx = group / 8;
            let z_shift = (group % 8) * 4;
            let z_quant = if z_idx < qz_data.len() {
                let packed_z = qz_data[z_idx];
                (packed_z >> z_shift) & 0xF
            } else {
                0
            };

            output[id] = (w_quant as f32 - z_quant as f32) * scale;
        }

        Tensor::from_vec(output, total_elements, &Device::Cpu)?
            .to_device(dev)?
            .to_dtype(candle_core::DType::F16)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, Tensor};

    #[test]
    fn test_cpu_fallback_dequant_awq() {
        // 16 u32 values = 128 elements of 4-bit weights
        let qweight = Tensor::from_slice(&[0x12345678u32; 16], (16,), &Device::Cpu).unwrap();
        // 1 scale for group 0
        let scales = Tensor::from_slice(&[2.0f32], (1,), &Device::Cpu).unwrap();
        // 1 zero point packed value (block 0 zero point is 3)
        let qzeros = Tensor::from_slice(&[3u32], (1,), &Device::Cpu).unwrap();

        let res = MetalQuantizer::dequantize_awq_cpu(&qweight, &scales, &qzeros, 128).unwrap();
        let res_vec = res.to_dtype(DType::F32).unwrap().to_vec1::<f32>().unwrap();

        assert_eq!(res_vec.len(), 128);
        // element index 0: w_quant = 0x8 = 8.0, z_quant = 3.0. scale = 2.0.
        // expected = (8.0 - 3.0) * 2.0 = 10.0
        assert!((res_vec[0] - 10.0).abs() < 1e-3);

        // element index 1: w_quant = 0x7 = 7.0, z_quant = 3.0. scale = 2.0.
        // expected = (7.0 - 3.0) * 2.0 = 8.0
        assert!((res_vec[1] - 8.0).abs() < 1e-3);
    }

    #[test]
    fn test_cpu_fallback_dequant_gptq() {
        let qweight = Tensor::from_slice(&[0x12345678u32; 16], (16,), &Device::Cpu).unwrap();
        let scales = Tensor::from_slice(&[2.0f32; 2], (2,), &Device::Cpu).unwrap();
        let qzeros = Tensor::from_slice(&[0x00000033u32], (1,), &Device::Cpu).unwrap();
        let g_idx = Tensor::from_slice(&[0u32; 128], (128,), &Device::Cpu).unwrap();

        let res =
            MetalQuantizer::dequantize_gptq_cpu(&qweight, &scales, &qzeros, &g_idx, 128).unwrap();
        let res_vec = res.to_dtype(DType::F32).unwrap().to_vec1::<f32>().unwrap();

        assert_eq!(res_vec.len(), 128);
        assert!((res_vec[0] - 10.0).abs() < 1e-3);
        assert!((res_vec[1] - 8.0).abs() < 1e-3);
    }
}
