#include <metal_stdlib>
using namespace metal;

// AWQ and GPTQ Metal compute kernels for Int4 dequantization
// Directly supports the Bloom engine's metal implementation

kernel void dequantize_awq_int4(
    device const uint *qweight [[buffer(0)]],
    device const half *scales [[buffer(1)]],
    device const uint *qzeros [[buffer(2)]],
    device half *output [[buffer(3)]],
    constant uint &total_elements [[buffer(4)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= total_elements) return;

    uint in_idx = id / 8;
    uint shift = (id % 8) * 4;
    
    uint packed_w = qweight[in_idx];
    uint w_quant = (packed_w >> shift) & 0xF;
    
    // AWQ default block size is 128
    uint block_idx = id / 128;
    
    uint z_idx = block_idx / 8;
    uint z_shift = (block_idx % 8) * 4;
    uint packed_z = qzeros[z_idx];
    uint z_quant = (packed_z >> z_shift) & 0xF;
    
    half scale = scales[block_idx];
    
    output[id] = (half(w_quant) - half(z_quant)) * scale;
}

kernel void dequantize_gptq_int4(
    device const uint *qweight [[buffer(0)]],
    device const half *scales [[buffer(1)]],
    device const uint *qzeros [[buffer(2)]],
    device const uint *g_idx [[buffer(3)]],
    device half *output [[buffer(4)]],
    constant uint &total_elements [[buffer(5)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= total_elements) return;

    uint in_idx = id / 8;
    // GPTQ packs columns differently sometimes, assuming standard packing:
    uint shift = (id % 8) * 4;
    
    uint packed_w = qweight[in_idx];
    uint w_quant = (packed_w >> shift) & 0xF;
    
    uint group = g_idx[id];
    
    uint z_idx = group / 8;
    uint z_shift = (group % 8) * 4;
    uint packed_z = qzeros[z_idx];
    uint z_quant = (packed_z >> z_shift) & 0xF;
    // Note: Some GPTQ implementations add 1 to the zero point.
    // z_quant = z_quant + 1;
    
    half scale = scales[group];
    
    output[id] = (half(w_quant) - half(z_quant)) * scale;
}
