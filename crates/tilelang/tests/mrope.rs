#![cfg(feature = "hardware-tests")]
// JIT-compiles kernels via the TileLang Python toolchain (numpy). Opt-in.

use bloomai_tilelang::{TileLangCompiler, TileLangKernel};

#[test]
fn test_mrope_basic() {
    let compiler = TileLangCompiler::new().unwrap();
    let so_path = compiler.compile_mrope().unwrap();
    assert!(so_path.exists());

    let kernel = unsafe { TileLangKernel::load(&so_path).unwrap() };

    let bs = 1;
    let num_heads = 2;
    let seq_len = 8;
    let head_dim = 128;

    let q = vec![1.0f32; bs * num_heads * seq_len * head_dim];
    let k = vec![2.0f32; bs * num_heads * seq_len * head_dim];
    let cos = vec![0.8f32; 3 * bs * seq_len * head_dim];
    let sin = vec![0.6f32; 3 * bs * seq_len * head_dim];

    let mut q_out = vec![0.0f32; bs * num_heads * seq_len * head_dim];
    let mut k_out = vec![0.0f32; bs * num_heads * seq_len * head_dim];

    let ret = kernel
        .mrope(
            &q, &k, &cos, &sin, &mut q_out, &mut k_out, bs, num_heads, num_heads, seq_len, head_dim,
        )
        .unwrap();

    assert_eq!(ret, 0, "mrope kernel returned error code");

    // Check outputs
    for &val in &q_out {
        assert!(!val.is_nan() && !val.is_infinite());
    }
    for &val in &k_out {
        assert!(!val.is_nan() && !val.is_infinite());
    }

    // Since q = 1, k = 2, cos = 0.8, sin = 0.6
    // For c < 64: q_out = 1 * 0.8 - 1 * 0.6 = 0.2
    // For c >= 64: q_out = 1 * 0.8 + 1 * 0.6 = 1.4
    // For c < 64: k_out = 2 * 0.8 - 2 * 0.6 = 0.4
    // For c >= 64: k_out = 2 * 0.8 + 2 * 0.6 = 2.8
    assert!((q_out[0] - 0.2).abs() < 1e-5);
    assert!((q_out[64] - 1.4).abs() < 1e-5);
    assert!((k_out[0] - 0.4).abs() < 1e-5);
    assert!((k_out[64] - 2.8).abs() < 1e-5);

    println!("mrope unit test passed successfully!");
}
