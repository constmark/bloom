#![cfg(feature = "hardware-tests")]
// JIT-compiles kernels via the TileLang Python toolchain (numpy). Opt-in.

use bloomai_tilelang::{TileLangCompiler, TileLangKernel};

#[test]
fn test_attention_mini() {
    let compiler = TileLangCompiler::new().unwrap();
    let seq_len = 16;
    let head_dim = 64;
    let so_path = compiler.compile_attention(seq_len, head_dim).unwrap();
    assert!(so_path.exists());

    let kernel = unsafe { TileLangKernel::load(&so_path).unwrap() };

    // Fill Q, K, V with realistic small patterns to avoid exp overflow
    let q: Vec<f32> = (0..seq_len * head_dim)
        .map(|i| ((i % 7) as f32 - 3.0) * 0.05)
        .collect();
    let k: Vec<f32> = (0..seq_len * head_dim)
        .map(|i| ((i % 5) as f32 - 2.0) * 0.05)
        .collect();
    let v: Vec<f32> = (0..seq_len * head_dim)
        .map(|i| ((i % 10) as f32) * 0.1)
        .collect();
    let mut output: Vec<f32> = vec![0.0; seq_len * head_dim];

    let ret = kernel
        .attention(&q, &k, &v, &mut output, seq_len, head_dim)
        .unwrap();
    assert_eq!(ret, 0, "Attention kernel returned error");

    // Verify output structure: should contain non-zero valid numbers
    let mut non_zero = 0;
    for &val in &output {
        assert!(
            !val.is_nan() && !val.is_infinite(),
            "Attention output contains invalid float values (got {})",
            val
        );
        if val.abs() > 1e-5 {
            non_zero += 1;
        }
    }
    assert!(non_zero > 0, "Attention output should not be all zeros");

    println!("Attention mini test passed!");
}

#[test]
fn test_attention_small() {
    let compiler = TileLangCompiler::new().unwrap();
    let seq_len = 128;
    let head_dim = 64;
    let so_path = compiler.compile_attention(seq_len, head_dim).unwrap();
    assert!(so_path.exists());

    let kernel = unsafe { TileLangKernel::load(&so_path).unwrap() };

    let q: Vec<f32> = (0..seq_len * head_dim)
        .map(|i| ((i % 7) as f32 - 3.0) * 0.05)
        .collect();
    let k: Vec<f32> = (0..seq_len * head_dim)
        .map(|i| ((i % 5) as f32 - 2.0) * 0.05)
        .collect();
    let v: Vec<f32> = (0..seq_len * head_dim)
        .map(|i| ((i % 10) as f32) * 0.1)
        .collect();
    let mut output: Vec<f32> = vec![0.0; seq_len * head_dim];

    let ret = kernel
        .attention(&q, &k, &v, &mut output, seq_len, head_dim)
        .unwrap();
    assert_eq!(ret, 0, "attention kernel returned error");

    let mut non_zero = 0;
    for &val in &output {
        assert!(
            !val.is_nan() && !val.is_infinite(),
            "Attention output contains invalid float values (got {})",
            val
        );
        if val.abs() > 1e-5 {
            non_zero += 1;
        }
    }
    assert!(non_zero > 0, "Attention output should not be all zeros");
}

#[test]
fn test_attention_large() {
    // This would have caused stack buffer overflow and crashed with segfault before the fix
    let compiler = TileLangCompiler::new().unwrap();
    let seq_len = 512;
    let head_dim = 64;
    let so_path = compiler.compile_attention(seq_len, head_dim).unwrap();
    assert!(so_path.exists());

    let kernel = unsafe { TileLangKernel::load(&so_path).unwrap() };

    let q: Vec<f32> = (0..seq_len * head_dim)
        .map(|i| ((i % 7) as f32 - 3.0) * 0.05)
        .collect();
    let k: Vec<f32> = (0..seq_len * head_dim)
        .map(|i| ((i % 5) as f32 - 2.0) * 0.05)
        .collect();
    let v: Vec<f32> = (0..seq_len * head_dim)
        .map(|i| ((i % 10) as f32) * 0.1)
        .collect();
    let mut output: Vec<f32> = vec![0.0; seq_len * head_dim];

    let ret = kernel
        .attention(&q, &k, &v, &mut output, seq_len, head_dim)
        .unwrap();
    assert_eq!(ret, 0, "attention kernel returned error");

    let mut non_zero = 0;
    for &val in &output {
        assert!(
            !val.is_nan() && !val.is_infinite(),
            "Attention output contains invalid float values (got {})",
            val
        );
        if val.abs() > 1e-5 {
            non_zero += 1;
        }
    }
    assert!(non_zero > 0, "Attention output should not be all zeros");
}
