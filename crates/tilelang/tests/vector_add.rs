#![cfg(feature = "hardware-tests")]
// JIT-compiles kernels via the TileLang Python toolchain (numpy). Opt-in.

use bloomai_tilelang::{TileLangCompiler, TileLangKernel};

#[test]
fn test_vector_add() {
    let compiler = TileLangCompiler::new().unwrap();
    let so_path = compiler.compile_vector_add(1024).unwrap();
    assert!(so_path.exists());

    let kernel = unsafe { TileLangKernel::load(&so_path).unwrap() };

    let a: Vec<f32> = vec![1.0; 1024];
    let b: Vec<f32> = vec![2.0; 1024];
    let mut c: Vec<f32> = vec![0.0; 1024];

    let ret = kernel.vector_add(&a, &b, &mut c).unwrap();
    assert_eq!(ret, 0, "CUDA kernel returned error");

    for v in &c {
        assert!((v - 3.0).abs() < 1e-5, "Expected 3.0, got {}", v);
    }
}
