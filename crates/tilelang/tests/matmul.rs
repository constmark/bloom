#![cfg(feature = "hardware-tests")]
// JIT-compiles kernels via the TileLang Python toolchain (numpy). Opt-in.

use bloomai_tilelang::{TileLangCompiler, TileLangKernel};

#[test]
fn test_matmul() {
    let compiler = TileLangCompiler::new().unwrap();
    let so_path = compiler.compile_matmul(64, 64, 64).unwrap();
    assert!(so_path.exists());

    let kernel = unsafe { TileLangKernel::load(&so_path).unwrap() };

    let m = 64;
    let n = 64;
    let k = 64;

    // Create test matrices
    let a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.1).collect();
    let b: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.1).collect();
    let mut c: Vec<f32> = vec![0.0; m * n];

    let ret = kernel.matmul(&a, &b, &mut c, m, n, k).unwrap();
    assert_eq!(ret, 0, "CUDA kernel returned error");

    // Verify result: c[i][j] = sum_k a[i][k] * b[k][j]
    for i in 0..m.min(3) {
        for j in 0..n.min(3) {
            let mut expected = 0.0f32;
            for l in 0..k {
                expected += a[i * k + l] * b[l * n + j];
            }
            let actual = c[i * n + j];
            // Float accumulation order differences cause small errors
            let rel_err = ((actual - expected).abs() / expected.abs().max(1.0)).abs();
            assert!(
                rel_err < 1e-3,
                "Mismatch at c[{}][{}]: expected {}, got {}, rel_err={}",
                i,
                j,
                expected,
                actual,
                rel_err
            );
        }
    }
    println!("Matmul test passed!");
}
