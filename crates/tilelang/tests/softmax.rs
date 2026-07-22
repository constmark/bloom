#![cfg(feature = "hardware-tests")]
// JIT-compiles kernels via the TileLang Python toolchain (numpy). Opt-in.

use bloomai_tilelang::{TileLangCompiler, TileLangKernel};

#[test]
fn test_softmax() {
    let compiler = TileLangCompiler::new().unwrap();
    let so_path = compiler.compile_softmax(256).unwrap();
    assert!(so_path.exists());

    let kernel = unsafe { TileLangKernel::load(&so_path).unwrap() };

    let input: Vec<f32> = (0..256).map(|i| (i as f32) * 0.01).collect();
    let mut output: Vec<f32> = vec![0.0; 256];

    let ret = kernel.softmax(&input, &mut output).unwrap();
    assert_eq!(ret, 0, "Softmax kernel returned error");

    // Verify softmax mathematical property: sum of outputs must be close to 1.0, and elements should be positive
    let mut sum = 0.0;
    for &val in &output {
        assert!(val > 0.0, "Softmax output must be positive, got {}", val);
        sum += val;
    }
    assert!((sum - 1.0).abs() < 1e-4, "Expected sum 1.0, got {}", sum);

    // Verify sorting order: higher input must produce higher softmax value
    for i in 1..256 {
        assert!(
            output[i] > output[i - 1],
            "Higher input must yield higher output: output[{}] = {}, output[{}] = {}",
            i,
            output[i],
            i - 1,
            output[i - 1]
        );
    }

    println!("Softmax test passed!");
}
