use anyhow::{Result, ensure};
use libloading::Library;
use std::path::Path;

type VectorAddFn = unsafe extern "C" fn(*const f32, *const f32, *mut f32, i32) -> i32;
type MatmulFn = unsafe extern "C" fn(*const f32, *const f32, *mut f32, i32, i32, i32) -> i32;
type SoftmaxFn = unsafe extern "C" fn(*const f32, *mut f32, i32) -> i32;
type AttentionFn =
    unsafe extern "C" fn(*const f32, *const f32, *const f32, *mut f32, i32, i32) -> i32;
type MropeFn = unsafe extern "C" fn(
    *const f32,
    *const f32,
    *const f32,
    *const f32,
    *mut f32,
    *mut f32,
    i32,
    i32,
    i32,
    i32,
    i32,
) -> i32;

pub struct TileLangKernel {
    #[allow(dead_code)]
    lib: Option<Library>,
    vector_add_fn: Option<VectorAddFn>,
    matmul_fn: Option<MatmulFn>,
    softmax_fn: Option<SoftmaxFn>,
    attention_fn: Option<AttentionFn>,
    mrope_fn: Option<MropeFn>,
}

impl TileLangKernel {
    /// Load a generated TileLang shared library.
    ///
    /// # Safety
    ///
    /// The caller must ensure `path` points to a trusted library compiled for the
    /// current process and ABI. The exported symbols must match the signatures
    /// expected by this wrapper.
    pub unsafe fn load(path: &Path) -> Result<Self> {
        unsafe {
            let lib = Library::new(path)?;

            let vector_add_fn: Option<VectorAddFn> = {
                lib.get::<VectorAddFn>(b"vector_add_launch")
                    .ok()
                    .map(|s| *s)
            };

            let matmul_fn: Option<MatmulFn> =
                { lib.get::<MatmulFn>(b"matmul_launch").ok().map(|s| *s) };

            let softmax_fn: Option<SoftmaxFn> =
                { lib.get::<SoftmaxFn>(b"softmax_launch").ok().map(|s| *s) };

            let attention_fn: Option<AttentionFn> =
                { lib.get::<AttentionFn>(b"attention_launch").ok().map(|s| *s) };

            let mrope_fn: Option<MropeFn> =
                { lib.get::<MropeFn>(b"mrope_launch").ok().map(|s| *s) };

            Ok(Self {
                lib: Some(lib),
                vector_add_fn,
                matmul_fn,
                softmax_fn,
                attention_fn,
                mrope_fn,
            })
        }
    }

    pub fn vector_add(&self, a: &[f32], b: &[f32], c: &mut [f32]) -> Result<i32> {
        let fn_ptr = self
            .vector_add_fn
            .ok_or_else(|| anyhow::anyhow!("vector_add not supported by this kernel"))?;
        ensure!(
            a.len() == b.len() && a.len() == c.len(),
            "vector_add buffer lengths must match"
        );
        let n = checked_elements(&[a.len()])? as i32;
        let ret = unsafe { (fn_ptr)(a.as_ptr(), b.as_ptr(), c.as_mut_ptr(), n) };
        Ok(ret)
    }

    pub fn matmul(
        &self,
        a: &[f32],
        b: &[f32],
        c: &mut [f32],
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<i32> {
        let fn_ptr = self
            .matmul_fn
            .ok_or_else(|| anyhow::anyhow!("matmul not supported by this kernel"))?;
        ensure!(
            a.len() == checked_elements(&[m, k])?,
            "matmul A shape mismatch"
        );
        ensure!(
            b.len() == checked_elements(&[k, n])?,
            "matmul B shape mismatch"
        );
        ensure!(
            c.len() == checked_elements(&[m, n])?,
            "matmul output shape mismatch"
        );
        let ret = unsafe {
            (fn_ptr)(
                a.as_ptr(),
                b.as_ptr(),
                c.as_mut_ptr(),
                m as i32,
                n as i32,
                k as i32,
            )
        };
        Ok(ret)
    }

    /// Apply softmax to input vector
    pub fn softmax(&self, input: &[f32], output: &mut [f32]) -> Result<i32> {
        let fn_ptr = self
            .softmax_fn
            .ok_or_else(|| anyhow::anyhow!("softmax not supported by this kernel"))?;
        ensure!(
            input.len() == output.len(),
            "softmax buffer lengths must match"
        );
        let n = checked_elements(&[input.len()])? as i32;
        let ret = unsafe { (fn_ptr)(input.as_ptr(), output.as_mut_ptr(), n) };
        Ok(ret)
    }

    /// Multi-head attention: Q, K, V are [seq_len * head_dim], output is [seq_len * head_dim]
    pub fn attention(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        output: &mut [f32],
        seq_len: usize,
        head_dim: usize,
    ) -> Result<i32> {
        let fn_ptr = self
            .attention_fn
            .ok_or_else(|| anyhow::anyhow!("attention not supported by this kernel"))?;
        let elements = checked_elements(&[seq_len, head_dim])?;
        ensure!(
            [q.len(), k.len(), v.len(), output.len()]
                .iter()
                .all(|&len| len == elements),
            "attention buffer shape mismatch"
        );
        let ret = unsafe {
            (fn_ptr)(
                q.as_ptr(),
                k.as_ptr(),
                v.as_ptr(),
                output.as_mut_ptr(),
                seq_len as i32,
                head_dim as i32,
            )
        };
        Ok(ret)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn mrope(
        &self,
        q: &[f32],
        k: &[f32],
        cos: &[f32],
        sin: &[f32],
        q_out: &mut [f32],
        k_out: &mut [f32],
        bs: usize,
        num_heads: usize,
        num_kv_heads: usize,
        seq_len: usize,
        head_dim: usize,
    ) -> Result<i32> {
        let fn_ptr = self
            .mrope_fn
            .ok_or_else(|| anyhow::anyhow!("mrope not supported by this kernel"))?;
        // The generated Qwen MRoPE kernel uses fixed 64-element rotation
        // partners and [24, 20, 20] sections. Other head sizes would read OOB.
        ensure!(head_dim == 128, "mrope requires head_dim=128");
        let q_elements = checked_elements(&[bs, num_heads, seq_len, head_dim])?;
        let k_elements = checked_elements(&[bs, num_kv_heads, seq_len, head_dim])?;
        let rotation_elements = checked_elements(&[3, bs, seq_len, head_dim])?;
        ensure!(
            q.len() == q_elements && q_out.len() == q_elements,
            "mrope Q shape mismatch"
        );
        ensure!(
            k.len() == k_elements && k_out.len() == k_elements,
            "mrope K shape mismatch"
        );
        ensure!(
            cos.len() == rotation_elements && sin.len() == rotation_elements,
            "mrope rotation shape mismatch"
        );
        let ret = unsafe {
            (fn_ptr)(
                q.as_ptr(),
                k.as_ptr(),
                cos.as_ptr(),
                sin.as_ptr(),
                q_out.as_mut_ptr(),
                k_out.as_mut_ptr(),
                bs as i32,
                num_heads as i32,
                num_kv_heads as i32,
                seq_len as i32,
                head_dim as i32,
            )
        };
        Ok(ret)
    }
}

// Generated kernels use signed 32-bit indices for both dimensions and products.
pub(crate) fn checked_elements(shape: &[usize]) -> Result<usize> {
    shape.iter().try_fold(1usize, |elements, &dimension| {
        ensure!(dimension > 0, "kernel dimensions must be positive");
        let elements = elements
            .checked_mul(dimension)
            .ok_or_else(|| anyhow::anyhow!("kernel shape overflows usize"))?;
        ensure!(
            elements <= i32::MAX as usize,
            "kernel shape exceeds the signed 32-bit ABI"
        );
        Ok(elements)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    unsafe extern "C" fn unreachable_vector(
        _: *const f32,
        _: *const f32,
        _: *mut f32,
        _: i32,
    ) -> i32 {
        panic!("invalid vector reached native code")
    }
    unsafe extern "C" fn unreachable_softmax(_: *const f32, _: *mut f32, _: i32) -> i32 {
        panic!("invalid softmax reached native code")
    }
    unsafe extern "C" fn unreachable_matmul(
        _: *const f32,
        _: *const f32,
        _: *mut f32,
        _: i32,
        _: i32,
        _: i32,
    ) -> i32 {
        panic!("invalid matmul reached native code")
    }
    unsafe extern "C" fn unreachable_attention(
        _: *const f32,
        _: *const f32,
        _: *const f32,
        _: *mut f32,
        _: i32,
        _: i32,
    ) -> i32 {
        panic!("invalid attention reached native code")
    }
    unsafe extern "C" fn unreachable_mrope(
        _: *const f32,
        _: *const f32,
        _: *const f32,
        _: *const f32,
        _: *mut f32,
        _: *mut f32,
        _: i32,
        _: i32,
        _: i32,
        _: i32,
        _: i32,
    ) -> i32 {
        panic!("invalid mrope reached native code")
    }

    #[test]
    fn malformed_shapes_never_enter_native_code() {
        let kernel = TileLangKernel {
            lib: None,
            vector_add_fn: Some(unreachable_vector),
            matmul_fn: Some(unreachable_matmul),
            softmax_fn: Some(unreachable_softmax),
            attention_fn: Some(unreachable_attention),
            mrope_fn: Some(unreachable_mrope),
        };
        assert!(kernel.vector_add(&[1.0], &[], &mut []).is_err());
        assert!(kernel.softmax(&[], &mut []).is_err());
        assert!(kernel.softmax(&[1.0], &mut []).is_err());
        assert!(kernel.matmul(&[], &[], &mut [], usize::MAX, 2, 2).is_err());
        assert!(kernel.attention(&[], &[], &[], &mut [], 1, 1).is_err());
        assert!(
            kernel
                .mrope(
                    &[0.0; 64],
                    &[0.0; 64],
                    &[0.0; 192],
                    &[0.0; 192],
                    &mut [0.0; 64],
                    &mut [0.0; 64],
                    1,
                    1,
                    1,
                    1,
                    64
                )
                .is_err()
        );
    }

    #[test]
    fn rejects_zero_overflow_and_signed_index_overflow() {
        for shape in [
            &[0][..],
            &[usize::MAX],
            &[65536, 65536],
            &[i32::MAX as usize, 2],
        ] {
            assert!(checked_elements(shape).is_err());
        }
        assert_eq!(checked_elements(&[2, 3, 128]).unwrap(), 768);
    }
}
