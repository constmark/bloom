use anyhow::Result;
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
    lib: Library,
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

        let mrope_fn: Option<MropeFn> = { lib.get::<MropeFn>(b"mrope_launch").ok().map(|s| *s) };

        Ok(Self {
            lib,
            vector_add_fn,
            matmul_fn,
            softmax_fn,
            attention_fn,
            mrope_fn,
        })
    }

    pub fn vector_add(&self, a: &[f32], b: &[f32], c: &mut [f32]) -> Result<i32> {
        let fn_ptr = self
            .vector_add_fn
            .ok_or_else(|| anyhow::anyhow!("vector_add not supported by this kernel"))?;
        assert_eq!(a.len(), b.len());
        assert_eq!(a.len(), c.len());
        let n = a.len() as i32;
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
        assert_eq!(a.len(), m * k);
        assert_eq!(b.len(), k * n);
        assert_eq!(c.len(), m * n);
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
        assert_eq!(input.len(), output.len());
        let n = input.len() as i32;
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
        assert_eq!(q.len(), seq_len * head_dim);
        assert_eq!(k.len(), seq_len * head_dim);
        assert_eq!(v.len(), seq_len * head_dim);
        assert_eq!(output.len(), seq_len * head_dim);
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
        assert_eq!(q.len(), bs * num_heads * seq_len * head_dim);
        assert_eq!(k.len(), bs * num_kv_heads * seq_len * head_dim);
        assert_eq!(cos.len(), 3 * bs * seq_len * head_dim);
        assert_eq!(sin.len(), 3 * bs * seq_len * head_dim);
        assert_eq!(q_out.len(), bs * num_heads * seq_len * head_dim);
        assert_eq!(k_out.len(), bs * num_kv_heads * seq_len * head_dim);
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
