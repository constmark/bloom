//! Checkpoint tokenizer: a thin wrapper over the Hugging Face `tokenizers` crate that loads
//! `tokenizer.json` directly and resolves Laya's special token ids. Mirrors the special-id
//! resolution in `laya-ts/src/tokenizer.ts`.

use crate::common::Tokenizer as TokTrait;
use crate::error::{LayaError, Result};
use std::path::Path;
use tokenizers::Tokenizer as HfTokenizer;

/// Fallback special ids of the Laya ModernBERT checkpoint (HF added tokens).
const FB_CLS: u32 = 50281;
const FB_SEP: u32 = 50282;
const FB_MASK: u32 = 50284;
const FB_PAD: u32 = 50283;
const FB_UNK: u32 = 50280;

/// Alias lookup order per special (ModernBERT `[X]` names first, Gemma `<x>` names after).
const CLS_ALIASES: &[&str] = &["[CLS]", "<bos>", "<s>"];
const SEP_ALIASES: &[&str] = &["[SEP]", "<eos>", "</s>"];
const PAD_ALIASES: &[&str] = &["[PAD]", "<pad>"];
const MASK_ALIASES: &[&str] = &["[MASK]", "<mask>"];
const UNK_ALIASES: &[&str] = &["[UNK]", "<unk>"];

/// A loaded checkpoint tokenizer.
pub struct LayaTokenizer {
    inner: HfTokenizer,
    cls: u32,
    sep: u32,
    mask: u32,
    pad: u32,
    #[allow(dead_code)]
    unk: u32,
    mask_token: String,
}

impl LayaTokenizer {
    /// Load a tokenizer from a `tokenizer.json` file.
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let inner = HfTokenizer::from_file(path.as_ref())
            .map_err(|e| LayaError::Tokenizer(e.to_string()))?;
        Ok(Self::from_hf(inner))
    }

    /// Encode with special tokens (CLS/SEP), truncated to `max_length`. Used for encoder
    /// embeddings (mean-pooling), matching HF `tok(text, truncation=True, max_length=...)`.
    pub fn encode_with_special(&self, text: &str, max_length: usize) -> Vec<u32> {
        match self.inner.encode(text, true) {
            Ok(enc) => {
                let mut ids = enc.get_ids().to_vec();
                ids.truncate(max_length);
                ids
            }
            Err(_) => Vec::new(),
        }
    }

    fn from_hf(inner: HfTokenizer) -> Self {
        let (cls, _) = resolve(&inner, CLS_ALIASES, FB_CLS);
        let (sep, _) = resolve(&inner, SEP_ALIASES, FB_SEP);
        let (pad, _) = resolve(&inner, PAD_ALIASES, FB_PAD);
        let (mask, mask_token) = resolve(&inner, MASK_ALIASES, FB_MASK);
        let (unk, _) = resolve(&inner, UNK_ALIASES, FB_UNK);
        LayaTokenizer {
            inner,
            cls,
            sep,
            mask,
            pad,
            unk,
            mask_token,
        }
    }
}

/// Resolve a special id + its token text by trying each alias in order (added tokens or vocab).
fn resolve(tok: &HfTokenizer, aliases: &[&str], fallback: u32) -> (u32, String) {
    for a in aliases {
        if let Some(id) = tok.token_to_id(a) {
            return (id, (*a).to_string());
        }
    }
    (fallback, aliases[0].to_string())
}

impl TokTrait for LayaTokenizer {
    fn cls_id(&self) -> u32 {
        self.cls
    }
    fn sep_id(&self) -> u32 {
        self.sep
    }
    fn mask_id(&self) -> u32 {
        self.mask
    }
    fn pad_id(&self) -> u32 {
        self.pad
    }
    fn mask_token(&self) -> &str {
        &self.mask_token
    }
    fn encode(&self, text: &str) -> Vec<u32> {
        // add_special_tokens = false, matching `tok(text, add_special_tokens=False)`.
        match self.inner.encode(text, false) {
            Ok(enc) => enc.get_ids().to_vec(),
            Err(_) => Vec::new(),
        }
    }
}
