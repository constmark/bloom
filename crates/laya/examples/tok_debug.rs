//! Debug: encode fixed strings with a checkpoint tokenizer. `LAYA_TOK=/path/to/tokenizer.json`.
use laya::common::Tokenizer;
use laya::tokenizer::LayaTokenizer;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::var("LAYA_TOK").expect("set LAYA_TOK=/path/to/tokenizer.json");
    let tok = LayaTokenizer::from_file(&path)?;
    println!(
        "cls={} sep={} mask={} pad={} mask_token={:?}",
        tok.cls_id(),
        tok.sep_id(),
        tok.mask_id(),
        tok.pad_id(),
        tok.mask_token()
    );
    for s in [
        "choice question: What is this?",
        " refund: money returned",
        "私は二重に請求されました",
    ] {
        println!("{s:?} -> {:?}", tok.encode(s));
    }
    Ok(())
}
