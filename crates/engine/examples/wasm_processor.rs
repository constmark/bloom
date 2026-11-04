//! Build the checked-in WAT fixture and execute it through ProcessorRegistry.
use anyhow::Result;
use bloomai_core::{Modality, ProcessorKind, ProcessorSpec};
use bloomai_engine::io::DataBlock;
use bloomai_engine::plugin::{PluginManager, WasmLimits};
use bloomai_engine::processor::ProcessorRegistry;

fn main() -> Result<()> {
    let root = tempfile::tempdir()?;
    let bytes = wat::parse_str(include_str!("wasm_processor/replace_z.wat"))?;
    std::fs::write(root.path().join("processor.wasm"), bytes)?;
    let manifest = serde_json::from_str(include_str!("wasm_processor/plugin.json"))?;
    let spec = ProcessorSpec {
        name: "bloom.example.replace-z".into(),
        kind: ProcessorKind::Custom("replace-z".into()),
        version: "1".into(),
        inputs: vec![Modality::Text],
        outputs: vec![Modality::Text],
        parameters: Default::default(),
    };
    let processor =
        PluginManager::load_wasm_processor(&manifest, root.path(), spec, WasmLimits::default())?;
    let mut registry = ProcessorRegistry::default();
    registry.register(Box::new(processor));
    let text = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "zebra 中文".into());
    let output = registry
        .get("bloom.example.replace-z")?
        .process(vec![DataBlock::Text(text)])?;
    println!("{}", serde_json::to_string(&output)?);
    Ok(())
}
