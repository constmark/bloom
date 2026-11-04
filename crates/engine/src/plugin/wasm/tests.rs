use super::*;
use crate::plugin::{PluginManager, PluginManifest};
use crate::processor::ProcessorRegistry;
use serde_json::json;
use std::sync::Arc;

const REPLACE_Z: &str = include_str!("../../../examples/wasm_processor/replace_z.wat");

fn manifest() -> PluginManifest {
    serde_json::from_value(json!({
        "metadata": {"name":"test.wasm", "version":"1.0.0", "description":"bounded WASM fixture",
            "author":"Bloom", "license":"Apache-2.0", "platforms":[PluginManager::current_platform()],
            "min_runtime_version":"0.1.0"},
        "entry_point":{"type":"wasm", "path":"processor.wasm"}
    })).unwrap()
}

fn spec() -> ProcessorSpec {
    ProcessorSpec {
        name: "test.wasm".into(),
        kind: ProcessorKind::Custom("inline-blocks".into()),
        version: "1".into(),
        inputs: vec![Modality::Multi],
        outputs: vec![Modality::Multi],
        parameters: Default::default(),
    }
}

fn load(wat: &str, limits: WasmLimits) -> Result<WasmProcessor> {
    let root = tempfile::tempdir()?;
    std::fs::write(root.path().join("processor.wasm"), wat::parse_str(wat)?)?;
    PluginManager::load_wasm_processor(&manifest(), root.path(), spec(), limits)
}

fn module(process: &str) -> String {
    format!(
        r#"(module
        (memory (export "memory") 1 128)
        (func (export "bloom_plugin_abi_version") (result i32) i32.const 1)
        (func (export "bloom_alloc") (param i32) (result i32) i32.const 0)
        (func (export "bloom_process") (param i32 i32) (result i64) {process}))"#
    )
}

fn input() -> Vec<DataBlock> {
    vec![DataBlock::Text("zebra 中文".into())]
}
fn output_module(output: &str) -> String {
    let escaped = output
        .as_bytes()
        .iter()
        .map(|byte| format!("\\{byte:02x}"))
        .collect::<String>();
    format!(
        r#"(module
        (memory (export "memory") 1 128)
        (data (i32.const 4096) "{escaped}")
        (func (export "bloom_plugin_abi_version") (result i32) i32.const 1)
        (func (export "bloom_alloc") (param i32) (result i32) i32.const 0)
        (func (export "bloom_process") (param i32 i32) (result i64)
            i64.const {}))"#,
        ((output.len() as u64) << 32) | 4096
    )
}

#[test]
fn manager_executes_real_wasm_through_the_existing_processor_registry() {
    let processor = load(REPLACE_Z, WasmLimits::default()).unwrap();
    let mut registry = ProcessorRegistry::default();
    registry.register(Box::new(processor));
    assert_eq!(registry.specs()[0].name, "test.wasm");
    for _ in 0..3 {
        let blocks = vec![
            DataBlock::Text("zebra 中文".into()),
            DataBlock::Tokens(vec![1, 2]),
            DataBlock::Tensor(vec![1.0, -2.0]),
        ];
        let actual = registry.get("test.wasm").unwrap().process(blocks).unwrap();
        assert_eq!(
            actual,
            vec![
                DataBlock::Text("Zebra 中文".into()),
                DataBlock::Tokens(vec![1, 2]),
                DataBlock::Tensor(vec![1.0, -2.0])
            ]
        );
    }
}

#[test]
fn guest_memory_growth_and_repeated_fuel_suspension_preserve_large_payloads() {
    let processor = load(REPLACE_Z, WasmLimits::default()).unwrap();
    let result = processor
        .process(vec![DataBlock::Text("z".repeat(70_000))])
        .unwrap();
    assert_eq!(result, vec![DataBlock::Text("Z".repeat(70_000))]);
}

#[test]
fn an_expired_deadline_returns_no_result_and_releases_admission() {
    let mut processor = load(REPLACE_Z, WasmLimits::default()).unwrap();
    processor.limits.execution_timeout = Duration::from_nanos(1);
    let error = processor.process(input()).unwrap_err();
    assert!(error.to_string().contains("deadline"));
    assert_eq!(processor.active_calls.load(Ordering::Acquire), 0);
}

#[test]
fn module_loading_rejects_start_functions_imports_and_incompatible_exports() {
    for source in [
        "(module (import \"wasi_snapshot_preview1\" \"fd_write\" (func)))".to_string(),
        "(module (import \"env\" \"memory\" (memory 1)))".to_string(),
        "(module (func $start (loop br 0)) (start $start))".to_string(),
        "(module)".to_string(),
        REPLACE_Z.replace("bloom_alloc", "wrong_alloc"),
        REPLACE_Z.replace("(result i32) i32.const 1)", "(result i32) i32.const 2)"),
    ] {
        assert!(
            load(&source, WasmLimits::default()).is_err(),
            "accepted {source}"
        );
    }
}

#[test]
fn traps_fuel_exhaustion_and_invalid_memory_never_publish_a_result() {
    for process in [
        "unreachable",
        "(loop br 0) i64.const 0",
        "i32.const 129 memory.grow drop i64.const 0",
        "i64.const 4294967297", // one byte at ptr 1: invalid JSON
        "i64.const 8589934591", // one byte at unsigned ptr MAX
        "i64.const -1",         // both pointer and length out of bounds
    ] {
        let limits = WasmLimits {
            fuel: 25_000,
            memory_bytes: PAGE_BYTES,
            io_bytes: 1024,
            ..Default::default()
        };
        let processor = load(&module(process), limits).unwrap();
        assert!(processor.process(input()).is_err(), "accepted {process}");
        assert_eq!(processor.active_calls.load(Ordering::Acquire), 0);
    }
    let bad_alloc = module("i64.const 0").replace("i32.const 0)", "i32.const -1)");
    assert!(
        load(&bad_alloc, WasmLimits::default())
            .unwrap()
            .process(input())
            .is_err()
    );
    let version_loop = REPLACE_Z.replace(
        "(result i32) i32.const 1)",
        "(result i32) (loop br 0) i32.const 1)",
    );
    assert!(
        load(
            &version_loop,
            WasmLimits {
                fuel: 1000,
                ..Default::default()
            }
        )
        .is_err()
    );
    assert!(
        load(
            &module("i64.const 0").replace(
                "(memory (export \"memory\") 1 128)",
                "(memory (export \"memory\") 2 128)"
            ),
            WasmLimits {
                memory_bytes: PAGE_BYTES,
                io_bytes: 1024,
                ..Default::default()
            }
        )
        .is_err()
    );
}

#[test]
fn envelope_semantics_and_inline_only_data_are_enforced() {
    for output in [
        json!({"schema_version":2,"object":ENVELOPE_OBJECT,"blocks":[{"Text":"no"}]}),
        json!({"schema_version":1,"object":"unknown","blocks":[{"Text":"no"}]}),
        json!({"schema_version":1,"object":ENVELOPE_OBJECT,"blocks":[{"Text":"no"}],"unexpected":true}),
        json!({"schema_version":1,"object":ENVELOPE_OBJECT,"blocks":[{"AudioFile":{"path":"/etc/passwd","language":null}}]}),
        json!({"schema_version":1,"object":ENVELOPE_OBJECT,"blocks":[{"Tensor":[]}]}),
        json!({"schema_version":1,"object":ENVELOPE_OBJECT,"blocks":[]}),
        json!({"schema_version":1,"object":ENVELOPE_OBJECT,"blocks":vec![json!({"Text":""}); 65]}),
    ] {
        let processor = load(&output_module(&output.to_string()), WasmLimits::default()).unwrap();
        assert!(processor.process(input()).is_err(), "accepted {output}");
    }
    let mut processor = load(REPLACE_Z, WasmLimits::default()).unwrap();
    processor.spec.outputs = vec![Modality::Audio];
    assert!(processor.process(input()).is_err());
    processor.spec.inputs = vec![Modality::Audio];
    assert!(processor.process(input()).is_err());
}

#[test]
fn input_serialization_is_bounded_and_invalid_values_are_rejected_before_execution() {
    let processor = load(
        REPLACE_Z,
        WasmLimits {
            io_bytes: 128,
            ..Default::default()
        },
    )
    .unwrap();
    for blocks in [
        vec![DataBlock::Text("z".repeat(4096))],
        vec![DataBlock::Tensor(vec![f32::NAN])],
        vec![DataBlock::Text("z".into()); MAX_BLOCKS + 1],
        vec![DataBlock::AudioFile {
            path: "/secret".into(),
            language: None,
        }],
    ] {
        assert!(processor.process(blocks).is_err());
    }
}

#[test]
fn cancellation_and_capacity_release_a_running_infinite_guest() {
    let processor = Arc::new(
        load(
            &module("(loop br 0) i64.const 0"),
            WasmLimits {
                fuel: 50_000_000,
                execution_timeout: Duration::from_secs(10),
                concurrent_calls: 1,
                ..Default::default()
            },
        )
        .unwrap(),
    );
    let cancelled = Arc::new(AtomicBool::new(false));
    let worker_processor = Arc::clone(&processor);
    let worker_cancelled = Arc::clone(&cancelled);
    let worker = std::thread::spawn(move || {
        worker_processor.process_cancellable(input(), &worker_cancelled)
    });
    let deadline = Instant::now() + Duration::from_secs(2);
    while processor.active_calls.load(Ordering::Acquire) == 0 {
        assert!(Instant::now() < deadline);
        std::thread::yield_now();
    }
    assert!(
        processor
            .process(input())
            .unwrap_err()
            .to_string()
            .contains("concurrency")
    );
    // Execution has no host imports; fuel suspension supplies the cancellation checkpoint.
    std::thread::sleep(Duration::from_millis(5));
    cancelled.store(true, Ordering::Release);
    let error = worker.join().unwrap().unwrap_err().to_string();
    assert!(error.contains("cancelled"), "{error}");
    assert_eq!(processor.active_calls.load(Ordering::Acquire), 0);
    assert!(
        processor
            .process_cancellable(input(), &cancelled)
            .unwrap_err()
            .to_string()
            .contains("cancelled")
    );
    assert_eq!(processor.active_calls.load(Ordering::Acquire), 0);
}

#[test]
fn file_and_limit_validation_are_fail_closed() {
    let root = tempfile::tempdir().unwrap();
    let path = root.path().join("processor.wasm");
    for bytes in [
        b"not wasm".as_slice(),
        b"\0asm\x01\0\0\0malformed".as_slice(),
    ] {
        std::fs::write(&path, bytes).unwrap();
        assert!(
            PluginManager::load_wasm_processor(
                &manifest(),
                root.path(),
                spec(),
                WasmLimits::default()
            )
            .is_err()
        );
    }
    std::fs::write(&path, wat::parse_str(REPLACE_Z).unwrap()).unwrap();
    assert!(
        PluginManager::load_wasm_processor(
            &manifest(),
            root.path(),
            spec(),
            WasmLimits {
                module_bytes: 8,
                ..Default::default()
            }
        )
        .is_err()
    );
    assert!(
        load(
            REPLACE_Z,
            WasmLimits {
                memory_bytes: usize::MAX,
                ..Default::default()
            }
        )
        .is_err()
    );
    assert!(
        load(
            REPLACE_Z,
            WasmLimits {
                fuel: 0,
                ..Default::default()
            }
        )
        .is_err()
    );
    assert!(
        load(
            REPLACE_Z,
            WasmLimits {
                concurrent_calls: 0,
                ..Default::default()
            }
        )
        .is_err()
    );
    #[cfg(unix)]
    {
        let link = root.path().join("linked.wasm");
        std::os::unix::fs::symlink(&path, &link).unwrap();
        let mut linked = manifest();
        linked.entry_point.path = "linked.wasm".into();
        assert!(
            PluginManager::load_wasm_processor(&linked, root.path(), spec(), WasmLimits::default())
                .is_err()
        );
    }
}

#[test]
fn allocator_and_abi_execution_share_fuel_and_cancellation_boundaries() {
    let allocator_loop = module("unreachable").replace(
        "(func (export \"bloom_alloc\") (param i32) (result i32) i32.const 0)",
        "(func (export \"bloom_alloc\") (param i32) (result i32) (loop br 0) i32.const 0)",
    );
    let processor = load(
        &allocator_loop,
        WasmLimits {
            fuel: 1000,
            ..Default::default()
        },
    )
    .unwrap();
    let error = processor.process(input()).unwrap_err().to_string();
    assert!(
        error.contains("fuel"),
        "allocator bypassed fuel accounting: {error}"
    );
    assert_eq!(processor.active_calls.load(Ordering::Acquire), 0);

    // A long, finite version function lets module loading negotiate normally.
    // Runtime cancellation must still interrupt the next instance's ABI call,
    // before the deliberately trapping process function is ever reached.
    let abi_loop = module("unreachable").replace(
        "(func (export \"bloom_plugin_abi_version\") (result i32) i32.const 1)",
        r#"(func (export "bloom_plugin_abi_version") (result i32) (local $remaining i32)
            i32.const 5000000 local.set $remaining
            (loop $busy
                local.get $remaining i32.const 1 i32.sub local.tee $remaining br_if $busy)
            i32.const 1)"#,
    );
    for source in [&allocator_loop, &abi_loop] {
        let mut processor = Arc::new(
            load(
                source,
                WasmLimits {
                    fuel: 50_000_000,
                    execution_timeout: Duration::from_secs(10),
                    ..Default::default()
                },
            )
            .unwrap(),
        );
        let cancelled = Arc::new(AtomicBool::new(false));
        let worker_processor = Arc::clone(&processor);
        let worker_cancelled = Arc::clone(&cancelled);
        let worker = std::thread::spawn(move || {
            worker_processor.process_cancellable(input(), &worker_cancelled)
        });
        let ready_deadline = Instant::now() + Duration::from_secs(2);
        while processor.active_calls.load(Ordering::Acquire) == 0 {
            assert!(Instant::now() < ready_deadline);
            std::thread::yield_now();
        }
        std::thread::sleep(Duration::from_millis(5));
        cancelled.store(true, Ordering::Release);
        let error = worker.join().unwrap().unwrap_err().to_string();
        assert!(
            error.contains("cancelled"),
            "guest phase ignored cancellation: {error}"
        );
        assert_eq!(processor.active_calls.load(Ordering::Acquire), 0);
        Arc::get_mut(&mut processor)
            .unwrap()
            .limits
            .execution_timeout = Duration::from_millis(5);
        let error = processor.process(input()).unwrap_err().to_string();
        assert!(
            error.contains("deadline"),
            "guest phase ignored its execution deadline: {error}"
        );
        assert_eq!(processor.active_calls.load(Ordering::Acquire), 0);
    }
}

#[test]
fn guest_tables_recursion_and_value_stacks_remain_bounded() {
    let excessive_table = module("unreachable").replace(
        "(memory (export \"memory\") 1 128)",
        "(memory (export \"memory\") 1 128) (table 4097 funcref)",
    );
    assert!(load(&excessive_table, WasmLimits::default()).is_err());
    let recursion = module("call $recurse").replace(
        "(func (export \"bloom_process\")",
        "(func $recurse (result i64) call $recurse) (func (export \"bloom_process\")",
    );
    let error = load(&recursion, WasmLimits::default())
        .unwrap()
        .process(input())
        .unwrap_err()
        .to_string();
    assert!(
        error.contains("stack"),
        "unbounded recursion did not hit its stack limit: {error}"
    );
    // 40,000 64-bit locals exceed the configured 256 KiB value stack without
    // requiring recursion, a huge module, or an enormous fuel budget.
    let locals = std::iter::repeat_n("i64", 40000)
        .collect::<Vec<_>>()
        .join(" ");
    let oversized_stack = module("i64.const 0").replace(
        "(param i32 i32) (result i64)",
        &format!("(param i32 i32) (result i64) (local {locals})"),
    );
    let result = load(&oversized_stack, WasmLimits::default())
        .and_then(|processor| processor.process(input()));
    assert!(result.is_err(), "oversized guest value stack was accepted");
}

#[test]
fn sequential_and_concurrent_calls_cannot_share_guest_global_state() {
    let source = module(
        r#"
        global.get $calls i32.const 1 i32.add global.set $calls
        global.get $calls i32.const 1 i32.ne if unreachable end
        local.get 1 i64.extend_i32_u i64.const 32 i64.shl
        local.get 0 i64.extend_i32_u i64.or"#,
    )
    .replace(
        "(memory (export \"memory\") 1 128)",
        "(memory (export \"memory\") 1 128) (global $calls (mut i32) (i32.const 0))",
    );
    let processor = Arc::new(load(&source, WasmLimits::default()).unwrap());
    for _ in 0..3 {
        assert_eq!(processor.process(input()).unwrap(), input());
    }
    let workers = (0..4)
        .map(|_| {
            let processor = Arc::clone(&processor);
            std::thread::spawn(move || processor.process(input()))
        })
        .collect::<Vec<_>>();
    for worker in workers {
        assert_eq!(worker.join().unwrap().unwrap(), input());
    }
    assert_eq!(processor.active_calls.load(Ordering::Acquire), 0);
}
