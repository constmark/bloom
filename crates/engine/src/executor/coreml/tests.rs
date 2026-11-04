use super::*;

#[test]
fn capability_only_declares_the_executable_numeric_cpu_surface() {
    let capability = CoreMlEngine.capability();
    assert_eq!(
        capability.maturity,
        crate::engine::BackendMaturity::Experimental
    );
    assert_eq!(capability.supported_modalities, [Modality::Multi]);
    assert!(
        !capability.supports_embeddings
            && !capability.supports_rerank
            && !capability.supports_structured_output
            && !capability.supports_streaming
    );
    assert_eq!(CoreMlEngine.default_device(), DeviceKind::Cpu);
    assert_eq!(
        CoreMlEngine.supported_devices(),
        if cfg!(target_os = "macos") {
            vec![DeviceKind::Cpu]
        } else {
            vec![]
        }
    );
}

#[test]
fn discovers_real_artifact_types_and_rejects_ambiguous_directories() {
    let directory = tempfile::tempdir().unwrap();
    let file = directory.path().join("model.mlmodel");
    std::fs::write(&file, b"fixture").unwrap();
    assert_eq!(CoreMlEngine::discover_model_file(&file).unwrap(), file);
    assert_eq!(
        CoreMlEngine::discover_model_file(directory.path()).unwrap(),
        file
    );
    let package = directory.path().join("model.mlpackage");
    std::fs::create_dir(&package).unwrap();
    assert_eq!(
        CoreMlEngine::discover_model_file(&package).unwrap(),
        package
    );
    assert!(CoreMlEngine::discover_model_file(directory.path()).is_err());
    std::fs::remove_dir(package).unwrap();
    let false_package = directory.path().join("false.mlpackage");
    std::fs::write(&false_package, b"not a package directory").unwrap();
    assert!(CoreMlEngine::discover_model_file(&false_package).is_err());
}

#[test]
fn numeric_contract_rejects_overflow_dimensions_wrong_identity_and_dtype() {
    assert_eq!(shape_elements(&[2, 3]).unwrap(), 6);
    for shape in [
        vec![],
        vec![0],
        vec![usize::MAX, 2],
        vec![MAX_COREML_VALUES + 1],
        vec![1; 17],
    ] {
        assert!(shape_elements(&shape).is_err());
    }
    let probe = serde_json::json!({"schema_version":1,"compute_units":"cpu_only","input_name":"input","input_shape":[3],"input_dtype":"float32","output_name":"output","output_kind":"double"});
    assert_eq!(
        serde_json::from_value::<CoreMlProbe>(probe.clone())
            .unwrap()
            .validate()
            .unwrap(),
        (3, 1)
    );
    for (field, value) in [
        ("schema_version", serde_json::json!(2)),
        ("compute_units", serde_json::json!("all")),
        ("input_dtype", serde_json::json!("int32")),
        ("input_shape", serde_json::json!([0])),
        ("output_shape", serde_json::json!([3])),
        ("input_name", serde_json::json!("bad\nname")),
    ] {
        let mut invalid = probe.clone();
        invalid[field] = value;
        assert!(
            serde_json::from_value::<CoreMlProbe>(invalid)
                .unwrap()
                .validate()
                .is_err()
        );
    }
}

#[test]
fn numeric_models_fail_closed_for_generation_controls() {
    let defaults = GenerationParams::default();
    validate_generation_params(&defaults).unwrap();
    validate_generation_params(&GenerationParams {
        response_format: Some(ResponseFormat::Text),
        ..defaults.clone()
    })
    .unwrap();
    for params in [
        GenerationParams {
            temperature: 0.0,
            ..defaults.clone()
        },
        GenerationParams {
            max_tokens: 1,
            ..defaults.clone()
        },
        GenerationParams {
            top_k: Some(1),
            ..defaults.clone()
        },
        GenerationParams {
            seed: Some(1),
            ..defaults.clone()
        },
        GenerationParams {
            response_format: Some(ResponseFormat::JsonObject),
            ..defaults.clone()
        },
        GenerationParams {
            logit_bias: Some([(1, 1.0)].into_iter().collect()),
            ..defaults.clone()
        },
    ] {
        assert!(validate_generation_params(&params).is_err());
    }
}

#[test]
fn unavailable_devices_fail_before_compiler_or_model_execution() {
    let directory = tempfile::tempdir().unwrap();
    std::fs::write(directory.path().join("model.mlmodel"), b"not a model").unwrap();
    for device in [DeviceKind::Gpu, DeviceKind::Npu] {
        let Err(error) = CoreMlEngine.load(directory.path(), device) else {
            panic!("unexpected accelerator load");
        };
        assert!(error.to_string().contains(if cfg!(target_os = "macos") {
            "requires --device cpu"
        } else {
            "requires macOS"
        }));
    }
}

#[cfg(unix)]
#[test]
fn worker_boundary_caps_diagnostics_and_kills_hung_processes() {
    let mut noisy = Command::new("sh");
    noisy.args(["-c", "printf 123456789"]);
    assert!(run_worker(&mut noisy, &[], Duration::from_secs(1), 4).is_err());
    let mut hung = Command::new("sleep");
    hung.arg("5");
    let started = std::time::Instant::now();
    assert!(run_worker(&mut hung, &[], Duration::from_millis(30), 1024).is_err());
    assert!(started.elapsed() < Duration::from_secs(2));
}

// Generate an 87-byte Core ML GLMRegressor using Apple's documented protobuf
// fields. This is the actual affine model y=2*x0-x1+0.5*x2+3, not a fake worker.
#[cfg(all(target_os = "macos", feature = "hardware-tests"))]
fn numeric_model_fixture(identity: bool) -> Vec<u8> {
    fn varint(mut value: usize) -> Vec<u8> {
        let mut result = Vec::new();
        while value > 127 {
            result.push(((value & 127) | 128) as u8);
            value >>= 7;
        }
        result.push(value as u8);
        result
    }
    fn field(index: usize, value: &[u8]) -> Vec<u8> {
        [varint(index << 3 | 2), varint(value.len()), value.to_vec()].concat()
    }
    fn integer(index: usize, value: usize) -> Vec<u8> {
        [varint(index << 3), varint(value)].concat()
    }
    fn feature(name: &[u8], kind: &[u8]) -> Vec<u8> {
        [field(1, name), field(3, kind)].concat()
    }
    let array = field(
        5,
        &[
            field(1, &[3]),
            integer(2, if identity { 65600 } else { 65568 }),
        ]
        .concat(),
    );
    if identity {
        let description = [
            field(1, &feature(b"input", &array)),
            field(10, &feature(b"output", &array)),
        ]
        .concat();
        return [integer(1, 1), field(2, &description), field(604, &[])].concat();
    }
    let description = [
        field(1, &feature(b"input", &array)),
        field(10, &feature(b"output", &field(2, &[]))),
        field(11, b"output"),
    ]
    .concat();
    let weights = field(
        1,
        &[2.0f64, -1.0, 0.5]
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>(),
    );
    let glm = [field(1, &weights), field(2, &3.0f64.to_le_bytes())].concat();
    [integer(1, 1), field(2, &description), field(300, &glm)].concat()
}

#[cfg(all(target_os = "macos", feature = "hardware-tests"))]
#[test]
fn real_coreml_cpu_tensor_runtime() {
    use sha2::{Digest, Sha256};
    let directory = tempfile::tempdir().unwrap();
    let model_path = directory.path().join("线性 fixture.mlmodel");
    let fixture = numeric_model_fixture(false);
    assert_eq!(fixture.len(), 87);
    std::fs::write(&model_path, &fixture).unwrap();
    let loaded_at = std::time::Instant::now();
    let model = CoreMlEngine.load(&model_path, DeviceKind::Cpu).unwrap();
    let load_ms = loaded_at.elapsed().as_millis();
    assert_eq!(model.actual_device(), Some(DeviceKind::Cpu));
    assert_eq!(model.metadata().manifest.parameters["bloom_task"], "tensor");
    let params = GenerationParams::default();
    let output = model
        .infer(
            ModelInput::Text {
                prompt: "[1,2,4]".into(),
            },
            &params,
        )
        .unwrap();
    assert_eq!(output.logits, Some(vec![5.0]));
    assert!(output.text.is_none());
    let mut chunks = Vec::new();
    model
        .infer_request(
            InferenceRequest {
                blocks: vec![DataBlock::Tensor(vec![-2.0, 3.0, 8.0])],
                params: crate::io::InferenceParams::default(),
            },
            &mut |chunk| {
                chunks.push(chunk);
                Ok(())
            },
        )
        .unwrap();
    assert!(matches!(&chunks[0],OutputChunk::Embedding(values) if values == &[0.0]));
    assert!(matches!(chunks[1], OutputChunk::End));
    assert_eq!(chunks.len(), 2);
    let mut streamed = Vec::new();
    model
        .infer_stream(
            ModelInput::Text {
                prompt: "[2,1,-4]".into(),
            },
            &params,
            &mut |chunk| {
                streamed.push(chunk);
                Ok(())
            },
        )
        .unwrap();
    assert!(matches!(&streamed[0],OutputChunk::Embedding(values) if values == &[4.0]));
    assert!(matches!(streamed[1], OutputChunk::End));
    assert_eq!(streamed.len(), 2);
    for prompt in ["Tell me a joke", "[1,2]", "[true,2,3]", "[1e100,2,3]"] {
        assert!(
            model
                .infer(
                    ModelInput::Text {
                        prompt: prompt.into()
                    },
                    &params
                )
                .is_err(),
            "accepted {prompt}"
        );
    }
    assert!(
        model
            .infer_request(
                InferenceRequest {
                    blocks: vec![DataBlock::Tensor(vec![f32::NAN, 1., 2.])],
                    params: crate::io::InferenceParams::default()
                },
                &mut |_| Ok(())
            )
            .is_err()
    );
    assert!(
        model
            .infer_request(
                InferenceRequest {
                    blocks: vec![
                        DataBlock::Tensor(vec![1., 2., 4.]),
                        DataBlock::Text("ignored?".into())
                    ],
                    params: crate::io::InferenceParams::default()
                },
                &mut |_| Ok(())
            )
            .is_err()
    );
    assert!(
        model
            .infer(
                ModelInput::Text {
                    prompt: "[1,2,4]".into()
                },
                &GenerationParams {
                    seed: Some(1),
                    ..params
                }
            )
            .is_err()
    );
    let error = model
        .infer_stream(
            ModelInput::Text {
                prompt: "[1,2,4]".into(),
            },
            &GenerationParams::default(),
            &mut |_| bail!("caller cancelled"),
        )
        .unwrap_err();
    assert_eq!(error.to_string(), "caller cancelled");
    let identity_fixture = numeric_model_fixture(true);
    let identity_path = directory.path().join("tensor identity.mlmodel");
    std::fs::write(&identity_path, &identity_fixture).unwrap();
    let identity = CoreMlEngine.load(&identity_path, DeviceKind::Cpu).unwrap();
    assert_eq!(
        identity
            .infer(
                ModelInput::Text {
                    prompt: "[-1.5,2,4]".into()
                },
                &GenerationParams::default()
            )
            .unwrap()
            .logits,
        Some(vec![-1.5, 2., 4.])
    );
    println!(
        "{}",
        serde_json::json!({"profile":"coreml-affine-cpu","fixture_bytes":fixture.len(),"fixture_sha256":format!("{:x}",Sha256::digest(&fixture)),"load_ms":load_ms,"device":"cpu","compute_units":"cpu_only","inputs":[[1,2,4],[-2,3,8],[2,1,-4]],"expected_outputs":[[5.0],[0.0],[4.0]],"array_fixture_bytes":identity_fixture.len(),"array_fixture_sha256":format!("{:x}",Sha256::digest(&identity_fixture)),"array_input":[-1.5,2.0,4.0],"array_output":[-1.5,2.0,4.0],"passed":true})
    );
}

#[test]
fn busy_prediction_is_rejected_before_spawning_another_worker() {
    let workspace = tempfile::tempdir().unwrap();
    let model = CoreMlModel {
        executable: workspace.path().join("worker-must-not-spawn"),
        compiled_model: workspace.path().join("model-does-not-exist.mlmodelc"),
        _workspace: workspace,
        metadata: ModelMetadata {
            id: "busy-fixture".into(),
            modality: Modality::Multi,
            quantized: false,
            manifest: ModelManifest::default(),
        },
        input_elements: 3,
        output_elements: 1,
        timeout: Duration::from_secs(1),
        prediction_worker: Mutex::new(()),
    };
    let active_worker = model.prediction_worker.lock().unwrap();
    let started = std::time::Instant::now();
    let error = model.predict(&[1.0, 2.0, 4.0]).unwrap_err().to_string();
    assert!(
        error.contains("worker is busy"),
        "did not reject before process spawn: {error}"
    );
    assert!(started.elapsed() < Duration::from_secs(1));
    drop(active_worker);
    let error = model.predict(&[1.0, 2.0, 4.0]).unwrap_err().to_string();
    assert!(
        !error.contains("worker is busy"),
        "admission was not released"
    );
    assert!(
        model.prediction_worker.try_lock().is_ok(),
        "failed prediction leaked worker admission"
    );
}
