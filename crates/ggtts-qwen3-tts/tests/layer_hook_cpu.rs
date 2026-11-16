#![cfg(feature = "ggml")]
//! CPU identity and selection checks; set GGTTS_TWO_LAYER_TALKER to a two-layer fixture.
//! Set GGTTS_DEVICE=cuda to run the same test on CUDA (compiled with the cuda feature).
use ggtts_core::steer::{write_safetensors, SteerFile, Steering};
use ggtts_qwen3_tts::cpu::Device;
use ggtts_qwen3_tts::talker::{language_id, Generation, Sampling, Talker};
const SEED: u64 = 20260905;
const STEPS: i32 = 4;
type Row = (i32, i32, usize, bool, Vec<f32>);
fn run(t: &mut Talker, layers: &[usize], steer: Option<&Steering>) -> (Generation, Vec<Row>) {
    let mut rows = Vec::new();
    let mut hook = |frame, token, layer, eos, values: &[f32]| rows.push((frame, token, layer, eos, values.to_vec()));
    let generated = t
        .generate_steered_with_layers(
            "Today is a good day.",
            None,
            "hk_cantonese_speaker",
            language_id("zh").unwrap(),
            SEED,
            STEPS,
            Sampling::default(),
            steer,
            layers,
            if layers.is_empty() { None } else { Some(&mut hook) },
            None,
            None,
        )
        .unwrap();
    (generated, rows)
}
#[test]
#[ignore = "requires GGTTS_TWO_LAYER_TALKER synthetic two-layer fixture"]
fn selected_layers_match_individual_captures_without_changing_generation() {
    let path = std::env::var("GGTTS_TWO_LAYER_TALKER").expect("set GGTTS_TWO_LAYER_TALKER");
    let device_name = std::env::var("GGTTS_DEVICE").unwrap_or_else(|_| "cpu".into());
    let device = Device::parse(&device_name).expect("GGTTS_DEVICE must be cpu or cuda");
    let mut talker = Talker::load_on(std::path::Path::new(&path), 4, device).unwrap();
    #[cfg(feature = "plant")]
    if let Ok(name) = std::env::var("GGTTS_LAYER_TEST_PLANT") {
        use ggtts_qwen3_tts::talker::Plant;
        let plant = match name.as_str() {
            "wrong-prefill" => Plant::LayerCaptureWrongPrefillRow,
            "shift" => Plant::LayerCaptureShift,
            "missing" => Plant::LayerCaptureMissingExport,
            "omit-eos" => Plant::LayerCaptureOmitEos,
            "draw" => Plant::LayerHookDraw,
            other => panic!("unknown test plant: {other}"),
        };
        let reference = if name == "omit-eos" {
            let mut eos =
                Talker::load_planted(std::path::Path::new(&path), 4, device, Plant::ForceEosAfterFirstDecode).unwrap();
            run(&mut eos, &[0, 1], None)
        } else {
            run(&mut talker, &[0, 1], None)
        };
        let mut subject = Talker::load_planted(std::path::Path::new(&path), 4, device, plant).unwrap();
        let observed = run(&mut subject, &[0, 1], None);
        assert_eq!(observed.0, reference.0, "capture changed generated codes or stop");
        assert_eq!(observed.1, reference.1, "capture records differ from reference");
        return;
    }
    let (plain, none) = run(&mut talker, &[], None);
    let (both, rows) = run(&mut talker, &[0, 1], None);
    let (first, zero) = run(&mut talker, &[0], None);
    let (second, one) = run(&mut talker, &[1], None);
    assert_eq!(plain, both, "capture changed generated codes or stop");
    assert_eq!(plain, first);
    assert_eq!(plain, second);
    assert!(none.is_empty());
    assert!(rows.len() >= 4, "must exercise prefill and decode at both layers");
    assert_eq!(rows.iter().filter(|r| r.2 == 0).cloned().collect::<Vec<_>>(), zero);
    assert_eq!(rows.iter().filter(|r| r.2 == 1).cloned().collect::<Vec<_>>(), one);
    assert!(rows
        .iter()
        .all(|r| !r.4.is_empty() && r.4.iter().all(|x| x.is_finite())));
    assert_eq!(zero[0].0, 0, "first capture must be prefill step");
    assert!(zero.iter().any(|r| r.0 > 0), "must exercise decode");
    assert_ne!(zero[0].4, one[0].4, "fixture must distinguish selected layers");
    for (a, b) in zero.iter().zip(&one) {
        assert_eq!((a.0, a.1, a.3), (b.0, b.1, b.3));
    }
    assert_eq!(zero.len(), one.len());
    // Steering is a separate graph placement mechanism: a layer-1-only add must
    // leave layer 0 unchanged and add the known vector to layer 1 at prefill.
    let v = vec![0.125_f32; zero[0].4.len()];
    let file = SteerFile::parse(&write_safetensors(&[("talker.layer.1", &v)])).unwrap();
    let steering = file.select(None, 1.0).unwrap();
    let (_, steered) = run(&mut talker, &[0, 1], Some(&steering));
    let s0 = steered.iter().find(|r| r.0 == 0 && r.2 == 0).unwrap();
    let s1 = steered.iter().find(|r| r.0 == 0 && r.2 == 1).unwrap();
    assert_eq!(s0.4, zero[0].4, "steering a later layer changed layer 0");
    for ((got, base), delta) in s1.4.iter().zip(&one[0].4).zip(&v) {
        assert!(
            (got - base - delta).abs() < 1e-4,
            "layer 1 does not contain its steering add"
        );
    }
    #[cfg(feature = "plant")]
    {
        use ggtts_qwen3_tts::talker::Plant;
        let mut planted =
            Talker::load_planted(std::path::Path::new(&path), 4, device, Plant::LayerCaptureShift).unwrap();
        let (_, shifted) = run(&mut planted, &[0, 1], None);
        assert_ne!(rows, shifted, "layer/step shift plant must be detected");
        for plant in [Plant::LayerCaptureWrongPrefillRow, Plant::LayerCaptureMissingExport] {
            let mut bad = Talker::load_planted(std::path::Path::new(&path), 4, device, plant).unwrap();
            let (_, observed) = run(&mut bad, &[0, 1], None);
            assert_ne!(rows, observed, "planted capture defect must be detected: {plant:?}");
        }
        let mut eos =
            Talker::load_planted(std::path::Path::new(&path), 4, device, Plant::ForceEosAfterFirstDecode).unwrap();
        let (_, complete) = run(&mut eos, &[0, 1], None);
        assert_eq!(
            complete.iter().filter(|r| r.3).count(),
            2,
            "EOS must be captured once per layer"
        );
        let mut omitted =
            Talker::load_planted(std::path::Path::new(&path), 4, device, Plant::LayerCaptureOmitEos).unwrap();
        let (_, bad) = run(&mut omitted, &[0, 1], None);
        assert_ne!(complete, bad, "omitted EOS must be detected");
        assert_eq!(bad.iter().filter(|r| r.3).count(), 0);
    }
    let mut callback = |_: i32, _: i32, _: usize, _: bool, _: &[f32]| {};
    let invalid = talker.generate_steered_with_layers(
        "Today is a good day.",
        None,
        "hk_cantonese_speaker",
        language_id("zh").unwrap(),
        SEED,
        STEPS,
        Sampling::default(),
        None,
        &[2],
        Some(&mut callback),
        None,
        None,
    );
    assert!(invalid.is_err(), "out-of-range layer must be refused");
    let missing = talker.generate_steered_with_layers(
        "Today is a good day.",
        None,
        "hk_cantonese_speaker",
        language_id("zh").unwrap(),
        SEED,
        STEPS,
        Sampling::default(),
        None,
        &[0],
        None,
        None,
        None,
    );
    assert!(missing.is_err(), "selection without callback must be refused");
}
