//! The ordinary runner contains fake Qwen-ASR success, failure and timeout without changing audio.
use std::path::{Path, PathBuf};

use ggtts_core::report::{AudioState, Status};
use ggtts_instruments::config::Config;
use ggtts_instruments::runner::{run, Inputs, EXIT_INCOMPLETE, EXIT_OK};

fn wrapper(path: &Path, script: &Path, result: &str) {
    let source = format!(
        "import runpy,sys,types,time\n\
         torch=types.ModuleType('torch'); torch.float32='f32'; torch.set_num_threads=lambda n: None; torch.cuda=types.SimpleNamespace(is_available=lambda: True)\n\
         class M:\n\
         \x20 def parameters(self): return iter([types.SimpleNamespace(device=types.SimpleNamespace(type='cuda'))])\n\
         \x20 @staticmethod\n\
         \x20 def from_pretrained(*a,**k): return M()\n\
         \x20 def transcribe(self,*a,**k):\n\
         \x20  {result}\n\
         q=types.ModuleType('qwen_asr'); q.Qwen3ASRModel=M\n\
         sys.modules['torch']=torch; sys.modules['qwen_asr']=q\n\
         runpy.run_path({:?},run_name='__main__')\n",
        script.to_string_lossy()
    );
    std::fs::write(path, source).unwrap();
}

fn config(wrapper: &Path, model: &Path, reference: &Path, required: bool, timeout: u64) -> Config {
    let mut c = Config::default();
    let python = if cfg!(windows) { "python" } else { "python3" };
    c.apply_instrument_flag(&format!(
        "asr-qwen,command={},interpreter={python},timeout_s={timeout},arg.model_dir={},arg.reference_file={}",
        wrapper.display(),
        model.display(),
        reference.display()
    ))
    .unwrap();
    if required {
        c.apply_require_flag("asr-qwen").unwrap();
    }
    c
}

fn sense_wrapper(path: &Path, script: &Path) {
    let source = format!(
        "import runpy,sys,types\nclass M:\n def __init__(self,**kw): pass\n def parameters(self): return iter([types.SimpleNamespace(device=types.SimpleNamespace(type='cuda'))])\n def generate(self,**kw): return [{{'text':'abcdef'}}]\nsys.modules['torch']=types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda:True))\nsys.modules['funasr']=types.SimpleNamespace(AutoModel=M)\nrunpy.run_path({:?},run_name='__main__')\n",
        script.to_string_lossy()
    );
    std::fs::write(path, source).unwrap();
}

#[test]
fn two_asr_instruments_stack_without_chaining_or_changing_audio() {
    let root = std::env::temp_dir().join(format!("ggtts-asr-stack-{}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    let qmodel = root.join("qmodel");
    let smodel = root.join("smodel");
    std::fs::create_dir_all(&qmodel).unwrap();
    std::fs::create_dir_all(&smodel).unwrap();
    std::fs::write(qmodel.join("config.json"), "{}").unwrap();
    std::fs::write(qmodel.join("model.safetensors"), b"synthetic").unwrap();
    for file in [
        "model.pt",
        "config.yaml",
        "am.mvn",
        "tokens.json",
        "chn_jpn_yue_eng_ko_spectok.bpe.model",
    ] {
        std::fs::write(smodel.join(file), b"synthetic").unwrap();
    }
    let reference = root.join("reference.txt");
    std::fs::write(&reference, "abcdef").unwrap();
    let audio = ggtts_core::flac::write_flac(&root.join("audio"), &[0.1; 2400], 24000).unwrap();
    let before = std::fs::read(&audio).unwrap();
    let inputs = Inputs {
        run: "test".into(),
        render: "r".into(),
        report_dir: root.join("report"),
        audio: audio.clone(),
        audio_state: AudioState::Trimmed,
        audio_seconds: Some(0.1),
        events: None,
        bin_dir: None,
        instruments_dir: Some(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../instruments")),
    };
    let base = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../instruments/examples/python");
    let qw = root.join("qwrapper.py");
    let sw = root.join("swrapper.py");
    wrapper(
        &qw,
        &base.join("asr-qwen/instrument.py"),
        "return [types.SimpleNamespace(text='abc')]",
    );
    sense_wrapper(&sw, &base.join("asr-sensevoice/instrument.py"));
    let python = if cfg!(windows) { "python" } else { "python3" };
    let qflag = format!(
        "asr-qwen,command={},interpreter={python},arg.model_dir={},arg.reference_file={}",
        qw.display(),
        qmodel.display(),
        reference.display()
    );
    let sflag = format!(
        "asr-sensevoice,command={},interpreter={python},arg.model_dir={},arg.reference_file={}",
        sw.display(),
        smodel.display(),
        reference.display()
    );
    assert!(Config::default().instruments.is_empty());
    let mut qonly = Config::default();
    qonly.apply_instrument_flag(&qflag).unwrap();
    qonly.apply_require_flag("asr-qwen").unwrap();
    let q = run(&qonly, &inputs).unwrap();
    assert_eq!(q.exit_code(), EXIT_OK, "{:?}", q.envelopes);
    assert_eq!(q.envelopes[0].metrics[0].value, Some(0.5));
    let mut sonly = Config::default();
    sonly.apply_instrument_flag(&sflag).unwrap();
    sonly.apply_require_flag("asr-sensevoice").unwrap();
    let s = run(&sonly, &inputs).unwrap();
    assert_eq!(s.exit_code(), EXIT_OK, "{:?}", s.envelopes);
    assert_eq!(s.envelopes[0].metrics[0].value, Some(0.0));
    let mut gpu = Config::default();
    gpu.set_device("cuda").unwrap();
    gpu.apply_instrument_flag(&qflag).unwrap();
    gpu.apply_instrument_flag(&sflag).unwrap();
    gpu.apply_require_flag("asr-qwen").unwrap();
    gpu.apply_require_flag("asr-sensevoice").unwrap();
    let result = run(&gpu, &inputs).unwrap();
    assert_eq!(result.exit_code(), EXIT_OK, "{:?}", result.envelopes);
    for envelope in &result.envelopes {
        assert_eq!(
            envelope.provenance.config.get("device").and_then(|v| v.as_str()),
            Some("cuda")
        );
    }
    gpu.apply_instrument_flag("asr-qwen,arg.device=cpu").unwrap();
    let result = run(&gpu, &inputs).unwrap();
    assert_eq!(result.exit_code(), EXIT_OK, "{:?}", result.envelopes);
    let q_override = result.envelopes.iter().find(|e| e.name == "asr-qwen").unwrap();
    let s_inherited = result.envelopes.iter().find(|e| e.name == "asr-sensevoice").unwrap();
    assert_eq!(
        q_override.provenance.config.get("device").and_then(|v| v.as_str()),
        Some("cpu")
    );
    assert_eq!(
        s_inherited.provenance.config.get("device").and_then(|v| v.as_str()),
        Some("cuda")
    );
    // The same hint also reaches a third-party child; the runner does not special-case ASR.
    let custom = root.join("custom.py");
    let sdk = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../instruments/python");
    std::fs::write(
        &custom,
        format!(
            "import os,sys\nsys.path.insert(0, {:?})\nfrom ggtts_instrument import run\ndef measure(c,e):\n assert os.environ.get('GGTTS_INSTRUMENT_DEVICE')=='cuda'\n e.metric('seen',1,'1','neutral',render=c.render)\nsys.exit(run(measure,'custom','1',hooks=[]))\n",
            sdk.to_string_lossy()
        ),
    )
    .unwrap();
    let mut c = Config::default();
    c.set_device("cuda").unwrap();
    c.apply_instrument_flag(&format!(
        "custom,command={},interpreter={python},version=1",
        custom.display()
    ))
    .unwrap();
    c.apply_require_flag("custom").unwrap();
    let result = run(&c, &inputs).unwrap();
    assert_eq!(result.exit_code(), EXIT_OK, "{:?}", result.envelopes);
    // Repeat the same stack as a no-op, then reverse it; both still consume the same input.
    for flags in [[&sflag, &qflag], [&sflag, &qflag], [&qflag, &sflag]] {
        let mut stacked = Config::default();
        for flag in flags {
            stacked.apply_instrument_flag(flag).unwrap();
        }
        stacked.apply_require_flag("asr-qwen").unwrap();
        stacked.apply_require_flag("asr-sensevoice").unwrap();
        let result = run(&stacked, &inputs).unwrap();
        assert_eq!(result.exit_code(), EXIT_OK, "{:?}", result.envelopes);
        assert_eq!(result.envelopes[0].name, flags[0].split(',').next().unwrap());
        assert_eq!(result.envelopes[1].name, flags[1].split(',').next().unwrap());
        let qstack = result.envelopes.iter().find(|e| e.name == "asr-qwen").unwrap();
        let sstack = result.envelopes.iter().find(|e| e.name == "asr-sensevoice").unwrap();
        assert_eq!(qstack.metrics, q.envelopes[0].metrics);
        assert_eq!(sstack.metrics, s.envelopes[0].metrics);
        assert_eq!(std::fs::read(&audio).unwrap(), before);
    }
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn completed_audio_is_unchanged_and_failures_respect_requirement() {
    let root = std::env::temp_dir().join(format!("ggtts-asr-qwen-runner-{}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    let model = root.join("model");
    std::fs::create_dir_all(&model).unwrap();
    std::fs::write(model.join("config.json"), "{}").unwrap();
    std::fs::write(model.join("model.safetensors"), b"synthetic").unwrap();
    let reference = root.join("reference.txt");
    std::fs::write(&reference, "abcdef").unwrap();
    let audio = ggtts_core::flac::write_flac(&root.join("audio"), &[0.1; 2400], 24000).unwrap();
    let before = std::fs::read(&audio).unwrap();
    let input = Inputs {
        run: "test".into(),
        render: "r".into(),
        report_dir: root.join("report"),
        audio: audio.clone(),
        audio_state: AudioState::Trimmed,
        audio_seconds: Some(0.1),
        events: None,
        bin_dir: None,
        instruments_dir: Some(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../instruments")),
    };
    let real =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../instruments/examples/python/asr-qwen/instrument.py");
    let prog = root.join("wrapper.py");
    let sibling = format!(
        "sibling,command={},version=t,arg.mode=ok",
        env!("CARGO_BIN_EXE_ggtts-instrument-selftest")
    );
    let mut baseline_config = Config::default();
    baseline_config.apply_instrument_flag(&sibling).unwrap();
    let baseline = run(&baseline_config, &input).unwrap();
    assert_eq!(baseline.envelopes[0].status, Status::Ok);
    wrapper(&prog, &real, "return [types.SimpleNamespace(text='abcdef')]");
    let mut stacked_config = config(&prog, &model, &reference, true, 10);
    stacked_config.apply_instrument_flag(&sibling).unwrap();
    let ok = run(&stacked_config, &input).unwrap();
    assert_eq!(ok.exit_code(), EXIT_OK);
    assert_eq!(ok.envelopes.len(), 2);
    assert_eq!(ok.envelopes[0].name, "asr-qwen");
    assert_eq!(ok.envelopes[1].name, "sibling");
    assert_eq!(ok.envelopes[1].metrics, baseline.envelopes[0].metrics);
    assert_eq!(ok.envelopes[0].status, Status::Ok);
    assert_eq!(ok.envelopes[0].metrics[0].value, Some(0.0));
    let envelope = std::fs::read_to_string(input.report_dir.join("instruments/asr-qwen/envelope.json")).unwrap();
    assert!(!envelope.contains("abcdef"));
    assert!(!ok.envelopes[0].provenance.config.contains_key("model_dir"));
    assert!(!ok.envelopes[0].provenance.config.contains_key("reference_file"));
    assert!(ok.envelopes[0]
        .provenance
        .inputs
        .iter()
        .all(|entry| !Path::new(&entry.path).is_absolute()));

    wrapper(&prog, &real, "return [types.SimpleNamespace(text='')]");
    let optional = run(&config(&prog, &model, &reference, false, 10), &input).unwrap();
    assert_eq!(optional.envelopes[0].status, Status::Failed);
    assert_eq!(optional.exit_code(), EXIT_OK);
    let required = run(&config(&prog, &model, &reference, true, 10), &input).unwrap();
    assert_eq!(required.envelopes[0].status, Status::Failed);
    assert_eq!(required.exit_code(), EXIT_INCOMPLETE);
    let envelope = std::fs::read_to_string(input.report_dir.join("instruments/asr-qwen/envelope.json")).unwrap();
    assert!(!envelope.contains("abcdef"));
    // The generic runner replaces failed instrument envelopes with its own
    // failure envelope, which currently contains configured and input paths.
    // Such reports are private; assert this limitation rather than hiding it.
    let failed: serde_json::Value = serde_json::from_str(&envelope).unwrap();
    assert_eq!(
        failed["provenance"]["config"]["args"]["model_dir"],
        model.to_string_lossy().as_ref()
    );

    wrapper(
        &prog,
        &real,
        "time.sleep(3); return [types.SimpleNamespace(text='abcdef')]",
    );
    let timed = run(&config(&prog, &model, &reference, true, 1), &input).unwrap();
    assert_eq!(timed.envelopes[0].status, Status::Timeout);
    assert_eq!(timed.exit_code(), EXIT_INCOMPLETE);
    let timeout_envelope =
        std::fs::read_to_string(input.report_dir.join("instruments/asr-qwen/envelope.json")).unwrap();
    let timeout_doc: serde_json::Value = serde_json::from_str(&timeout_envelope).unwrap();
    assert_eq!(
        timeout_doc["provenance"]["config"]["args"]["model_dir"],
        model.to_string_lossy().as_ref()
    );
    assert!(!timeout_envelope.contains("abcdef"));
    assert_eq!(std::fs::read(&audio).unwrap(), before);
    let _ = std::fs::remove_dir_all(root);
}

