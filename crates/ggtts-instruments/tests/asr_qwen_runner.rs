//! The ordinary runner contains fake Qwen-ASR success, failure and timeout without changing audio.
use std::path::{Path, PathBuf};

use ggtts_core::report::{AudioState, Status};
use ggtts_instruments::config::Config;
use ggtts_instruments::runner::{run, Inputs, EXIT_INCOMPLETE, EXIT_OK};

fn wrapper(path: &Path, script: &Path, result: &str) {
    let source = format!(
        "import runpy,sys,types,time\n\
         torch=types.ModuleType('torch'); torch.float32='f32'; torch.set_num_threads=lambda n: None\n\
         class M:\n\
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
    assert!(!envelope.contains(&root.to_string_lossy().to_string()));

    wrapper(&prog, &real, "return [types.SimpleNamespace(text='')]");
    let optional = run(&config(&prog, &model, &reference, false, 10), &input).unwrap();
    assert_eq!(optional.envelopes[0].status, Status::Failed);
    assert_eq!(optional.exit_code(), EXIT_OK);
    let required = run(&config(&prog, &model, &reference, true, 10), &input).unwrap();
    assert_eq!(required.envelopes[0].status, Status::Failed);
    assert_eq!(required.exit_code(), EXIT_INCOMPLETE);
    let envelope = std::fs::read_to_string(input.report_dir.join("instruments/asr-qwen/envelope.json")).unwrap();
    assert!(!envelope.contains("abcdef"));
    assert!(!envelope.contains(&root.to_string_lossy().to_string()));

    wrapper(
        &prog,
        &real,
        "time.sleep(3); return [types.SimpleNamespace(text='abcdef')]",
    );
    let timed = run(&config(&prog, &model, &reference, true, 1), &input).unwrap();
    assert_eq!(timed.envelopes[0].status, Status::Timeout);
    assert_eq!(timed.exit_code(), EXIT_INCOMPLETE);
    assert_eq!(std::fs::read(&audio).unwrap(), before);
    let _ = std::fs::remove_dir_all(root);
}
