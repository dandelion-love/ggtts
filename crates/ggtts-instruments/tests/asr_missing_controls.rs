//! Missing ASR reference and missing output must never become a passing required measurement.
use std::path::PathBuf;

use ggtts_core::report::{AudioState, Status};
use ggtts_instruments::config::{Config, OnError};
use ggtts_instruments::runner::{run, Inputs, EXIT_INCOMPLETE, EXIT_OK};

#[test]
fn missing_reference_and_output_respect_requirement_and_on_error() {
    if std::env::var_os("GGTTS_SKIP_PYTHON").is_some() {
        return;
    }
    let root = std::env::temp_dir().join(format!("ggtts-asr-missing-{}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    let model = root.join("model");
    std::fs::create_dir_all(&model).unwrap();
    std::fs::write(model.join("config.json"), "{}").unwrap();
    std::fs::write(model.join("model.safetensors"), b"synthetic").unwrap();
    let missing_ref = root.join("absent-reference.txt");
    let python = if cfg!(windows) { "python" } else { "python3" };
    let wrapper = root.join("fake-asr.py");
    let instrument = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../instruments/examples/python/asr-qwen/instrument.py");
    std::fs::write(
        &wrapper,
        format!(
            "import runpy,sys,types\n\
             torch=types.ModuleType('torch'); torch.float32='f32'; torch.set_num_threads=lambda n: None\n\
             class M:\n\
             \x20 @staticmethod\n\
             \x20 def from_pretrained(*a,**k): return M()\n\
             \x20 def transcribe(self,*a,**k): return [types.SimpleNamespace(text='abcdef')]\n\
             q=types.ModuleType('qwen_asr'); q.Qwen3ASRModel=M\n\
             sys.modules['torch']=torch; sys.modules['qwen_asr']=q\n\
             runpy.run_path({:?},run_name='__main__')\n",
            instrument.to_string_lossy()
        ),
    )
    .unwrap();
    let audio = ggtts_core::flac::write_flac(&root.join("audio"), &[0.1; 2400], 24000).unwrap();
    let before = std::fs::read(&audio).unwrap();
    let mut inputs = Inputs {
        run: "missing-controls".into(),
        render: "fixture".into(),
        report_dir: root.join("report"),
        audio: audio.clone(),
        audio_state: AudioState::Unknown,
        audio_seconds: Some(0.1),
        events: None,
        bin_dir: None,
        instruments_dir: Some(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../instruments")),
    };
    let mut missing = Config::default();
    missing
        .apply_instrument_flag(&format!(
            "asr-qwen,command={},interpreter={python},arg.model_dir={},arg.reference_file={}",
            wrapper.display(),
            model.display(),
            missing_ref.display()
        ))
        .unwrap();
    let optional = run(&missing, &inputs).unwrap();
    assert_eq!(optional.envelopes[0].status, Status::Failed);
    assert_eq!(optional.exit_code(), EXIT_OK);
    missing.apply_require_flag("asr-qwen").unwrap();
    inputs.report_dir = root.join("report-required-reference");
    let required = run(&missing, &inputs).unwrap();
    assert_eq!(required.envelopes[0].status, Status::Failed);
    assert_eq!(required.exit_code(), EXIT_INCOMPLETE);
    assert!(required.envelopes[0].metrics.is_empty());
    let log = std::fs::read_to_string(inputs.report_dir.join("instruments/asr-qwen/asr-qwen.log")).unwrap();
    assert!(log.contains("local model directory and reference file are required"));

    // Only create the missing reference; the same instrument, model and audio must now succeed.
    std::fs::write(&missing_ref, "abcdef").unwrap();
    inputs.report_dir = root.join("report-reference-present");
    let control = run(&missing, &inputs).unwrap();
    assert_eq!(control.exit_code(), EXIT_OK);
    assert_eq!(control.envelopes[0].status, Status::Ok);
    assert_eq!(control.envelopes[0].metrics[0].value, Some(0.0));

    // A process which exits successfully without writing its envelope is invalid, never zero.
    let script = root.join("no-envelope.py");
    std::fs::write(&script, "# deliberately write no envelope\n").unwrap();
    let flag = format!(
        "no-output,command={},interpreter={python},version=test",
        script.display()
    );
    let sibling = format!(
        "sibling,command={},version=test,arg.mode=ok",
        env!("CARGO_BIN_EXE_ggtts-instrument-selftest")
    );
    let mut absent = Config::default();
    absent.apply_instrument_flag(&flag).unwrap();
    absent.apply_instrument_flag(&sibling).unwrap();
    inputs.report_dir = root.join("report-optional-output");
    let optional_output = run(&absent, &inputs).unwrap();
    assert_eq!(optional_output.envelopes[0].status, Status::Invalid);
    assert_eq!(optional_output.envelopes[1].status, Status::Ok);
    assert_eq!(optional_output.exit_code(), EXIT_OK);
    absent.apply_require_flag("no-output").unwrap();
    inputs.report_dir = root.join("report-required-output");
    let required_output = run(&absent, &inputs).unwrap();
    assert_eq!(required_output.envelopes[0].status, Status::Invalid);
    assert_eq!(required_output.exit_code(), EXIT_INCOMPLETE);
    assert!(required_output.envelopes[0].metrics.is_empty());
    absent.on_error = OnError::Abort;
    absent.apply_require_flag("sibling").unwrap();
    inputs.report_dir = root.join("report-aborted");
    let aborted = run(&absent, &inputs).unwrap();
    assert_eq!(aborted.envelopes[0].status, Status::Invalid);
    assert_eq!(aborted.envelopes[1].status, Status::Skipped);
    assert_eq!(aborted.exit_code(), EXIT_INCOMPLETE);
    assert_eq!(std::fs::read(&audio).unwrap(), before);
    let _ = std::fs::remove_dir_all(root);
}
