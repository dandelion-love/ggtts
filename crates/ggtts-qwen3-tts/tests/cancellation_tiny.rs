//! Deterministic CPU-only stop at real Qwen frame-loop entry.
#![cfg(all(feature = "ggml", feature = "plant"))]

use ggtts_core::cancel::{CancelOutcome, CancelSignal};
use ggtts_core::engine::{ChunkRequest, Codes, Engine, EngineError, EngineInfo, Generated, Pcm};
use ggtts_core::hooks::Hooks;
use ggtts_core::pipeline::{self, DurationJudge, Given, Observers, Output, RenderError, RenderOptions, StreamMode};
use ggtts_core::profile::Profile;
use ggtts_core::trace::NullEvents;
use ggtts_qwen3_tts::engine::{Qwen3Tts, Voice};
use sha2::{Digest, Sha256};
use std::path::PathBuf;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    mpsc, Arc, Mutex,
};
use std::time::Duration;

fn fixture(var: &str, expected: &str) -> PathBuf {
    let path = PathBuf::from(std::env::var(var).expect("local fixture path required"));
    let data = std::fs::read(&path).expect("fixture missing");
    let digest = Sha256::digest(data)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect::<String>();
    assert_eq!(digest, expected, "fixture hash mismatch");
    path
}

struct Counted {
    inner: Qwen3Tts,
    decodes: Arc<AtomicUsize>,
}
impl Engine for Counted {
    fn info(&self) -> &EngineInfo {
        self.inner.info()
    }
    fn profile(&self) -> &Profile {
        self.inner.profile()
    }
    fn capabilities(&self) -> ggtts_core::capabilities::Capabilities {
        self.inner.capabilities()
    }
    fn generate(&mut self, r: &ChunkRequest<'_>) -> Result<Generated, EngineError> {
        self.inner.generate(r)
    }
    fn generate_cancellable(
        &mut self,
        r: &ChunkRequest<'_>,
        signal: &CancelSignal,
    ) -> Result<CancelOutcome<Generated>, EngineError> {
        self.inner.generate_cancellable(r, signal)
    }
    fn decode(&mut self, codes: &Codes) -> Result<Pcm, EngineError> {
        self.decodes.fetch_add(1, Ordering::SeqCst);
        self.inner.decode(codes)
    }
}

#[test]
#[ignore = "requires verified local tiny talker and codec; no download"]
fn real_cpu_frame_loop_stop_then_fresh_render() {
    let talker = fixture(
        "GGTTS_TINY_TALKER",
        "75b126568ce4c90c99b4fc4e44db6c5953803bff486db00be51bc85284932c33",
    );
    let codec = fixture(
        "GGTTS_CODEC",
        "7771c653aea5d26cdbf461cf00279b78e6c97af5f313da3a8cb3d2b61b837824",
    );
    let signal = CancelSignal::new();
    let worker_signal = signal.clone();
    let (entered_tx, entered_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel();
    let release = Arc::new(Mutex::new(release_rx));
    let worker = std::thread::spawn(move || {
        let mut engine = Counted {
            inner: Qwen3Tts::load(
                &talker,
                &codec,
                2,
                Voice {
                    max_frames: 4,
                    ..Voice::default()
                },
            )
            .unwrap(),
            decodes: Arc::new(AtomicUsize::new(0)),
        };
        engine.inner.set_cancel_loop_probe(Some(Arc::new(move || {
            entered_tx.send(()).unwrap();
            release
                .lock()
                .unwrap()
                .recv_timeout(Duration::from_secs(15))
                .expect("barrier timed out");
        })));
        let chunks = pipeline::exact::chunks("Hello world.").unwrap();
        let mut opts = RenderOptions::for_profile(engine.profile());
        pipeline::exact::options(&mut opts);
        let mut events = NullEvents;
        let mut hooks = Hooks::new();
        let mut obs = Observers::new(&mut events, &mut hooks).with_cancel(&worker_signal);
        let mut delivered = 0;
        let mut sink = |_: pipeline::StreamPiece<'_>| {
            delivered += 1;
            Ok(())
        };
        let stopped = pipeline::render_plan(
            &mut engine,
            &chunks,
            &Given(1),
            &mut DurationJudge,
            &opts,
            &mut obs,
            Output::Stream {
                sink: &mut sink,
                level: StreamMode::Raw,
            },
        );
        assert!(matches!(stopped, Err(RenderError::Stopped)));
        assert_eq!(engine.decodes.load(Ordering::SeqCst), 0);
        assert_eq!(delivered, 0);
        engine.inner.set_cancel_loop_probe(None);
        let next = CancelSignal::new();
        let mut events = NullEvents;
        let mut hooks = Hooks::new();
        let mut obs = Observers::new(&mut events, &mut hooks).with_cancel(&next);
        let mut delivered = 0;
        let mut sink = |_: pipeline::StreamPiece<'_>| {
            delivered += 1;
            Ok(())
        };
        let read = pipeline::render_plan(
            &mut engine,
            &chunks,
            &Given(1),
            &mut DurationJudge,
            &opts,
            &mut obs,
            Output::Stream {
                sink: &mut sink,
                level: StreamMode::Raw,
            },
        )
        .unwrap();
        assert!(delivered > 0 && !read.samples.is_empty());
        assert_eq!(engine.decodes.load(Ordering::SeqCst), 1);
    });
    entered_rx
        .recv_timeout(Duration::from_secs(45))
        .expect("real Qwen frame loop was not reached");
    signal.cancel();
    release_tx.send(()).unwrap();
    worker.join().unwrap();
}
