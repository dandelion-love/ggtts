use ggtts_core::cancel::{CancelOutcome, CancelSignal};
use ggtts_core::engine::{ChunkRequest, Codes, Engine, EngineError, EngineInfo, Generated, Pcm};
use ggtts_core::hooks::Hooks;
use ggtts_core::pipeline::{self, DurationJudge, Given, Observers, Output, RenderError, RenderOptions};
use ggtts_core::profile::Profile;
use ggtts_core::trace::NullEvents;
use ggtts_fake::FakeEngine;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    mpsc, Arc,
};
use std::thread;

fn request() -> ChunkRequest<'static> {
    ChunkRequest {
        text: "test",
        seed: 1,
        chunk: 0,
        draw: 0,
        instruction: None,
        steer: None,
    }
}

#[test]
fn unsupported_engine_refuses_before_generation_even_if_already_cancelled() {
    let mut engine = FakeEngine::default();
    let signal = CancelSignal::new();
    signal.cancel();
    assert_eq!(
        engine.generate_cancellable(&request(), &signal).unwrap(),
        CancelOutcome::Unsupported
    );
    assert!(engine.calls.is_empty());
}

struct BlockingFake {
    info: EngineInfo,
    started: mpsc::Sender<()>,
    decoded: Arc<AtomicUsize>,
}

impl Engine for BlockingFake {
    fn info(&self) -> &EngineInfo {
        &self.info
    }
    fn profile(&self) -> &Profile {
        &ggtts_qwen3_tts::PROFILE
    }
    fn generate(&mut self, _: &ChunkRequest<'_>) -> Result<Generated, EngineError> {
        unreachable!("cancellable test path only")
    }
    fn decode(&mut self, _: &Codes) -> Result<Pcm, EngineError> {
        self.decoded.fetch_add(1, Ordering::SeqCst);
        panic!("stopped generation must not decode")
    }
    fn generate_cancellable(
        &mut self,
        _: &ChunkRequest<'_>,
        signal: &CancelSignal,
    ) -> Result<CancelOutcome<Generated>, EngineError> {
        self.started.send(()).unwrap();
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        while !signal.is_cancelled() && std::time::Instant::now() < deadline {
            std::thread::yield_now();
        }
        if !signal.is_cancelled() {
            return Err(EngineError("test cancellation timed out".into()));
        }
        Ok(CancelOutcome::Stopped)
    }
}

#[test]
fn pipeline_stops_blocked_generation_without_decode_or_delivery() {
    let signal = CancelSignal::new();
    let worker_signal = signal.clone();
    let decoded = Arc::new(AtomicUsize::new(0));
    let worker_decoded = decoded.clone();
    let (started, waiting) = mpsc::channel();
    let worker = thread::spawn(move || {
        let mut engine = BlockingFake {
            info: EngineInfo {
                family: "test".into(),
                backend: "fake".into(),
                parity: "none".into(),
            },
            started,
            decoded: worker_decoded,
        };
        let chunks = pipeline::exact::chunks("test").unwrap();
        let opts = RenderOptions::for_profile(engine.profile());
        let mut events = NullEvents;
        let mut hooks = Hooks::new();
        let mut obs = Observers::new(&mut events, &mut hooks).with_cancel(&worker_signal);
        pipeline::render_plan(
            &mut engine,
            &chunks,
            &Given(1),
            &mut DurationJudge,
            &opts,
            &mut obs,
            Output::Whole,
        )
    });
    waiting.recv_timeout(std::time::Duration::from_secs(5)).unwrap();
    signal.cancel();
    assert!(matches!(worker.join().unwrap(), Err(RenderError::Stopped)));
    assert_eq!(decoded.load(Ordering::SeqCst), 0);
    assert!(!CancelSignal::new().is_cancelled());
}

struct PassingFake(ggtts_fake::FakeEngine);
impl Engine for PassingFake {
    fn info(&self) -> &EngineInfo {
        self.0.info()
    }
    fn profile(&self) -> &Profile {
        self.0.profile()
    }
    fn generate(&mut self, req: &ChunkRequest<'_>) -> Result<Generated, EngineError> {
        self.0.generate(req)
    }
    fn decode(&mut self, codes: &Codes) -> Result<Pcm, EngineError> {
        self.0.decode(codes)
    }
    fn generate_cancellable(
        &mut self,
        req: &ChunkRequest<'_>,
        _: &CancelSignal,
    ) -> Result<CancelOutcome<Generated>, EngineError> {
        self.0.generate(req).map(CancelOutcome::Completed)
    }
}

struct CancelDuringJudge(CancelSignal);
impl pipeline::Judge for CancelDuringJudge {
    fn judge(&mut self, profile: &Profile, samples: &[f32], rate: u32, text: &str) -> ggtts_core::audio::Judgement {
        self.0.cancel();
        pipeline::Judge::judge(&mut DurationJudge, profile, samples, rate, text)
    }
}

#[test]
fn cancellation_during_judge_prevents_keep_and_stream_callback() {
    let signal = CancelSignal::new();
    let mut engine = PassingFake(FakeEngine::default());
    let chunks = pipeline::exact::chunks("test").unwrap();
    let opts = RenderOptions::for_profile(engine.profile());
    let mut events = NullEvents;
    let mut hooks = Hooks::new();
    let mut obs = Observers::new(&mut events, &mut hooks).with_cancel(&signal);
    let mut called = 0;
    let mut sink = |_: pipeline::StreamPiece<'_>| {
        called += 1;
        Ok(())
    };
    let result = pipeline::render_plan(
        &mut engine,
        &chunks,
        &Given(1),
        &mut CancelDuringJudge(signal.clone()),
        &opts,
        &mut obs,
        Output::Stream {
            sink: &mut sink,
            level: pipeline::StreamMode::Raw,
        },
    );
    assert!(matches!(result, Err(RenderError::Stopped)));
    assert_eq!(called, 0);
}
