//! Tensor hooks on the fake engine: the engine computes only what a hook asks for, the same seed renders identically
//! with and without each hook, and the identity check goes red on a planted export with a side effect.
//!
//! Also the documented failure behaviour, since a tensor hook has no isolation: a panic is caught and the render goes
//! on unchanged, a hang hangs the render (no timeout), and an abort takes the process with it.

mod common;

use std::cell::RefCell;
use std::io::{self, Write};
use std::rc::Rc;
use std::sync::mpsc;
use std::time::Duration;

use ggtts_core::engine::{ChunkRequest, Codes, Engine, EngineError, EngineInfo, Generated, Pcm, Stop};
use ggtts_core::hooks::{Hooks, Step, StopTrace, TensorHook, Wants};
use ggtts_core::pipeline::{render_story, render_story_hooked, Read};
use ggtts_core::profile::Profile;
use ggtts_core::trace::NullEvents;
use ggtts_fake::{fake_logits, fake_p_stop, FakeEngine};

/// Counts the steps it is handed and keeps what it saw.
struct Probe {
    wants: Wants,
    steps: usize,
    logits_sum: f64,
}

impl Probe {
    fn boxed(p_stop: bool, logits: bool) -> Box<Self> {
        Box::new(Self {
            wants: Wants { p_stop, logits },
            steps: 0,
            logits_sum: 0.0,
        })
    }
}

impl TensorHook for Probe {
    fn name(&self) -> &str {
        "probe"
    }
    fn wants(&self) -> Wants {
        self.wants
    }
    fn step(&mut self, s: &Step<'_>) {
        self.steps += 1;
        if let Some(l) = s.logits {
            self.logits_sum += l.iter().map(|&x| f64::from(x)).sum::<f64>();
        }
    }
}

/// Render every recorded case with `hooks()` on a fresh engine; the reads and the engines.
fn render_all(mut hooks: impl FnMut() -> Hooks) -> Vec<(Read, FakeEngine, Hooks)> {
    common::cases()
        .iter()
        .map(|c| {
            let mut e = c.engine();
            let (chunks, opts) = (c.chunks(&e), c.opts(&e));
            let mut h = hooks();
            let read = render_story_hooked(&mut e, &c.story_id, &chunks, &opts, &mut NullEvents, &mut h).unwrap();
            h.finish();
            (read, e, h)
        })
        .collect()
}

fn plain_reads() -> Vec<Read> {
    common::cases()
        .iter()
        .map(|c| {
            let mut e = c.engine();
            let (chunks, opts) = (c.chunks(&e), c.opts(&e));
            render_story(&mut e, &c.story_id, &chunks, &opts).unwrap()
        })
        .collect()
}

#[test]
fn nothing_is_computed_for_hooks_unless_a_hook_asks() {
    for (_, e, _) in render_all(Hooks::new) {
        assert_eq!(
            (e.exported.p_stop, e.exported.logits),
            (0, 0),
            "computed with no hook registered"
        );
    }
    for (_, e, _) in render_all(|| {
        let mut h = Hooks::new();
        h.add(Probe::boxed(true, false));
        h
    }) {
        assert!(e.exported.p_stop > 0);
        assert_eq!(
            e.exported.logits, 0,
            "logits computed although only p_stop was asked for"
        );
    }
    for (_, e, _) in render_all(|| {
        let mut h = Hooks::new();
        h.add(Probe::boxed(false, true));
        h
    }) {
        assert_eq!(
            e.exported.p_stop, 0,
            "p_stop computed although only logits were asked for"
        );
        assert!(e.exported.logits > 0);
    }
}

/// Every request every case makes, as `(text, seed, chunk, draw)`, from a plain render's calls.
fn requests() -> Vec<Vec<(String, u64, u32, u32)>> {
    common::cases()
        .iter()
        .map(|c| {
            let mut e = c.engine();
            let (chunks, opts) = (c.chunks(&e), c.opts(&e));
            render_story(&mut e, &c.story_id, &chunks, &opts).unwrap();
            let mut seen = std::collections::HashMap::<String, u32>::new();
            e.calls
                .iter()
                .map(|call| {
                    let draw = seen.entry(call.text.clone()).or_insert(0);
                    *draw += 1;
                    let chunk = chunks.iter().position(|p| p.text == call.text).unwrap() as u32;
                    (call.text.clone(), call.seed, chunk, *draw - 1)
                })
                .collect()
        })
        .collect()
}

/// The first request whose tokens differ between an engine with no hooks and one with `hooks()`.
fn first_token_diff(planted: bool, mut hooks: impl FnMut() -> Hooks) -> Option<String> {
    for (case, reqs) in common::cases().iter().zip(requests()) {
        let (mut a, mut b) = (case.engine(), case.engine());
        b.plant_hook_export_draw = planted;
        let mut h = hooks();
        for (text, seed, chunk, draw) in &reqs {
            let req = ChunkRequest {
                text,
                seed: *seed,
                chunk: *chunk,
                draw: *draw,
                instruction: None,
                steer: None,
            };
            let x = a.generate_hooked(&req, &mut NullEvents, &mut Hooks::new()).unwrap();
            let y = b.generate_hooked(&req, &mut NullEvents, &mut h).unwrap();
            if x != y {
                return Some(format!("{}: chunk {chunk} draw {draw}", case.id));
            }
        }
    }
    None
}

/// A named way to build a fresh set of hooks.
type MakeHooks = (&'static str, fn() -> Hooks);

fn each_hook() -> Vec<MakeHooks> {
    vec![
        ("stop-trace", || {
            let mut h = Hooks::new();
            h.add(Box::new(StopTrace::new(Vec::new())));
            h
        }),
        ("a p_stop probe", || {
            let mut h = Hooks::new();
            h.add(Probe::boxed(true, false));
            h
        }),
        ("a logits probe", || {
            let mut h = Hooks::new();
            h.add(Probe::boxed(false, true));
            h
        }),
        ("all of them", || {
            let mut h = Hooks::new();
            h.add(Box::new(StopTrace::new(Vec::new())));
            h.add(Probe::boxed(true, true));
            h
        }),
    ]
}

#[test]
fn the_same_seed_renders_identically_with_and_without_each_hook() {
    let plain = plain_reads();
    for (name, hooks) in each_hook() {
        assert_eq!(first_token_diff(false, hooks), None, "{name} changed a token");
        let hooked = render_all(hooks);
        for ((a, (b, _, h)), case) in plain.iter().zip(&hooked).zip(common::cases()) {
            assert_eq!(a.samples, b.samples, "{name}: {} audio differs", case.id);
            assert_eq!(a.chunks, b.chunks, "{name}: {} chunk records differ", case.id);
            assert!(h.failures().is_empty(), "{name}: {:?}", h.failures());
        }
    }
}

#[test]
fn the_identity_check_goes_red_on_a_planted_export_with_a_side_effect() {
    let hooks = || {
        let mut h = Hooks::new();
        h.add(Probe::boxed(true, false));
        h
    };
    assert!(
        first_token_diff(true, hooks).is_some(),
        "an export that draws from the sampler stream was not seen"
    );
    // The plant fires only when a hook asks: with none, the planted engine is still identical.
    assert_eq!(first_token_diff(true, Hooks::new), None);
}

/// A buffer a boxed hook can write to while the test keeps a handle on it.
#[derive(Clone, Default)]
struct Shared(Rc<RefCell<Vec<u8>>>);

impl Write for Shared {
    fn write(&mut self, b: &[u8]) -> io::Result<usize> {
        self.0.borrow_mut().extend_from_slice(b);
        Ok(b.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

#[test]
fn the_stop_trace_hook_writes_the_values_the_engine_has() {
    let case = &common::cases()[0];
    let mut e = case.engine();
    let req = ChunkRequest {
        text: "多謝。",
        seed: 9,
        chunk: 2,
        draw: 1,
        instruction: None,
        steer: None,
    };
    let buf = Shared::default();
    let mut h = Hooks::new();
    h.add(Box::new(StopTrace::new(buf.clone())));
    let g = e.generate_hooked(&req, &mut NullEvents, &mut h).unwrap();
    h.finish();
    assert!(h.failures().is_empty(), "{:?}", h.failures());
    let n = g.codes.n_frames() as u64;
    let want: String = (0..n)
        .map(|f| {
            format!(
                "{{\"chunk\":2,\"draw\":1,\"seed\":9,\"frame\":{f},\"token\":{},\"p_stop\":{:e},\"p_stop_semantics\":\"fake-synthetic/1\"}}
",
                g.codes.frames[(f * 16) as usize],
                fake_p_stop(f, n)
            )
        })
        .collect();
    assert!(n > 0);
    assert_eq!(String::from_utf8(buf.0.borrow().clone()).unwrap(), want);
}

#[test]
fn a_logits_hook_sees_the_engine_logits() {
    let case = &common::cases()[0];
    let mut e = case.engine();
    let req = ChunkRequest {
        text: "多謝。",
        seed: 9,
        chunk: 0,
        draw: 0,
        instruction: None,
        steer: None,
    };
    let seen = Rc::new(RefCell::new(Vec::<f32>::new()));
    struct Logits(Rc<RefCell<Vec<f32>>>);
    impl TensorHook for Logits {
        fn name(&self) -> &str {
            "logits"
        }
        fn wants(&self) -> Wants {
            Wants {
                p_stop: false,
                logits: true,
            }
        }
        fn step(&mut self, s: &Step<'_>) {
            assert!(s.p_stop.is_none(), "p_stop was handed over unasked");
            self.0.borrow_mut().extend_from_slice(s.logits.unwrap());
        }
    }
    let mut h = Hooks::new();
    h.add(Box::new(Logits(seen.clone())));
    let g = e.generate_hooked(&req, &mut NullEvents, &mut h).unwrap();
    let want: Vec<f32> = (0..g.codes.n_frames() as u64).flat_map(|f| fake_logits(9, f)).collect();
    assert_eq!(*seen.borrow(), want);
}

/// Panics on its `at`th step.
struct Panics {
    at: usize,
    seen: usize,
}

impl TensorHook for Panics {
    fn name(&self) -> &str {
        "panics"
    }
    fn wants(&self) -> Wants {
        Wants {
            p_stop: true,
            logits: false,
        }
    }
    fn step(&mut self, _s: &Step<'_>) {
        self.seen += 1;
        if self.seen == self.at {
            panic!("planted panic");
        }
    }
}

#[test]
fn a_panicking_hook_is_switched_off_and_the_render_goes_on_unchanged() {
    let plain = plain_reads();
    let hooked = render_all(|| {
        let mut h = Hooks::new();
        h.add(Box::new(Panics { at: 3, seen: 0 }));
        h.add(Probe::boxed(true, false));
        h
    });
    for ((a, (b, _, h)), case) in plain.iter().zip(&hooked).zip(common::cases()) {
        assert_eq!(a.samples, b.samples, "{}: a hook's panic changed the audio", case.id);
        assert_eq!(h.failures().len(), 1, "{}: {:?}", case.id, h.failures());
        assert_eq!(h.failures()[0].hook, "panics");
        assert!(h.failures()[0].message.contains("planted panic"));
        assert!(!h.wants().any(), "every hook is finished");
    }
}

/// Blocks on its first step until released.
struct Hangs(mpsc::Receiver<()>);

impl TensorHook for Hangs {
    fn name(&self) -> &str {
        "hangs"
    }
    fn wants(&self) -> Wants {
        Wants {
            p_stop: true,
            logits: false,
        }
    }
    fn step(&mut self, _s: &Step<'_>) {
        let _ = self.0.recv();
    }
}

#[test]
fn a_hanging_hook_hangs_the_render_there_is_no_timeout() {
    let (release, rx) = mpsc::channel::<()>();
    let (done_tx, done) = mpsc::channel::<Vec<f32>>();
    let worker = std::thread::spawn(move || {
        let case = &common::cases()[0];
        let mut e = case.engine();
        let (chunks, opts) = (case.chunks(&e), case.opts(&e));
        let mut h = Hooks::new();
        h.add(Box::new(Hangs(rx)));
        let read = render_story_hooked(&mut e, &case.story_id, &chunks, &opts, &mut NullEvents, &mut h).unwrap();
        done_tx.send(read.samples).unwrap();
    });
    assert!(
        done.recv_timeout(Duration::from_millis(500)).is_err(),
        "the render finished while its hook was blocked: something is timing hooks out"
    );
    drop(release); // the hook's recv returns, and every later step passes straight through
    let samples = done
        .recv_timeout(Duration::from_secs(60))
        .expect("the render finishes once released");
    worker.join().unwrap();
    assert_eq!(samples, plain_reads()[0].samples);
}

/// Aborts the process on its first step.
struct Aborts;

impl TensorHook for Aborts {
    fn name(&self) -> &str {
        "aborts"
    }
    fn wants(&self) -> Wants {
        Wants {
            p_stop: true,
            logits: false,
        }
    }
    fn step(&mut self, _s: &Step<'_>) {
        std::process::abort();
    }
}

const ABORT_CHILD: &str = "GGTTS_HOOK_ABORT_CHILD";

/// Run only as the child of the abort test: renders with a hook that aborts, then prints a marker it never reaches.
#[test]
fn abort_child() {
    if std::env::var(ABORT_CHILD).is_err() {
        return;
    }
    let case = &common::cases()[0];
    let mut e = case.engine();
    let (chunks, opts) = (case.chunks(&e), case.opts(&e));
    let mut h = Hooks::new();
    h.add(Box::new(Aborts));
    let _ = render_story_hooked(&mut e, &case.story_id, &chunks, &opts, &mut NullEvents, &mut h);
    println!("RENDER-FINISHED");
}

#[test]
fn an_aborting_hook_takes_the_render_and_the_process_with_it() {
    let out = std::process::Command::new(std::env::current_exe().unwrap())
        .args(["abort_child", "--exact", "--nocapture", "--test-threads", "1"])
        .env(ABORT_CHILD, "1")
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(!out.status.success(), "the process survived an abort in a hook");
    assert!(
        !stdout.contains("RENDER-FINISHED"),
        "the render went on after the abort"
    );
    // The no-op: the same child without the abort switch finishes.
    let ok = std::process::Command::new(std::env::current_exe().unwrap())
        .args(["abort_child", "--exact", "--test-threads", "1"])
        .output()
        .unwrap();
    assert!(ok.status.success());
}

/// An engine that exports nothing to hooks: the trait's default.
struct Plain {
    info: EngineInfo,
}

impl Engine for Plain {
    fn info(&self) -> &EngineInfo {
        &self.info
    }
    fn profile(&self) -> &Profile {
        &ggtts_qwen3_tts::PROFILE
    }
    fn generate(&mut self, _req: &ChunkRequest<'_>) -> Result<Generated, EngineError> {
        Ok(Generated {
            codes: Codes {
                codebooks: 1,
                frames: vec![1],
            },
            stop: Stop::Eos,
        })
    }
    fn decode(&mut self, _codes: &Codes) -> Result<Pcm, EngineError> {
        Ok(Pcm {
            samples: vec![0.0],
            sample_rate: 24000,
        })
    }
}

#[test]
fn an_engine_that_exports_nothing_refuses_a_hook_rather_than_starve_it() {
    let mut e = Plain {
        info: EngineInfo {
            family: "plain".into(),
            backend: "none".into(),
            parity: "none".into(),
        },
    };
    let req = ChunkRequest {
        text: "x",
        seed: 1,
        chunk: 0,
        draw: 0,
        instruction: None,
        steer: None,
    };
    assert!(e.generate_hooked(&req, &mut NullEvents, &mut Hooks::new()).is_ok());
    let mut h = Hooks::new();
    h.add(Probe::boxed(true, false));
    let err = e.generate_hooked(&req, &mut NullEvents, &mut h).unwrap_err();
    assert!(err.0.contains("exports nothing"), "{err}");
}
