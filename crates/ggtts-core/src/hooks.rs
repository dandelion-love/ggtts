//! Tensor hooks: Rust code that runs inside the engine, once per generated frame, and reads values the event stream
//! does not carry, such as the talker's logits.
//!
//! **What an engine exports.** A hook says what it [`Wants`]. [`Hooks::wants`] is the union over the registered hooks,
//! and an engine computes only that. With no hook registered the union is empty and the engine computes nothing for
//! hooks at all.
//!
//! **Read-only, by convention.** A hook gets a [`Step`] holding shared references, so safe Rust cannot write through
//! them. That is a convention, not a proof (`unsafe` code could). What enforces it is the identity check: the same
//! requests with and without a hook must give the same tokens and audio. The fake engine's planted export draw shows
//! that check going red (`tests/tensor_hooks.rs`).
//!
//! **No isolation.** A hook runs in the engine's process and thread:
//! - a **panic** in a hook is caught, the hook is switched off for the rest of the render, and the failure is kept
//!   in [`Hooks::failures`]. The render's output is unchanged;
//! - a **hang** in a hook hangs the render. There is no timeout;
//! - an **abort** (or anything else that kills the process) kills the render with it.
//!
//! Each of the three is tested in `tests/tensor_hooks.rs`. An instrument that needs a timeout or a crash boundary
//! belongs on the event stream, out of process.
//!
//! [`StopTrace`] is the worked example: the per-step stop-probability trace, as a hook.

use std::io::{self, Write};
use std::panic::{catch_unwind, AssertUnwindSafe};

/// What a hook asks the engine to export at each step. An engine computes the union of what registered hooks want,
/// and nothing else.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Wants {
    /// The end-of-speech probability under the distribution the sampler draws from.
    pub p_stop: bool,
    /// The codebook-0 logits as the sampler sees them, before temperature.
    pub logits: bool,
}

impl Wants {
    pub const NONE: Wants = Wants {
        p_stop: false,
        logits: false,
    };

    /// Anything at all.
    pub fn any(self) -> bool {
        self.p_stop || self.logits
    }

    fn union(self, o: Wants) -> Wants {
        Wants {
            p_stop: self.p_stop || o.p_stop,
            logits: self.logits || o.logits,
        }
    }

    fn meets(self, o: Wants) -> bool {
        (self.p_stop && o.p_stop) || (self.logits && o.logits)
    }
}

/// One generated frame, as a hook sees it. A value no registered hook asked for is `None`.
#[derive(Debug, Clone, Copy)]
pub struct Step<'a> {
    pub chunk: u32,
    pub draw: u32,
    pub seed: u64,
    pub frame: u32,
    /// The codebook-0 token drawn at this step.
    pub token: u32,
    pub p_stop: Option<f64>,
    /// Versioned producer meaning of `p_stop`; absent when the producer cannot identify it.
    pub p_stop_semantics: Option<&'static str>,
    pub logits: Option<&'a [f32]>,
}

/// A selected talker layer at one sampled step, borrowed only for the callback.
/// Step zero is the final prefill row; later steps are decode rows. The producer
/// identifies EOS explicitly, and supplies values after residual and steering,
/// before final normalization. Values are f32; hooks must copy anything retained.
#[derive(Debug, Clone, Copy)]
pub struct LayerStep<'a> {
    pub chunk: u32,
    pub draw: u32,
    pub seed: u64,
    pub frame: u32,
    pub token: u32,
    pub is_eos: bool,
    pub layer: usize,
    pub values: &'a [f32],
}

/// An in-process instrument.
pub trait TensorHook {
    /// A short name, for its failures.
    fn name(&self) -> &str;
    /// What it reads. The engine exports nothing it does not ask for.
    fn wants(&self) -> Wants;
    /// Zero-based talker layer indices requested; read once at registration.
    fn layers(&self) -> Vec<usize> {
        Vec::new()
    }
    /// Called for each requested layer and sampled step. Borrowed values expire on return.
    fn layer(&mut self, _s: &LayerStep<'_>) {}
    /// Called once per generated frame, after the frame's token is drawn and before the next step.
    fn step(&mut self, s: &Step<'_>);
    /// Called once when the render is done. An error is recorded as the hook's failure.
    fn finish(&mut self) -> Result<(), String> {
        Ok(())
    }
}

/// A hook that panicked or failed to finish.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HookFailure {
    pub hook: String,
    pub message: String,
}

struct Slot {
    hook: Box<dyn TensorHook>,
    wants: Wants,
    layers: Vec<usize>,
    alive: bool,
}

/// The hooks registered for a render. Empty by default, and then [`wants`](Hooks::wants) is [`Wants::NONE`].
#[derive(Default)]
pub struct Hooks {
    slots: Vec<Slot>,
    failures: Vec<HookFailure>,
}

impl std::fmt::Debug for Hooks {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let names: Vec<&str> = self.slots.iter().map(|s| s.hook.name()).collect();
        f.debug_struct("Hooks")
            .field("hooks", &names)
            .field("failures", &self.failures)
            .finish()
    }
}

fn panic_text(p: &(dyn std::any::Any + Send)) -> String {
    p.downcast_ref::<&str>()
        .map(|s| s.to_string())
        .or_else(|| p.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "a panic with no message".into())
}

impl Hooks {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a hook. Its wants are read once, here.
    pub fn add(&mut self, hook: Box<dyn TensorHook>) {
        let wants = hook.wants();
        let mut layers = hook.layers();
        layers.sort_unstable();
        layers.dedup();
        self.slots.push(Slot {
            hook,
            wants,
            layers,
            alive: true,
        });
    }

    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }

    /// What the engine must export: the union over the hooks still running.
    pub fn wants(&self) -> Wants {
        self.slots
            .iter()
            .filter(|s| s.alive)
            .fold(Wants::NONE, |w, s| w.union(s.wants))
    }

    /// Sorted, deduplicated union of selected layers for live hooks.
    pub fn layers(&self) -> Vec<usize> {
        let mut layers: Vec<_> = self
            .slots
            .iter()
            .filter(|s| s.alive)
            .flat_map(|s| s.layers.iter().copied())
            .collect();
        layers.sort_unstable();
        layers.dedup();
        layers
    }

    /// Deliver one selected layer to interested hooks. A panicking hook is
    /// disabled and its failure recorded, as for the ordinary step callback.
    pub fn layer(&mut self, s: &LayerStep<'_>) {
        for slot in self
            .slots
            .iter_mut()
            .filter(|x| x.alive && x.layers.binary_search(&s.layer).is_ok())
        {
            if let Err(p) = catch_unwind(AssertUnwindSafe(|| slot.hook.layer(s))) {
                slot.alive = false;
                self.failures.push(HookFailure {
                    hook: slot.hook.name().to_string(),
                    message: format!(
                        "panicked at chunk {} frame {} layer {}: {}",
                        s.chunk,
                        s.frame,
                        s.layer,
                        panic_text(&*p)
                    ),
                });
            }
        }
    }

    /// Hand `s` to every running hook that wants something it carries. A hook that panics is switched off and its
    /// failure kept; the others, and the render, go on.
    pub fn step(&mut self, s: &Step<'_>) {
        let carried = Wants {
            p_stop: s.p_stop.is_some(),
            logits: s.logits.is_some(),
        };
        for slot in self.slots.iter_mut().filter(|x| x.alive && x.wants.meets(carried)) {
            let hook = &mut slot.hook;
            if let Err(p) = catch_unwind(AssertUnwindSafe(|| hook.step(s))) {
                slot.alive = false;
                self.failures.push(HookFailure {
                    hook: slot.hook.name().to_string(),
                    message: format!("panicked at chunk {} frame {}: {}", s.chunk, s.frame, panic_text(&*p)),
                });
            }
        }
    }

    /// Call every running hook's [`finish`](TensorHook::finish), once. Its error, or its panic, is kept as a failure.
    pub fn finish(&mut self) {
        for slot in self.slots.iter_mut().filter(|x| x.alive) {
            let hook = &mut slot.hook;
            let r = catch_unwind(AssertUnwindSafe(|| hook.finish()));
            let message = match r {
                Ok(Ok(())) => None,
                Ok(Err(e)) => Some(e),
                Err(p) => Some(format!("panicked in finish: {}", panic_text(&*p))),
            };
            slot.alive = false;
            if let Some(message) = message {
                self.failures.push(HookFailure {
                    hook: slot.hook.name().to_string(),
                    message,
                });
            }
        }
    }

    /// Every hook that panicked or failed to finish, in order.
    pub fn failures(&self) -> &[HookFailure] {
        &self.failures
    }
}

/// The worked example: the per-step stop-probability trace, one JSON object per generated frame,
/// `{"chunk":C,"draw":D,"seed":S,"frame":F,"token":T,"p_stop":P}` with `P` in exponent form.
/// A known producer meaning is included as `p_stop_semantics`; older records have no such declaration.
///
/// It wants `p_stop` and nothing else, so an engine computes only that. A write error is kept and returned by
/// [`finish`](TensorHook::finish), so a trace that could not be written is never mistaken for a complete one.
pub struct StopTrace<W: Write> {
    w: W,
    error: Option<io::Error>,
}

impl<W: Write> StopTrace<W> {
    pub fn new(w: W) -> Self {
        Self { w, error: None }
    }

    /// The first write error, if any; leaves it in place.
    pub fn error(&self) -> Option<&io::Error> {
        self.error.as_ref()
    }

    pub fn into_inner(self) -> W {
        self.w
    }
}

impl<W: Write> TensorHook for StopTrace<W> {
    fn name(&self) -> &str {
        "stop-trace"
    }
    fn wants(&self) -> Wants {
        Wants {
            p_stop: true,
            logits: false,
        }
    }
    fn step(&mut self, s: &Step<'_>) {
        let Some(p) = s.p_stop else { return };
        if self.error.is_some() {
            return;
        }
        let semantics = s.p_stop_semantics.map_or_else(String::new, |v| {
            format!(",\"p_stop_semantics\":{}", serde_json::json!(v))
        });
        let r = writeln!(
            self.w,
            "{{\"chunk\":{},\"draw\":{},\"seed\":{},\"frame\":{},\"token\":{},\"p_stop\":{:e}{}}}",
            s.chunk, s.draw, s.seed, s.frame, s.token, p, semantics
        )
        .and_then(|_| self.w.flush());
        if let Err(e) = r {
            self.error = Some(e);
        }
    }
    fn finish(&mut self) -> Result<(), String> {
        match self.error.take() {
            Some(e) => Err(format!("trace: {e}")),
            None => self.w.flush().map_err(|e| format!("trace: {e}")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Count(Wants, usize);
    impl TensorHook for Count {
        fn name(&self) -> &str {
            "count"
        }
        fn wants(&self) -> Wants {
            self.0
        }
        fn step(&mut self, _s: &Step<'_>) {
            self.1 += 1;
        }
    }

    fn step(p: Option<f64>) -> Step<'static> {
        Step {
            chunk: 0,
            draw: 0,
            seed: 1,
            frame: 2,
            token: 3,
            p_stop: p,
            p_stop_semantics: None,
            logits: None,
        }
    }

    struct Layers {
        selected: Vec<usize>,
        seen: std::rc::Rc<std::cell::RefCell<Vec<(usize, u32, bool, Vec<f32>)>>>,
        panic: bool,
    }
    impl TensorHook for Layers {
        fn name(&self) -> &str {
            "layers"
        }
        fn wants(&self) -> Wants {
            Wants::NONE
        }
        fn layers(&self) -> Vec<usize> {
            self.selected.clone()
        }
        fn step(&mut self, _: &Step<'_>) {}
        fn layer(&mut self, s: &LayerStep<'_>) {
            assert!(!self.panic, "planted hook panic");
            self.seen
                .borrow_mut()
                .push((s.layer, s.frame, s.is_eos, s.values.to_vec()));
        }
    }

    #[test]
    fn layer_union_dispatch_and_failure_isolation() {
        let seen = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
        let mut h = Hooks::new();
        assert!(h.layers().is_empty());
        h.add(Box::new(Layers {
            selected: vec![3, 1, 3],
            seen: seen.clone(),
            panic: false,
        }));
        h.add(Box::new(Layers {
            selected: vec![2, 1],
            seen: seen.clone(),
            panic: true,
        }));
        assert_eq!(h.layers(), vec![1, 2, 3]);
        let mut s = LayerStep {
            chunk: 4,
            draw: 5,
            seed: 6,
            frame: 0,
            token: 7,
            is_eos: false,
            layer: 1,
            values: &[1.0, 2.0],
        };
        h.layer(&s);
        assert_eq!(h.failures().len(), 1);
        assert_eq!(h.layers(), vec![1, 3]);
        s.layer = 3;
        s.frame = 1;
        s.is_eos = true;
        s.values = &[3.0];
        h.layer(&s);
        assert_eq!(
            *seen.borrow(),
            vec![(1, 0, false, vec![1.0, 2.0]), (3, 1, true, vec![3.0])]
        );
    }

    #[test]
    fn no_hooks_want_nothing() {
        assert_eq!(Hooks::new().wants(), Wants::NONE);
        assert!(!Hooks::new().wants().any());
    }

    #[test]
    fn wants_is_the_union_of_the_running_hooks() {
        let mut h = Hooks::new();
        h.add(Box::new(Count(
            Wants {
                p_stop: true,
                logits: false,
            },
            0,
        )));
        h.add(Box::new(Count(
            Wants {
                p_stop: false,
                logits: true,
            },
            0,
        )));
        assert_eq!(
            h.wants(),
            Wants {
                p_stop: true,
                logits: true
            }
        );
    }

    #[test]
    fn stop_trace_writes_the_documented_line() {
        let mut t = StopTrace::new(Vec::new());
        t.step(&step(Some(0.25)));
        t.step(&step(None));
        assert!(t.finish().is_ok());
        assert_eq!(
            String::from_utf8(t.into_inner()).unwrap(),
            "{\"chunk\":0,\"draw\":0,\"seed\":1,\"frame\":2,\"token\":3,\"p_stop\":2.5e-1}\n"
        );
    }

    #[test]
    fn stop_trace_labels_known_semantics_and_escapes_them() {
        let mut t = StopTrace::new(Vec::new());
        let mut s = step(Some(0.25));
        s.p_stop_semantics = Some("producer\"/1");
        t.step(&s);
        t.finish().unwrap();
        let bytes = t.into_inner();
        let value: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(value["p_stop_semantics"], "producer\"/1");
        assert_eq!(value["p_stop"], 0.25);
    }

    struct Broken;
    impl Write for Broken {
        fn write(&mut self, _b: &[u8]) -> io::Result<usize> {
            Err(io::Error::other("disk full"))
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    #[test]
    fn a_trace_that_could_not_be_written_fails_at_finish() {
        let mut h = Hooks::new();
        h.add(Box::new(StopTrace::new(Broken)));
        h.step(&step(Some(0.5)));
        h.finish();
        assert_eq!(h.failures().len(), 1);
        assert!(h.failures()[0].message.contains("disk full"));
    }
}
