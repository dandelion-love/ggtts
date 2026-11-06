//! A template tensor hook: stop pressure early in each draw, from `p_stop`.
//!
//! The hook is compiled only with the `enabled` feature (see the README for what a tensor hook may and may not do).

#[cfg(feature = "enabled")]
pub use hook::{EarlyStop, Summary};

#[cfg(feature = "enabled")]
mod hook {
    use std::cell::RefCell;
    use std::collections::BTreeMap;
    use std::rc::Rc;

    use ggtts_core::hooks::{Step, TensorHook, Wants};

    /// What the hook found, readable after the render through the handle [`EarlyStop::new`] returns.
    #[derive(Debug, Clone, Default, PartialEq)]
    pub struct Summary {
        /// The highest `p_stop` in the first half of any draw; `None` when no draw had a `p_stop`.
        pub p_stop_early_max: Option<f64>,
        /// Frames in a draw's first half at or above the threshold.
        pub early_stop_frames: u64,
        pub draws: u64,
    }

    /// The hook. It keeps each draw's `p_stop` values until the render finishes, then summarises.
    pub struct EarlyStop {
        threshold: f64,
        draws: BTreeMap<(u32, u32), Vec<f64>>,
        out: Rc<RefCell<Summary>>,
    }

    impl EarlyStop {
        /// The hook, and a handle to its summary.
        pub fn new(threshold: f64) -> (Self, Rc<RefCell<Summary>>) {
            let out = Rc::new(RefCell::new(Summary::default()));
            (
                Self {
                    threshold,
                    draws: BTreeMap::new(),
                    out: out.clone(),
                },
                out,
            )
        }
    }

    impl TensorHook for EarlyStop {
        fn name(&self) -> &str {
            "early-stop"
        }
        fn wants(&self) -> Wants {
            Wants {
                p_stop: true,
                logits: false,
            }
        }
        fn step(&mut self, s: &Step<'_>) {
            if let Some(p) = s.p_stop {
                self.draws.entry((s.chunk, s.draw)).or_default().push(p);
            }
        }
        fn finish(&mut self) -> Result<(), String> {
            let mut sum = Summary {
                draws: self.draws.len() as u64,
                ..Summary::default()
            };
            for ps in self.draws.values() {
                for &p in &ps[..ps.len() / 2] {
                    sum.p_stop_early_max = Some(sum.p_stop_early_max.map_or(p, |m: f64| m.max(p)));
                    sum.early_stop_frames += u64::from(p >= self.threshold);
                }
            }
            *self.out.borrow_mut() = sum;
            Ok(())
        }
    }
}
