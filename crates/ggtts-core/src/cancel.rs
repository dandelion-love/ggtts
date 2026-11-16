//! A per-render, one-way cancellation signal shared across threads.
//!
//! Clones share ownership, so dropping a caller's handle cannot invalidate a
//! signal already held by a running render. A signal is never reset: create a
//! fresh one for each render to prevent a late stop affecting the next read.

use std::collections::HashMap;
use std::sync::{
    atomic::{AtomicBool, AtomicU64, Ordering},
    Arc, Mutex,
};

#[derive(Clone, Default, Debug)]
pub struct CancelSignal(Arc<AtomicBool>);

impl CancelSignal {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn cancel(&self) {
        self.0.store(true, Ordering::Release);
    }

    pub fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::Acquire)
    }
}

/// A cancelled draw is distinct from a normal model stop. Unsupported
/// engines refuse the request before doing work rather than silently ignoring it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CancelOutcome<T> {
    Completed(T),
    Stopped,
    Unsupported,
}

/// Process-local opaque, non-reused IDs. A render clones its signal before
/// releasing the registry lock. Removing an ID prevents future lookups but
/// cannot invalidate the clone held by a running render.
pub struct CancelRegistry {
    next: AtomicU64,
    signals: Mutex<HashMap<u64, CancelSignal>>,
}

impl Default for CancelRegistry {
    fn default() -> Self {
        Self {
            next: AtomicU64::new(1),
            signals: Mutex::new(HashMap::new()),
        }
    }
}

impl CancelRegistry {
    pub fn create(&self) -> Option<u64> {
        let id = self
            .next
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_add(1))
            .ok()?;
        let signal = CancelSignal::new();
        self.signals
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .insert(id, signal);
        Some(id)
    }

    pub fn get(&self, id: u64) -> Option<CancelSignal> {
        self.signals.lock().unwrap_or_else(|p| p.into_inner()).get(&id).cloned()
    }

    pub fn cancel(&self, id: u64) -> bool {
        match self.get(id) {
            Some(signal) => {
                signal.cancel();
                true
            }
            None => false,
        }
    }

    pub fn remove(&self, id: u64) -> bool {
        self.signals
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .remove(&id)
            .is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::mpsc;
    use std::thread;

    #[test]
    fn cancellation_crosses_threads_and_each_render_has_its_own_signal() {
        let signal = CancelSignal::new();
        let worker_signal = signal.clone();
        let (started, waiting) = mpsc::channel();
        let worker = thread::spawn(move || {
            started.send(()).unwrap();
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
            while !worker_signal.is_cancelled() && std::time::Instant::now() < deadline {
                thread::yield_now();
            }
            worker_signal.is_cancelled()
        });
        waiting.recv_timeout(std::time::Duration::from_secs(5)).unwrap();
        signal.cancel();
        assert!(worker.join().unwrap());
        assert!(!CancelSignal::new().is_cancelled());
    }

    #[test]
    fn dropping_a_handle_does_not_invalidate_an_inflight_clone() {
        let signal = CancelSignal::new();
        let copy = signal.clone();
        drop(signal);
        assert!(!copy.is_cancelled());
        copy.cancel();
        assert!(copy.is_cancelled());
    }
    #[test]
    fn registry_supports_cancel_before_start_and_free_during_render() {
        let registry = CancelRegistry::default();
        let first = registry.create().unwrap();
        assert!(registry.cancel(first));
        let held = registry.get(first).unwrap();
        assert!(held.is_cancelled());
        assert!(registry.remove(first));
        assert!(!registry.cancel(first));
        assert!(held.is_cancelled());
        let next = registry.create().unwrap();
        assert_ne!(first, next);
        assert!(!registry.get(next).unwrap().is_cancelled());
    }
}
