//! Raw, unsafe bindings to ggml at [`GGML_COMMIT`], with ggml built from source by CMake (CPU, static).
//!
//! Everything is behind the `ggml` feature; without it the crate holds only the pin. Nothing here is safe: a
//! tensor is a raw `*mut ggml_tensor`, and a failed `GGML_ASSERT` aborts the process rather than returning an
//! error, so callers validate before they call. See `README.md` for the build and for what in ggml's C API fights
//! Rust.

/// The ggml commit the bindings were generated from and the submodule is pinned to.
pub const GGML_COMMIT: &str = "af97976c7810cdabb1863172f31c432dab767de7";

#[cfg(feature = "ggml")]
#[allow(
    non_upper_case_globals,
    non_camel_case_types,
    non_snake_case,
    dead_code,
    clippy::all,
    clippy::pedantic
)]
mod raw {
    include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
}

#[cfg(feature = "ggml")]
pub use raw::*;

/// The CPU threads to run a graph with now: `configured`, but never more than `allowed`, the CPUs that may be used
/// at this moment. A `configured` of 0 or less is returned as it is: unset, so the count the engine was loaded with
/// stands, as before this cap existed.
///
/// More threads than allowed CPUs does not just waste them: ggml's threads wait for each other at every step, so a
/// thread the kernel has parked stalls the rest. Measured on an Android phone whose off-screen app is confined to
/// three cores: four threads did not finish a line in 400 s that three threads rendered in 87 s.
pub fn cap_threads(configured: i32, allowed: usize) -> i32 {
    if configured <= 0 {
        return configured;
    }
    configured.min(allowed.clamp(1, i32::MAX as usize) as i32)
}

/// The CPUs the calling thread may run on, as `std::thread::available_parallelism` reports them, read afresh.
///
/// * **Linux and Android:** the calling thread's affinity mask (`sched_getaffinity(0)`), and cgroup CPU quotas. A
///   cpuset applies to every thread of a process, so on Android, where an app is moved to a smaller cpuset when it
///   leaves the screen, the render thread's mask follows it.
/// * **Windows:** the number of logical processors (`GetSystemInfo`). It does **not** see affinity or job-object
///   limits, so on Windows this cap never lowers the count below the machine's processors.
/// * **macOS:** the number of logical CPUs.
pub fn allowed_cpus_now() -> usize {
    std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1)
}

/// How long a read of the allowed CPUs is reused.
pub const ALLOWED_CPUS_TTL: std::time::Duration = std::time::Duration::from_millis(100);

/// The cache rule, apart from the clock and the reader: reuse `last` while it is younger than [`ALLOWED_CPUS_TTL`],
/// else call `read`. Returns the count and the entry to keep.
pub fn cached_cpus(
    last: Option<(std::time::Instant, usize)>,
    now: std::time::Instant,
    read: impl FnOnce() -> usize,
) -> (usize, (std::time::Instant, usize)) {
    match last {
        Some((at, n)) if now.saturating_duration_since(at) < ALLOWED_CPUS_TTL => (n, (at, n)),
        _ => {
            let n = read();
            (n, (now, n))
        }
    }
}

/// [`allowed_cpus_now`], re-read at most every 100 ms: it re-reads cgroup files (about 23 us a call), and a render
/// computes many graphs a second. A change of cpuset is seen within 100 ms. The cache is per thread, because on Linux
/// and Android the read is the calling thread's affinity mask: two engines on threads with different masks each see
/// their own.
pub fn allowed_cpus() -> usize {
    use std::cell::Cell;
    thread_local! {
        static LAST: Cell<Option<(std::time::Instant, usize)>> = const { Cell::new(None) };
    }
    LAST.with(|last| {
        let (n, keep) = cached_cpus(last.get(), std::time::Instant::now(), allowed_cpus_now);
        last.set(Some(keep));
        n
    })
}

/// [`cap_threads`] against [`allowed_cpus`]: the count the engines run a graph with.
pub fn threads_now(configured: i32) -> i32 {
    cap_threads(configured, allowed_cpus())
}

/// Whether to say that the cap lowered `configured` to `running`: only when it did, and only the first time `said`
/// is consulted for it (it is set then).
pub fn warn_once(configured: i32, running: i32, said: &std::sync::atomic::AtomicBool) -> bool {
    running < configured && !said.swap(true, std::sync::atomic::Ordering::Relaxed)
}

/// Before a graph compute: set every CPU backend on `sched` to [`threads_now`]`(configured)`. The count can change
/// between computes, because the CPUs a process may use can change while it runs. The thread count changes speed
/// only: the engines' output is the same at any count. A `configured` of 0 or less leaves the backends as loaded.
/// The first time the cap lowers the count, it says so once on stderr.
///
/// # Safety
/// `sched` must be a live scheduler.
#[cfg(feature = "ggml")]
pub unsafe fn cap_sched_threads(sched: ggml_backend_sched_t, configured: i32) {
    static WARNED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    if configured <= 0 {
        return;
    }
    let n = threads_now(configured);
    if warn_once(configured, n, &WARNED) {
        eprintln!(
            "ggtts: {configured} CPU threads were asked for, but only {n} CPUs may be used now; running {n} \
             (more threads than CPUs would stall). Said once."
        );
    }
    for i in 0..ggml_backend_sched_get_n_backends(sched) {
        let b = ggml_backend_sched_get_backend(sched, i);
        if !b.is_null() && ggml_backend_is_cpu(b) {
            ggml_backend_cpu_set_n_threads(b, n);
        }
    }
}

/// `GGML_PAD(x, n)`: a function-like macro, which bindgen does not translate. `n` must be a power of two.
pub const fn pad_to(x: i64, n: i64) -> i64 {
    (x + n - 1) & !(n - 1)
}

/// `GGML_ROPE_TYPE_NEOX`, `GGML_ROPE_TYPE_MROPE`: rope modes are `#define`d ints, re-stated here so code that
/// does not enable `ggml` can still name them in tests.
pub mod rope_mode {
    pub const NORMAL: i32 = 0;
    pub const NEOX: i32 = 2;
    pub const MROPE: i32 = 8;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pad_matches_the_macro() {
        assert_eq!(pad_to(0, 32), 0);
        assert_eq!(pad_to(1, 32), 32);
        assert_eq!(pad_to(256, 256), 256);
        assert_eq!(pad_to(257, 256), 512);
    }

    #[test]
    fn threads_are_capped_at_the_allowed_cpus() {
        assert_eq!(cap_threads(4, 8), 4, "room: the configured count");
        assert_eq!(cap_threads(4, 3), 3, "an off-screen app on three cores");
        assert_eq!(cap_threads(4, 1), 1);
        assert_eq!(cap_threads(4, 0), 1, "at least one");
        assert_eq!(cap_threads(0, 8), 0, "unset stays unset: the loaded count stands");
        assert_eq!(cap_threads(-2, 8), -2);
        assert_eq!(cap_threads(8, usize::MAX), 8);
    }

    /// A thread confined to one CPU must read one allowed CPU, so the cap gives it one thread, through every reader
    /// the engines use: the fresh read, the per-thread cached one, and `threads_now`. This is the case the cap exists
    /// for, made on any Linux machine (CI included) by restricting this test's own thread: a reader of the machine's
    /// processor count, a cache shared across threads, or a `threads_now` that skips the cap would fail here.
    #[cfg(any(target_os = "linux", target_os = "android"))]
    #[test]
    fn a_thread_confined_to_one_cpu_is_capped_to_one_thread() {
        extern "C" {
            fn sched_setaffinity(pid: i32, cpusetsize: usize, mask: *const u64) -> i32;
        }
        let read = std::thread::spawn(|| {
            // CPU 0 only, for this spawned thread (pid 0 is the calling thread); other tests are not affected.
            let mut mask = [0u64; 16];
            mask[0] = 1;
            let rc = unsafe { sched_setaffinity(0, std::mem::size_of_val(&mask), mask.as_ptr()) };
            assert_eq!(rc, 0, "sched_setaffinity");
            (allowed_cpus_now(), allowed_cpus(), threads_now(8), threads_now(0))
        })
        .join()
        .unwrap();
        assert_eq!(read, (1, 1, 1, 0), "one allowed CPU, so one thread; unset stays unset");
    }

    /// The cache's rule: a read younger than the TTL is reused, an older one is read again.
    #[test]
    fn the_cpu_count_is_reread_once_the_cache_is_stale() {
        use std::time::{Duration, Instant};
        let t0 = Instant::now();
        let fresh = Some((t0, 3));
        assert_eq!(cached_cpus(None, t0, || 8), (8, (t0, 8)), "nothing cached: read");
        let soon = t0 + Duration::from_millis(50);
        assert_eq!(
            cached_cpus(fresh, soon, || 8),
            (3, (t0, 3)),
            "younger than the TTL: reused, not read"
        );
        let later = t0 + ALLOWED_CPUS_TTL;
        assert_eq!(
            cached_cpus(fresh, later, || 8),
            (8, (later, 8)),
            "as old as the TTL: read again"
        );
    }

    /// The warning: said once, and only when the cap lowered the count.
    #[test]
    fn the_cap_is_said_once_and_only_when_it_lowers_the_count() {
        use std::sync::atomic::AtomicBool;
        let said = AtomicBool::new(false);
        assert!(!warn_once(4, 4, &said), "not lowered: nothing to say");
        assert!(warn_once(4, 3, &said), "lowered: said");
        assert!(!warn_once(4, 3, &said), "said once");
        assert!(!warn_once(8, 1, &said));
    }

    /// The cap on a live scheduler whose CPU backend was configured for far more threads than this process may use:
    /// the call is safe there and the graph's result is unchanged. (The CPU backend keeps its thread count private,
    /// so the count itself is checked through `cap_threads` above, and its effect by a run on a confined device.)
    #[cfg(feature = "ggml")]
    #[test]
    fn capping_a_live_scheduler_keeps_its_results() {
        unsafe {
            let cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, std::ptr::null());
            assert!(!cpu.is_null());
            ggml_backend_cpu_set_n_threads(cpu, 64);
            let mut backends = [cpu];
            let sched = ggml_backend_sched_new(backends.as_mut_ptr(), std::ptr::null_mut(), 1, 64, false, true);
            assert!(!sched.is_null());
            let params = ggml_init_params {
                mem_size: 1 << 20,
                mem_buffer: std::ptr::null_mut(),
                no_alloc: true,
            };
            let ctx = ggml_init(params);
            let a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1024);
            let b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1024);
            let c = ggml_add(ctx, a, b);
            let gf = ggml_new_graph(ctx);
            ggml_build_forward_expand(gf, c);
            assert!(ggml_backend_sched_alloc_graph(sched, gf));
            let av: Vec<f32> = (0..1024).map(|i| i as f32).collect();
            let bv = vec![0.5f32; 1024];
            ggml_backend_tensor_set(a, av.as_ptr().cast(), 0, 4096);
            ggml_backend_tensor_set(b, bv.as_ptr().cast(), 0, 4096);
            cap_sched_threads(sched, 64);
            assert_eq!(ggml_backend_sched_graph_compute(sched, gf), GGML_STATUS_SUCCESS);
            let mut out = vec![0f32; 1024];
            ggml_backend_tensor_get(c, out.as_mut_ptr().cast(), 0, 4096);
            assert!(out.iter().enumerate().all(|(i, v)| *v == i as f32 + 0.5));
            ggml_backend_sched_free(sched);
            ggml_backend_free(cpu);
            ggml_free(ctx);
        }
    }

    #[test]
    fn pin_is_a_full_sha() {
        assert_eq!(GGML_COMMIT.len(), 40);
        assert!(GGML_COMMIT.bytes().all(|b| b.is_ascii_hexdigit()));
    }

    #[cfg(feature = "ggml")]
    #[test]
    fn rope_modes_match_the_header() {
        assert_eq!(rope_mode::NEOX, GGML_ROPE_TYPE_NEOX as i32);
        assert_eq!(rope_mode::MROPE, GGML_ROPE_TYPE_MROPE as i32);
    }

    #[cfg(feature = "ggml")]
    #[test]
    fn links_and_runs_one_op() {
        // A 2x2 add through a no_alloc=false context: proves the archives link and the bindings' layouts are sane.
        unsafe {
            let params = ggml_init_params {
                mem_size: 1 << 20,
                mem_buffer: std::ptr::null_mut(),
                no_alloc: false,
            };
            let ctx = ggml_init(params);
            assert!(!ctx.is_null());
            let a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
            let b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
            let pa = (*a).data as *mut f32;
            let pb = (*b).data as *mut f32;
            for i in 0..4 {
                *pa.add(i) = i as f32;
                *pb.add(i) = 10.0;
            }
            let c = ggml_add(ctx, a, b);
            let gf = ggml_new_graph(ctx);
            ggml_build_forward_expand(gf, c);
            ggml_graph_compute_with_ctx(ctx, gf, 1);
            let pc = (*c).data as *const f32;
            let got: Vec<f32> = (0..4).map(|i| *pc.add(i)).collect();
            assert_eq!(got, vec![10.0, 11.0, 12.0, 13.0]);
            ggml_free(ctx);
        }
    }
}
