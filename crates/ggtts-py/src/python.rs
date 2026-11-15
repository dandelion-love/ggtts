//! The native module `ggtts._native`: thin glue from Python to [`crate::core`]. The Python package wraps it
//! (`python/ggtts/__init__.py`), and that wrapper is the documented API.

use std::path::PathBuf;

use pyo3::exceptions::PyException;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyByteArray};

use crate::core::{self, Fail};

pyo3::create_exception!(
    ggtts,
    GgttsError,
    PyException,
    "An invalid argument, an engine refusal or a caught panic: `code` is the C-ABI's code (ERR_*), `message` the text."
);

/// A code and a message as a [`GgttsError`] carrying both.
fn error(py: Python<'_>, (code, message): Fail) -> PyErr {
    let err = GgttsError::new_err(format!("{message} (code {code})"));
    let value = err.value(py);
    // Setting attributes on a fresh exception instance cannot fail short of memory exhaustion; the message still
    // carries the code if it does.
    let _ = value.setattr("code", code);
    let _ = value.setattr("message", message);
    err
}

/// Samples as a bytearray of native-endian float32, which `numpy.frombuffer` and `array.array('f')` read as they are.
fn pcm<'py>(py: Python<'py>, samples: &[f32]) -> Bound<'py, PyByteArray> {
    let bytes: Vec<u8> = samples.iter().flat_map(|x| x.to_ne_bytes()).collect();
    PyByteArray::new(py, &bytes)
}

/// A loaded engine. Calls on it from several threads are safe: one renders at a time, and a second call meanwhile is
/// `ERR_BUSY`.
#[pyclass(name = "Engine", module = "ggtts._native", frozen)]
struct PyEngine {
    handle: core::Handle,
    loaded: core::Loaded,
}

#[pymethods]
impl PyEngine {
    /// Render `text` (one paragraph per non-empty line) at `seed`: `(pcm, sample_rate)`, `pcm` a bytearray of
    /// float32 mono samples. The GIL is released while it renders.
    fn render<'py>(&self, py: Python<'py>, text: String, seed: u64) -> PyResult<(Bound<'py, PyByteArray>, u32)> {
        let r = py.allow_threads(|| self.handle.render(&text, seed));
        let (samples, rate) = r.map_err(|e| error(py, e))?;
        Ok((pcm(py, &samples), rate))
    }

    /// Render `text`, calling `on_piece(pcm, sample_rate)` with each kept chunk's piece, in order, before the next
    /// chunk is generated. `on_piece` returning `False` stops the render, and so does an exception it raises, which is
    /// then raised here. Returns `True` when the whole text was rendered, `False` when stopped. The GIL is released
    /// between pieces.
    fn render_stream(&self, py: Python<'_>, text: String, seed: u64, on_piece: PyObject) -> PyResult<bool> {
        let mut raised: Option<PyErr> = None;
        let r = py.allow_threads(|| {
            self.handle.render_stream(&text, seed, &mut |samples, rate| {
                Python::with_gil(|py| match on_piece.call1(py, (pcm(py, samples), rate)) {
                    Ok(v) => !v.bind(py).is(PyBool::new(py, false)),
                    Err(e) => {
                        raised = Some(e);
                        false
                    }
                })
            })
        });
        if let Some(e) = raised {
            return Err(e);
        }
        r.map_err(|e| error(py, e))
    }

    /// Drop the engine. Later calls raise `GgttsError` with `ERR_HANDLE`. A render in progress on another thread
    /// finishes first, and the engine goes when it returns. Returns whether it was open.
    fn close(&self) -> bool {
        self.handle.close()
    }

    #[getter]
    fn closed(&self) -> bool {
        self.handle.is_closed()
    }

    #[getter]
    fn engine(&self) -> &str {
        &self.loaded.engine
    }

    #[getter]
    fn family(&self) -> &str {
        &self.loaded.family
    }

    #[getter]
    fn backend(&self) -> &str {
        &self.loaded.backend
    }

    #[getter]
    fn parity(&self) -> &str {
        &self.loaded.parity
    }

    #[getter]
    fn speaker(&self) -> Option<&str> {
        self.loaded.speaker.as_deref()
    }

    #[getter]
    fn instruction(&self) -> Option<&str> {
        self.loaded.instruction.as_deref()
    }

    #[getter]
    fn device(&self) -> &str {
        self.loaded.device.as_str()
    }

    /// `(role, path, source)` for each model file, `source` being `path` (given) or `cache` (the pinned default).
    #[getter]
    fn model_files(&self) -> Vec<(String, String, String)> {
        self.loaded
            .files
            .iter()
            .map(|f| (f.role.clone(), f.path.to_string_lossy().into_owned(), f.source.clone()))
            .collect()
    }
}

/// Load an engine. See `ggtts.load` for the arguments. The GIL is released while it loads (and downloads).
#[pyfunction]
#[pyo3(signature = (engine, model=None, codec=None, voice=None, speaker=None, instruction=None, device="cpu",
                    threads=None, offline=false, progress=None))]
#[allow(clippy::too_many_arguments)]
fn load(
    py: Python<'_>,
    engine: String,
    model: Option<PathBuf>,
    codec: Option<PathBuf>,
    voice: Option<PathBuf>,
    speaker: Option<String>,
    instruction: Option<String>,
    device: &str,
    threads: Option<i32>,
    offline: bool,
    progress: Option<PyObject>,
) -> PyResult<PyEngine> {
    let device = core::parse_device(device).map_err(|e| error(py, e))?;
    let req = core::LoadRequest {
        engine,
        model,
        codec,
        voice,
        speaker,
        instruction,
        device,
        threads: threads.unwrap_or(0),
        offline,
    };
    let r = py.allow_threads(|| {
        // A download of gigabytes is never silent: to `progress` when given, else to stderr.
        core::load(&req, &mut |message| match &progress {
            Some(f) => Python::with_gil(|py| {
                if let Err(e) = f.call1(py, (message,)) {
                    e.write_unraisable(py, Some(f.bind(py)));
                }
            }),
            None => eprintln!("ggtts: {message}"),
        })
    });
    let (handle, loaded) = r.map_err(|e| error(py, e))?;
    Ok(PyEngine { handle, loaded })
}

/// Write `pcm` (native-endian float32 mono samples, as `bytes`) as 16-bit FLAC at `sample_rate`. The suffix is
/// forced to `.flac`; returns the path written.
#[pyfunction]
fn write_flac(py: Python<'_>, path: PathBuf, pcm: &[u8], sample_rate: u32) -> PyResult<String> {
    if pcm.len() % 4 != 0 {
        return Err(error(
            py,
            (
                core::ERR_ARG,
                format!("pcm is {} bytes, not a whole number of float32 samples", pcm.len()),
            ),
        ));
    }
    let samples: Vec<f32> = pcm
        .chunks_exact(4)
        .map(|b| f32::from_ne_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    let r = py.allow_threads(|| core::guard(|| core::write_flac(&path, &samples, sample_rate)));
    r.map(|p| p.to_string_lossy().into_owned()).map_err(|e| error(py, e))
}

/// The engines this build can load: `fake` always, the model engines with the ggml feature, the planted test
/// engines with the plant feature.
fn built_engines() -> Vec<&'static str> {
    let mut v = vec!["fake"];
    if cfg!(feature = "ggml") {
        v.extend_from_slice(core::ENGINES);
    }
    if cfg!(feature = "plant") {
        v.extend_from_slice(&["panic", "slow", "panic-load"]);
    }
    v
}

#[pymodule]
#[pyo3(name = "_native")]
fn native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("GgttsError", m.py().get_type::<GgttsError>())?;
    m.add_class::<PyEngine>()?;
    m.add_function(wrap_pyfunction!(load, m)?)?;
    m.add_function(wrap_pyfunction!(write_flac, m)?)?;
    m.add("OK", core::OK)?;
    m.add("ERR_ARG", core::ERR_ARG)?;
    m.add("ERR_HANDLE", core::ERR_HANDLE)?;
    m.add("ERR_ENGINE", core::ERR_ENGINE)?;
    m.add("ERR_PANIC", core::ERR_PANIC)?;
    m.add("ERR_STOPPED", core::ERR_STOPPED)?;
    m.add("ERR_BUSY", core::ERR_BUSY)?;
    m.add("DEFAULT_SEED", core::DEFAULT_SEED)?;
    m.add("ENGINES", built_engines())?;
    m.add("VERSION", ggtts_core::VERSION)?;
    m.add("COMMIT", ggtts_core::SOURCE_COMMIT)?;
    Ok(())
}
