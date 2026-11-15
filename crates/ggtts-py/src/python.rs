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
    "An invalid argument, an engine refusal or a caught panic: `code` is the C-ABI's code (ERR_*), `message` the \
     text, and `key` the schema key of the one option refused (`render.steer`), or None."
);

/// A code and a message as a [`GgttsError`] carrying both, and the schema key of the option it refused, if one.
fn keyed_error(py: Python<'_>, (code, message): Fail, key: Option<String>) -> PyErr {
    let err = GgttsError::new_err(format!("{message} (code {code})"));
    let value = err.value(py);
    // Setting attributes on a fresh exception instance cannot fail short of memory exhaustion; the message still
    // carries the code if it does.
    let _ = value.setattr("code", code);
    let _ = value.setattr("message", message);
    let _ = value.setattr("key", key);
    err
}

/// A code and a message as a [`GgttsError`], with no refused key.
fn error(py: Python<'_>, fail: Fail) -> PyErr {
    keyed_error(py, fail, None)
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
    /// float32 mono samples. `seed` is the story seed's salt (`render.seed_salt`), `DEFAULT_SEED` when neither it nor
    /// `options` gives one. `options` is a render document (JSON), laid over the load's render defaults. The GIL is
    /// released while it renders.
    #[pyo3(signature = (text, seed=None, options=None))]
    fn render<'py>(
        &self,
        py: Python<'py>,
        text: String,
        seed: Option<u64>,
        options: Option<String>,
    ) -> PyResult<(Bound<'py, PyByteArray>, u32)> {
        let r = py.allow_threads(|| core::guard_keyed(|| self.handle.render_with(&text, seed, options.as_deref())));
        let (samples, rate) = r.map_err(|(e, key)| keyed_error(py, e, key))?;
        Ok((pcm(py, &samples), rate))
    }

    /// Render `text`, calling `on_piece(pcm, sample_rate)` with each kept chunk's piece, in order, before the next
    /// chunk is generated. `on_piece` returning `False` stops the render, and so does an exception it raises, which is
    /// then raised here. Returns `True` when the whole text was rendered, `False` when stopped. `seed` and `options`
    /// are as `render`'s. The GIL is released between pieces.
    #[pyo3(signature = (text, on_piece, seed=None, options=None))]
    fn render_stream(
        &self,
        py: Python<'_>,
        text: String,
        on_piece: PyObject,
        seed: Option<u64>,
        options: Option<String>,
    ) -> PyResult<bool> {
        let mut raised: Option<PyErr> = None;
        let r = py.allow_threads(|| {
            core::guard_keyed(|| {
                self.handle
                    .render_stream_with(&text, seed, options.as_deref(), &mut |samples, rate| {
                        Python::with_gil(|py| match on_piece.call1(py, (pcm(py, samples), rate)) {
                            Ok(v) => !v.bind(py).is(PyBool::new(py, false)),
                            Err(e) => {
                                raised = Some(e);
                                false
                            }
                        })
                    })
            })
        });
        if let Some(e) = raised {
            return Err(e);
        }
        r.map_err(|(e, key)| keyed_error(py, e, key))
    }

    /// The engine's capability statement as JSON (`ggtts-capabilities/1`): what it supports, and the speakers its
    /// model has. The C-ABI's `ggtts_capabilities`, from the same shared piece.
    fn capabilities(&self, py: Python<'_>) -> PyResult<String> {
        self.handle.capabilities().map_err(|e| error(py, e))
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

/// Load an engine. See `ggtts.load` for the arguments; an argument left None is not given (the load document's value,
/// else the default). `options` is a load document (JSON). The GIL is released while it loads (and downloads).
#[pyfunction]
#[pyo3(signature = (engine=None, model=None, codec=None, voice=None, speaker=None, instruction=None, device=None,
                    threads=None, offline=None, options=None, progress=None))]
#[allow(clippy::too_many_arguments)]
fn load(
    py: Python<'_>,
    engine: Option<String>,
    model: Option<PathBuf>,
    codec: Option<PathBuf>,
    voice: Option<PathBuf>,
    speaker: Option<String>,
    instruction: Option<String>,
    device: Option<&str>,
    threads: Option<i32>,
    offline: Option<bool>,
    options: Option<String>,
    progress: Option<PyObject>,
) -> PyResult<PyEngine> {
    let device = device.map(core::parse_device).transpose().map_err(|e| error(py, e))?;
    let req = core::LoadRequest {
        engine,
        model,
        codec,
        voice,
        speaker,
        instruction,
        device,
        threads,
        offline,
        options,
    };
    let r = py.allow_threads(|| {
        core::guard_keyed(|| {
            // A download of gigabytes is never silent: to `progress` when given, else to stderr.
            core::load(&req, &mut |message| match &progress {
                Some(f) => Python::with_gil(|py| {
                    if let Err(e) = f.call1(py, (message,)) {
                        e.write_unraisable(py, Some(f.bind(py)));
                    }
                }),
                None => eprintln!("ggtts: {message}"),
            })
        })
    });
    let (handle, loaded) = r.map_err(|(e, key)| keyed_error(py, e, key))?;
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

/// The options schema as JSON: every option a document may give, as the C-ABI's `ggtts_options_schema` states it.
#[pyfunction]
fn options_schema() -> String {
    core::options_schema()
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
    m.add_function(wrap_pyfunction!(options_schema, m)?)?;
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
