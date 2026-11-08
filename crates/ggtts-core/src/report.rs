//! The report envelope `ggtts-report/2`, and the merger that turns a run's envelopes into one report.
//!
//! Every instrument writes one envelope. It says:
//! - who it is: name, version and identity;
//! - what it read: hooks, plus provenance (configuration, versions, seeds, input hashes, raw or trimmed audio);
//! - how it ended: its status;
//! - what it measured: metrics, each optionally scoped to one render and chunk so results can be joined per clip.
//!
//! The merger reads every envelope a run produced and writes `report.json`, `report.md` and `report.html`, with a
//! completeness section for the run's required instruments.
//!
//! - **Nothing is dropped.** An instrument that failed, timed out, was skipped or truncated stays visible. An
//!   envelope that cannot be read, does not validate, belongs to another run or repeats another's identity becomes
//!   an `invalid` entry naming its source and every reason.
//! - **Rendering and measuring are judged apart.** A required instrument that is missing, or present with any status
//!   but `ok`, makes the measurement INCOMPLETE: never a pass, never a zero. A metric the instrument could not
//!   compute is `null`, not 0.
//!
//! [`validate`] is the runtime check. The published schema, `schemas/ggtts-report-2.schema.json`, says the same for
//! instruments in other languages, and `tests/report.rs` holds the two in agreement on every valid case and every
//! planted defect.
//!
//! Metrics are a list, so adding a metric never changes the format. A new field is a new version, and readers accept
//! the version before it ([`READS`]).

use std::fmt::Write as _;
use std::io;
use std::path::Path;

use serde_json::{json, Map, Value};

/// The envelope and report format this module writes.
pub const FORMAT: &str = "ggtts-report/2";

/// The formats [`validate`] accepts: the current one first, then at most the one before it.
pub const READS: &[&str] = &[FORMAT, "ggtts-report/1"];

/// How an instrument ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Status {
    Ok,
    Failed,
    Timeout,
    Skipped,
    /// It ended without reading all its input, for example a stream without `end`.
    Truncated,
    /// Its envelope could not be read or did not validate. Set by the merger; an instrument may also report it.
    Invalid,
}

impl Status {
    pub const ALL: [Status; 6] = [
        Status::Ok,
        Status::Failed,
        Status::Timeout,
        Status::Skipped,
        Status::Truncated,
        Status::Invalid,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            Status::Ok => "ok",
            Status::Failed => "failed",
            Status::Timeout => "timeout",
            Status::Skipped => "skipped",
            Status::Truncated => "truncated",
            Status::Invalid => "invalid",
        }
    }

    fn parse(s: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|v| v.as_str() == s)
    }
}

/// Which way a metric improves: `Neutral` for a descriptive value, `NotApplicable` for a diagnostic.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    Higher,
    Lower,
    Neutral,
    NotApplicable,
}

impl Direction {
    pub const ALL: [Direction; 4] = [
        Direction::Higher,
        Direction::Lower,
        Direction::Neutral,
        Direction::NotApplicable,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            Direction::Higher => "higher",
            Direction::Lower => "lower",
            Direction::Neutral => "neutral",
            Direction::NotApplicable => "n/a",
        }
    }
}

/// Whether the audio an instrument measured was trimmed of edge silence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AudioState {
    Raw,
    Trimmed,
    /// Audio was measured but its trim state was not established.
    Unknown,
    /// The instrument measured no audio.
    NotApplicable,
}

impl AudioState {
    pub fn as_str(self) -> &'static str {
        match self {
            AudioState::Raw => "raw",
            AudioState::Trimmed => "trimmed",
            AudioState::Unknown => "unknown",
            AudioState::NotApplicable => "n/a",
        }
    }
}

/// One measured value.
#[derive(Debug, Clone, PartialEq)]
pub struct Metric {
    pub name: String,
    /// `None` when the instrument could not compute it: a missing value, never zero.
    pub value: Option<f64>,
    pub unit: String,
    pub direction: Direction,
    /// A two-sided interval around `value`, when the instrument states one.
    pub interval: Option<(f64, f64)>,
    /// The render this value is about, when it is about one render.
    pub render: Option<String>,
    /// The chunk this value is about, within `render`.
    pub chunk: Option<u64>,
}

/// A file, with its media type and content hash.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Artifact {
    pub path: String,
    pub kind: String,
    pub sha256: String,
}

/// An input the instrument read, by hash.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InputFile {
    pub path: String,
    pub sha256: String,
}

/// An effective seed: the render, the chunk when per chunk, and the seed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Seed {
    pub render: String,
    pub chunk: Option<u64>,
    pub seed: u64,
}

/// Who the envelope belongs to.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Identity {
    /// The run's id; every envelope of one run carries the same one.
    pub run: String,
    /// This instrument instance's id, unique within the run.
    pub instrument: String,
}

/// What the result depends on. A value nobody knows is `None` (null), never a guess.
#[derive(Debug, Clone, PartialEq)]
pub struct Provenance {
    /// The instrument's effective configuration, defaults included.
    pub config: Map<String, Value>,
    pub ggtts: Option<String>,
    pub engine: Option<String>,
    pub backend: Option<String>,
    pub model_sha256: Option<String>,
    pub seeds: Vec<Seed>,
    pub inputs: Vec<InputFile>,
    pub audio: AudioState,
}

/// One instrument's envelope.
#[derive(Debug, Clone, PartialEq)]
pub struct Envelope {
    pub name: String,
    pub version: String,
    /// The format the envelope was written in, one of [`READS`].
    pub format: String,
    pub identity: Identity,
    pub provenance: Provenance,
    pub hooks: Vec<String>,
    pub status: Status,
    pub metrics: Vec<Metric>,
    pub artifacts: Vec<Artifact>,
    /// The instrument's log, when it has one.
    pub log: Option<String>,
    /// Why it did not end `ok`, in one line.
    pub error: Option<String>,
}

const NAME_RULE: &str = "^[a-z0-9][a-z0-9_.-]{0,63}$";

/// An instrument or metric name: lower case, digits, `_`, `.` and `-`, starting with a letter or digit.
fn is_name(s: &str) -> bool {
    s.bytes()
        .next()
        .is_some_and(|c| c.is_ascii_lowercase() || c.is_ascii_digit())
        && s.len() <= 64
        && s.bytes()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || matches!(c, b'_' | b'.' | b'-'))
}

fn is_sha256(s: &str) -> bool {
    s.len() == 64 && s.bytes().all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c))
}

/// Collects every reason an object fails, each prefixed with where it is.
struct Check<'v, 'w> {
    at: String,
    m: &'v Map<String, Value>,
    why: &'w mut Vec<String>,
}

impl<'v, 'w> Check<'v, 'w> {
    fn new(at: &str, v: &'v Value, why: &'w mut Vec<String>, what: &str) -> Option<Self> {
        match v {
            Value::Object(m) => Some(Self { at: at.into(), m, why }),
            _ => {
                why.push(format!("{at}must be an object ({what})"));
                None
            }
        }
    }
    fn fail(&mut self, s: String) {
        self.why.push(format!("{}{s}", self.at));
    }
    fn only(&mut self, allowed: &[&str]) {
        let extra: Vec<String> = self
            .m
            .keys()
            .filter(|k| !allowed.contains(&k.as_str()))
            .cloned()
            .collect();
        for k in extra {
            self.fail(format!("`{k}` is not a field here"));
        }
    }
    fn text(&mut self, k: &str) -> Option<String> {
        match self.m.get(k) {
            Some(Value::String(s)) if !s.is_empty() => Some(s.clone()),
            Some(_) => {
                self.fail(format!("`{k}` must be a non-empty string"));
                None
            }
            None => {
                self.fail(format!("`{k}` is missing"));
                None
            }
        }
    }
    /// A required field that is a non-empty string or null.
    fn text_or_null(&mut self, k: &str) -> Option<String> {
        match self.m.get(k) {
            Some(Value::Null) => None,
            Some(Value::String(s)) if !s.is_empty() => Some(s.clone()),
            Some(_) => {
                self.fail(format!("`{k}` must be a non-empty string or null"));
                None
            }
            None => {
                self.fail(format!("`{k}` is missing (null when unknown)"));
                None
            }
        }
    }
    fn name(&mut self, k: &str) -> Option<String> {
        let n = self.text(k)?;
        if is_name(&n) {
            Some(n)
        } else {
            self.fail(format!("`{k}` must match {NAME_RULE}"));
            None
        }
    }
    fn sha(&mut self, k: &str) -> Option<String> {
        match self.m.get(k).and_then(Value::as_str) {
            Some(s) if is_sha256(s) => Some(s.into()),
            _ => {
                self.fail(format!("`{k}` must be 64 lower-case hex digits"));
                None
            }
        }
    }
    fn list(&mut self, k: &str) -> &'v [Value] {
        match self.m.get(k) {
            Some(Value::Array(a)) => a,
            _ => {
                self.fail(format!("`{k}` is missing or not a list"));
                &[]
            }
        }
    }
    fn uint_or_absent(&mut self, k: &str) -> Option<u64> {
        match self.m.get(k) {
            None => None,
            Some(v) => v.as_u64().or_else(|| {
                self.fail(format!("`{k}` must be a non-negative integer"));
                None
            }),
        }
    }
}

/// Check one envelope against a supported report version. Every reason it fails, not only the first.
pub fn validate(v: &Value) -> Result<Envelope, Vec<String>> {
    let mut why = Vec::new();
    let Some(mut c) = Check::new("", v, &mut why, "an envelope") else {
        return Err(why);
    };
    c.only(&[
        "$schema",
        "format",
        "name",
        "version",
        "identity",
        "provenance",
        "hooks",
        "status",
        "metrics",
        "artifacts",
        "log",
        "error",
    ]);
    let format = c.text("format");
    if let Some(f) = &format {
        if !READS.contains(&f.as_str()) {
            c.fail(format!("format {f:?} is not one this reads ({})", READS.join(", ")));
        }
    }
    let name = c.name("name");
    let version = c.text("version");
    let status = match c.m.get("status").and_then(Value::as_str) {
        Some(s) => Status::parse(s).or_else(|| {
            c.fail(format!(
                "status {s:?} is not one of {}",
                names(&Status::ALL.map(Status::as_str))
            ));
            None
        }),
        None => {
            c.fail("`status` is missing or not a string".into());
            None
        }
    };
    let mut hooks = Vec::new();
    for h in c.list("hooks") {
        match h.as_str() {
            Some(s) if !s.is_empty() => hooks.push(s.to_string()),
            _ => c.fail("every hook is a non-empty string".into()),
        }
    }
    let metric_values = c.list("metrics");
    let artifact_values = c.list("artifacts");
    let log = c.text_or_null("log");
    let error = match c.m.get("error") {
        None | Some(Value::Null) => None,
        Some(Value::String(s)) => Some(s.clone()),
        Some(_) => {
            c.fail("`error` must be a string or null".into());
            None
        }
    };
    let (id_v, prov_v) = (c.m.get("identity"), c.m.get("provenance"));
    if id_v.is_none() {
        c.fail("`identity` is missing".into());
    }
    if prov_v.is_none() {
        c.fail("`provenance` is missing".into());
    }

    let identity = id_v.and_then(|v| identity(v, &mut why));
    let provenance = prov_v.and_then(|v| provenance(v, &mut why));
    if format.as_deref() == Some("ggtts-report/1")
        && provenance.as_ref().is_some_and(|p| p.audio == AudioState::Unknown)
    {
        why.push("provenance: unknown audio state requires ggtts-report/2".into());
    }
    let mut metrics = Vec::new();
    for (i, x) in metric_values.iter().enumerate() {
        if let Some(m) = metric(&format!("metrics[{i}]: "), x, &mut why) {
            metrics.push(m);
        }
    }
    let mut artifacts = Vec::new();
    for (i, x) in artifact_values.iter().enumerate() {
        if let Some(mut a) = Check::new(&format!("artifacts[{i}]: "), x, &mut why, "{path, type, sha256}") {
            a.only(&["path", "type", "sha256"]);
            if let (Some(path), Some(kind), Some(sha256)) = (a.text("path"), a.text("type"), a.sha("sha256")) {
                artifacts.push(Artifact { path, kind, sha256 });
            }
        }
    }
    if !why.is_empty() {
        return Err(why);
    }
    Ok(Envelope {
        name: name.unwrap(),
        version: version.unwrap(),
        format: format.unwrap(),
        identity: identity.unwrap(),
        provenance: provenance.unwrap(),
        hooks,
        status: status.unwrap(),
        metrics,
        artifacts,
        log,
        error,
    })
}

fn names(v: &[&str]) -> String {
    v.join(", ")
}

fn identity(v: &Value, why: &mut Vec<String>) -> Option<Identity> {
    let mut c = Check::new("identity: ", v, why, "{run, instrument}")?;
    c.only(&["run", "instrument"]);
    let (run, instrument) = (c.text("run"), c.text("instrument"));
    Some(Identity {
        run: run?,
        instrument: instrument?,
    })
}

fn provenance(v: &Value, why: &mut Vec<String>) -> Option<Provenance> {
    let mut c = Check::new("provenance: ", v, why, "the provenance object")?;
    c.only(&[
        "config",
        "ggtts",
        "engine",
        "backend",
        "model_sha256",
        "seeds",
        "inputs",
        "audio",
    ]);
    let config = match c.m.get("config") {
        Some(Value::Object(m)) => Some(m.clone()),
        _ => {
            c.fail("`config` must be an object (the effective configuration)".into());
            None
        }
    };
    let (ggtts, engine, backend) = (
        c.text_or_null("ggtts"),
        c.text_or_null("engine"),
        c.text_or_null("backend"),
    );
    let model_sha256 = match c.m.get("model_sha256") {
        Some(Value::Null) => None,
        Some(_) => c.sha("model_sha256"),
        None => {
            c.fail("`model_sha256` is missing (null when there is no model)".into());
            None
        }
    };
    let audio = match c.m.get("audio").and_then(Value::as_str) {
        Some("raw") => Some(AudioState::Raw),
        Some("trimmed") => Some(AudioState::Trimmed),
        Some("unknown") => Some(AudioState::Unknown),
        Some("n/a") => Some(AudioState::NotApplicable),
        _ => {
            c.fail("`audio` must be \"raw\", \"trimmed\", \"unknown\" or \"n/a\"".into());
            None
        }
    };
    let (seed_values, input_values) = (c.list("seeds"), c.list("inputs"));
    let mut seeds = Vec::new();
    let mut inputs = Vec::new();
    for (i, x) in seed_values.iter().enumerate() {
        if let Some(mut s) = Check::new(&format!("provenance: seeds[{i}]: "), x, why, "{render, chunk, seed}") {
            s.only(&["render", "chunk", "seed"]);
            let (render, chunk) = (s.text("render"), s.uint_or_absent("chunk"));
            let seed = s.m.get("seed").and_then(Value::as_u64);
            if seed.is_none() {
                s.fail("`seed` must be a non-negative integer".into());
            }
            if let (Some(render), Some(seed)) = (render, seed) {
                seeds.push(Seed { render, chunk, seed });
            }
        }
    }
    for (i, x) in input_values.iter().enumerate() {
        if let Some(mut s) = Check::new(&format!("provenance: inputs[{i}]: "), x, why, "{path, sha256}") {
            s.only(&["path", "sha256"]);
            if let (Some(path), Some(sha256)) = (s.text("path"), s.sha("sha256")) {
                inputs.push(InputFile { path, sha256 });
            }
        }
    }
    Some(Provenance {
        config: config?,
        ggtts,
        engine,
        backend,
        model_sha256,
        seeds,
        inputs,
        audio: audio?,
    })
}

fn metric(at: &str, v: &Value, why: &mut Vec<String>) -> Option<Metric> {
    let mut c = Check::new(at, v, why, "a metric")?;
    c.only(&["name", "value", "unit", "direction", "interval", "render", "chunk"]);
    let name = c.name("name");
    let value = match c.m.get("value") {
        Some(Value::Null) => Some(None),
        Some(v) if v.is_number() => Some(v.as_f64()),
        _ => {
            c.fail("`value` must be a number, or null when it could not be computed".into());
            None
        }
    };
    // A unit may be "1" for a ratio or a count, never empty: a metric without a unit cannot be compared.
    let unit = c.text("unit");
    let direction = match c.m.get("direction").and_then(Value::as_str) {
        Some(d) => Direction::ALL.into_iter().find(|x| x.as_str() == d),
        None => None,
    };
    if direction.is_none() {
        c.fail(format!(
            "`direction` must be one of {}",
            names(&Direction::ALL.map(Direction::as_str))
        ));
    }
    let interval = match c.m.get("interval") {
        None => Some(None),
        Some(Value::Array(a)) if a.len() == 2 => match (a[0].as_f64(), a[1].as_f64()) {
            (Some(lo), Some(hi)) if lo <= hi => Some(Some((lo, hi))),
            _ => {
                c.fail("`interval` must be [low, high] numbers with low <= high".into());
                None
            }
        },
        Some(_) => {
            c.fail("`interval` must be [low, high]".into());
            None
        }
    };
    let render = match c.m.get("render") {
        None => Some(None),
        Some(Value::String(s)) if !s.is_empty() => Some(Some(s.clone())),
        Some(_) => {
            c.fail("`render` must be a non-empty string".into());
            None
        }
    };
    let chunk = c.uint_or_absent("chunk");
    if chunk.is_some() && render == Some(None) {
        c.fail("`chunk` needs the `render` it belongs to".into());
    }
    Some(Metric {
        name: name?,
        value: value?,
        unit: unit?,
        direction: direction?,
        interval: interval?,
        render: render?,
        chunk,
    })
}

impl Envelope {
    /// The envelope as JSON, in the field order of the schema.
    pub fn to_json(&self) -> Value {
        let metrics: Vec<Value> = self
            .metrics
            .iter()
            .map(|m| {
                let mut o = json!({
                    "name": m.name, "value": m.value, "unit": m.unit, "direction": m.direction.as_str(),
                });
                if let Some((lo, hi)) = m.interval {
                    o["interval"] = json!([lo, hi]);
                }
                if let Some(r) = &m.render {
                    o["render"] = json!(r);
                }
                if let Some(c) = m.chunk {
                    o["chunk"] = json!(c);
                }
                o
            })
            .collect();
        let p = &self.provenance;
        let seeds: Vec<Value> = p
            .seeds
            .iter()
            .map(|s| {
                let mut o = json!({"render": s.render, "seed": s.seed});
                if let Some(c) = s.chunk {
                    o["chunk"] = json!(c);
                }
                o
            })
            .collect();
        let inputs: Vec<Value> = p
            .inputs
            .iter()
            .map(|i| json!({"path": i.path, "sha256": i.sha256}))
            .collect();
        let artifacts: Vec<Value> = self
            .artifacts
            .iter()
            .map(|a| json!({"path": a.path, "type": a.kind, "sha256": a.sha256}))
            .collect();
        let mut o = json!({
            "format": self.format, "name": self.name, "version": self.version,
            "identity": {"run": self.identity.run, "instrument": self.identity.instrument},
            "provenance": {
                "config": p.config, "ggtts": p.ggtts, "engine": p.engine, "backend": p.backend,
                "model_sha256": p.model_sha256, "seeds": seeds, "inputs": inputs, "audio": p.audio.as_str(),
            },
            "hooks": self.hooks, "status": self.status.as_str(), "metrics": metrics, "artifacts": artifacts,
            "log": self.log,
        });
        if let Some(e) = &self.error {
            o["error"] = json!(e);
        }
        o
    }
}

/// One envelope as the merger found it.
#[derive(Debug, Clone, PartialEq)]
pub struct Input {
    /// Where it came from, usually its file path. Named in the report, and the fallback name of an invalid one.
    pub source: String,
    pub text: String,
}

/// One entry of a merged report: a valid envelope, or an `invalid` stand-in for one that was not.
#[derive(Debug, Clone, PartialEq)]
pub struct Entry {
    pub source: String,
    pub envelope: Envelope,
    /// Every reason the envelope is invalid; empty when it is not.
    pub invalid: Vec<String>,
}

/// One required instrument, as the run found it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Requirement {
    pub name: String,
    /// At least one entry carries the name.
    pub present: bool,
    /// Every entry with the name is a valid envelope.
    pub valid: bool,
    /// The statuses of its entries, in report order; empty when absent.
    pub statuses: Vec<Status>,
}

impl Requirement {
    /// Present, valid, and every one of its results `ok`.
    pub fn met(&self) -> bool {
        self.present && self.valid && self.statuses.iter().all(|s| *s == Status::Ok)
    }
}

/// A run's merged report.
#[derive(Debug, Clone, PartialEq)]
pub struct Report {
    pub run: String,
    /// Ordered by instrument name, then source, so the report does not depend on the order envelopes were found.
    pub entries: Vec<Entry>,
    /// The run's required instruments, in the order the run declared them.
    pub required: Vec<Requirement>,
}

/// The name an invalid envelope is listed under: its own `name` when that is a valid name, else its source's
/// file stem made into one.
fn fallback_name(source: &str, v: Option<&Value>) -> String {
    if let Some(n) = v
        .and_then(|v| v.get("name"))
        .and_then(Value::as_str)
        .filter(|n| is_name(n))
    {
        return n.to_string();
    }
    let stem = Path::new(source)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or(source)
        .to_ascii_lowercase();
    let n: String = stem
        .chars()
        .map(|c| {
            if c.is_ascii_lowercase() || c.is_ascii_digit() || matches!(c, '_' | '.' | '-') {
                c
            } else {
                '-'
            }
        })
        .collect::<String>()
        .trim_start_matches(['_', '.', '-'])
        .chars()
        .take(64)
        .collect();
    if n.is_empty() {
        "unnamed".into()
    } else {
        n
    }
}

/// The stand-in for an envelope that is invalid: the run's identity, nothing measured, every reason kept.
fn invalid_entry(run: &str, source: &str, v: Option<&Value>, why: Vec<String>) -> Entry {
    let s = |k: &str| {
        v.and_then(|v| v.get(k))
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty())
            .map(String::from)
    };
    let name = fallback_name(source, v);
    let instrument = v
        .and_then(|v| v.pointer("/identity/instrument"))
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
        .map_or_else(|| name.clone(), String::from);
    Entry {
        source: source.into(),
        envelope: Envelope {
            name,
            version: s("version").unwrap_or_else(|| "unknown".into()),
            format: FORMAT.into(),
            identity: Identity {
                run: run.into(),
                instrument,
            },
            provenance: Provenance {
                config: Map::new(),
                ggtts: None,
                engine: None,
                backend: None,
                model_sha256: None,
                seeds: Vec::new(),
                inputs: Vec::new(),
                audio: AudioState::NotApplicable,
            },
            hooks: Vec::new(),
            status: Status::Invalid,
            metrics: Vec::new(),
            artifacts: Vec::new(),
            log: s("log"),
            error: Some(format!("invalid envelope: {}", why.join("; "))),
        },
        invalid: why,
    }
}

/// Merge the envelopes of run `run`, against the instruments the run declared required. Every input becomes exactly
/// one entry. An envelope is `invalid` when it cannot be read, does not validate, names another run, or shares its
/// instrument identity with another envelope of this run (all of them, so neither copy is trusted).
pub fn merge(run: &str, inputs: &[Input], required: &[&str]) -> Report {
    let mut entries: Vec<Entry> = inputs
        .iter()
        .map(|i| {
            let parsed: Result<Value, _> = serde_json::from_str(&i.text);
            let checked = match &parsed {
                Ok(v) => validate(v),
                Err(e) => Err(vec![format!("not JSON: {e}")]),
            };
            match checked {
                Ok(e) if e.identity.run != run => invalid_entry(
                    run,
                    &i.source,
                    parsed.as_ref().ok(),
                    vec![format!("it belongs to run {:?}, not {run:?}", e.identity.run)],
                ),
                Ok(envelope) => Entry {
                    source: i.source.clone(),
                    envelope,
                    invalid: Vec::new(),
                },
                Err(why) => invalid_entry(run, &i.source, parsed.as_ref().ok(), why),
            }
        })
        .collect();
    let ids: Vec<String> = entries.iter().map(|e| e.envelope.identity.instrument.clone()).collect();
    for (k, e) in entries.iter_mut().enumerate() {
        let id = &ids[k];
        if e.invalid.is_empty() && ids.iter().filter(|x| *x == id).count() > 1 {
            let why = vec![format!("instrument identity {id:?} appears more than once in this run")];
            let v = serde_json::from_str::<Value>(&inputs[k].text).ok();
            *e = invalid_entry(run, &e.source, v.as_ref(), why);
        }
    }
    entries.sort_by(|a, b| (&a.envelope.name, &a.source).cmp(&(&b.envelope.name, &b.source)));
    let required = required
        .iter()
        .map(|name| {
            let mine: Vec<&Entry> = entries.iter().filter(|e| e.envelope.name == *name).collect();
            Requirement {
                name: name.to_string(),
                present: !mine.is_empty(),
                valid: mine.iter().all(|e| e.invalid.is_empty()),
                statuses: mine.iter().map(|e| e.envelope.status).collect(),
            }
        })
        .collect();
    Report {
        run: run.into(),
        entries,
        required,
    }
}

impl Report {
    /// How many instruments ended in each status, in [`Status::ALL`] order.
    pub fn counts(&self) -> [(Status, usize); 6] {
        Status::ALL.map(|s| (s, self.entries.iter().filter(|e| e.envelope.status == s).count()))
    }

    /// Every required instrument met. A run with nothing required is complete.
    pub fn complete(&self) -> bool {
        self.required.iter().all(Requirement::met)
    }

    fn verdict(&self) -> &'static str {
        if self.complete() {
            "complete"
        } else {
            "incomplete"
        }
    }

    /// `report.json`.
    pub fn to_json(&self) -> Value {
        let mut summary = Map::new();
        for (s, n) in self.counts() {
            summary.insert(s.as_str().into(), n.into());
        }
        let completeness: Vec<Value> = self
            .required
            .iter()
            .map(|r| {
                json!({"name": r.name, "present": r.present, "valid": r.valid,
                       "statuses": r.statuses.iter().map(|s| s.as_str()).collect::<Vec<_>>(), "met": r.met()})
            })
            .collect();
        let instruments: Vec<Value> = self
            .entries
            .iter()
            .map(|e| {
                let mut o = e.envelope.to_json();
                // A v2 aggregate normalizes accepted v1 entries to the current envelope format.
                o["format"] = json!(FORMAT);
                o["source"] = json!(e.source);
                if !e.invalid.is_empty() {
                    o["invalid"] = json!(e.invalid);
                }
                o
            })
            .collect();
        json!({"format": FORMAT, "run": self.run, "measurement": self.verdict(), "summary": summary,
               "completeness": completeness, "instruments": instruments})
    }

    fn summary_line(&self) -> String {
        let counts: Vec<String> = self
            .counts()
            .iter()
            .filter(|(_, n)| *n > 0)
            .map(|(st, n)| format!("{n} {}", st.as_str()))
            .collect();
        format!(
            "{} instruments: {}.",
            self.entries.len(),
            if counts.is_empty() {
                "none".into()
            } else {
                counts.join(", ")
            }
        )
    }

    fn requirement_row(r: &Requirement) -> [String; 4] {
        let statuses: Vec<&str> = r.statuses.iter().map(|s| s.as_str()).collect();
        [
            r.name.clone(),
            if r.present { "yes" } else { "**no**" }.into(),
            if r.valid { "yes" } else { "**no**" }.into(),
            if statuses.is_empty() {
                "-".into()
            } else {
                statuses.join(", ")
            },
        ]
    }

    /// `report.md`: the measurement verdict and completeness, one row per instrument, then each metric and file.
    pub fn to_markdown(&self) -> String {
        let mut s = format!(
            "# Instrument report\n\nRun `{}`. {}\n\n",
            self.run.replace('`', "'"),
            self.summary_line()
        );
        let _ = writeln!(s, "## Measurement: **{}**\n", self.verdict().to_uppercase());
        if self.required.is_empty() {
            s.push_str("No instrument was declared required.\n");
        } else {
            s.push_str("| Required instrument | Present | Valid | Status |\n|---|---|---|---|\n");
            for r in &self.required {
                let [a, b, c, d] = Self::requirement_row(r);
                let _ = writeln!(s, "| {} | {b} | {c} | {} |", md_cell(&a), md_cell(&d));
            }
        }
        s.push_str("\n## Instruments\n\n| Instrument | Version | Status | Metrics | Notes |\n|---|---|---|---|---|\n");
        for e in &self.entries {
            let env = &e.envelope;
            let _ = writeln!(
                s,
                "| {} | {} | **{}** | {} | {} |",
                md_cell(&env.name),
                md_cell(&env.version),
                env.status.as_str(),
                env.metrics.len(),
                md_cell(env.error.as_deref().unwrap_or(""))
            );
        }
        for e in self.entries.iter().filter(|e| !e.envelope.metrics.is_empty()) {
            let _ = write!(
                s,
                "\n### {}\n\n| Metric | Value | Unit | Better | Interval | Scope |\n|---|---|---|---|---|---|\n",
                md_cell(&e.envelope.name)
            );
            for m in &e.envelope.metrics {
                let [v, iv, sc] = metric_cells(m);
                let _ = writeln!(
                    s,
                    "| {} | {v} | {} | {} | {iv} | {} |",
                    md_cell(&m.name),
                    md_cell(&m.unit),
                    m.direction.as_str(),
                    md_cell(&sc)
                );
            }
        }
        let files: Vec<(&Entry, String, String)> = self.files();
        if !files.is_empty() {
            s.push_str("\n## Files\n\n");
            for (e, what, path) in files {
                let _ = writeln!(
                    s,
                    "- {}: {what} `{}`",
                    md_cell(&e.envelope.name),
                    path.replace('`', "'")
                );
            }
        }
        s
    }

    /// `report.html`: the same content as the Markdown, self-contained, every value escaped.
    pub fn to_html(&self) -> String {
        let mut s = String::from(
            "<!doctype html>\n<html lang=\"en\"><head><meta charset=\"utf-8\"><title>Instrument report</title>\n\
             <style>body{font:14px/1.4 system-ui,sans-serif;margin:16px}table{border-collapse:collapse}\
             td,th{border:1px solid #999;padding:2px 8px;text-align:left}\
             .failed,.timeout,.truncated,.invalid,.incomplete{color:#b00}.skipped{color:#777}</style></head><body>\n\
             <h1>Instrument report</h1>\n",
        );
        let _ = writeln!(
            s,
            "<p>Run <code>{}</code>. {}</p>",
            esc(&self.run),
            esc(&self.summary_line())
        );
        let _ = writeln!(
            s,
            "<h2>Measurement: <span class=\"{v}\">{}</span></h2>",
            self.verdict().to_uppercase(),
            v = self.verdict()
        );
        if self.required.is_empty() {
            s.push_str("<p>No instrument was declared required.</p>\n");
        } else {
            s.push_str("<table><tr><th>Required instrument</th><th>Present</th><th>Valid</th><th>Status</th></tr>\n");
            for r in &self.required {
                let [a, b, c, d] = Self::requirement_row(r);
                let _ = writeln!(
                    s,
                    "<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>",
                    esc(&a),
                    esc(&b.replace("**", "")),
                    esc(&c.replace("**", "")),
                    esc(&d)
                );
            }
            s.push_str("</table>\n");
        }
        s.push_str("<h2>Instruments</h2>\n<table><tr><th>Instrument</th><th>Version</th><th>Status</th><th>Metrics</th><th>Notes</th></tr>\n");
        for e in &self.entries {
            let env = &e.envelope;
            let _ = writeln!(
                s,
                "<tr><td>{}</td><td>{}</td><td class=\"{st}\">{st}</td><td>{}</td><td>{}</td></tr>",
                esc(&env.name),
                esc(&env.version),
                env.metrics.len(),
                esc(env.error.as_deref().unwrap_or("")),
                st = env.status.as_str(),
            );
        }
        s.push_str("</table>\n");
        for e in self.entries.iter().filter(|e| !e.envelope.metrics.is_empty()) {
            let _ = writeln!(
                s,
                "<h3>{}</h3>\n<table><tr><th>Metric</th><th>Value</th><th>Unit</th><th>Better</th><th>Interval</th><th>Scope</th></tr>",
                esc(&e.envelope.name)
            );
            for m in &e.envelope.metrics {
                let [v, iv, sc] = metric_cells(m);
                let _ = writeln!(
                    s,
                    "<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>",
                    esc(&m.name),
                    esc(&v),
                    esc(&m.unit),
                    esc(m.direction.as_str()),
                    esc(&iv),
                    esc(&sc)
                );
            }
            s.push_str("</table>\n");
        }
        let files = self.files();
        if !files.is_empty() {
            s.push_str("<h2>Files</h2>\n<ul>\n");
            for (e, what, path) in files {
                let _ = writeln!(
                    s,
                    "<li>{}: {} <code>{}</code></li>",
                    esc(&e.envelope.name),
                    esc(&what),
                    esc(&path)
                );
            }
            s.push_str("</ul>\n");
        }
        s.push_str("</body></html>\n");
        s
    }

    /// `(entry, what, path)` for each artifact and log, in report order.
    fn files(&self) -> Vec<(&Entry, String, String)> {
        let mut v = Vec::new();
        for e in &self.entries {
            for a in &e.envelope.artifacts {
                v.push((e, format!("{} sha256 {}", a.kind, &a.sha256[..12]), a.path.clone()));
            }
            if let Some(l) = &e.envelope.log {
                v.push((e, "log".into(), l.clone()));
            }
        }
        v
    }

    /// Write `report.json`, `report.md` and `report.html` into `dir`.
    pub fn write(&self, dir: &Path) -> io::Result<()> {
        std::fs::create_dir_all(dir)?;
        let mut j = serde_json::to_string_pretty(&self.to_json()).map_err(io::Error::other)?;
        j.push('\n');
        std::fs::write(dir.join("report.json"), j)?;
        std::fs::write(dir.join("report.md"), self.to_markdown())?;
        std::fs::write(dir.join("report.html"), self.to_html())
    }
}

/// A metric's value (`missing` for null), interval and scope, as table text.
fn metric_cells(m: &Metric) -> [String; 3] {
    let v = m.value.map_or_else(|| "missing".to_string(), |v| v.to_string());
    let iv = m.interval.map_or(String::new(), |(lo, hi)| format!("[{lo}, {hi}]"));
    let sc = match (&m.render, m.chunk) {
        (Some(r), Some(c)) => format!("{r} chunk {c}"),
        (Some(r), None) => r.clone(),
        _ => "run".into(),
    };
    [v, iv, sc]
}

/// A Markdown table cell: no pipe or newline can break the row.
fn md_cell(s: &str) -> String {
    s.replace('\\', "\\\\").replace('|', "\\|").replace(['\n', '\r'], " ")
}

/// HTML text escaping.
fn esc(s: &str) -> String {
    let mut o = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '&' => o.push_str("&amp;"),
            '<' => o.push_str("&lt;"),
            '>' => o.push_str("&gt;"),
            '"' => o.push_str("&quot;"),
            '\'' => o.push_str("&#39;"),
            c => o.push(c),
        }
    }
    o
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_the_current_format_and_at_most_the_previous_one() {
        assert_eq!(READS[0], FORMAT);
        assert!(READS.len() <= 2, "{READS:?}");
        if let [cur, prev] = READS {
            let v = |f: &str| {
                f.rsplit_once('/')
                    .map(|(a, n)| (a.to_string(), n.parse::<u32>().unwrap()))
            };
            let ((a, n), (b, m)) = (v(cur).unwrap(), v(prev).unwrap());
            assert!(a == b && m + 1 == n, "{READS:?}");
        }
    }

    #[test]
    fn names_are_lower_case_tokens() {
        for ok in ["timings", "stop-probability", "cer.judge_a", "0x"] {
            assert!(is_name(ok), "{ok}");
        }
        for bad in ["", "Timings", "-x", "a b", "a/b", "<script>"] {
            assert!(!is_name(bad), "{bad}");
        }
    }

    #[test]
    fn escaping_leaves_no_markup() {
        assert_eq!(esc("<a href=\"x\">&'"), "&lt;a href=&quot;x&quot;&gt;&amp;&#39;");
        assert_eq!(md_cell("a|b\nc"), "a\\|b c");
    }
}
