//! Which instruments run, and how: from flags, from an `instruments.yaml`, or both.
//!
//! The two map 1:1, and a flag overrides the file:
//!
//! | Flag | YAML |
//! |---|---|
//! | `--instruments FILE` | (the file itself) |
//! | `--instrument NAME` | `instruments: [NAME]` |
//! | `--instrument NAME,timeout_s=S` | `- {name: NAME, timeout_s: S}` |
//! | `--instrument NAME,command=PATH,version=V` | `- {name: NAME, command: PATH, version: V}` |
//! | `--instrument NAME,arg.K=V` | `- {name: NAME, args: {K: V}}` |
//! | `--instrument NAME,needs_events=true` | `- {name: NAME, needs_events: true}` |
//! | `--instrument NAME,interpreter=PROG` | `- {name: NAME, interpreter: PROG}` |
//! | `--require NAME` | `require: [NAME]` |
//! | `--on-instrument-error continue\|abort` | `on_error: continue\|abort` |
//! | `--instrument-timeout S` | `timeout_s: S` |
//!
//! An `--instrument` naming an instrument the file already lists overrides only the keys it gives; a new name is
//! added after the file's. `--require` adds to the file's list. The file is validated against
//! `docs/instruments/instruments.schema.json`: an unknown key, a wrong type or a bad value is refused with its path
//! (`instruments[1].timeout_s: ...`), never ignored.
//!
//! **Required instruments and inputs.** `require` names the instruments the run's *measurement* depends on, and
//! `needs_events` says an instrument needs the event stream. A required instrument that is missing, failed, timed
//! out, skipped (for example for want of its input, or a truncated event stream) or invalid makes the measurement
//! INCOMPLETE ([`crate::runner::RunSummary::complete`]), never a pass and never a zero.

use std::path::Path;

use yaml_rust2::{Yaml, YamlLoader};

/// The config format this crate reads.
pub const FORMAT: &str = "ggtts-instruments/1";
/// The default timeout for one instrument, in seconds.
pub const DEFAULT_TIMEOUT_S: f64 = 60.0;

/// The keys a config file may have at the top. The published schema lists the same keys; a test holds them equal.
pub const TOP_KEYS: [&str; 5] = ["format", "on_error", "timeout_s", "require", "instruments"];
/// The keys one instrument entry may have.
pub const ENTRY_KEYS: [&str; 7] = [
    "name",
    "timeout_s",
    "command",
    "version",
    "args",
    "needs_events",
    "interpreter",
];

/// What the runner does after an instrument fails or times out. It governs the run only: whether the measurement
/// is complete is decided by `require`, never by this.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OnError {
    /// Record it and run the rest. The default: a contributed instrument must never cost the run.
    #[default]
    Continue,
    /// Record it and mark the rest `skipped`. The render's output is kept either way.
    Abort,
}

impl OnError {
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "continue" => Some(OnError::Continue),
            "abort" => Some(OnError::Abort),
            _ => None,
        }
    }
}

/// One instrument to run.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Entry {
    pub name: String,
    /// Overrides the run's timeout for this instrument.
    pub timeout_s: Option<f64>,
    /// An instrument outside the catalogue: the program to start. Needs `version`.
    pub command: Option<String>,
    pub version: Option<String>,
    /// Passed to the instrument as `--K V`, in key order.
    pub args: Vec<(String, String)>,
    /// It needs the run's event stream; without one, or with a truncated one, it is `skipped`. `None`: the
    /// catalogue's value, or no.
    pub needs_events: Option<bool>,
    /// Run `command` (or a shipped script) with this program: `python3 instrument.py ...`. A script cannot be
    /// started directly on Windows, so a `.py` instrument names its interpreter.
    pub interpreter: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct Config {
    pub on_error: OnError,
    pub timeout_s: Option<f64>,
    /// The instruments the measurement depends on, in the order first named.
    pub require: Vec<String>,
    pub instruments: Vec<Entry>,
}

impl Config {
    /// The timeout for `entry`, in seconds.
    pub fn timeout_for(&self, entry: &Entry) -> f64 {
        entry.timeout_s.or(self.timeout_s).unwrap_or(DEFAULT_TIMEOUT_S)
    }

    /// Read and validate a config file.
    pub fn from_file(path: &Path) -> Result<Self, String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
        Self::from_yaml(&text).map_err(|e| format!("{}: {e}", path.display()))
    }

    /// Read and validate a config from YAML text. An empty document is an empty config.
    pub fn from_yaml(text: &str) -> Result<Self, String> {
        let docs = YamlLoader::load_from_str(text).map_err(|e| format!("not YAML: {e}"))?;
        if docs.len() > 1 {
            return Err("one YAML document expected".into());
        }
        let Some(doc) = docs.into_iter().next() else {
            return Ok(Self::default());
        };
        let top = match doc {
            Yaml::Null => return Ok(Self::default()),
            Yaml::Hash(h) => h,
            _ => return Err("the top level must be a mapping".into()),
        };
        let mut c = Self::default();
        for (k, v) in &top {
            let k = key(k, "")?;
            match k.as_str() {
                "format" => {
                    let f = string(v, "format")?;
                    if f != FORMAT {
                        return Err(format!("format: expected {FORMAT:?}, got {f:?}"));
                    }
                }
                "on_error" => {
                    let s = string(v, "on_error")?;
                    c.on_error = OnError::parse(&s).ok_or(format!("on_error: {s:?} is not continue or abort"))?;
                }
                "timeout_s" => c.timeout_s = Some(seconds(v, "timeout_s")?),
                "require" if v.is_null() => {}
                "require" => {
                    let Yaml::Array(items) = v else {
                        return Err("require: must be a list of names".into());
                    };
                    for (i, item) in items.iter().enumerate() {
                        let n = string(item, &format!("require[{i}]"))?;
                        check_name(&n, &format!("require[{i}]"))?;
                        if !c.require.contains(&n) {
                            c.require.push(n);
                        }
                    }
                }
                // `instruments:` with every entry commented out, as in the template, is an empty list.
                "instruments" if v.is_null() => {}
                "instruments" => {
                    let Yaml::Array(items) = v else {
                        return Err("instruments: must be a list".into());
                    };
                    for (i, item) in items.iter().enumerate() {
                        let e = entry(item, &format!("instruments[{i}]"))?;
                        if c.instruments.iter().any(|x| x.name == e.name) {
                            return Err(format!("instruments[{i}].name: {:?} is listed twice", e.name));
                        }
                        c.instruments.push(e);
                    }
                }
                other => return Err(format!("{other}: unknown key (allowed: {})", TOP_KEYS.join(", "))),
            }
        }
        Ok(c)
    }

    /// Apply one `--instrument` flag: override the named entry's given keys, or add it.
    pub fn apply_instrument_flag(&mut self, spec: &str) -> Result<(), String> {
        let e = parse_spec(spec)?;
        match self.instruments.iter_mut().find(|x| x.name == e.name) {
            Some(x) => {
                if e.timeout_s.is_some() {
                    x.timeout_s = e.timeout_s;
                }
                if e.command.is_some() {
                    x.command = e.command;
                }
                if e.version.is_some() {
                    x.version = e.version;
                }
                if e.needs_events.is_some() {
                    x.needs_events = e.needs_events;
                }
                if e.interpreter.is_some() {
                    x.interpreter = e.interpreter;
                }
                for (k, v) in e.args {
                    match x.args.iter_mut().find(|(xk, _)| *xk == k) {
                        Some(slot) => slot.1 = v,
                        None => x.args.push((k, v)),
                    }
                }
                x.args.sort();
            }
            None => self.instruments.push(e),
        }
        Ok(())
    }

    pub fn apply_on_error_flag(&mut self, s: &str) -> Result<(), String> {
        self.on_error = OnError::parse(s).ok_or(format!("--on-instrument-error: {s:?} is not continue or abort"))?;
        Ok(())
    }

    pub fn apply_timeout_flag(&mut self, s: &str) -> Result<(), String> {
        self.timeout_s = Some(positive(s.parse::<f64>().ok(), "--instrument-timeout")?);
        Ok(())
    }

    pub fn apply_require_flag(&mut self, name: &str) -> Result<(), String> {
        check_name(name, "--require")?;
        if !self.require.iter().any(|r| r == name) {
            self.require.push(name.to_string());
        }
        Ok(())
    }
}

/// The instrument flags a binary accepts, collected in order so that a flag always overrides the file.
#[derive(Debug, Clone, Default)]
pub struct Flags {
    pub file: Option<std::path::PathBuf>,
    flags: Vec<(String, String)>,
}

impl Flags {
    /// The flags this module owns. A binary's parser hands each of them, with its value, to [`Flags::take`].
    pub const NAMES: [&'static str; 5] = [
        "--instruments",
        "--instrument",
        "--require",
        "--on-instrument-error",
        "--instrument-timeout",
    ];

    /// Record one flag. `false` if the flag is not an instrument flag.
    pub fn take(&mut self, flag: &str, value: String) -> bool {
        match flag {
            "--instruments" => self.file = Some(value.into()),
            "--instrument" | "--require" | "--on-instrument-error" | "--instrument-timeout" => {
                self.flags.push((flag.to_string(), value))
            }
            _ => return false,
        }
        true
    }

    /// Whether any instrument is asked for, or required.
    pub fn any(&self) -> bool {
        self.file.is_some() || self.flags.iter().any(|(f, _)| f == "--instrument" || f == "--require")
    }

    /// The config: the file's, then every flag over it.
    pub fn resolve(&self) -> Result<Config, String> {
        let mut c = match &self.file {
            Some(p) => Config::from_file(p)?,
            None => Config::default(),
        };
        for (flag, value) in &self.flags {
            match flag.as_str() {
                "--instrument" => c.apply_instrument_flag(value)?,
                "--require" => c.apply_require_flag(value)?,
                "--on-instrument-error" => c.apply_on_error_flag(value)?,
                "--instrument-timeout" => c.apply_timeout_flag(value)?,
                _ => unreachable!("only instrument flags are recorded"),
            }
        }
        Ok(c)
    }
}

fn parse_spec(spec: &str) -> Result<Entry, String> {
    let mut parts = spec.split(',');
    let name = parts.next().unwrap_or_default().trim().to_string();
    check_name(&name, "--instrument")?;
    let mut e = Entry {
        name,
        ..Entry::default()
    };
    for p in parts {
        let (k, v) = p
            .split_once('=')
            .ok_or(format!("--instrument {spec}: {p:?} is not KEY=VALUE"))?;
        match k {
            "timeout_s" => {
                e.timeout_s = Some(positive(
                    v.parse::<f64>().ok(),
                    &format!("--instrument {spec}: timeout_s"),
                )?)
            }
            "command" => e.command = Some(v.to_string()),
            "version" => e.version = Some(v.to_string()),
            "interpreter" => e.interpreter = Some(v.to_string()),
            "needs_events" => {
                e.needs_events = Some(match v {
                    "true" => true,
                    "false" => false,
                    _ => return Err(format!("--instrument {spec}: needs_events must be true or false")),
                })
            }
            _ => match k.strip_prefix("arg.") {
                Some(a) if !a.is_empty() => e.args.push((a.to_string(), v.to_string())),
                _ => {
                    return Err(format!(
                        "--instrument {spec}: unknown key {k:?} (allowed: timeout_s, command, version, \
                         needs_events, interpreter, arg.NAME)"
                    ))
                }
            },
        }
    }
    e.args.sort();
    Ok(e)
}

fn entry(item: &Yaml, at: &str) -> Result<Entry, String> {
    let h = match item {
        Yaml::String(s) => {
            check_name(s, at)?;
            return Ok(Entry {
                name: s.clone(),
                ..Entry::default()
            });
        }
        Yaml::Hash(h) => h,
        _ => return Err(format!("{at}: must be a name or a mapping")),
    };
    let mut e = Entry::default();
    for (k, v) in h {
        let k = key(k, at)?;
        let path = format!("{at}.{k}");
        match k.as_str() {
            "name" => e.name = string(v, &path)?,
            "timeout_s" => e.timeout_s = Some(seconds(v, &path)?),
            "command" => e.command = Some(string(v, &path)?),
            "version" => e.version = Some(string(v, &path)?),
            "interpreter" => e.interpreter = Some(string(v, &path)?),
            "needs_events" => match v {
                Yaml::Boolean(b) => e.needs_events = Some(*b),
                _ => return Err(format!("{path}: must be true or false")),
            },
            "args" => {
                let Yaml::Hash(a) = v else {
                    return Err(format!("{path}: must be a mapping"));
                };
                for (ak, av) in a {
                    let ak = key(ak, &path)?;
                    let av = match av {
                        Yaml::String(s) => s.clone(),
                        Yaml::Integer(n) => n.to_string(),
                        Yaml::Real(r) => r.clone(),
                        Yaml::Boolean(b) => b.to_string(),
                        _ => return Err(format!("{path}.{ak}: must be a string, number or boolean")),
                    };
                    e.args.push((ak, av));
                }
            }
            other => {
                return Err(format!(
                    "{at}.{other}: unknown key (allowed: {})",
                    ENTRY_KEYS.join(", ")
                ))
            }
        }
    }
    check_name(&e.name, &format!("{at}.name"))?;
    if e.command.is_some() && e.version.is_none() {
        return Err(format!("{at}.version: an instrument with a command needs a version"));
    }
    e.args.sort();
    Ok(e)
}

/// A name is also a directory name in the report, so it is kept to `[a-z0-9-]`.
fn check_name(name: &str, at: &str) -> Result<(), String> {
    if name.is_empty()
        || !name
            .chars()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-')
    {
        return Err(format!("{at}: {name:?} is not a name of a-z, 0-9 and -"));
    }
    Ok(())
}

fn key(k: &Yaml, at: &str) -> Result<String, String> {
    match k {
        Yaml::String(s) => Ok(s.clone()),
        _ => Err(format!("{at}: keys must be strings")),
    }
}

fn string(v: &Yaml, at: &str) -> Result<String, String> {
    match v {
        Yaml::String(s) => Ok(s.clone()),
        _ => Err(format!("{at}: must be a string")),
    }
}

fn seconds(v: &Yaml, at: &str) -> Result<f64, String> {
    let n = match v {
        Yaml::Integer(n) => Some(*n as f64),
        Yaml::Real(r) => r.parse::<f64>().ok(),
        _ => None,
    };
    positive(n, at)
}

fn positive(n: Option<f64>, at: &str) -> Result<f64, String> {
    match n {
        Some(x) if x.is_finite() && x > 0.0 => Ok(x),
        _ => Err(format!("{at}: must be a positive number of seconds")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flags_and_yaml_map_one_to_one() {
        let yaml = Config::from_yaml(
            "on_error: abort\ntimeout_s: 5\nrequire: [other]\ninstruments:\n  - selftest\n  - name: other\n    \
             command: /bin/x\n    version: '2'\n    timeout_s: 1.5\n    needs_events: true\n    interpreter: python3\n    args: {mode: ok, n: 3}\n",
        )
        .unwrap();
        let mut flags = Config::default();
        flags.apply_on_error_flag("abort").unwrap();
        flags.apply_timeout_flag("5").unwrap();
        flags.apply_require_flag("other").unwrap();
        flags.apply_instrument_flag("selftest").unwrap();
        flags
            .apply_instrument_flag("other,command=/bin/x,version=2,timeout_s=1.5,needs_events=true,interpreter=python3,arg.mode=ok,arg.n=3")
            .unwrap();
        assert_eq!(yaml, flags);
    }

    #[test]
    fn a_flag_overrides_the_file() {
        let mut c =
            Config::from_yaml("on_error: abort\ninstruments:\n  - {name: selftest, timeout_s: 9, args: {mode: ok}}\n")
                .unwrap();
        c.apply_on_error_flag("continue").unwrap();
        c.apply_instrument_flag("selftest,arg.mode=crash").unwrap();
        c.apply_instrument_flag("added").unwrap();
        assert_eq!(c.on_error, OnError::Continue);
        assert_eq!(
            c.instruments[0].timeout_s,
            Some(9.0),
            "a key the flag does not give is kept"
        );
        assert_eq!(c.instruments[0].args, vec![("mode".into(), "crash".into())]);
        assert_eq!(c.instruments[1].name, "added");
    }

    #[test]
    fn a_bad_file_is_refused_with_its_path() {
        for (yaml, want) in [
            ("instrumnets: []", "instrumnets: unknown key"),
            ("on_error: stop", "on_error:"),
            ("timeout_s: -1", "timeout_s:"),
            (
                "instruments: [{name: a, timeout: 3}]",
                "instruments[0].timeout: unknown key",
            ),
            ("instruments: [{name: a, command: x}]", "instruments[0].version:"),
            (
                "instruments: [{name: a, needs_events: yes please}]",
                "instruments[0].needs_events:",
            ),
            ("require: [Bad]", "require[0]:"),
            ("instruments: [{name: Bad_Name}]", "instruments[0].name:"),
            ("instruments: [a, a]", "instruments[1].name:"),
            ("format: ggtts-instruments/0", "format:"),
            ("- a list", "the top level"),
        ] {
            let e = Config::from_yaml(yaml).unwrap_err();
            assert!(e.starts_with(want), "{yaml:?}: {e}");
        }
        assert_eq!(
            Config::from_yaml("").unwrap(),
            Config::default(),
            "an empty file is an empty config"
        );
    }

    #[test]
    fn the_published_schema_lists_exactly_the_keys_the_reader_accepts() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../docs/instruments/instruments.schema.json");
        let schema: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        let keys = |v: &serde_json::Value| {
            let mut k: Vec<String> = v["properties"].as_object().unwrap().keys().cloned().collect();
            k.sort();
            k
        };
        let sorted = |a: &[&str]| {
            let mut v: Vec<String> = a.iter().map(|s| s.to_string()).collect();
            v.sort();
            v
        };
        assert_eq!(keys(&schema), sorted(&TOP_KEYS));
        assert_eq!(keys(&schema["$defs"]["entry"]), sorted(&ENTRY_KEYS));
        assert_eq!(schema["additionalProperties"], false);
        assert_eq!(schema["$defs"]["entry"]["additionalProperties"], false);
        assert_eq!(schema["properties"]["format"]["const"], FORMAT);
    }
}
