//! The instruments this build ships, and the template `instruments.yaml` that lists them.
//!
//! The template is generated from this list, and a test holds the committed `instruments/instruments.yaml` equal to
//! it, so a new instrument cannot ship without its line in the template.

/// One shipped instrument.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Known {
    pub name: &'static str,
    pub version: &'static str,
    /// The program: a binary looked up beside the running binary, or, with an `interpreter`, a script's path inside
    /// the shipped `instruments/` directory.
    pub program: &'static str,
    /// The interpreter that runs `program`, when it is a script. `python` means `GGTTS_PYTHON` if set, else
    /// `python3` (`python` on Windows). A script is never started directly: Windows cannot run a `.py` file.
    pub interpreter: Option<&'static str>,
    /// What it measures, in one line.
    pub about: &'static str,
    /// Example `args`, for the template.
    pub args: &'static [(&'static str, &'static str)],
    /// It needs the run's event stream (see [`crate::config::Entry::needs_events`]).
    pub needs_events: bool,
}

/// A shipped Python instrument: its script under `instruments/`, run by the Python interpreter.
const fn python(name: &'static str, program: &'static str, about: &'static str, needs_events: bool) -> Known {
    Known {
        name,
        version: "0.1.0",
        program,
        interpreter: Some("python"),
        about,
        args: &[],
        needs_events,
    }
}

pub const CATALOGUE: &[Known] = &[
    Known {
        name: "selftest",
        version: env!("CARGO_PKG_VERSION"),
        program: "ggtts-instrument-selftest",
        interpreter: None,
        about: "the runner's own check: counts the run's events and audio bytes; arg mode=crash|hang|bad-envelope|\
                tamper plants a failure",
        args: &[("mode", "ok")],
        needs_events: false,
    },
    python(
        "p-stop-early",
        "templates/python/step/instrument.py",
        "per step: frames whose stop probability rose early, before the chunk's last frames (template)",
        true,
    ),
    python(
        "draws",
        "templates/python/chunk/instrument.py",
        "per chunk: draws, redraws and the judge's verdicts (template)",
        true,
    ),
    python(
        "edge-trim",
        "templates/python/trim/instrument.py",
        "per chunk: the lead and tail seconds trimmed from each kept draw (template)",
        true,
    ),
    python(
        "render-totals",
        "templates/python/render/instrument.py",
        "per render: frames, seconds and chunks, and truncated when the stream has no end (template)",
        true,
    ),
    python(
        "audio-levels",
        "templates/python/audio/instrument.py",
        "the rendered audio: peak and RMS level (template)",
        false,
    ),
    python(
        "internal-pauses",
        "examples/python/internal-pauses/instrument.py",
        "the rendered audio: silences inside the speech, between its first and last loud frame (example)",
        false,
    ),
    python(
        "pitch-range",
        "examples/python/pitch-range/instrument.py",
        "voiced pitch median and 5th-to-95th percentile range using the lab's Pitchy detector",
        false,
    ),
];

pub fn find(name: &str) -> Option<&'static Known> {
    CATALOGUE.iter().find(|k| k.name == name)
}

/// The template config: every shipped instrument, commented out, with what it measures.
pub fn template_yaml() -> String {
    let mut s = String::from(
        "# ggtts instruments: copy this file, uncomment what you want, and pass it with --instruments FILE.\n\
         # Every key has a flag (--instrument NAME,KEY=VALUE, --on-instrument-error, --instrument-timeout), and a flag\n\
         # overrides the file. Schema: docs/instruments/instruments.schema.json.\n\
         format: ggtts-instruments/1\n\
         on_error: continue   # or abort: the render's output is kept either way\n\
         timeout_s: 60        # per instrument, unless its entry sets timeout_s\n\
         require: []          # instruments the measurement depends on: any not ok makes it INCOMPLETE (exit 3)\n\
         instruments:\n",
    );
    for k in CATALOGUE {
        s.push_str(&format!("  # {}: {}\n", k.name, k.about));
        if k.args.is_empty() {
            s.push_str(&format!("  # - {}\n", k.name));
        } else {
            let args: Vec<String> = k.args.iter().map(|(a, v)| format!("{a}: {v}")).collect();
            s.push_str(&format!("  # - {{name: {}, args: {{{}}}}}\n", k.name, args.join(", ")));
        }
    }
    s.push_str(
        "  # An instrument of your own: a program that takes --events, --audio, --envelope, --name and its args.\n",
    );
    s.push_str("  # - {name: mine, command: /path/to/mine, version: '0.1', timeout_s: 30}\n");
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;
    use std::path::Path;

    #[test]
    fn the_committed_template_lists_every_instrument() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../instruments/instruments.yaml");
        // GGTTS_BLESS=1 rewrites the committed template; the diff is then reviewed like any other change.
        if std::env::var_os("GGTTS_BLESS").is_some() {
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(&path, template_yaml()).unwrap();
        }
        let committed = std::fs::read_to_string(&path).unwrap().replace("\r\n", "\n");
        assert_eq!(
            committed,
            template_yaml(),
            "instruments/instruments.yaml is stale: regenerate it from catalogue::template_yaml()"
        );
        for k in CATALOGUE {
            assert!(
                committed.contains(&format!("# {}: ", k.name)),
                "{} is not in the template",
                k.name
            );
        }
    }

    #[test]
    fn the_template_is_a_valid_config_and_every_entry_uncommented_is_too() {
        assert_eq!(Config::from_yaml(&template_yaml()).unwrap().instruments, vec![]);
        let uncommented = template_yaml().replace("  # - ", "  - ");
        let c = Config::from_yaml(&uncommented).unwrap();
        assert_eq!(c.instruments.len(), CATALOGUE.len() + 1);
    }
}
