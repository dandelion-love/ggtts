//! `ggtts-event-template --events E.jsonl --envelope ENV.json [--name N] [--run R] [--render R] [--KEY VALUE]...`
//!
//! The command line the runner starts an event instrument with. Writes the envelope, as `failed` when the trace
//! cannot be read, and exits 0 only when it says `ok`.

use std::collections::BTreeMap;
use std::process::ExitCode;

use ggtts_instrument_event_template::{envelope, measure, sha256, Context, NAME};

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut got: BTreeMap<String, String> = BTreeMap::new();
    let mut i = 0;
    while i < args.len() {
        let Some(k) = args[i].strip_prefix("--") else {
            eprintln!("expected --KEY VALUE, got {:?}", args[i]);
            return ExitCode::from(2);
        };
        let Some(v) = args.get(i + 1) else {
            eprintln!("--{k} needs a value");
            return ExitCode::from(2);
        };
        got.insert(k.replace('-', "_"), v.clone());
        i += 2;
    }
    let Some(out) = got.remove("envelope") else {
        eprintln!("--envelope PATH is required");
        return ExitCode::from(2);
    };
    let events = got.remove("events");
    got.remove("audio");
    let mut config = BTreeMap::new();
    let name = got.remove("name").unwrap_or_else(|| NAME.to_string());
    let run = got.remove("run").unwrap_or_else(|| {
        config.insert("run_default".to_string(), "local".to_string());
        "local".into()
    });
    let render = got.remove("render").unwrap_or_else(|| {
        config.insert("render_default".to_string(), "render".to_string());
        "render".into()
    });
    config.extend(got);
    let mut ctx = Context {
        name,
        run,
        render,
        config,
        inputs: Vec::new(),
    };
    let measured = match &events {
        None => Err("--events is required".to_string()),
        Some(p) => match std::fs::read(p) {
            Err(e) => Err(format!("{p}: {e}")),
            Ok(bytes) => {
                let file = std::path::Path::new(p)
                    .file_name()
                    .map_or(p.clone(), |f| f.to_string_lossy().into());
                ctx.inputs.push((file, sha256(&bytes)));
                measure(&String::from_utf8_lossy(&bytes)).map_err(|e| format!("not a trace: {e}"))
            }
        },
    };
    let env = envelope(&ctx, measured.as_ref().map_err(|e| e.clone()));
    let ok = env["status"] == "ok";
    let text = serde_json::to_string_pretty(&env).expect("JSON") + "\n";
    if let Err(e) = std::fs::write(&out, text) {
        eprintln!("{out}: {e}");
        return ExitCode::from(1);
    }
    if ok {
        ExitCode::SUCCESS
    } else {
        ExitCode::from(1)
    }
}
