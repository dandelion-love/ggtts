//! Engine capabilities (`ggtts_core::capabilities`) and the one check every front end makes (`options::check`, and
//! `render_plan`'s own): each engine asked for each optional capability renders or is refused with the uniform
//! message; a refused render draws nothing; and an option only the environment set is ignored, with a warning and a
//! record, where the same option by flag is refused.

use std::cell::Cell;
use std::rc::Rc;

use ggtts_core::capabilities::Capabilities;
use ggtts_core::engine::{ChunkRequest, Codes, Engine, EngineError, EngineInfo, Generated, Pcm};
use ggtts_core::hooks::{Hooks, Step, TensorHook, Wants};
use ggtts_core::options::{self, Options, Source};
use ggtts_core::pipeline::{
    plan_story, render_plan, DurationJudge, Observers, Output, RenderError, RenderOptions, StorySeeds,
};
use ggtts_core::profile::Profile;
use ggtts_core::steer::{write_safetensors, SteerFile, Steering};
use ggtts_core::trace::{Events, NullEvents};
use ggtts_fake::FakeEngine;

const ENGINES: [&str; 3] = ["qwen3-tts", "kitten-tts-2", "fake"];

/// The fake engine with the capabilities it is given, counting its draws. What it is asked for and supports is
/// rendered as the fake renders without it.
struct Counting {
    inner: FakeEngine,
    caps: Capabilities,
    draws: Rc<Cell<usize>>,
}

impl Counting {
    fn new(caps: Capabilities) -> (Self, Rc<Cell<usize>>) {
        let draws = Rc::new(Cell::new(0));
        let e = Self {
            inner: FakeEngine::default(),
            caps,
            draws: draws.clone(),
        };
        (e, draws)
    }
}

impl Engine for Counting {
    fn capabilities(&self) -> Capabilities {
        self.caps.clone()
    }
    fn info(&self) -> &EngineInfo {
        self.inner.info()
    }
    fn profile(&self) -> &Profile {
        self.inner.profile()
    }
    fn generate(&mut self, req: &ChunkRequest<'_>) -> Result<Generated, EngineError> {
        self.generate_hooked(req, &mut NullEvents, &mut Hooks::new())
    }
    fn generate_hooked(
        &mut self,
        req: &ChunkRequest<'_>,
        events: &mut dyn Events,
        hooks: &mut Hooks,
    ) -> Result<Generated, EngineError> {
        self.draws.set(self.draws.get() + 1);
        let plain = ChunkRequest {
            instruction: None,
            steer: None,
            ..*req
        };
        self.inner.generate_hooked(&plain, events, hooks)
    }
    fn decode(&mut self, codes: &Codes) -> Result<Pcm, EngineError> {
        self.inner.decode(codes)
    }
}

/// A hook wanting `wants`, reading nothing.
struct Wanting(Wants);

impl TensorHook for Wanting {
    fn name(&self) -> &str {
        "wanting"
    }
    fn wants(&self) -> Wants {
        self.0
    }
    fn step(&mut self, _: &Step<'_>) {}
}

fn steering() -> Steering {
    SteerFile::parse(&write_safetensors(&[("talker.layer.3", &[1.0, 2.0])]))
        .unwrap()
        .select(None, 1.0)
        .unwrap()
}

/// What a render asks for, one optional capability at a time.
#[derive(Clone, Copy, Debug)]
enum Ask {
    Instruction,
    Steering,
    PStop,
    Logits,
}

const ASKS: [Ask; 4] = [Ask::Instruction, Ask::Steering, Ask::PStop, Ask::Logits];

/// A render of one short story on `engine`, asking for `ask`: the read, or the error.
fn render_asking(engine: &mut dyn Engine, ask: Option<Ask>) -> Result<usize, RenderError> {
    let profile = *engine.profile();
    let chunks = plan_story(&profile, &["今日天氣好好，我哋去公園行下。".to_string()]).unwrap();
    let mut opts = RenderOptions::for_profile(&profile);
    let mut hooks = Hooks::new();
    match ask {
        Some(Ask::Instruction) => opts.instruction = Some("Speak slowly.".into()),
        Some(Ask::Steering) => opts.steer = Some(steering()),
        Some(Ask::PStop) => hooks.add(Box::new(Wanting(Wants {
            p_stop: true,
            logits: false,
        }))),
        Some(Ask::Logits) => hooks.add(Box::new(Wanting(Wants {
            p_stop: false,
            logits: true,
        }))),
        None => {}
    }
    let seeds = StorySeeds {
        story_id: "t",
        salt: "",
    };
    let read = render_plan(
        engine,
        &chunks,
        &seeds,
        &mut DurationJudge,
        &opts,
        &mut Observers::new(&mut NullEvents, &mut hooks),
        Output::Whole,
    )?;
    Ok(read.samples.len())
}

fn supports(c: &Capabilities, ask: Ask) -> bool {
    match ask {
        Ask::Instruction => c.instruction,
        Ask::Steering => c.steering,
        Ask::PStop => c.hooks.p_stop,
        Ask::Logits => c.hooks.logits,
    }
}

fn refusal(c: &Capabilities, ask: Ask) -> String {
    match ask {
        Ask::Instruction => c.refusal("render.instruction", None),
        Ask::Steering => c.refusal("render.steer", None),
        Ask::PStop => c.refusal("hooks", Some("p_stop")),
        Ask::Logits => c.refusal("hooks", Some("logits")),
    }
    .to_string()
}

/// AC1 of #277: every engine, asked for every optional capability, renders when it supports it and is refused with
/// the uniform message when it does not. The checks before load (`options::check` on the engine's name) say the same
/// for the options a front end takes.
#[test]
fn each_engine_renders_what_it_supports_and_refuses_the_rest_in_one_wording() {
    let mut table = Vec::new();
    for name in ENGINES {
        let caps = Capabilities::of(name).unwrap();
        let (mut e, _) = Counting::new(caps.clone());
        let plain = render_asking(&mut e, None).unwrap();
        for ask in ASKS {
            let (mut e, _) = Counting::new(caps.clone());
            match (supports(&caps, ask), render_asking(&mut e, Some(ask))) {
                (true, Ok(n)) => assert_eq!(n, plain, "{name} {ask:?}: rendered differently"),
                (false, Err(RenderError::Refused(r))) => assert_eq!(r.to_string(), refusal(&caps, ask)),
                (s, got) => panic!("{name} {ask:?}: supported {s}, got {got:?}"),
            }
            table.push(format!(
                "{name} {ask:?}: {}",
                if supports(&caps, ask) { "renders" } else { "refused" }
            ));
        }
        // The options a front end takes, before load.
        for (args, key, supported) in [
            (
                &["--instruction", "Speak slowly."][..],
                "render.instruction",
                caps.instruction,
            ),
            (&["--speaker", "hk_voice_777"][..], "load.speaker", caps.speakers),
            (&["--device", "cuda"][..], "load.device", caps.devices.contains(&"cuda")),
        ] {
            let args: Vec<String> = args.iter().map(|s| s.to_string()).collect();
            let o = Options::from_args(&args).unwrap();
            match options::check(&o, &caps) {
                Ok(_) => assert!(supported, "{name} {key}: accepted"),
                Err(r) => {
                    assert!(!supported, "{name} {key}: refused: {r}");
                    assert_eq!(r.key, key);
                    assert_eq!(r, caps.refusal(key, o.load.device.as_deref()));
                }
            }
        }
    }
    println!("{}", table.join("\n"));
}

/// AC2 of #277: steering on kitten-tts-2 and on the fake engine, an instruction on a checkpoint that takes none, and
/// a hook wanting logits on an engine that exports nothing are each refused before the first draw.
#[test]
fn an_unsupported_render_is_refused_before_anything_is_drawn() {
    let no_instruction = Capabilities {
        instruction: false,
        no_instruction_why: Some("this checkpoint is a base model, which takes no style instruction".into()),
        ..Capabilities::of("qwen3-tts").unwrap()
    };
    for (caps, ask, want) in [
        (Capabilities::of("kitten-tts-2").unwrap(), Ask::Steering, "render.steer"),
        (Capabilities::of("fake").unwrap(), Ask::Steering, "render.steer"),
        (no_instruction, Ask::Instruction, "render.instruction"),
        (Capabilities::none("bare"), Ask::Logits, "hooks"),
    ] {
        let (mut e, draws) = Counting::new(caps.clone());
        match render_asking(&mut e, Some(ask)) {
            Err(RenderError::Refused(r)) => assert_eq!(r.key, want, "{}: {r}", caps.family),
            other => panic!("{} {ask:?}: {other:?}", caps.family),
        }
        assert_eq!(draws.get(), 0, "{} {ask:?}: drew before refusing", caps.family);
    }
}

/// The capabilities an engine says it has are those of its name.
#[test]
fn the_fake_engine_has_the_capabilities_of_its_name() {
    assert_eq!(FakeEngine::default().capabilities(), Capabilities::of("fake").unwrap());
    assert!(Capabilities::of("no-such-engine").is_none());
}

/// AC6 of #277: an option the engine lacks that came only from the environment is ignored, with one warning naming
/// its variable, and recorded; the render goes on. The same option by flag is refused. No engine-specific option has
/// an environment spelling yet, so the test marks the device as the environment's, as one would arrive.
#[test]
fn an_environment_value_the_engine_lacks_is_ignored_and_a_flag_is_refused() {
    let caps = Capabilities::of("fake").unwrap();
    let args = vec!["--device".to_string(), "cuda".to_string()];
    let by_flag = Options::from_args(&args).unwrap();
    assert_eq!(options::check(&by_flag, &caps).unwrap_err().key, "load.device");
    let mut ambient = by_flag.clone();
    ambient.marks.sources.insert("load.device".into(), Source::Environment);
    let checked = options::check(&ambient, &caps).unwrap();
    assert_eq!(checked.options.load.device, None, "the ambient value is reset");
    assert_eq!(checked.warnings.len(), 1, "{:?}", checked.warnings);
    assert!(checked.warnings[0].contains("load.device"), "{:?}", checked.warnings);
    let record = options::record(&checked.options, FakeEngine::default().info());
    assert_eq!(record["ignored"][0]["key"], "load.device", "{record}");
    // And the render goes on.
    let (mut e, draws) = Counting::new(caps);
    render_asking(&mut e, None).unwrap();
    assert!(draws.get() > 0);
}
