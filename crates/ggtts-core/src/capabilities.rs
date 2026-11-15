//! What an engine supports, and the one refusal for an option it does not.
//!
//! An engine declares its [`Capabilities`] ([`Engine::capabilities`](crate::engine::Engine::capabilities)): whether
//! it takes a style instruction, applies steering, has named speakers, the devices it runs on, the values it exports
//! to tensor hooks and whether it emits step events. [`Capabilities::of`] gives the same by engine name, for a check
//! before anything is loaded.
//!
//! [`options::check`](crate::options::check) checks an options value against them before a front end loads or
//! renders, and [`Capabilities::check_render`] checks the pipeline's options and hooks before
//! [`render_plan`](crate::pipeline::render_plan) draws anything. Both refuse with [`Capabilities::refusal`]: one
//! message per option, under the option's schema key, that every front end shows as it is. The engines' own
//! per-request refusals stay, as a second line of defence.

use crate::hooks::Wants;
use crate::options::Refusal;
use crate::pipeline::RenderOptions;

/// What one engine supports beyond rendering text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Capabilities {
    /// The engine's family name, as [`EngineInfo::family`](crate::engine::EngineInfo::family).
    pub family: String,
    /// Takes a non-empty style instruction.
    pub instruction: bool,
    /// Why this engine or checkpoint takes no instruction, when it can say more than its family name: a Qwen3-TTS
    /// checkpoint names its type and hidden size. Read only when `instruction` is false.
    pub no_instruction_why: Option<String>,
    /// Applies steering vectors.
    pub steering: bool,
    /// Has named speakers to choose from.
    pub speakers: bool,
    /// The devices it runs on, by their option value.
    pub devices: Vec<&'static str>,
    /// The values it exports to tensor hooks.
    pub hooks: Wants,
    /// Emits a `step` event per frame into a trace.
    pub step_events: bool,
}

impl Capabilities {
    /// An engine with nothing optional: no instruction, steering or speakers, the CPU only, nothing exported to hooks
    /// and no step events. What [`Engine::capabilities`](crate::engine::Engine::capabilities) says by default.
    pub fn none(family: &str) -> Self {
        Self {
            family: family.to_string(),
            instruction: false,
            no_instruction_why: None,
            steering: false,
            speakers: false,
            devices: vec!["cpu"],
            hooks: Wants::NONE,
            step_events: false,
        }
    }

    /// The capabilities of the engine named `name`, before it is loaded; `None` for a name that is no engine. A
    /// loaded engine may say less: a Qwen3-TTS checkpoint that takes no instruction says so when it is opened.
    pub fn of(name: &str) -> Option<Self> {
        let both = Wants {
            p_stop: true,
            logits: true,
        };
        match name {
            "qwen3-tts" => Some(Self {
                instruction: true,
                steering: true,
                speakers: true,
                devices: vec!["cpu", "cuda"],
                hooks: both,
                step_events: true,
                ..Self::none(name)
            }),
            "kitten-tts-2" => Some(Self {
                devices: vec!["cpu", "cuda"],
                ..Self::none(name)
            }),
            "fake" => Some(Self {
                hooks: both,
                step_events: true,
                ..Self::none(name)
            }),
            _ => None,
        }
    }

    /// The refusal of the option `key` (its schema name) on this engine, given as `value` where the message names
    /// it. The one wording every front end shows.
    pub fn refusal(&self, key: &str, value: Option<&str>) -> Refusal {
        let family = &self.family;
        let message = match key {
            "render.instruction" => match &self.no_instruction_why {
                Some(why) => why.clone(),
                None => format!(
                    "the {family} engine takes no style instruction, which is only supported by the qwen3-tts \
                     engine's custom_voice (1.7B) and voice_design checkpoints"
                ),
            },
            "render.steer" => {
                format!("the {family} engine cannot apply steering, which is only supported by the qwen3-tts engine")
            }
            "load.speaker" => format!("a speaker is only supported by the qwen3-tts engine, not {family}"),
            "load.device" => format!(
                "the {family} engine runs on {}, not {}",
                self.devices.join(" or "),
                value.unwrap_or("that device")
            ),
            "hooks" => format!(
                "the {family} engine exports {} to tensor hooks, and a hook wants {}",
                exported(self.hooks),
                value.unwrap_or("more")
            ),
            other => format!("the {family} engine has no {other}"),
        };
        Refusal {
            key: key.to_string(),
            message,
        }
    }

    /// Refuses a non-empty `instruction` this engine does not take. An empty one is no instruction.
    pub fn check_instruction(&self, instruction: Option<&str>) -> Result<(), Refusal> {
        match instruction.filter(|i| !i.is_empty()) {
            Some(_) if !self.instruction => Err(self.refusal("render.instruction", None)),
            _ => Ok(()),
        }
    }

    /// Refuses what a render asks of this engine that it does not support: a non-empty instruction, steering, and a
    /// hook value it does not export. [`render_plan`](crate::pipeline::render_plan) calls it before anything is drawn.
    pub fn check_render(&self, opts: &RenderOptions, wants: Wants) -> Result<(), Refusal> {
        self.check_instruction(opts.instruction.as_deref())?;
        if opts.steer.is_some() && !self.steering {
            return Err(self.refusal("render.steer", None));
        }
        let missing: Vec<&str> = [
            (wants.p_stop && !self.hooks.p_stop, "p_stop"),
            (wants.logits && !self.hooks.logits, "logits"),
        ]
        .into_iter()
        .filter_map(|(lacking, name)| lacking.then_some(name))
        .collect();
        if !missing.is_empty() {
            return Err(self.refusal("hooks", Some(&missing.join(" and "))));
        }
        Ok(())
    }
}

fn exported(w: Wants) -> &'static str {
    match (w.p_stop, w.logits) {
        (true, true) => "p_stop and logits",
        (true, false) => "only p_stop",
        (false, true) => "only logits",
        (false, false) => "nothing",
    }
}
