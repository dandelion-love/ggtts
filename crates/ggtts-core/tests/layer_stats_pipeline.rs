use ggtts_core::audio::{Judgement, Verdict};
use ggtts_core::engine::{ChunkRequest, Codes, Engine, EngineError, EngineInfo, Generated, Pcm};
use ggtts_core::hooks::{Hooks, LayerStep, Step};
use ggtts_core::layer_stats::LayerStats;
use ggtts_core::pipeline::{render_plan, Judge, Observers, Output, RenderOptions};
use ggtts_core::profile::Profile;
use ggtts_core::text::PlannedChunk;
use ggtts_core::trace::{Events, NullEvents};
use ggtts_fake::FakeEngine;

struct LayerEngine {
    inner: FakeEngine,
    omit: bool,
    wrong_token: bool,
}
impl Engine for LayerEngine {
    fn capabilities(&self) -> ggtts_core::capabilities::Capabilities {
        ggtts_core::capabilities::Capabilities::of("qwen3-tts").unwrap()
    }
    fn info(&self) -> &EngineInfo {
        self.inner.info()
    }
    fn profile(&self) -> &Profile {
        self.inner.profile()
    }
    fn supports_layer_hooks(&self) -> bool {
        true
    }
    fn generate(&mut self, req: &ChunkRequest<'_>) -> Result<Generated, EngineError> {
        self.inner.generate(req)
    }
    fn decode(&mut self, codes: &Codes) -> Result<Pcm, EngineError> {
        self.inner.decode(codes)
    }
    fn generate_hooked(
        &mut self,
        req: &ChunkRequest<'_>,
        _: &mut dyn Events,
        hooks: &mut Hooks,
    ) -> Result<Generated, EngineError> {
        let g = self.inner.generate(req)?;
        for frame in 0..2 {
            let token = frame + 10;
            hooks.step(&Step {
                chunk: req.chunk,
                draw: req.draw,
                seed: req.seed,
                frame,
                token,
                p_stop: Some(0.0),
                p_stop_semantics: None,
                logits: None,
            });
            for layer in hooks.layers() {
                if self.omit && layer == 1 {
                    continue;
                }
                let values = [req.draw as f32 + frame as f32, layer as f32];
                hooks.layer(&LayerStep {
                    chunk: req.chunk,
                    draw: req.draw,
                    seed: req.seed,
                    frame,
                    token: token + u32::from(self.wrong_token),
                    is_eos: frame == 1,
                    layer,
                    values: &values,
                });
            }
        }
        Ok(g)
    }
}
struct RejectFirst(usize);
impl Judge for RejectFirst {
    fn judge(&mut self, _: &Profile, _: &[f32], _: u32, _: &str) -> Judgement {
        self.0 += 1;
        Judgement {
            verdict: if self.0 == 1 { Some(Verdict::Silent) } else { None },
            s_per_char: 1.0,
        }
    }
}
fn chunks() -> Vec<PlannedChunk> {
    ["First.", "Second."]
        .into_iter()
        .map(|text| PlannedChunk {
            para: 0,
            text: text.into(),
            punct: ".".into(),
            last_in_para: false,
        })
        .collect()
}
fn execute(
    omit: bool,
    wrong_token: bool,
) -> (
    Result<ggtts_core::pipeline::Read, ggtts_core::pipeline::RenderError>,
    LayerStats,
) {
    let mut engine = LayerEngine {
        inner: FakeEngine::default(),
        omit,
        wrong_token,
    };
    let mut opts = RenderOptions::for_profile(engine.profile());
    opts.redraw = 1;
    let stats = LayerStats::new(vec![0, 1]).unwrap();
    let mut hooks = Hooks::new();
    hooks.add(Box::new(stats.clone()));
    let mut events = NullEvents;
    let mut obs = Observers::new(&mut events, &mut hooks).with_layer_stats(stats.clone());
    let result = render_plan(
        &mut engine,
        &chunks(),
        &|chunk: usize, _: &str, draw: u32| 100 + chunk as u32 * 10 + draw,
        &mut RejectFirst(0),
        &opts,
        &mut obs,
        Output::Whole,
    );
    (result, stats)
}
#[test]
fn multi_chunk_redraw_keeps_only_the_heard_draw_and_its_seed() {
    let (result, stats) = execute(false, false);
    let read = result.unwrap();
    let rows = stats.records().unwrap();
    assert_eq!(rows.len(), 2);
    assert_eq!((rows[0].chunk, rows[0].draw, rows[0].seed), (0, 1, 101));
    assert_eq!((rows[1].chunk, rows[1].draw, rows[1].seed), (1, 0, 110));
    assert_eq!(read.chunks[0].redraws, 1);
    assert_eq!(rows[0].layers[0].mean, vec![1.5, 0.0]);
    assert_eq!(rows[1].layers[0].mean, vec![0.5, 0.0]);
}
#[test]
fn missing_layer_and_mismatched_tokens_fail_and_remain_incomplete() {
    for (omit, wrong_token) in [(true, false), (false, true)] {
        let (result, stats) = execute(omit, wrong_token);
        assert!(result.is_err());
        assert!(stats.records().is_err());
    }
}
