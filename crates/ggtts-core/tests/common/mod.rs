//! The recorded render cases (`fixtures/pipeline/render.json`) as stories for the fake engine, with their scripts
//! and shapes.

#![allow(dead_code)]

use std::collections::HashMap;
use std::path::Path;

use ggtts_core::engine::Engine;
use ggtts_core::pipeline::{plan_story, render_story_traced, RenderOptions};
use ggtts_core::text::PlannedChunk;
use ggtts_core::trace::{emit_header, Header, Recorder};
use ggtts_fake::{FakeEngine, Shape};
use serde_json::{Map, Value};

pub struct Case {
    pub id: String,
    pub story_id: String,
    pub paragraphs: Vec<String>,
    pub redraw: u32,
    pub seed_salt: String,
    pub script: HashMap<String, Vec<f64>>,
    pub shape: HashMap<String, Shape>,
}

impl Case {
    pub fn engine(&self) -> FakeEngine {
        FakeEngine::new(self.script.clone(), self.shape.clone())
    }
    pub fn chunks(&self, engine: &FakeEngine) -> Vec<PlannedChunk> {
        plan_story(engine.profile(), &self.paragraphs).unwrap()
    }
    pub fn opts(&self, engine: &FakeEngine) -> RenderOptions {
        RenderOptions {
            redraw: self.redraw,
            seed_salt: self.seed_salt.clone(),
            ..RenderOptions::for_profile(engine.profile())
        }
    }
    /// Render traced at `level` on a fresh engine; the trace's lines.
    pub fn trace(&self, level: u8) -> Vec<Map<String, Value>> {
        let mut engine = self.engine();
        let chunks = self.chunks(&engine);
        let opts = self.opts(&engine);
        let mut rec = Recorder::new(level);
        emit_header(&mut rec, &Header::for_engine(&engine, level));
        render_story_traced(&mut engine, &self.story_id, &chunks, &opts, &mut rec).expect("renders");
        rec.lines
    }
}

pub fn cases() -> Vec<Case> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/pipeline/render.json");
    let v: Value = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
    v["cases"]
        .as_array()
        .unwrap()
        .iter()
        .map(|c| Case {
            id: c["id"].as_str().unwrap().to_string(),
            story_id: c["story_id"].as_str().unwrap().to_string(),
            paragraphs: c["paragraphs"]
                .as_array()
                .unwrap()
                .iter()
                .map(|p| p.as_str().unwrap().to_string())
                .collect(),
            redraw: c["redraw"].as_u64().unwrap() as u32,
            seed_salt: c["seed_salt"].as_str().unwrap().to_string(),
            script: c["script"]
                .as_object()
                .unwrap()
                .iter()
                .map(|(k, v)| {
                    (
                        k.clone(),
                        v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect(),
                    )
                })
                .collect(),
            shape: c["shape"]
                .as_object()
                .unwrap()
                .iter()
                .map(|(k, v)| {
                    (
                        k.clone(),
                        Shape {
                            amplitude_scale: v["amplitude_scale"].as_f64(),
                            spike: v["spike"].as_f64(),
                            sample_rate: v["sample_rate"].as_u64().map(|n| n as u32),
                        },
                    )
                })
                .collect(),
        })
        .collect()
}
