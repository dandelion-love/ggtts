//! #420: one steering scale per layer (`render.steer_scales`, paired with `render.steer_layers`). A file of
//! unit-length directions at per-layer scales s x |v_L| restates one scale s over the unnormalised file, within f32
//! rounding of the added vectors; the per-layer scales are refused when they cannot mean that; and the record says
//! which form a render used.

use ggtts_core::options::Options;
use ggtts_core::steer::{write_safetensors, SteerFile, Steering};

const N: usize = 64;

/// A deterministic vector of `n` values in [-amp, amp].
fn vector(seed: u64, n: usize, amp: f32) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((s >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0) * amp
        })
        .collect()
}

fn file(layers: &[(u32, &[f32])]) -> SteerFile {
    let names: Vec<String> = layers.iter().map(|(l, _)| format!("talker.layer.{l}")).collect();
    let tensors: Vec<(&str, &[f32])> = names.iter().zip(layers).map(|(n, (_, v))| (n.as_str(), *v)).collect();
    SteerFile::parse(&write_safetensors(&tensors)).unwrap()
}

/// `v` at unit length (computed in f64, stored in f32) and its length.
fn unit(v: &[f32]) -> (Vec<f32>, f64) {
    let norm = v.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>().sqrt();
    (v.iter().map(|x| (f64::from(*x) / norm) as f32).collect(), norm)
}

/// The largest difference between two added vectors, in units in the last place of f32.
fn max_ulp(a: &[(u32, Vec<f32>)], b: &[(u32, Vec<f32>)]) -> u32 {
    a.iter()
        .zip(b)
        .flat_map(|((la, va), (lb, vb))| {
            assert_eq!(la, lb);
            va.iter()
                .zip(vb)
                .map(|(x, y)| (x.to_bits() as i64 - y.to_bits() as i64).unsigned_abs() as u32)
        })
        .max()
        .unwrap()
}

/// AC2: a unit-vector file at scales s x |v_L| adds, at every layer, what one scale s adds over the raw file, within
/// 2 f32 places; a planted scale 1% off is far outside that.
#[test]
fn a_unit_file_at_per_layer_scales_restates_one_scale_over_the_raw_file() {
    let raw: Vec<(u32, Vec<f32>)> = (21..26).map(|l| (l, vector(l as u64, N, 3.0 + l as f32))).collect();
    let units: Vec<(u32, Vec<f32>, f64)> = raw.iter().map(|(l, v)| (*l, unit(v).0, unit(v).1)).collect();
    let raw_file = file(&raw.iter().map(|(l, v)| (*l, v.as_slice())).collect::<Vec<_>>());
    let unit_file = file(&units.iter().map(|(l, u, _)| (*l, u.as_slice())).collect::<Vec<_>>());
    let layers: Vec<u32> = raw.iter().map(|(l, _)| *l).collect();
    for s in [0.1875, 0.375, -0.25] {
        let want = raw_file.select(Some(&layers), s).unwrap().scaled();
        let alphas: Vec<f64> = units.iter().map(|(_, _, n)| s * n).collect();
        let got = unit_file.select_scaled(&layers, &alphas).unwrap();
        assert!(
            max_ulp(&got.scaled(), &want) <= 2,
            "scale {s}: {} ulp",
            max_ulp(&got.scaled(), &want)
        );
        // Planted: one scale 1% off.
        let mut off = alphas.clone();
        off[2] *= 1.01;
        let bad = unit_file.select_scaled(&layers, &off).unwrap();
        assert!(max_ulp(&bad.scaled(), &want) > 1000, "a 1% scale error must be seen");
    }
}

/// Each scale goes with the layer at its place in `steer_layers`, whatever their order; the layers apply ascending.
#[test]
fn each_scale_goes_with_its_layer() {
    let v = [1.0f32, 2.0];
    let f = file(&[(3, &v), (5, &v), (9, &v)]);
    let s = f.select_scaled(&[9, 3], &[2.0, 0.5]).unwrap();
    assert_eq!(s.layer_indices(), vec![3, 9]);
    assert_eq!(s.scales.as_deref(), Some(&[0.5, 2.0][..]));
    assert_eq!(s.scaled(), vec![(3, vec![0.5, 1.0]), (9, vec![2.0, 4.0])]);
    // A negative scale reverses that layer's push, as render.steer_scale allows one.
    let neg = f.select_scaled(&[5], &[-1.0]).unwrap();
    assert_eq!(neg.scaled(), vec![(5, vec![-1.0, -2.0])]);
}

/// AC3: refused by the file selection: lengths that differ, a scale that is not finite, one that overflows f32.
#[test]
fn per_layer_scales_that_cannot_apply_are_refused() {
    let v = [1.0f32, 2.0];
    let f = file(&[(3, &v), (5, &v)]);
    assert!(f
        .select_scaled(&[3, 5], &[1.0])
        .unwrap_err()
        .0
        .contains("one scale per layer"));
    assert!(f
        .select_scaled(&[3], &[f64::NAN])
        .unwrap_err()
        .0
        .contains("not a finite number"));
    assert!(f
        .select_scaled(&[3], &[1e39])
        .unwrap_err()
        .0
        .contains("not a finite number in f32"));
    assert!(f
        .select_scaled(&[4], &[1.0])
        .unwrap_err()
        .0
        .contains("not in the steering file"));
}

fn steer_path(name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("ggtts-steer-scales-{name}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("steer.safetensors");
    let v = [1.0f32, 2.0];
    std::fs::write(
        &path,
        write_safetensors(&[("talker.layer.3", &v), ("talker.layer.5", &v)]),
    )
    .unwrap();
    path
}

fn args(list: &[&str]) -> Result<Options, ggtts_core::options::Refusal> {
    let a: Vec<String> = list.iter().map(|s| s.to_string()).collect();
    Options::from_args(&a)
}

/// AC1 and AC3 through the options value: the flag and the document are read; each refusal is under
/// `render.steer_scales`, and comes before the file is read where it needs no file.
#[test]
fn the_option_is_read_and_its_refusals_name_it() {
    let path = steer_path("opts");
    let p = path.to_str().unwrap();
    let mut o = args(&["--steer", p, "--steer-layers", "5,3", "--steer-scales", "2,0.5"]).unwrap();
    assert_eq!(o.render.steer_scales.as_deref(), Some(&[2.0, 0.5][..]));
    o.read_steering().unwrap();
    let s: &Steering = o.render.steering.as_ref().unwrap();
    assert_eq!(
        (s.layer_indices(), s.scales.clone()),
        (vec![3, 5], Some(vec![0.5, 2.0]))
    );
    let doc = format!(r#"{{"render": {{"steer": {p:?}, "steer_layers": [3], "steer_scales": [1.5]}}}}"#);
    let mut d = Options::from_json(&doc).unwrap();
    d.read_steering().unwrap();
    assert_eq!(d.render.steering.unwrap().scales, Some(vec![1.5]));

    // Without steer_layers: refused by the table (requires), under the key.
    assert_eq!(
        args(&["--steer", p, "--steer-scales", "1"]).unwrap_err().key,
        "render.steer_scales"
    );
    // A value that is not a finite number.
    for bad in ["nan", "inf", "1,x", ""] {
        let r = args(&["--steer", p, "--steer-layers", "3", "--steer-scales", bad]).unwrap_err();
        assert_eq!(r.key, "render.steer_scales", "{bad}");
    }
    let refused = |extra: &[&str]| {
        let mut all = vec!["--steer", p];
        all.extend_from_slice(extra);
        let mut o = args(&all).unwrap();
        let r = o.read_steering().unwrap_err();
        assert_eq!(r.key, "render.steer_scales", "{extra:?}: {r}");
        r.message
    };
    assert!(refused(&["--steer-layers", "3,5", "--steer-scales", "1"]).contains("one per layer"));
    assert!(refused(&["--steer-layers", "3", "--steer-scales", "1", "--steer-scale", "2"]).contains("never both"));
    std::fs::remove_dir_all(path.parent().unwrap()).ok();
}

/// AC4: the record says which form a render used: `scales` with per-layer scales, `scale` otherwise, never both.
#[test]
fn the_record_carries_per_layer_scales() {
    let v = [1.0f32, 2.0];
    let f = file(&[(3, &v), (5, &v)]);
    let one = f.select(None, 0.5).unwrap().record();
    assert_eq!(one["scale"], 0.5);
    assert!(one.get("scales").is_none());
    let per = f.select_scaled(&[5, 3], &[2.0, 0.5]).unwrap().record();
    assert_eq!(per["scales"], serde_json::json!([0.5, 2.0]));
    assert_eq!(per["layers"], serde_json::json!([3, 5]));
    assert!(per.get("scale").is_none());
}
