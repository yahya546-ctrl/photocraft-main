//! Learned Select Subject (desktop only).
//!
//! With a network the app downloads a small salient-object ONNX model once (U²-Netp, Apache-2.0,
//! about 5 MB) into the settings directory and registers it with
//! `photocraft_algo::segment::subject::set_matter`; Select Subject then uses it. Without a
//! network, or if anything here fails, nothing is registered and the classical pipeline in
//! `photocraft-algo` runs as before, so the tool always works.
//!
//! The model runs through Microsoft's ONNX Runtime (the `ort` crate, "download-binaries" and
//! "copy-dylibs" features): the build downloads the matching `onnxruntime` shared library and
//! copies it beside the compiled binary, so it must ship together with `photocraft.exe`.
//!
//! Environment: `PHOTOCRAFT_SUBJECT_MODEL=off` disables the download and the model;
//! `PHOTOCRAFT_SUBJECT_MODEL_URL` (https only) and `PHOTOCRAFT_SUBJECT_MODEL_FILE` (a local
//! `.onnx` path, no download) choose another U²-Net-compatible model (input 320x320).

use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::Duration;

use ndarray::Array4;
use ort::{Environment, ExecutionProvider, GraphOptimizationLevel, Session, SessionBuilder, Value};
use photocraft_algo::segment::RgbImage;
use photocraft_algo::segment::subject::{SubjectMatter, set_matter};

/// Network input side (U²-Net family).
const SIZE: usize = 320;
const DEFAULT_URL: &str = "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2netp.onnx";
const FILE_NAME: &str = "u2netp.onnx";
/// Largest model file accepted from the network.
const MAX_BYTES: u64 = 256 * 1024 * 1024;
const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const STD: [f32; 3] = [0.229, 0.224, 0.225];

struct OnnxMatter {
    // Kept alongside the session only so the ONNX Runtime environment it was built from is never
    // dropped first; the session does not borrow it directly.
    _environment: Arc<Environment>,
    session: Mutex<Session>,
}

impl SubjectMatter for OnnxMatter {
    fn matte(&self, img: &RgbImage) -> Option<Vec<f32>> {
        if img.w < 4 || img.h < 4 || img.px.len() != img.w * img.h {
            return None;
        }
        let input = preprocess(img)?;
        let session = self.session.lock().unwrap_or_else(PoisonError::into_inner);
        let raw = infer(&session, input).ok()?;
        if raw.len() != SIZE * SIZE {
            return None;
        }
        let (lo, hi) = raw.iter().fold((f32::MAX, f32::MIN), |(a, b), v| (a.min(*v), b.max(*v)));
        let range = hi - lo;
        if !range.is_finite() || range < 1e-6 {
            return None;
        }
        let map: Vec<[f32; 1]> = raw.iter().map(|v| [(v - lo) / range]).collect();
        let mut out = Vec::with_capacity(img.w * img.h);
        for y in 0..img.h {
            for x in 0..img.w {
                let v = bilerp(&map, SIZE, SIZE, (x as f32 + 0.5) / img.w as f32, (y as f32 + 0.5) / img.h as f32);
                out.push(v[0].clamp(0.0, 1.0));
            }
        }
        Some(out)
    }
}

/// `img` stretched to `SIZE` x `SIZE`, scaled by its maximum and normalised (rembg's recipe),
/// as a 1x3xSIZExSIZE array.
fn preprocess(img: &RgbImage) -> Option<Array4<f32>> {
    let mut chw = vec![0.0f32; 3 * SIZE * SIZE];
    let mut max = 1e-6f32;
    let mut resized = Vec::with_capacity(SIZE * SIZE);
    for y in 0..SIZE {
        for x in 0..SIZE {
            let p = bilerp(&img.px, img.w, img.h, (x as f32 + 0.5) / SIZE as f32, (y as f32 + 0.5) / SIZE as f32);
            max = max.max(p[0]).max(p[1]).max(p[2]);
            resized.push(p);
        }
    }
    for (i, p) in resized.iter().enumerate() {
        for c in 0..3 {
            if let Some(slot) = chw.get_mut(c * SIZE * SIZE + i) {
                *slot = (p[c] / max - MEAN[c]) / STD[c];
            }
        }
    }
    Array4::from_shape_vec((1, 3, SIZE, SIZE), chw).ok()
}

/// Bilinear sample at normalised coordinates `u`, `v` (`0..=1`, pixel centres at half steps).
fn bilerp<const N: usize>(src: &[[f32; N]], w: usize, h: usize, u: f32, v: f32) -> [f32; N] {
    if w == 0 || h == 0 {
        return [0.0; N];
    }
    let fx = (u * w as f32 - 0.5).clamp(0.0, (w - 1) as f32);
    let fy = (v * h as f32 - 0.5).clamp(0.0, (h - 1) as f32);
    let (x0, y0) = (fx.floor() as usize, fy.floor() as usize);
    let (x1, y1) = ((x0 + 1).min(w - 1), (y0 + 1).min(h - 1));
    let (tx, ty) = (fx - x0 as f32, fy - y0 as f32);
    let at = |x: usize, y: usize| src.get(y * w + x).copied().unwrap_or([0.0; N]);
    let (a, b, c, d) = (at(x0, y0), at(x1, y0), at(x0, y1), at(x1, y1));
    std::array::from_fn(|k| (a[k] * (1.0 - tx) + b[k] * tx) * (1.0 - ty) + (c[k] * (1.0 - tx) + d[k] * tx) * ty)
}

/// Runs `session` on `input` and returns its first (fused) output, flattened.
fn infer(session: &Session, input: Array4<f32>) -> Result<Vec<f32>, String> {
    let array = input.into_dyn();
    let value = Value::from_array(session.allocator(), &array).map_err(|e| e.to_string())?;
    let outputs = session.run(vec![value]).map_err(|e| e.to_string())?;
    let first = outputs.first().ok_or_else(|| "the model returned no output".to_string())?;
    let tensor: ort::tensor::OrtOwnedTensor<f32, _> = first.try_extract().map_err(|e| e.to_string())?;
    Ok(tensor.view().iter().copied().collect())
}

fn load(path: &Path) -> Result<OnnxMatter, String> {
    let environment = Environment::builder()
        .with_name("photocraft-subject")
        .with_execution_providers([ExecutionProvider::CPU(Default::default())])
        .build()
        .map_err(|e| e.to_string())?
        .into_arc();
    let session = SessionBuilder::new(&environment)
        .map_err(|e| e.to_string())?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(|e| e.to_string())?
        .with_model_from_file(path)
        .map_err(|e| e.to_string())?;
    Ok(OnnxMatter { _environment: environment, session: Mutex::new(session) })
}

fn download(url: &str, dest: &Path) -> Result<(), String> {
    if !url.starts_with("https://") {
        return Err("the model URL must be https".into());
    }
    let response = ureq::get(url).timeout(Duration::from_secs(180)).call().map_err(|e| e.to_string())?;
    let mut bytes = Vec::new();
    response.into_reader().take(MAX_BYTES + 1).read_to_end(&mut bytes).map_err(|e| e.to_string())?;
    if bytes.len() as u64 > MAX_BYTES || bytes.len() < 1024 {
        return Err("the downloaded model has an unexpected size".into());
    }
    let part = dest.with_extension("part");
    std::fs::write(&part, &bytes).map_err(|e| e.to_string())?;
    std::fs::rename(&part, dest).map_err(|e| e.to_string())
}

fn model_path(config_dir: Option<PathBuf>) -> Option<PathBuf> {
    config_dir.map(|d| d.join("Models").join(FILE_NAME))
}

fn register(path: &Path) -> bool {
    match load(path) {
        Ok(m) => {
            set_matter(Some(Arc::new(m)));
            log::info!("Select Subject: learned model loaded from {path:?}");
            true
        }
        Err(e) => {
            log::warn!("Select Subject: cannot load the model {path:?}: {e}");
            false
        }
    }
}

/// Loads the model in the background (downloading it first when it is not on disk and the
/// network answers). Never blocks start-up and never fails the app.
pub fn start(config_dir: Option<PathBuf>) {
    if std::env::var("PHOTOCRAFT_SUBJECT_MODEL").is_ok_and(|v| v.eq_ignore_ascii_case("off") || v == "0") {
        return;
    }
    let local = std::env::var_os("PHOTOCRAFT_SUBJECT_MODEL_FILE").map(PathBuf::from);
    let url = std::env::var("PHOTOCRAFT_SUBJECT_MODEL_URL").ok().filter(|u| !u.trim().is_empty()).unwrap_or_else(|| DEFAULT_URL.to_string());
    let spawned = std::thread::Builder::new().name("subject-model".into()).spawn(move || {
        if let Some(file) = local {
            register(&file);
            return;
        }
        let Some(path) = model_path(config_dir) else { return };
        if !path.exists() {
            if let Some(parent) = path.parent()
                && let Err(e) = std::fs::create_dir_all(parent)
            {
                log::warn!("Select Subject: cannot create {parent:?}: {e}");
                return;
            }
            if let Err(e) = download(&url, &path) {
                // Offline (or blocked): the classical Select Subject stays in use; the next start tries again.
                log::info!("Select Subject: model not downloaded ({e}); using the classical method");
                return;
            }
        }
        if !register(&path) {
            // A damaged file (or one the runtime can't load) would fail on every start: remove it
            // so the next start downloads it again.
            let _ = std::fs::remove_file(&path);
        }
    });
    if let Err(e) = spawned {
        log::warn!("Select Subject: cannot start the model loader: {e}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bilerp_keeps_a_constant_and_interpolates_between_pixels() {
        let flat = vec![[0.5f32]; 4];
        assert!((bilerp(&flat, 2, 2, 0.3, 0.9)[0] - 0.5).abs() < 1e-6);
        let ramp = vec![[0.0f32], [1.0]];
        assert!((bilerp(&ramp, 2, 1, 0.5, 0.5)[0] - 0.5).abs() < 1e-6);
        assert_eq!(bilerp::<1>(&[], 0, 0, 0.5, 0.5), [0.0]);
    }

    #[test]
    fn preprocess_makes_a_square_input_and_rejects_nothing_valid() {
        let img = RgbImage::from_fn(10, 7, |x, y| [x as f32 / 10.0, y as f32 / 7.0, 0.5]);
        assert_eq!(preprocess(&img).map(|t| t.shape().to_vec()), Some(vec![1, 3, SIZE, SIZE]));
    }

    #[test]
    fn a_missing_model_or_a_non_https_url_is_an_error_not_a_panic() {
        assert!(load(Path::new("/nonexistent/model.onnx")).is_err());
        assert!(download("http://example.invalid/m.onnx", Path::new("/tmp/never")).is_err());
    }
}
