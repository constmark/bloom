//! Experimental MiniMax H3 file generation through an external sd.cpp runner.
//!
//! This is a dedicated CLI workflow, not a registered Bloom model backend.
//! The external runner owns CUDA and model loading. Success requires a decoded
//! video and stereo audio; ordinary engine tests do not establish model quality.

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};

use crate::core::process::run_bounded_output_cancellable;

const LOG_LIMIT: usize = 64 * 1024;
const PROBE_LIMIT: usize = 256 * 1024;

#[derive(Debug, Clone, Serialize)]
pub struct H3Request {
    pub sd_cli: PathBuf,
    pub diffusion_model: PathBuf,
    pub llm: PathBuf,
    pub vae: PathBuf,
    pub audio_vae: PathBuf,
    pub prompt: String,
    pub output: PathBuf,
    pub width: u32,
    pub height: u32,
    pub frames: u32,
    pub steps: u32,
    pub seed: u32,
    pub max_vram: f64,
    pub threads: u32,
    pub timeout_secs: u64,
    pub ffmpeg: PathBuf,
    pub ffprobe: PathBuf,
}

impl H3Request {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            !self.prompt.trim().is_empty()
                && self.prompt.len() <= 16 * 1024
                && !self.prompt.contains('\0'),
            "prompt must contain 1–16384 bytes and no NUL"
        );
        ensure!(
            self.width >= 32
                && self.height >= 32
                && self.width <= 1024
                && self.height <= 1024
                && self.width.is_multiple_of(32)
                && self.height.is_multiple_of(32),
            "H3 width and height must be multiples of 32 between 32 and 1024"
        );
        ensure!(
            (5..=243).contains(&self.frames) && (self.frames - 5).is_multiple_of(17),
            "H3 frame count must be 17k+5, between 5 and 243 (24 fps)"
        );
        ensure!(
            u64::from(self.width) * u64::from(self.height) * u64::from(self.frames)
                <= 64 * 1024 * 1024,
            "video exceeds the 64 megapixel output limit"
        );
        ensure!((1..=100).contains(&self.steps), "steps must be 1–100");
        ensure!((1..=64).contains(&self.threads), "threads must be 1–64");
        ensure!(
            self.max_vram.is_finite() && self.max_vram > 0.0 && self.max_vram <= 64.0,
            "max-vram must be greater than 0 and at most 64 GiB"
        );
        ensure!(
            (1..=86400).contains(&self.timeout_secs),
            "timeout must be 1–86400 seconds"
        );
        ensure!(
            self.output.extension().is_some_and(|ext| ext == "mp4"),
            "output must end in .mp4"
        );
        ensure!(
            !self.output.try_exists()?,
            "refusing to overwrite {}",
            self.output.display()
        );
        ensure!(
            !manifest_path(&self.output).try_exists()?,
            "output manifest already exists"
        );
        for (name, path) in [
            ("sd-cli", &self.sd_cli),
            ("diffusion model", &self.diffusion_model),
            ("text encoder", &self.llm),
            ("video VAE", &self.vae),
            ("audio VAE", &self.audio_vae),
        ] {
            let metadata = std::fs::metadata(path)
                .with_context(|| format!("missing {name}: {}", path.display()))?;
            ensure!(
                metadata.is_file() && metadata.len() > 0,
                "{name} must be a nonempty regular file: {}",
                path.display()
            );
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct H3MediaInfo {
    pub width: u32,
    pub height: u32,
    pub frames: u32,
    pub fps: u32,
    pub audio_channels: u32,
    pub audio_sample_rate: u32,
    pub duration_secs: f64,
}

#[derive(Debug, Serialize)]
pub struct H3Artifacts {
    pub schema_version: u32,
    pub runtime: &'static str,
    pub backend: &'static str,
    pub params_backend: &'static str,
    pub request: H3Request,
    pub media: H3MediaInfo,
    pub elapsed_secs: f64,
    pub output_bytes: u64,
    pub output_sha256: String,
    pub manifest: PathBuf,
    /// Bounded diagnostics; not a substitute for verifying model provenance.
    pub runtime_stdout: String,
    pub runtime_stderr: String,
    pub runtime_log_truncated: bool,
}

fn manifest_path(output: &Path) -> PathBuf {
    let mut path = output.as_os_str().to_os_string();
    path.push(".json");
    path.into()
}

fn generation_command(request: &H3Request, raw: &Path) -> Command {
    let mut command = Command::new(&request.sd_cli);
    command
        .args(["-M", "vid_gen", "--diffusion-model"])
        .arg(&request.diffusion_model)
        .arg("--llm")
        .arg(&request.llm)
        .arg("--vae")
        .arg(&request.vae)
        .arg("--audio-vae")
        .arg(&request.audio_vae)
        .arg("--prompt")
        .arg(&request.prompt)
        .arg("--output")
        .arg(raw)
        .arg("--width")
        .arg(request.width.to_string())
        .arg("--height")
        .arg(request.height.to_string())
        .arg("--video-frames")
        .arg(request.frames.to_string())
        .arg("--steps")
        .arg(request.steps.to_string())
        .arg("--seed")
        .arg(request.seed.to_string())
        .arg("--threads")
        .arg(request.threads.to_string())
        .arg("--max-vram")
        .arg(request.max_vram.to_string())
        .args([
            "--backend",
            "cuda0",
            "--params-backend",
            "disk",
            "--mmap",
            "--rng",
            "cpu",
            "--fps",
            "24",
            "--cfg-scale",
            "1",
            "--sampling-method",
            "euler",
            "--flash-attn",
            "on",
            "--vae-tiling",
            "--model-args",
            "minimax_h3_attn_chunk_budget=0.25",
        ]);
    command
}

fn run_stage(
    command: &mut Command,
    stage: &str,
    deadline: Instant,
    cancelled: &AtomicBool,
    stdout_limit: usize,
) -> Result<crate::core::process::BoundedProcessOutput> {
    let remaining = deadline
        .checked_duration_since(Instant::now())
        .context("H3 task deadline exceeded")?;
    crate::validate_runner(Path::new(command.get_program()))?;
    command
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    tracing::info!(stage, "MiniMax H3 task");
    let result = run_bounded_output_cancellable(
        command,
        &[],
        stdout_limit,
        LOG_LIMIT,
        remaining,
        Some(cancelled),
    )
    .with_context(|| stage.to_string())?;
    ensure!(
        result.status.success(),
        "{stage} failed ({}). stdout: {}\nstderr: {}",
        result.status,
        String::from_utf8_lossy(&result.stdout),
        String::from_utf8_lossy(&result.stderr)
    );
    Ok(result)
}

fn probe(
    request: &H3Request,
    path: &Path,
    deadline: Instant,
    cancelled: &AtomicBool,
) -> Result<serde_json::Value> {
    let mut command = Command::new(&request.ffprobe);
    command
        .args([
            "-v",
            "error",
            "-count_frames",
            "-show_streams",
            "-of",
            "json",
        ])
        .arg(path);
    let result = run_stage(
        &mut command,
        "probe media",
        deadline,
        cancelled,
        PROBE_LIMIT,
    )?;
    ensure!(!result.stdout_exceeded, "ffprobe response exceeds limit");
    serde_json::from_slice(&result.stdout).context("invalid ffprobe JSON")
}

fn validate_media(value: &serde_json::Value, request: &H3Request) -> Result<H3MediaInfo> {
    let streams = value["streams"]
        .as_array()
        .context("ffprobe streams missing")?;
    let videos: Vec<_> = streams
        .iter()
        .filter(|s| s["codec_type"] == "video")
        .collect();
    let audios: Vec<_> = streams
        .iter()
        .filter(|s| s["codec_type"] == "audio")
        .collect();
    ensure!(
        videos.len() == 1 && audios.len() == 1,
        "expected exactly one video and one audio stream"
    );
    let video = videos[0];
    let audio = audios[0];
    ensure!(
        video["width"].as_u64() == Some(u64::from(request.width))
            && video["height"].as_u64() == Some(u64::from(request.height)),
        "generated video dimensions differ from request"
    );
    let frames: u32 = video["nb_read_frames"]
        .as_str()
        .context("decoded frame count missing")?
        .parse()?;
    ensure!(
        frames == request.frames,
        "decoded frame count {frames} differs from requested {}",
        request.frames
    );
    let fps = video["avg_frame_rate"]
        .as_str()
        .context("frame rate missing")?;
    let (n, d) = fps.split_once('/').context("invalid frame rate")?;
    let n: u64 = n.parse()?;
    let d: u64 = d.parse()?;
    ensure!(
        d > 0 && d.checked_mul(24) == Some(n),
        "H3 must produce 24 fps"
    );
    let video_duration: f64 = video["duration"]
        .as_str()
        .context("video duration missing")?
        .parse()?;
    let audio_duration: f64 = audio["duration"]
        .as_str()
        .context("audio duration missing")?
        .parse()?;
    let expected = f64::from(frames) / 24.0;
    ensure!(
        video_duration.is_finite() && (video_duration - expected).abs() <= 1.0 / 24.0,
        "unexpected video duration"
    );
    ensure!(
        audio_duration.is_finite()
            && audio_duration > 0.0
            && (audio_duration - expected).abs() <= 0.25,
        "audio duration does not match video"
    );
    ensure!(
        audio["channels"].as_u64() == Some(2),
        "H3 stereo audio stream missing"
    );
    let sample_rate: u32 = audio["sample_rate"]
        .as_str()
        .context("audio sample rate missing")?
        .parse()?;
    ensure!(
        sample_rate == 32000,
        "expected H3 32 kHz audio, found {sample_rate}"
    );
    Ok(H3MediaInfo {
        width: request.width,
        height: request.height,
        frames,
        fps: 24,
        audio_channels: 2,
        audio_sample_rate: sample_rate,
        duration_secs: video_duration,
    })
}

/// Generate and validate an MP4. The total deadline includes conversion and
/// validation. No existing output is replaced, and failed temporary files are
/// removed. A memory cgroup may additionally constrain the external runtime;
/// max_vram is sd.cpp's managed allocation budget, not a hard process limit.
pub fn generate_h3(mut request: H3Request, cancelled: &AtomicBool) -> Result<H3Artifacts> {
    request.validate()?;
    ensure!(!cancelled.load(Ordering::Acquire), "H3 task cancelled");
    let started = Instant::now();
    let deadline = started + Duration::from_secs(request.timeout_secs);
    for path in [
        &mut request.sd_cli,
        &mut request.diffusion_model,
        &mut request.llm,
        &mut request.vae,
        &mut request.audio_vae,
    ] {
        *path = path.canonicalize()?;
    }
    let parent = request
        .output
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    std::fs::create_dir_all(parent)?;
    let parent = parent.canonicalize()?;
    request.output = parent.join(
        request
            .output
            .file_name()
            .context("output filename missing")?,
    );
    let work = tempfile::Builder::new()
        .prefix(".bloom-h3-")
        .tempdir_in(&parent)?;
    let raw = work.path().join("raw.avi");
    let mp4 = work.path().join("video.mp4");
    let runtime = run_stage(
        &mut generation_command(&request, &raw),
        "sd.cpp inference",
        deadline,
        cancelled,
        LOG_LIMIT,
    )?;
    let raw_metadata = std::fs::metadata(&raw).context("sd.cpp exited without raw.avi")?;
    ensure!(
        raw_metadata.is_file()
            && raw_metadata.len() > 0
            && raw_metadata.len() <= 1024 * 1024 * 1024,
        "sd.cpp output is missing or outside the 1 byte–1 GiB limit"
    );
    let raw_probe = probe(&request, &raw, deadline, cancelled)?;
    let streams = raw_probe["streams"]
        .as_array()
        .context("raw video streams missing")?;
    ensure!(
        streams.iter().any(|s| s["codec_type"] == "video"),
        "sd.cpp output has no video"
    );
    let embedded_audio = streams.iter().any(|s| s["codec_type"] == "audio");
    let wav = raw.with_extension("wav");
    if !embedded_audio {
        ensure!(
            wav.is_file(),
            "sd.cpp output has no audio stream or WAV sidecar"
        );
    }
    let mut convert = Command::new(&request.ffmpeg);
    convert
        .args(["-nostdin", "-v", "error", "-xerror", "-i"])
        .arg(&raw);
    if !embedded_audio {
        convert.arg("-i").arg(&wav);
    }
    convert
        .args([
            "-map",
            "0:v:0",
            "-map",
            if embedded_audio { "0:a:0" } else { "1:a:0" },
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-threads",
            "2",
            "-movflags",
            "+faststart",
            "-n",
        ])
        .arg(&mp4);
    run_stage(&mut convert, "encode MP4", deadline, cancelled, LOG_LIMIT)?;
    let media = validate_media(&probe(&request, &mp4, deadline, cancelled)?, &request)?;
    let mut decode = Command::new(&request.ffmpeg);
    decode
        .args(["-nostdin", "-v", "error", "-xerror", "-i"])
        .arg(&mp4)
        .args([
            "-map", "0:v:0", "-map", "0:a:0", "-threads", "2", "-f", "null", "-",
        ]);
    run_stage(
        &mut decode,
        "decode complete audio and video",
        deadline,
        cancelled,
        LOG_LIMIT,
    )?;
    let output_bytes = std::fs::metadata(&mp4)?.len();
    ensure!(
        output_bytes > 0 && output_bytes <= 1024 * 1024 * 1024,
        "MP4 size outside 1 byte–1 GiB limit"
    );
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    std::io::copy(&mut std::fs::File::open(&mp4)?, &mut hasher)?;
    let manifest = manifest_path(&request.output);
    let artifacts = H3Artifacts {
        schema_version: 1,
        runtime: "stable-diffusion.cpp",
        backend: "cuda0",
        params_backend: "disk",
        media,
        elapsed_secs: started.elapsed().as_secs_f64(),
        output_bytes,
        output_sha256: format!("{:x}", hasher.finalize()),
        manifest: manifest.clone(),
        request,
        runtime_stdout: String::from_utf8_lossy(&runtime.stdout).into_owned(),
        runtime_stderr: String::from_utf8_lossy(&runtime.stderr).into_owned(),
        runtime_log_truncated: runtime.stdout_exceeded || runtime.stderr_exceeded,
    };
    ensure!(
        !cancelled.load(Ordering::Acquire),
        "H3 task cancelled before publishing output"
    );
    ensure!(
        Instant::now() < deadline,
        "H3 task deadline exceeded before publishing output"
    );
    let metadata_file = work.path().join("artifact.json");
    std::fs::write(&metadata_file, serde_json::to_vec_pretty(&artifacts)?)?;
    // The temporary directory is on the destination filesystem. Hard linking
    // publishes atomically and fails if another writer already created a file.
    std::fs::hard_link(&mp4, &artifacts.request.output)
        .context("publish MP4 without overwriting")?;
    if let Err(error) = std::fs::hard_link(&metadata_file, &manifest) {
        // This path was created above, never an output belonging to a caller.
        let _ = std::fs::remove_file(&artifacts.request.output);
        return Err(error).context("publish H3 manifest without overwriting");
    }
    Ok(artifacts)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request(dir: &Path) -> H3Request {
        let component = dir.join("component");
        std::fs::write(&component, b"fixture").unwrap();
        H3Request {
            sd_cli: component.clone(),
            diffusion_model: component.clone(),
            llm: component.clone(),
            vae: component.clone(),
            audio_vae: component,
            prompt: "a cat; $(touch must-not-exist)".into(),
            output: dir.join("result.mp4"),
            width: 320,
            height: 192,
            frames: 22,
            steps: 20,
            seed: 42,
            max_vram: 4.0,
            threads: 2,
            timeout_secs: 60,
            ffmpeg: "ffmpeg".into(),
            ffprobe: "ffprobe".into(),
        }
    }

    #[test]
    fn rejects_misaligned_shapes_missing_components_and_existing_output() {
        let dir = tempfile::tempdir().unwrap();
        let mut request = request(dir.path());
        assert!(request.validate().is_ok());
        request.frames = 24;
        assert!(request.validate().is_err());
        request.frames = 22;
        request.width = 321;
        assert!(request.validate().is_err());
        request.width = 320;
        std::fs::write(&request.output, b"existing").unwrap();
        assert!(request.validate().is_err());
        std::fs::remove_file(&request.output).unwrap();
        std::fs::remove_file(&request.audio_vae).unwrap();
        assert!(request.validate().is_err());
    }

    #[test]
    fn command_preserves_prompt_as_one_argument_and_disk_residency() {
        let dir = tempfile::tempdir().unwrap();
        let request = request(dir.path());
        let command = generation_command(&request, &dir.path().join("raw.avi"));
        let args: Vec<_> = command
            .get_args()
            .map(|x| x.to_string_lossy().into_owned())
            .collect();
        assert!(
            args.windows(2)
                .any(|a| a == ["--prompt", request.prompt.as_str()])
        );
        assert!(args.windows(2).any(|a| a == ["--params-backend", "disk"]));
        assert!(!args.iter().any(|a| a == "--offload-to-cpu"));
    }

    #[test]
    fn requires_complete_matching_stereo_video() {
        let dir = tempfile::tempdir().unwrap();
        let request = request(dir.path());
        let mut probe = serde_json::json!({"streams": [
            {"codec_type":"video","width":320,"height":192,"nb_read_frames":"22","avg_frame_rate":"24/1","duration":"0.916667"},
            {"codec_type":"audio","channels":2,"sample_rate":"32000","duration":"0.916667"}
        ]});
        assert!(validate_media(&probe, &request).is_ok());
        probe["streams"][0]["nb_read_frames"] = "21".into();
        assert!(validate_media(&probe, &request).is_err());
        probe["streams"][0]["nb_read_frames"] = "22".into();
        probe["streams"][1]["channels"] = 1.into();
        assert!(validate_media(&probe, &request).is_err());
        probe["streams"].as_array_mut().unwrap().pop();
        assert!(validate_media(&probe, &request).is_err());
    }

    #[cfg(unix)]
    #[test]
    fn successful_worker_without_video_is_rejected() {
        use std::os::unix::fs::PermissionsExt;
        let _lock = crate::test_env::ENV_LOCK.lock().unwrap();
        let _strict = crate::test_env::EnvVarGuard::set("BLOOM_STRICT_SECURITY", "0");
        let dir = tempfile::tempdir().unwrap();
        let request = request(dir.path());
        std::fs::write(&request.sd_cli, b"#!/bin/sh\nexit 0\n").unwrap();
        std::fs::set_permissions(&request.sd_cli, std::fs::Permissions::from_mode(0o700)).unwrap();
        let output = request.output.clone();
        let error = generate_h3(request, &AtomicBool::new(false)).unwrap_err();
        assert!(error.to_string().contains("without raw.avi"));
        assert!(!output.exists());
        assert!(!manifest_path(&output).exists());
        assert!(!dir.path().read_dir().unwrap().any(|entry| {
            entry
                .unwrap()
                .file_name()
                .to_string_lossy()
                .starts_with(".bloom-h3-")
        }));
    }

    /// This validates the media adapter with a synthetic clip, NOT H3 inference.
    #[cfg(unix)]
    #[test]
    #[ignore = "requires installed ffmpeg (libx264) and ffprobe; synthetic media only"]
    fn synthetic_audio_video_survives_conversion_and_validation() {
        use std::os::unix::fs::PermissionsExt;
        let _lock = crate::test_env::ENV_LOCK.lock().unwrap();
        let _strict = crate::test_env::EnvVarGuard::set("BLOOM_STRICT_SECURITY", "0");
        let dir = tempfile::tempdir().unwrap();
        let request = request(dir.path());
        std::fs::write(&request.sd_cli, br##"#!/bin/sh
while [ "$#" -gt 0 ]; do
    if [ "$1" = "--output" ]; then shift; output="$1"; fi
    shift
done
exec ffmpeg -nostdin -v error -f lavfi -i 'color=c=blue:s=320x192:r=24:d=0.916667' -f lavfi -i 'sine=frequency=440:sample_rate=32000:duration=0.916667' -map 0:v -map 1:a -frames:v 22 -ac 2 -c:v mjpeg -c:a pcm_s16le -threads 1 -n "$output"
"##).unwrap();
        std::fs::set_permissions(&request.sd_cli, std::fs::Permissions::from_mode(0o700)).unwrap();
        let artifact = generate_h3(request, &AtomicBool::new(false)).unwrap();
        assert_eq!(artifact.media.frames, 22);
        assert_eq!(artifact.media.audio_channels, 2);
        assert!(artifact.request.output.is_file());
        assert!(artifact.manifest.is_file());
        assert_eq!(artifact.output_sha256.len(), 64);
    }
}
