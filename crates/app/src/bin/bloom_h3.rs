//! Dedicated experimental H3 file-generation command. CUDA is provided by the
//! external sd.cpp executable, so this binary needs no Bloom CUDA build feature.

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::{Context, Result};
use bloomai_engine::executor::minimax_h3::{H3Request, generate_h3};
use clap::Parser;

#[derive(Debug, Parser)]
#[command(
    name = "bloom_h3",
    about = "Experimental MiniMax H3 audio/video generation through an external CUDA sd.cpp runner"
)]
struct Args {
    /// Explicit H3-capable CUDA sd-cli executable.
    #[arg(long)]
    sd_cli: PathBuf,
    #[arg(long)]
    diffusion_model: PathBuf,
    /// MiniMax H3 variant of the Qwen3-VL text encoder.
    #[arg(long)]
    llm: PathBuf,
    #[arg(long)]
    vae: PathBuf,
    #[arg(long)]
    audio_vae: PathBuf,
    #[arg(long)]
    prompt: String,
    /// New MP4 filename. Also writes FILE.mp4.json; refuses to overwrite either.
    #[arg(long)]
    output: PathBuf,
    #[arg(long, default_value_t = 320)]
    width: u32,
    #[arg(long, default_value_t = 192)]
    height: u32,
    /// Must be 17k+5. The model always runs at 24 fps.
    #[arg(long, default_value_t = 22)]
    frames: u32,
    #[arg(long, default_value_t = 20)]
    steps: u32,
    #[arg(long, default_value_t = 42)]
    seed: u32,
    /// sd.cpp managed GPU budget in GiB, not a hard process memory limit.
    #[arg(long, default_value_t = 4.0)]
    max_vram: f64,
    #[arg(long, default_value_t = 2)]
    threads: u32,
    /// Total deadline, including conversion and validation.
    #[arg(long, default_value_t = 7200)]
    timeout_secs: u64,
    #[arg(long, default_value = "ffmpeg")]
    ffmpeg: PathBuf,
    #[arg(long, default_value = "ffprobe")]
    ffprobe: PathBuf,
}

impl From<Args> for H3Request {
    fn from(args: Args) -> Self {
        Self {
            sd_cli: args.sd_cli,
            diffusion_model: args.diffusion_model,
            llm: args.llm,
            vae: args.vae,
            audio_vae: args.audio_vae,
            prompt: args.prompt,
            output: args.output,
            width: args.width,
            height: args.height,
            frames: args.frames,
            steps: args.steps,
            seed: args.seed,
            max_vram: args.max_vram,
            threads: args.threads,
            timeout_secs: args.timeout_secs,
            ffmpeg: args.ffmpeg,
            ffprobe: args.ffprobe,
        }
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .with_writer(std::io::stderr)
        .init();
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?
        .block_on(run(args.into()))
}

async fn run(request: H3Request) -> Result<()> {
    request.validate()?;
    // Install handlers before starting a worker in its own process group.
    // The default SIGINT action would skip Drop and orphan that worker.
    let mut signals = ShutdownSignals::install()?;
    let cancelled = Arc::new(AtomicBool::new(false));
    let worker_cancelled = Arc::clone(&cancelled);
    let mut task = tokio::task::spawn_blocking(move || generate_h3(request, &worker_cancelled));
    let result = tokio::select! {
        result = &mut task => result,
        signal = signals.wait() => {
            cancelled.store(true, Ordering::Release);
            eprintln!("Stopping H3 worker and releasing its GPU resources...");
            let result = task.await;
            signal?;
            result
        }
    }
    .context("H3 generation task panicked")??;
    println!("{}", serde_json::to_string_pretty(&result)?);
    Ok(())
}

#[cfg(unix)]
struct ShutdownSignals {
    interrupt: tokio::signal::unix::Signal,
    terminate: tokio::signal::unix::Signal,
    hangup: tokio::signal::unix::Signal,
}

#[cfg(unix)]
impl ShutdownSignals {
    fn install() -> Result<Self> {
        use tokio::signal::unix::{SignalKind, signal};
        Ok(Self {
            interrupt: signal(SignalKind::interrupt())?,
            terminate: signal(SignalKind::terminate())?,
            hangup: signal(SignalKind::hangup())?,
        })
    }

    async fn wait(&mut self) -> Result<()> {
        tokio::select! {
            signal = self.interrupt.recv() => signal,
            signal = self.terminate.recv() => signal,
            signal = self.hangup.recv() => signal,
        }
        .context("H3 signal listener stopped")
    }
}

#[cfg(windows)]
struct ShutdownSignals {
    interrupt: tokio::signal::windows::CtrlC,
    break_signal: tokio::signal::windows::CtrlBreak,
}

#[cfg(windows)]
impl ShutdownSignals {
    fn install() -> Result<Self> {
        Ok(Self {
            interrupt: tokio::signal::windows::ctrl_c()?,
            break_signal: tokio::signal::windows::ctrl_break()?,
        })
    }

    async fn wait(&mut self) -> Result<()> {
        tokio::select! {
            signal = self.interrupt.recv() => signal,
            signal = self.break_signal.recv() => signal,
        }
        .context("H3 signal listener stopped")
    }
}

#[cfg(not(any(unix, windows)))]
struct ShutdownSignals;

#[cfg(not(any(unix, windows)))]
impl ShutdownSignals {
    fn install() -> Result<Self> {
        anyhow::bail!("H3 CLI cancellation requires a Unix or Windows signal driver")
    }

    async fn wait(&mut self) -> Result<()> {
        std::future::pending().await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cli_requires_all_components_and_uses_low_vram_defaults() {
        assert!(Args::try_parse_from(["bloom_h3", "--prompt", "cat"]).is_err());
        let args = Args::try_parse_from([
            "bloom_h3",
            "--sd-cli",
            "/runtime/sd-cli",
            "--diffusion-model",
            "dit.gguf",
            "--llm",
            "te.gguf",
            "--vae",
            "video.safetensors",
            "--audio-vae",
            "audio.safetensors",
            "--prompt",
            "A cat plays a piano.",
            "--output",
            "cat.mp4",
        ])
        .unwrap();
        let request: H3Request = args.into();
        assert_eq!(
            (request.width, request.height, request.frames),
            (320, 192, 22)
        );
        assert_eq!(request.steps, 20);
        assert_eq!(request.max_vram, 4.0);
    }
}
