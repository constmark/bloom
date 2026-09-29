//! Ownership and decoding at external inference process boundaries.
use std::io::{ErrorKind, Read};
use std::ops::{Deref, DerefMut};
use std::process::{Child, Command, ExitStatus, Stdio};

use crate::{OutputChunk, model::OutputSink};
use anyhow::{Context, Result, bail, ensure};

/// Every early return kills and reaps the process before releasing its runtime.
pub(crate) struct ChildGuard {
    child: Child,
    reaped: bool,
}

impl ChildGuard {
    pub(crate) fn spawn(command: &mut Command) -> Result<Self> {
        Ok(Self {
            child: command
                .spawn()
                .context("failed to spawn inference process")?,
            reaped: false,
        })
    }

    pub(crate) fn wait(&mut self) -> std::io::Result<ExitStatus> {
        let status = self.child.wait()?;
        self.reaped = true;
        Ok(status)
    }
}

impl Deref for ChildGuard {
    type Target = Child;
    fn deref(&self) -> &Child {
        &self.child
    }
}
impl DerefMut for ChildGuard {
    fn deref_mut(&mut self) -> &mut Child {
        &mut self.child
    }
}
impl Drop for ChildGuard {
    fn drop(&mut self) {
        if !self.reaped {
            let _ = self.child.kill();
            let _ = self.child.wait();
        }
    }
}

pub(crate) fn stream_text_process(command: &mut Command, sink: &mut dyn OutputSink) -> Result<()> {
    command.stdout(Stdio::piped()).stderr(Stdio::inherit());
    let mut child = ChildGuard::spawn(command)?;
    let mut stdout = child
        .stdout
        .take()
        .context("inference process has no stdout")?;
    stream_utf8(&mut stdout, sink)?;
    let status = child
        .wait()
        .context("failed to wait for inference process")?;
    ensure!(status.success(), "inference process exited with {status}");
    sink.on_chunk(OutputChunk::End)
}

fn stream_utf8(reader: &mut impl Read, sink: &mut dyn OutputSink) -> Result<()> {
    let mut pending = Vec::with_capacity(4);
    let mut byte = [0u8; 1];
    loop {
        match reader.read(&mut byte) {
            Ok(0) => {
                ensure!(
                    pending.is_empty(),
                    "inference process ended with truncated UTF-8"
                );
                return Ok(());
            }
            Ok(_) => pending.push(byte[0]),
            Err(error) if error.kind() == ErrorKind::Interrupted => continue,
            Err(error) => return Err(error).context("failed to read inference output"),
        }
        match std::str::from_utf8(&pending) {
            Ok(text) => {
                sink.on_chunk(OutputChunk::TextDelta(text.to_string()))?;
                pending.clear();
            }
            Err(error) if error.error_len().is_none() && pending.len() < 4 => {}
            Err(error) => bail!("inference process emitted invalid UTF-8: {error}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn utf8_accepts_multibyte_and_rejects_malformed_and_truncated_output() {
        let mut text = String::new();
        stream_utf8(&mut "你好 🌱".as_bytes(), &mut |chunk| {
            if let OutputChunk::TextDelta(delta) = chunk {
                text.push_str(&delta);
            }
            Ok(())
        })
        .unwrap();
        assert_eq!(text, "你好 🌱");
        for bytes in [&[0xff][..], &[0xe4, 0xbd], &[0xe4, 0xff]] {
            assert!(stream_utf8(&mut &bytes[..], &mut |_| Ok(())).is_err());
        }
    }

    #[cfg(unix)]
    #[test]
    fn failed_process_does_not_emit_success_end() {
        let mut command = Command::new("sh");
        command.args(["-c", "printf hello; exit 7"]);
        let mut ended = false;
        assert!(
            stream_text_process(&mut command, &mut |chunk| {
                ended |= matches!(chunk, OutputChunk::End);
                Ok(())
            })
            .is_err()
        );
        assert!(!ended);
    }

    #[cfg(unix)]
    #[test]
    fn rejecting_output_kills_and_reaps_the_worker() {
        let dir = tempfile::tempdir().unwrap();
        let pid_path = dir.path().join("pid");
        let mut command = Command::new("sh");
        command
            .args([
                "-c",
                "echo $$ > \"$1\"; printf hello; exec sleep 30",
                "test",
            ])
            .arg(&pid_path);
        let start = std::time::Instant::now();
        assert!(stream_text_process(&mut command, &mut |_| anyhow::bail!("cancelled")).is_err());
        assert!(start.elapsed() < std::time::Duration::from_secs(5));
        let pid: libc::pid_t = std::fs::read_to_string(pid_path)
            .unwrap()
            .trim()
            .parse()
            .unwrap();
        // SAFETY: waitpid receives a valid PID and no output pointer; the child
        // must already have been reaped by the guard.
        assert_eq!(
            unsafe { libc::waitpid(pid, std::ptr::null_mut(), libc::WNOHANG) },
            -1
        );
        assert_eq!(
            std::io::Error::last_os_error().raw_os_error(),
            Some(libc::ECHILD)
        );
    }
}
