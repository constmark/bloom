//! Ownership and decoding at external inference process boundaries.
use std::io::{ErrorKind, Read, Write};
use std::ops::{Deref, DerefMut};
use std::process::{Child, Command, ExitStatus, Stdio};
use std::thread;
use std::time::{Duration, Instant};

use crate::{OutputChunk, model::OutputSink};
use anyhow::{Context, Result, anyhow, bail, ensure};

/// Every early return kills and reaps the process before releasing its runtime.
pub(crate) struct ChildGuard {
    child: Child,
    reaped: bool,
    #[cfg(unix)]
    process_group: Option<libc::pid_t>,
}

/// Write a request to a spawned inference process while preserving ownership
/// on error. A broken pipe (or a missing stdin handle) means the worker has
/// already failed; explicitly kill and reap it before returning so callers do
/// not leave a zombie behind on an early I/O error.
#[cfg(all(test, unix))]
pub(crate) fn write_stdin(child: &mut Child, request: &[u8]) -> Result<()> {
    let Some(mut stdin) = child.stdin.take() else {
        let _ = child.kill();
        let _ = child.wait();
        bail!("inference process has no stdin");
    };
    if let Err(error) = stdin.write_all(request) {
        let _ = child.kill();
        let _ = child.wait();
        return Err(error).context("failed to write inference request");
    }
    Ok(())
}

impl ChildGuard {
    pub(crate) fn spawn(command: &mut Command) -> Result<Self> {
        Ok(Self {
            child: command
                .spawn()
                .context("failed to spawn inference process")?,
            reaped: false,
            #[cfg(unix)]
            process_group: None,
        })
    }

    pub(crate) fn spawn_bounded(command: &mut Command) -> Result<Self> {
        #[cfg(unix)]
        {
            use std::os::unix::process::CommandExt;
            command.process_group(0);
        }
        let child = Self::spawn(command)?;
        #[cfg(unix)]
        {
            let mut child = child;
            child.process_group = Some(child.id() as libc::pid_t);
            Ok(child)
        }
        #[cfg(not(unix))]
        {
            Ok(child)
        }
    }

    pub(crate) fn kill(&mut self) -> std::io::Result<()> {
        #[cfg(unix)]
        if let Some(group) = self.process_group {
            // SAFETY: spawn_bounded created a separate process group whose
            // positive ID is the child PID. A negative PID targets only that
            // group, including descendants holding inherited pipe handles.
            if unsafe { libc::kill(-group, libc::SIGKILL) } == 0 {
                return Ok(());
            }
            let error = std::io::Error::last_os_error();
            if error.raw_os_error() != Some(libc::ESRCH) {
                return Err(error);
            }
        }
        self.child.kill()
    }

    pub(crate) fn wait(&mut self) -> std::io::Result<ExitStatus> {
        let status = self.child.wait()?;
        self.reaped = true;
        Ok(status)
    }

    /// Poll the child without blocking, recording ownership when the process
    /// has already been reaped by the platform.  Callers that use bounded
    /// subprocess runners need this instead of calling `Child::try_wait`
    /// through the `Deref` implementation; otherwise `Drop` would attempt a
    /// second wait after a successful poll.
    pub(crate) fn try_wait(&mut self) -> std::io::Result<Option<ExitStatus>> {
        let status = self.child.try_wait()?;
        if status.is_some() {
            self.reaped = true;
        }
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
            let _ = self.kill();
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

/// Drain a child-process pipe while retaining only a bounded prefix.
///
/// The pipe is always drained to EOF so a noisy worker cannot block on a full
/// stderr/stdout pipe, but callers never have to retain unbounded diagnostics
/// in memory. The boolean reports whether bytes beyond `limit` were observed.
pub(crate) fn read_bounded(
    mut reader: impl Read,
    limit: usize,
) -> std::io::Result<(Vec<u8>, bool)> {
    let mut retained = Vec::with_capacity(limit.min(8192));
    let mut buffer = [0u8; 8192];
    let mut exceeded = false;
    loop {
        let read = reader.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        if retained.len() < limit {
            let keep = (limit - retained.len()).min(read);
            retained.extend_from_slice(&buffer[..keep]);
            exceeded |= keep < read;
        } else {
            exceeded = true;
        }
    }
    Ok((retained, exceeded))
}

/// The retained result of a child process whose pipes were drained to EOF.
///
/// The limits apply only to the retained prefixes; reader threads continue to
/// drain both pipes so a noisy worker cannot block on a full stdout/stderr
/// pipe while the parent waits for it to exit.
#[derive(Debug)]
pub(crate) struct BoundedProcessOutput {
    pub(crate) status: ExitStatus,
    pub(crate) stdout: Vec<u8>,
    pub(crate) stderr: Vec<u8>,
    pub(crate) stdout_exceeded: bool,
    pub(crate) stderr_exceeded: bool,
}

/// Run a short-lived external process with bounded output and a hard deadline.
///
/// If `request` is empty, stdin is closed before waiting. Otherwise it is
/// written on a worker thread so a child that stops consuming input remains
/// interruptible by `timeout`. The process is killed and reaped on timeout or
/// polling failure, and all pipe readers are joined before returning. On Unix,
/// the bounded worker owns a process group so descendants cannot keep its
/// inherited pipes open after timeout. Pipe draining shares the same deadline.
pub(crate) fn run_bounded_output(
    command: &mut Command,
    request: &[u8],
    stdout_limit: usize,
    stderr_limit: usize,
    timeout: Duration,
) -> Result<BoundedProcessOutput> {
    let mut child = ChildGuard::spawn_bounded(command)?;
    let stdout = child
        .stdout
        .take()
        .context("inference process has no stdout")?;
    let stderr = child
        .stderr
        .take()
        .context("inference process has no stderr")?;
    let stdin = child.stdin.take();
    if !request.is_empty() && stdin.is_none() {
        bail!("inference process has no stdin");
    }
    let deadline = Instant::now()
        .checked_add(timeout)
        .ok_or_else(|| anyhow!("inference process timeout is too large"))?;

    let stdout_reader = thread::spawn(move || read_bounded(stdout, stdout_limit));
    let stderr_reader = thread::spawn(move || read_bounded(stderr, stderr_limit));
    let stdin_writer = if request.is_empty() {
        drop(stdin);
        None
    } else {
        let mut stdin = stdin.expect("checked non-empty request has stdin");
        let request = request.to_vec();
        Some(thread::spawn(move || stdin.write_all(&request)))
    };

    let mut exit_status = None;
    let status = loop {
        let polled = match exit_status {
            Some(status) => Ok(Some(status)),
            None => child.try_wait(),
        };
        match polled {
            Ok(status) => {
                exit_status = status;
                if let Some(status) = status
                    && stdout_reader.is_finished()
                    && stderr_reader.is_finished()
                    && stdin_writer
                        .as_ref()
                        .is_none_or(|writer| writer.is_finished())
                {
                    break status;
                }
                if Instant::now() < deadline {
                    thread::sleep(Duration::from_millis(10));
                    continue;
                }
                let _ = child.kill();
                let _ = child.wait();
                discard_process_threads(stdout_reader, stderr_reader, stdin_writer);
                return Err(anyhow!(
                    "inference process timed out after {} seconds",
                    timeout.as_secs()
                ));
            }
            Err(error) => {
                let _ = child.kill();
                let _ = child.wait();
                discard_process_threads(stdout_reader, stderr_reader, stdin_writer);
                return Err(error).context("failed to poll inference process");
            }
        }
    };

    let stdout_result = join_process_reader(stdout_reader, "stdout");
    let stderr_result = join_process_reader(stderr_reader, "stderr");
    let stdin_result = stdin_writer.map(|writer| {
        writer
            .join()
            .map_err(|_| anyhow!("inference process stdin writer panicked"))
    });
    let (stdout, stdout_exceeded) = stdout_result?;
    let (stderr, stderr_exceeded) = stderr_result?;
    if let Some(writer) = stdin_result {
        writer??;
    }

    Ok(BoundedProcessOutput {
        status,
        stdout,
        stderr,
        stdout_exceeded,
        stderr_exceeded,
    })
}

type ProcessReader = thread::JoinHandle<std::io::Result<(Vec<u8>, bool)>>;
type ProcessWriter = thread::JoinHandle<std::io::Result<()>>;

fn join_process_reader(reader: ProcessReader, stream: &str) -> Result<(Vec<u8>, bool)> {
    reader
        .join()
        .map_err(|_| anyhow!("inference process {stream} reader panicked"))?
        .with_context(|| format!("failed to read inference process {stream}"))
}

fn discard_process_threads(
    stdout_reader: ProcessReader,
    stderr_reader: ProcessReader,
    stdin_writer: Option<ProcessWriter>,
) {
    let _ = stdout_reader.join();
    let _ = stderr_reader.join();
    if let Some(writer) = stdin_writer {
        let _ = writer.join();
    }
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

    #[test]
    fn bounded_reader_drains_input_and_retains_only_the_prefix() {
        let (bytes, exceeded) = read_bounded("abcdef".as_bytes(), 3).unwrap();
        assert_eq!(bytes, b"abc");
        assert!(exceeded);

        let (bytes, exceeded) = read_bounded("abc".as_bytes(), 3).unwrap();
        assert_eq!(bytes, b"abc");
        assert!(!exceeded);
    }

    #[cfg(unix)]
    #[test]
    fn bounded_timeout_closes_descendant_pipes_even_after_parent_exit() {
        for script in ["sleep 30 <&0 & wait", "sleep 30 <&0 & exit 0"] {
            let mut command = Command::new("sh");
            command
                .args(["-c", script])
                .stdin(Stdio::piped())
                .stdout(Stdio::piped())
                .stderr(Stdio::piped());
            let started = Instant::now();
            // The unread request also exercises cancellation of a blocked
            // stdin writer while descendants retain all three pipe handles.
            let error = run_bounded_output(
                &mut command,
                &vec![b'x'; 1024 * 1024],
                16,
                16,
                Duration::from_secs(1),
            )
            .unwrap_err();
            assert!(error.to_string().contains("timed out after 1 seconds"));
            assert!(started.elapsed() < Duration::from_secs(5));
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

    #[cfg(unix)]
    #[test]
    fn missing_stdin_kills_and_reaps_the_worker() {
        let mut command = Command::new("sleep");
        command.arg("30");
        let mut child = command.spawn().unwrap();
        let pid = child.id() as libc::pid_t;

        assert!(write_stdin(&mut child, b"request").is_err());
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
