#![allow(clippy::type_complexity)]
#![cfg_attr(not(test), warn(clippy::unwrap_used))]
//! Cross-platform local desktop launcher.
//!
//! `bloom_desktop` owns the lifetime of a sibling `bloom_server` process,
//! waits for its HTTP listener to become reachable, and opens the embedded UI
//! using the host operating system's browser launcher.  The server arguments
//! are passed after `--`, which keeps this launcher independent from the
//! server's rapidly growing configuration surface:
//!
//! ```text
//! bloom_desktop -- --models-dir ~/.bloom/models --device auto
//! ```

use std::ffi::OsString;
use std::io::{self, Read};
use std::net::{TcpStream, ToSocketAddrs};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitStatus, Stdio};
use std::sync::mpsc::{self, Receiver, TryRecvError};
use std::thread;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow, bail};
use clap::Parser;

const DEFAULT_HOST: &str = "127.0.0.1";
const DEFAULT_PORT: u16 = 3000;
const DEFAULT_WAIT_SECONDS: u64 = 30;
const POLL_INTERVAL: Duration = Duration::from_millis(100);
const SHUTDOWN_GRACE: Duration = Duration::from_secs(35);

#[derive(Debug, Parser)]
#[command(
    name = "bloom_desktop",
    version,
    about = "Launch a local Bloom server and open its graphical UI"
)]
struct Args {
    /// Path to the bloom_server executable. By default, look beside this executable.
    #[arg(long, env = "BLOOM_SERVER_BIN")]
    server_bin: Option<PathBuf>,

    /// Do not open a browser after the server is reachable.
    #[arg(long, default_value_t = false)]
    no_browser: bool,

    /// Maximum time to wait for the server listener, in seconds.
    #[arg(long, default_value_t = DEFAULT_WAIT_SECONDS)]
    wait_seconds: u64,

    /// Server arguments. Put `--` before the first server option.
    #[arg(trailing_var_arg = true)]
    server_args: Vec<OsString>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ServerTarget {
    host: String,
    port: u16,
}

#[derive(Debug)]
enum SignalEvent {
    Shutdown,
    ListenerError(String),
}

fn main() -> Result<()> {
    let args = Args::parse();
    run(args)
}

fn run(args: Args) -> Result<()> {
    let target = server_target(&args.server_args)?;
    let server_bin = args
        .server_bin
        .or_else(default_server_binary)
        .ok_or_else(|| {
            anyhow!(
                "could not find bloom_server beside bloom_desktop; pass --server-bin /path/to/bloom_server"
            )
        })?;
    if !server_bin.is_file() {
        bail!(
            "bloom_server executable does not exist: {}",
            server_bin.display()
        );
    }

    let mut child = spawn_server(&server_bin, &args.server_args)?;
    let signals = start_signal_listener()?;
    match wait_for_server(
        &mut child,
        &target,
        Duration::from_secs(args.wait_seconds),
        &signals,
    ) {
        Ok(true) => {}
        Ok(false) => {
            let status = wait_for_child_after_shutdown(&mut child, &signals)?;
            return report_exit(status);
        }
        Err(error) => {
            terminate_child(&mut child);
            return Err(error);
        }
    }

    let url = format!("http://{}:{}/", display_host(&target.host), target.port);
    if !args.no_browser
        && let Err(error) = open_browser(&url)
    {
        eprintln!("warning: could not open {url}: {error}; open it manually");
    }
    eprintln!("Bloom UI is available at {url}; press Ctrl-C to stop the server.");

    let status = wait_for_child(&mut child, &signals)?;
    report_exit(status)
}

fn spawn_server(server_bin: &Path, server_args: &[OsString]) -> Result<Child> {
    Command::new(server_bin)
        .args(server_args)
        .stdin(Stdio::inherit())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .spawn()
        .with_context(|| format!("failed to start bloom_server at {}", server_bin.display()))
}

/// Wait for the server to publish a usable HTTP listener.
///
/// The boolean result is `false` when a process signal requested shutdown
/// before readiness. Keeping this loop synchronous lets the parent reap the
/// exact child process even when the launcher itself receives SIGTERM.
fn wait_for_server(
    child: &mut Child,
    target: &ServerTarget,
    timeout: Duration,
    signals: &Receiver<SignalEvent>,
) -> Result<bool> {
    let deadline = Instant::now()
        .checked_add(timeout)
        .ok_or_else(|| anyhow!("server readiness timeout is too large"))?;
    loop {
        if let Some(event) = receive_signal(signals)?
            && matches!(event, SignalEvent::Shutdown)
        {
            request_graceful_shutdown(child)?;
            return Ok(false);
        }
        if let Some(status) = child
            .try_wait()
            .context("failed to inspect bloom_server while waiting for readiness")?
        {
            bail!("bloom_server exited before becoming ready ({status})");
        }
        if readiness_probe(target).is_ok() {
            return Ok(true);
        }
        if Instant::now() >= deadline {
            bail!(
                "timed out after {}s waiting for Bloom at {}:{}",
                timeout.as_secs(),
                target.host,
                target.port
            );
        }
        thread::sleep(POLL_INTERVAL);
    }
}

fn readiness_probe(target: &ServerTarget) -> io::Result<()> {
    let probe_host = local_probe_host(&target.host);
    let host_header = display_host(&probe_host);
    let addrs = (probe_host.as_str(), target.port)
        .to_socket_addrs()?
        .collect::<Vec<_>>();
    if addrs.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::AddrNotAvailable,
            "no server address",
        ));
    }
    let mut last_error = None;
    for addr in addrs {
        let mut stream = match TcpStream::connect_timeout(&addr, Duration::from_millis(500)) {
            Ok(stream) => stream,
            Err(error) => {
                last_error = Some(error);
                continue;
            }
        };
        stream.set_read_timeout(Some(Duration::from_millis(500)))?;
        stream.set_write_timeout(Some(Duration::from_millis(500)))?;
        use std::io::Write;
        write!(
            stream,
            "GET /ready HTTP/1.1\r\nHost: {}\r\nConnection: close\r\n\r\n",
            host_header
        )?;
        let mut response = [0_u8; 256];
        let count = stream.read(&mut response)?;
        let line = std::str::from_utf8(&response[..count]).unwrap_or_default();
        let status = line
            .strip_prefix("HTTP/")
            .and_then(|rest| rest.split_whitespace().nth(1))
            .and_then(|value| value.parse::<u16>().ok());
        return match status {
            Some(200 | 503) => Ok(()),
            Some(code) => Err(io::Error::other(format!(
                "Bloom readiness returned HTTP {code}"
            ))),
            None => Err(io::Error::other("Bloom readiness response was malformed")),
        };
    }
    Err(last_error.unwrap_or_else(|| io::Error::other("could not connect to Bloom")))
}

fn terminate_child(child: &mut Child) {
    let _ = child.kill();
    let _ = child.wait();
}

fn wait_for_child(child: &mut Child, signals: &Receiver<SignalEvent>) -> Result<ExitStatus> {
    loop {
        if let Some(status) = child.try_wait().context("failed to inspect bloom_server")? {
            return Ok(status);
        }
        if let Some(event) = receive_signal(signals)?
            && matches!(event, SignalEvent::Shutdown)
        {
            request_graceful_shutdown(child)?;
            return wait_for_child_after_shutdown(child, signals);
        }
        thread::sleep(POLL_INTERVAL);
    }
}

fn wait_for_child_after_shutdown(
    child: &mut Child,
    signals: &Receiver<SignalEvent>,
) -> Result<ExitStatus> {
    let deadline = Instant::now() + SHUTDOWN_GRACE;
    loop {
        if let Some(status) = child.try_wait().context("failed to inspect bloom_server")? {
            return Ok(status);
        }
        match receive_signal(signals)? {
            Some(SignalEvent::Shutdown) => {
                eprintln!("received a repeated shutdown signal; forcing bloom_server to exit");
                terminate_child(child);
                return child
                    .try_wait()
                    .context("failed to reap bloom_server after forced shutdown")?
                    .ok_or_else(|| anyhow!("bloom_server did not exit after forced shutdown"));
            }
            Some(SignalEvent::ListenerError(error)) => {
                terminate_child(child);
                bail!("desktop signal listener failed during shutdown: {error}");
            }
            None => {}
        }
        if Instant::now() >= deadline {
            eprintln!("bloom_server did not exit within {SHUTDOWN_GRACE:?}; forcing shutdown");
            terminate_child(child);
            return child
                .try_wait()
                .context("failed to reap bloom_server after graceful timeout")?
                .ok_or_else(|| anyhow!("bloom_server did not exit after forced shutdown"));
        }
        thread::sleep(POLL_INTERVAL);
    }
}

fn receive_signal(signals: &Receiver<SignalEvent>) -> Result<Option<SignalEvent>> {
    match signals.try_recv() {
        Ok(event @ SignalEvent::Shutdown) => Ok(Some(event)),
        Ok(SignalEvent::ListenerError(error)) => {
            bail!("desktop signal listener failed: {error}");
        }
        Err(TryRecvError::Empty) => Ok(None),
        Err(TryRecvError::Disconnected) => {
            bail!("desktop signal listener stopped unexpectedly");
        }
    }
}

fn start_signal_listener() -> Result<Receiver<SignalEvent>> {
    let (sender, receiver) = mpsc::channel();
    thread::Builder::new()
        .name("bloom-desktop-signals".to_string())
        .spawn(move || {
            let runtime = match tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
            {
                Ok(runtime) => runtime,
                Err(error) => {
                    let _ = sender.send(SignalEvent::ListenerError(format!(
                        "failed to create signal runtime: {error}"
                    )));
                    return;
                }
            };
            let panic_sender = sender.clone();
            let error_sender = sender.clone();
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                runtime.block_on(async move {
                    if let Err(error) = process_signal_loop(sender).await {
                        let _ = error_sender.send(SignalEvent::ListenerError(error));
                    }
                });
            }));
            if result.is_err() {
                // A signal driver panic must never leave the server orphaned.
                // The parent turns this event into a bounded forced cleanup.
                let _ = panic_sender.send(SignalEvent::ListenerError(
                    "signal listener panicked".to_string(),
                ));
            }
        })
        .context("failed to start desktop signal listener")?;
    Ok(receiver)
}

async fn process_signal_loop(sender: mpsc::Sender<SignalEvent>) -> Result<(), String> {
    #[cfg(unix)]
    {
        use tokio::signal::unix::{SignalKind, signal};
        let mut interrupt = signal(SignalKind::interrupt()).map_err(|error| error.to_string())?;
        let mut terminate = signal(SignalKind::terminate()).map_err(|error| error.to_string())?;
        let mut hangup = signal(SignalKind::hangup()).map_err(|error| error.to_string())?;
        loop {
            tokio::select! {
                signal = interrupt.recv() => signal.map(|_| ()).ok_or_else(|| "SIGINT listener closed".to_string()),
                signal = terminate.recv() => signal.map(|_| ()).ok_or_else(|| "SIGTERM listener closed".to_string()),
                signal = hangup.recv() => signal.map(|_| ()).ok_or_else(|| "SIGHUP listener closed".to_string()),
            }?;
            if sender.send(SignalEvent::Shutdown).is_err() {
                return Ok(());
            }
        }
    }
    #[cfg(windows)]
    {
        use tokio::signal::windows::{ctrl_break, ctrl_c, ctrl_close, ctrl_shutdown};
        let mut interrupt = ctrl_c().map_err(|error| error.to_string())?;
        let mut break_signal = ctrl_break().map_err(|error| error.to_string())?;
        let mut close = ctrl_close().map_err(|error| error.to_string())?;
        let mut shutdown = ctrl_shutdown().map_err(|error| error.to_string())?;
        loop {
            tokio::select! {
                signal = interrupt.recv() => signal,
                signal = break_signal.recv() => signal,
                signal = close.recv() => signal,
                signal = shutdown.recv() => signal,
            }
            .ok_or_else(|| "Windows signal listener closed".to_string())?;
            if sender.send(SignalEvent::Shutdown).is_err() {
                return Ok(());
            }
        }
    }
    #[cfg(not(any(unix, windows)))]
    {
        loop {
            tokio::signal::ctrl_c()
                .await
                .map_err(|error| error.to_string())?;
            if sender.send(SignalEvent::Shutdown).is_err() {
                return Ok(());
            }
        }
    }
}

#[cfg(unix)]
fn request_graceful_shutdown(child: &Child) -> Result<()> {
    let result = unsafe { libc::kill(child.id() as libc::pid_t, libc::SIGTERM) };
    if result == 0 {
        return Ok(());
    }
    let error = io::Error::last_os_error();
    if error.raw_os_error() == Some(libc::ESRCH) {
        Ok(())
    } else {
        Err(error).context("failed to forward SIGTERM to bloom_server")
    }
}

#[cfg(not(unix))]
fn request_graceful_shutdown(_child: &Child) -> Result<()> {
    // Windows console signals are delivered to the child console process as
    // well. The bounded wait below provides a force-kill fallback.
    Ok(())
}

fn report_exit(status: ExitStatus) -> Result<()> {
    if status.success() {
        Ok(())
    } else {
        bail!("bloom_server exited with status {status}");
    }
}

fn default_server_binary() -> Option<PathBuf> {
    let current = std::env::current_exe().ok()?;
    let name = if cfg!(windows) {
        "bloom_server.exe"
    } else {
        "bloom_server"
    };
    current.parent().map(|parent| parent.join(name))
}

fn server_target(args: &[OsString]) -> Result<ServerTarget> {
    let mut host = DEFAULT_HOST.to_string();
    let mut port = DEFAULT_PORT;
    let mut index = 0;
    while index < args.len() {
        let arg = args[index].to_string_lossy();
        if arg == "--host" {
            index += 1;
            host = args
                .get(index)
                .ok_or_else(|| anyhow!("--host requires a value"))?
                .to_string_lossy()
                .into_owned();
        } else if let Some(value) = arg.strip_prefix("--host=") {
            host = value.to_string();
        } else if arg == "--port" {
            index += 1;
            port = args
                .get(index)
                .ok_or_else(|| anyhow!("--port requires a value"))?
                .to_string_lossy()
                .parse()
                .context("--port must be a valid u16")?;
        } else if let Some(value) = arg.strip_prefix("--port=") {
            port = value.parse().context("--port must be a valid u16")?;
        }
        index += 1;
    }
    if host.trim().is_empty() {
        bail!("--host cannot be empty");
    }
    if host
        .chars()
        .any(|character| character.is_ascii_control() || character.is_whitespace())
    {
        bail!("--host cannot contain whitespace or control characters");
    }
    if port == 0 {
        bail!("--port 0 is not supported by bloom_desktop; choose an explicit port");
    }
    Ok(ServerTarget { host, port })
}

fn display_host(host: &str) -> String {
    let host = local_probe_host(host);
    if host.contains(':') && !host.starts_with('[') {
        format!("[{host}]")
    } else {
        host
    }
}

fn local_probe_host(host: &str) -> String {
    match host.trim() {
        // Wildcard bind addresses are not usable browser destinations.
        "0.0.0.0" => "127.0.0.1".to_string(),
        "::" => "::1".to_string(),
        other => other.to_string(),
    }
}

fn open_browser(url: &str) -> io::Result<()> {
    #[cfg(target_os = "windows")]
    {
        spawn_detached(
            // `explorer.exe` accepts a URL directly; avoid `cmd /C start`,
            // whose shell parsing would treat a malformed host as syntax.
            Command::new("explorer.exe")
                .arg(url)
                .stdin(Stdio::null())
                .stdout(Stdio::null())
                .stderr(Stdio::null()),
        )
    }
    #[cfg(target_os = "macos")]
    {
        spawn_detached(
            Command::new("open")
                .arg(url)
                .stdin(Stdio::null())
                .stdout(Stdio::null())
                .stderr(Stdio::null()),
        )
    }
    #[cfg(all(unix, not(target_os = "macos")))]
    {
        for program in ["xdg-open", "gio"] {
            let mut command = Command::new(program);
            if program == "gio" {
                command.arg("open");
            }
            match spawn_detached(
                command
                    .arg(url)
                    .stdin(Stdio::null())
                    .stdout(Stdio::null())
                    .stderr(Stdio::null()),
            ) {
                Ok(_) => return Ok(()),
                Err(error) if error.kind() == io::ErrorKind::NotFound => continue,
                Err(error) => return Err(error),
            }
        }
        Err(io::Error::new(
            io::ErrorKind::NotFound,
            "neither xdg-open nor gio is available",
        ))
    }
    #[cfg(not(any(unix, target_os = "windows")))]
    {
        let _ = url;
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "this platform has no supported browser launcher",
        ))
    }
}

fn spawn_detached(command: &mut Command) -> io::Result<()> {
    let mut child = command.spawn()?;
    thread::spawn(move || {
        let _ = child.wait();
    });
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use std::net::TcpListener;
    use std::thread;

    fn os_args(args: &[&str]) -> Vec<OsString> {
        args.iter().map(OsString::from).collect()
    }

    #[test]
    fn target_defaults_to_loopback_port() {
        assert_eq!(
            server_target(&[]).expect("default target"),
            ServerTarget {
                host: DEFAULT_HOST.to_string(),
                port: DEFAULT_PORT,
            }
        );
    }

    #[test]
    fn target_accepts_separate_and_equals_values() {
        assert_eq!(
            server_target(&os_args(&["--host", "0.0.0.0", "--port=3010"])).expect("target"),
            ServerTarget {
                host: "0.0.0.0".to_string(),
                port: 3010,
            }
        );
    }

    #[test]
    fn target_rejects_missing_or_invalid_values() {
        assert!(server_target(&os_args(&["--host"])).is_err());
        assert!(server_target(&os_args(&["--port", "bad"])).is_err());
        assert!(server_target(&os_args(&["--port", "0"])).is_err());
        assert!(server_target(&os_args(&["--host="])).is_err());
        assert!(server_target(&os_args(&["--host", "127.0.0.1\n"])).is_err());
    }

    #[test]
    fn display_host_brackets_ipv6() {
        assert_eq!(display_host("::1"), "[::1]");
        assert_eq!(display_host("127.0.0.1"), "127.0.0.1");
        assert_eq!(display_host("0.0.0.0"), "127.0.0.1");
        assert_eq!(display_host("::"), "[::1]");
    }

    #[test]
    fn readiness_probe_accepts_empty_catalog_status() {
        let listener = TcpListener::bind(("127.0.0.1", 0)).expect("bind probe listener");
        let port = listener.local_addr().expect("probe address").port();
        let worker = thread::spawn(move || {
            let (mut stream, _) = listener.accept().expect("accept probe request");
            let mut request = [0_u8; 256];
            let _ = stream.read(&mut request);
            stream
                .write_all(b"HTTP/1.1 503 Service Unavailable\r\nContent-Length: 0\r\n\r\n")
                .expect("write probe response");
        });
        readiness_probe(&ServerTarget {
            host: DEFAULT_HOST.to_string(),
            port,
        })
        .expect("503 is a valid empty-catalog readiness response");
        worker.join().expect("probe worker");
    }

    #[test]
    fn readiness_probe_rejects_unexpected_http_status() {
        let listener = TcpListener::bind(("127.0.0.1", 0)).expect("bind probe listener");
        let port = listener.local_addr().expect("probe address").port();
        let worker = thread::spawn(move || {
            let (mut stream, _) = listener.accept().expect("accept probe request");
            let mut request = [0_u8; 256];
            let _ = stream.read(&mut request);
            stream
                .write_all(b"HTTP/1.1 500 Internal Server Error\r\nContent-Length: 0\r\n\r\n")
                .expect("write probe response");
        });
        assert!(
            readiness_probe(&ServerTarget {
                host: DEFAULT_HOST.to_string(),
                port,
            })
            .is_err()
        );
        worker.join().expect("probe worker");
    }
}
