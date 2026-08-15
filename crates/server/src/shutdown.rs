//! Cross-platform process signals and bounded graceful-shutdown coordination.

use std::time::Duration;

use tokio::sync::watch;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ShutdownSignal {
    Interrupt,
    #[cfg(unix)]
    Terminate,
}

impl ShutdownSignal {
    pub(crate) fn label(self) -> &'static str {
        match self {
            Self::Interrupt => "interrupt",
            #[cfg(unix)]
            Self::Terminate => "terminate",
        }
    }
}

#[cfg(unix)]
pub(crate) struct ShutdownSignalListener {
    interrupt: tokio::signal::unix::Signal,
    terminate: tokio::signal::unix::Signal,
}

#[cfg(unix)]
impl ShutdownSignalListener {
    pub(crate) fn install() -> std::io::Result<Self> {
        Ok(Self {
            interrupt: tokio::signal::unix::signal(tokio::signal::unix::SignalKind::interrupt())?,
            terminate: tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())?,
        })
    }

    pub(crate) async fn recv(&mut self) -> std::io::Result<ShutdownSignal> {
        tokio::select! {
            signal = self.interrupt.recv() => signal
                .map(|()| ShutdownSignal::Interrupt)
                .ok_or_else(|| std::io::Error::other("SIGINT listener closed unexpectedly")),
            signal = self.terminate.recv() => signal
                .map(|()| ShutdownSignal::Terminate)
                .ok_or_else(|| std::io::Error::other("SIGTERM listener closed unexpectedly")),
        }
    }
}

#[cfg(windows)]
pub(crate) struct ShutdownSignalListener {
    interrupt: tokio::signal::windows::CtrlC,
}

#[cfg(windows)]
impl ShutdownSignalListener {
    pub(crate) fn install() -> std::io::Result<Self> {
        Ok(Self {
            interrupt: tokio::signal::windows::ctrl_c()?,
        })
    }

    pub(crate) async fn recv(&mut self) -> std::io::Result<ShutdownSignal> {
        self.interrupt
            .recv()
            .await
            .map(|()| ShutdownSignal::Interrupt)
            .ok_or_else(|| std::io::Error::other("Ctrl-C listener closed unexpectedly"))
    }
}

#[cfg(not(any(unix, windows)))]
pub(crate) struct ShutdownSignalListener;

#[cfg(not(any(unix, windows)))]
impl ShutdownSignalListener {
    pub(crate) fn install() -> std::io::Result<Self> {
        Ok(Self)
    }

    pub(crate) async fn recv(&mut self) -> std::io::Result<ShutdownSignal> {
        tokio::signal::ctrl_c().await?;
        Ok(ShutdownSignal::Interrupt)
    }
}

pub(crate) async fn wait_for_shutdown_notification(mut receiver: watch::Receiver<bool>) {
    while !*receiver.borrow() {
        if receiver.changed().await.is_err() {
            break;
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) enum ShutdownWait<T> {
    Completed(T),
    TimedOut,
}

pub(crate) async fn wait_for_server_or_shutdown_timeout<F, T>(
    server: F,
    shutdown_receiver: watch::Receiver<bool>,
    timeout: Duration,
) -> ShutdownWait<T>
where
    F: std::future::Future<Output = T>,
{
    let deadline = async move {
        wait_for_shutdown_notification(shutdown_receiver).await;
        tokio::time::sleep(timeout).await;
    };
    tokio::pin!(server);
    tokio::pin!(deadline);
    tokio::select! {
        result = &mut server => ShutdownWait::Completed(result),
        () = &mut deadline => ShutdownWait::TimedOut,
    }
}
