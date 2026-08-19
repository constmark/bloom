//! Process entry point for the Bloom HTTP server.

fn main() -> anyhow::Result<()> {
    // SAFETY: this is the process entry point and no threads have been started.
    unsafe { bloom_server::run_cli() }
}
