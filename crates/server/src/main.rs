//! Process entry point for the Bloom HTTP server.

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    bloom_server::run_cli().await
}
