# syntax=docker/dockerfile:1

FROM rust:1.97.1-bookworm AS builder
WORKDIR /src

RUN apt-get update \
    && apt-get install -y --no-install-recommends libssl-dev pkg-config \
    && rm -rf /var/lib/apt/lists/*
COPY . .
RUN rustup target add wasm32-unknown-unknown \
    && cargo install dioxus-cli --version 0.7.10 --locked
RUN ./scripts/build_ui.sh
RUN cargo build --locked --release --bin bloom_server --features serve-ui \
    && cargo build --locked --release --bin bloom_infer --bin bloom_bench --bin inspect_gguf
RUN mkdir -p /tmp/bloom-doctor \
    && BLOOM_CONFIG_HOME=/tmp/bloom-doctor ./target/release/bloom_server --doctor > /tmp/bloom-doctor.txt \
    && grep -Fq "[PASS] embedded_ui:" /tmp/bloom-doctor.txt \
    && rm -rf /tmp/bloom-doctor /tmp/bloom-doctor.txt

FROM debian:bookworm-slim AS runtime
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=builder /src/target/release/bloom_server /usr/local/bin/bloom_server
COPY --from=builder /src/target/release/bloom_infer /usr/local/bin/bloom_infer
COPY --from=builder /src/target/release/bloom_bench /usr/local/bin/bloom_bench
COPY --from=builder /src/target/release/inspect_gguf /usr/local/bin/inspect_gguf

ENV RUST_LOG=info
EXPOSE 3000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -fsS http://127.0.0.1:3000/health >/dev/null || exit 1

STOPSIGNAL SIGTERM
ENTRYPOINT ["bloom_server"]
CMD ["--host", "0.0.0.0", "--port", "3000"]
