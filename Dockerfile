# syntax=docker/dockerfile:1

FROM rust:1-bookworm AS builder
WORKDIR /src

COPY . .
RUN cargo build --release --bin bloom_server --bin bloom_infer --bin bloom_bench --bin inspect_gguf

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

ENTRYPOINT ["bloom_server"]
CMD ["--host", "0.0.0.0", "--port", "3000"]
