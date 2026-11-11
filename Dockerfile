# syntax=docker/dockerfile:1@sha256:ecfaec9ed6d810b56388c508f4121597bfbba70d41a6dfeee4d8cad5f295fc32

FROM rust:1.97.1-bookworm@sha256:0e2bcaef56d041a486784e54104a81aebe0da44bd03019bd70bc0401e42e4a97 AS builder
WORKDIR /src

RUN apt-get -o Acquire::Retries=3 -o Acquire::http::Timeout=30 update \
    && apt-get -o Acquire::Retries=3 -o Acquire::http::Timeout=30 install -y --no-install-recommends ca-certificates curl libssl-dev pkg-config python3 \
    && rm -rf /var/lib/apt/lists/*
ARG TARGETARCH
COPY scripts/install_ui_toolchain_linux.sh /usr/local/src/install_ui_toolchain_linux.sh
RUN --mount=type=cache,id=bloom-cargo-registry,target=/usr/local/cargo/registry \
    rustup target add wasm32-unknown-unknown \
    && cargo install dioxus-cli --version 0.7.10 --locked --features no-downloads \
    && TARGETARCH="$TARGETARCH" /usr/local/src/install_ui_toolchain_linux.sh
RUN toolchain_dir="$(dirname "$(dirname "$(rustup which rustc)")")" \
    && rustup toolchain link bloom-container "$toolchain_dir"
ENV RUSTUP_TOOLCHAIN=bloom-container \
    NO_DOWNLOADS=1

COPY . .
RUN --mount=type=cache,id=bloom-cargo-registry,target=/usr/local/cargo/registry \
    native_target="$(rustc -vV | sed -n 's/^host: //p')" \
    && cargo metadata --locked --format-version 1 --features serve-ui \
        --filter-platform "$native_target" > /tmp/bloom-native-metadata.json \
    && cargo metadata --manifest-path ui/Cargo.toml --locked --format-version 1 \
        --filter-platform wasm32-unknown-unknown > /tmp/bloom-ui-metadata.json \
    && python3 scripts/generate_sbom.py --metadata /tmp/bloom-native-metadata.json \
        --metadata /tmp/bloom-ui-metadata.json --policy config/dependency-policy.json \
        --target "$native_target" --embedded-ui true --output /src/BLOOM-SBOM.cdx.json
RUN --mount=type=cache,id=bloom-cargo-registry,target=/usr/local/cargo/registry \
    --mount=type=cache,id=bloom-ui-target-${TARGETARCH},target=/src/ui/target \
    ./scripts/build_ui.sh
RUN --mount=type=cache,id=bloom-cargo-registry,target=/usr/local/cargo/registry \
    cargo build --locked --release --bin bloom_server --features serve-ui \
    && cargo build --locked --release --bin bloom_infer --bin bloom_bench --bin inspect_gguf
RUN mkdir -p /tmp/bloom-doctor \
    && BLOOM_CONFIG_HOME=/tmp/bloom-doctor ./target/release/bloom_server --doctor > /tmp/bloom-doctor.txt \
    && grep -Fq "[PASS] embedded_ui:" /tmp/bloom-doctor.txt \
    && rm -rf /tmp/bloom-doctor /tmp/bloom-doctor.txt

FROM debian:bookworm-slim@sha256:abd67ffcfa541b485a3dff59865ab629aa048a6c613e639d36e7456b0b229241 AS runtime
RUN apt-get -o Acquire::Retries=3 -o Acquire::http::Timeout=30 update \
    && apt-get -o Acquire::Retries=3 -o Acquire::http::Timeout=30 install -y --no-install-recommends ca-certificates curl \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 10001 bloom \
    && useradd --uid 10001 --gid bloom --create-home --no-log-init \
        --home-dir /home/bloom --shell /usr/sbin/nologin bloom \
    && install -d -o bloom -g bloom /var/lib/bloom/models

WORKDIR /app
COPY --from=builder /src/target/release/bloom_server /usr/local/bin/bloom_server
COPY --from=builder /src/target/release/bloom_infer /usr/local/bin/bloom_infer
COPY --from=builder /src/target/release/bloom_bench /usr/local/bin/bloom_bench
COPY --from=builder /src/target/release/inspect_gguf /usr/local/bin/inspect_gguf
COPY --from=builder /src/BLOOM-SBOM.cdx.json /usr/share/bloom/BLOOM-SBOM.cdx.json
COPY --from=builder /src/config/dependency-policy.json /usr/share/bloom/BLOOM-DEPENDENCY-POLICY.json

ENV HOME=/home/bloom \
    BLOOM_CONFIG_HOME=/var/lib/bloom \
    BLOOM_MODELS_DIR=/var/lib/bloom/models \
    BLOOM_STRICT_MEMORY_BUDGET=1 \
    BLOOM_STRICT_SECURITY=1 \
    RUST_LOG=info
EXPOSE 3000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -fsS http://127.0.0.1:3000/health >/dev/null || exit 1

STOPSIGNAL SIGTERM
USER 10001:10001
ENTRYPOINT ["bloom_server"]
CMD ["--host", "0.0.0.0", "--port", "3000"]
