# Bloom UI

Bloom UI is a [Dioxus](https://dioxuslabs.com) client for `bloom_server`.
It currently compiles to a WebAssembly application and can later share the same
codebase with a desktop client.

The UI is fully decoupled from the backend. It communicates only through the
OpenAI-compatible HTTP API, including streaming `/v1/chat/completions`,
`/v1/models`, and `/health`. Deploy it as a standalone static site or embed it
in the server binary.

## Features

- Streaming chat with token-by-token rendering
- Configurable server URL and API key, persisted in `localStorage`
- Connection test and live health status
- `max_tokens` and `temperature` controls
- Message history, empty states, and error banners
- Responsive layout for desktop and mobile browsers

## Development

Install a [`dx`](https://github.com/DioxusLabs/dioxus/tree/master/packages/cli)
version compatible with `dioxus = 0.7`:

```bash
cargo install dioxus-cli
```

Start the backend and UI in separate terminals:

```bash
# Terminal 1: standalone backend
cargo run --bin bloom_server -- --model /path/to/model

# Terminal 2: UI development server
just ui-dev
```

The UI connects to `http://127.0.0.1:3000` by default. Change the base URL in
Settings when the backend is hosted elsewhere.

## Deployment

### Standalone static site

Build static assets and deploy `ui/dist/` to a static host, CDN, or web server:

```bash
just ui-build
```

Configure the deployed UI to use the URL of your separately running
`bloom_server` instance.

### Embedded in bloom_server

Build the UI and embed it in a single server binary:

```bash
just server-ui
./target/release/bloom_server --model /path/to/model
```

Open `http://127.0.0.1:3000/` for the UI. The server continues to expose its API
under `/v1/*`.

The `serve-ui` feature is disabled by default and requires `ui/dist/` to exist
at compile time. `just server-ui` performs the required steps in order.

## Layout

```text
ui/
|-- Cargo.toml       # Independent WASM crate
|-- Dioxus.toml
|-- index.html       # Application mount point
|-- assets/style.css # Global styles
`-- src/
    |-- main.rs      # Components and application state
    `-- api.rs       # OpenAI client and SSE parser
```

`ui/dist/` and `ui/target/` are generated artifacts and are not committed.
