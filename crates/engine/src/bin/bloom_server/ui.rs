//! Optional embedded UI for `bloom_server`.
//!
//! Enabled with the `serve-ui` cargo feature. The Dioxus frontend is built
//! separately in `ui/` (`dx build --release`, output copied to `ui/dist`)
//! and embedded into the binary with `rust-embed`, so a single self-contained
//! `bloom_server` can serve both the OpenAI-compatible API under `/v1/*` and
//! the chat UI at `/`.
//!
//! The feature is off by default: the backend stays deployable on its own,
//! and the frontend can also be hosted independently (it talks to the API
//! over plain HTTP with CORS).

#[cfg(feature = "serve-ui")]
use axum::{
    body::Body,
    http::{header, StatusCode, Uri},
    response::{IntoResponse, Response},
    routing::get,
    Router,
};
#[cfg(feature = "serve-ui")]
use rust_embed::Embed;

#[cfg(feature = "serve-ui")]
#[derive(Embed)]
#[folder = "../../ui/dist"]
struct UiAssets;

#[cfg(not(feature = "serve-ui"))]
use axum::Router;

/// Build the router that serves the embedded UI, or `None` when the
/// `serve-ui` feature is disabled.
#[cfg(feature = "serve-ui")]
pub fn ui_router() -> Option<Router> {
    Some(
        Router::new()
            .route("/", get(serve_index))
            .fallback(serve_static),
    )
}

#[cfg(not(feature = "serve-ui"))]
pub fn ui_router() -> Option<Router> {
    None
}

#[cfg(feature = "serve-ui")]
async fn serve_index() -> Response {
    serve_embedded("index.html")
}

#[cfg(feature = "serve-ui")]
async fn serve_static(uri: Uri) -> Response {
    let path = uri.path().trim_start_matches('/');
    // SPA fallback: unknown non-asset paths serve the app shell.
    if path.is_empty() {
        return serve_embedded("index.html");
    }
    match serve_embedded(path) {
        resp if resp.status() == StatusCode::NOT_FOUND => serve_embedded("index.html"),
        resp => resp,
    }
}

#[cfg(feature = "serve-ui")]
fn serve_embedded(path: &str) -> Response {
    match UiAssets::get(path) {
        Some(content) => {
            let mime = mime_guess::from_path(path).first_or_octet_stream();
            (
                [(header::CONTENT_TYPE, mime.as_ref())],
                Body::from(content.data.into_owned()),
            )
                .into_response()
        }
        None => (StatusCode::NOT_FOUND, "not found").into_response(),
    }
}
