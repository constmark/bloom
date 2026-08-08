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

use std::io;
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

#[cfg(any(feature = "serve-ui", test))]
use axum::http::{HeaderName, HeaderValue};
#[cfg(feature = "serve-ui")]
use axum::{
    body::Body,
    http::{header, HeaderMap, Method, StatusCode, Uri},
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

/// Report whether this binary contains the embedded browser application.
pub const fn embedded_ui_available() -> bool {
    cfg!(feature = "serve-ui")
}

/// Return a browser-reachable local URL for the bound listener.
pub fn browser_url(bound: SocketAddr) -> String {
    let ip = match bound.ip() {
        IpAddr::V4(ip) if ip.is_unspecified() => IpAddr::V4(Ipv4Addr::LOCALHOST),
        IpAddr::V6(ip) if ip.is_unspecified() => IpAddr::V6(Ipv6Addr::LOCALHOST),
        ip => ip,
    };
    format!("http://{}/", SocketAddr::new(ip, bound.port()))
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct BrowserCommandSpec {
    program: &'static str,
    args: Vec<String>,
}

fn browser_commands(url: &str) -> Vec<BrowserCommandSpec> {
    if cfg!(target_os = "windows") {
        vec![BrowserCommandSpec {
            program: "cmd",
            args: vec![
                "/C".to_string(),
                "start".to_string(),
                String::new(),
                url.to_string(),
            ],
        }]
    } else if cfg!(target_os = "macos") {
        vec![BrowserCommandSpec {
            program: "open",
            args: vec![url.to_string()],
        }]
    } else {
        vec![
            BrowserCommandSpec {
                program: "xdg-open",
                args: vec![url.to_string()],
            },
            BrowserCommandSpec {
                program: "gio",
                args: vec!["open".to_string(), url.to_string()],
            },
            BrowserCommandSpec {
                program: "sensible-browser",
                args: vec![url.to_string()],
            },
        ]
    }
}

fn try_browser_commands(
    url: &str,
    mut run: impl FnMut(&BrowserCommandSpec) -> io::Result<()>,
) -> io::Result<()> {
    let commands = browser_commands(url);
    let mut last_error = None;
    for command in &commands {
        match run(command) {
            Ok(()) => return Ok(()),
            Err(error) => last_error = Some(error),
        }
    }
    let attempted = commands
        .iter()
        .map(|command| command.program)
        .collect::<Vec<_>>()
        .join(", ");
    let kind = last_error
        .as_ref()
        .map(io::Error::kind)
        .unwrap_or(io::ErrorKind::NotFound);
    Err(io::Error::new(
        kind,
        format!("no browser launcher succeeded (tried: {attempted})"),
    ))
}

const BROWSER_HANDOFF_WAIT: Duration = Duration::from_millis(200);
const BROWSER_HANDOFF_POLL: Duration = Duration::from_millis(20);

fn run_browser_command(spec: &BrowserCommandSpec) -> io::Result<()> {
    let mut child = Command::new(spec.program)
        .args(&spec.args)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()?;
    let deadline = Instant::now() + BROWSER_HANDOFF_WAIT;
    loop {
        match child.try_wait()? {
            Some(status) if status.success() => return Ok(()),
            Some(status) => {
                return Err(io::Error::other(format!(
                    "{} exited with {status}",
                    spec.program
                )))
            }
            None if Instant::now() >= deadline => {
                std::thread::spawn(move || {
                    let _ = child.wait();
                });
                return Ok(());
            }
            None => std::thread::sleep(BROWSER_HANDOFF_POLL),
        }
    }
}

/// Ask the operating system to open the local Bloom application URL.
pub fn launch_browser(url: &str) -> io::Result<()> {
    try_browser_commands(url, run_browser_command)
}

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
async fn serve_static(method: Method, uri: Uri, headers: HeaderMap) -> Response {
    if method != Method::GET {
        return StatusCode::NOT_FOUND.into_response();
    }
    let path = uri.path().trim_start_matches('/');
    if path.is_empty() {
        return serve_embedded("index.html");
    }
    match serve_embedded(path) {
        resp if resp.status() == StatusCode::NOT_FOUND
            && should_serve_spa_fallback(path, headers.get(header::ACCEPT)) =>
        {
            serve_embedded("index.html")
        }
        resp if resp.status() == StatusCode::NOT_FOUND => StatusCode::NOT_FOUND.into_response(),
        resp => resp,
    }
}

#[cfg(any(feature = "serve-ui", test))]
fn should_serve_spa_fallback(path: &str, accept: Option<&HeaderValue>) -> bool {
    let first_segment = path.split('/').next().unwrap_or_default();
    let reserved = matches!(first_segment, "v1" | "api" | "health" | "ready" | "metrics");
    let looks_like_asset = path
        .rsplit('/')
        .next()
        .is_some_and(|segment| segment.contains('.'));
    let accepts_html = accept
        .and_then(|value| value.to_str().ok())
        .is_some_and(|value| {
            value.split(',').any(|media_range| {
                media_range
                    .split(';')
                    .next()
                    .unwrap_or_default()
                    .trim()
                    .eq_ignore_ascii_case("text/html")
            })
        });
    !reserved && !looks_like_asset && accepts_html
}

#[cfg(feature = "serve-ui")]
fn serve_embedded(path: &str) -> Response {
    let response = match UiAssets::get(path) {
        Some(content) => {
            let mime = mime_guess::from_path(path).first_or_octet_stream();
            (
                [(header::CONTENT_TYPE, mime.as_ref())],
                Body::from(content.data.into_owned()),
            )
                .into_response()
        }
        None => (StatusCode::NOT_FOUND, "not found").into_response(),
    };
    apply_ui_security_headers(response)
}

#[cfg(any(feature = "serve-ui", test))]
const UI_CONTENT_SECURITY_POLICY: &str = "default-src 'self'; base-uri 'none'; object-src 'none'; frame-ancestors 'none'; script-src 'self' 'unsafe-inline' 'wasm-unsafe-eval'; style-src 'self' 'unsafe-inline'; connect-src 'self' http: https:; img-src 'self' data: blob:; font-src 'self'; form-action 'none'";

#[cfg(any(feature = "serve-ui", test))]
fn ui_security_headers() -> [(HeaderName, HeaderValue); 5] {
    [
        (
            HeaderName::from_static("content-security-policy"),
            HeaderValue::from_static(UI_CONTENT_SECURITY_POLICY),
        ),
        (
            HeaderName::from_static("x-content-type-options"),
            HeaderValue::from_static("nosniff"),
        ),
        (
            HeaderName::from_static("x-frame-options"),
            HeaderValue::from_static("DENY"),
        ),
        (
            HeaderName::from_static("referrer-policy"),
            HeaderValue::from_static("no-referrer"),
        ),
        (
            HeaderName::from_static("permissions-policy"),
            HeaderValue::from_static("camera=(), microphone=(), geolocation=()"),
        ),
    ]
}

#[cfg(feature = "serve-ui")]
fn apply_ui_security_headers(mut response: Response) -> Response {
    response.headers_mut().extend(ui_security_headers());
    response
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spa_fallback_requires_html_navigation_outside_protocol_namespaces() {
        let html = HeaderValue::from_static("text/html,application/xhtml+xml;q=0.9");
        let any = HeaderValue::from_static("*/*");

        assert!(should_serve_spa_fallback("conversations/42", Some(&html)));
        assert!(!should_serve_spa_fallback("conversations/42", Some(&any)));
        assert!(!should_serve_spa_fallback("assets/missing.js", Some(&html)));
        for path in [
            "v1/not-found",
            "api/not-found",
            "health/not-found",
            "ready/not-found",
            "metrics/not-found",
        ] {
            assert!(!should_serve_spa_fallback(path, Some(&html)));
        }
    }

    #[cfg(feature = "serve-ui")]
    #[tokio::test]
    async fn embedded_ui_does_not_shadow_unknown_api_or_asset_routes() {
        use tower::ServiceExt as _;

        let app = ui_router().unwrap();
        for path in ["/v1/not-found", "/api/not-found", "/assets/missing.js"] {
            let response = app
                .clone()
                .oneshot(
                    axum::http::Request::builder()
                        .uri(path)
                        .header(header::ACCEPT, "text/html")
                        .body(Body::empty())
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::NOT_FOUND, "path: {path}");
        }

        let navigation = app
            .clone()
            .oneshot(
                axum::http::Request::builder()
                    .uri("/conversations/42")
                    .header(header::ACCEPT, "text/html")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(navigation.status(), StatusCode::OK);
        assert_eq!(navigation.headers()[header::CONTENT_TYPE], "text/html");

        let non_browser = app
            .oneshot(
                axum::http::Request::builder()
                    .uri("/conversations/42")
                    .header(header::ACCEPT, "application/json")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(non_browser.status(), StatusCode::NOT_FOUND);
    }

    #[test]
    fn browser_url_replaces_unspecified_addresses_and_brackets_ipv6() {
        let ipv4 = browser_url("0.0.0.0:3000".parse().unwrap());
        let ipv6 = browser_url("[::]:4123".parse().unwrap());
        let explicit = browser_url("192.0.2.10:8080".parse().unwrap());

        assert_eq!(ipv4, "http://127.0.0.1:3000/");
        assert_eq!(ipv6, "http://[::1]:4123/");
        assert_eq!(explicit, "http://192.0.2.10:8080/");
    }

    #[test]
    fn browser_launch_falls_back_and_reports_exhaustion() {
        let commands = browser_commands("http://127.0.0.1:3000/");
        let mut attempts = Vec::new();
        let success_program = commands.last().unwrap().program;
        try_browser_commands("http://127.0.0.1:3000/", |command| {
            attempts.push(command.program);
            if command.program == success_program {
                Ok(())
            } else {
                Err(io::Error::from(io::ErrorKind::NotFound))
            }
        })
        .unwrap();
        assert_eq!(
            attempts,
            commands
                .iter()
                .map(|command| command.program)
                .collect::<Vec<_>>()
        );

        let error = try_browser_commands("http://127.0.0.1:3000/", |_| {
            Err(io::Error::from(io::ErrorKind::NotFound))
        })
        .unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::NotFound);
        for command in commands {
            assert!(error.to_string().contains(command.program));
        }
    }

    #[cfg(unix)]
    #[test]
    fn browser_command_reports_an_immediate_nonzero_exit() {
        let command = BrowserCommandSpec {
            program: "/bin/sh",
            args: vec!["-c".to_string(), "exit 9".to_string()],
        };

        let error = run_browser_command(&command).unwrap_err();
        assert!(error.to_string().contains("exit status: 9"));
    }

    #[test]
    fn embedded_ui_headers_constrain_active_content_and_embedding() {
        let headers = ui_security_headers()
            .into_iter()
            .collect::<axum::http::HeaderMap>();
        let csp = headers["content-security-policy"].to_str().unwrap();

        assert!(csp.contains("object-src 'none'"));
        assert!(csp.contains("frame-ancestors 'none'"));
        assert!(csp.contains("form-action 'none'"));
        assert_eq!(headers["x-content-type-options"], "nosniff");
        assert_eq!(headers["x-frame-options"], "DENY");
        assert_eq!(headers["referrer-policy"], "no-referrer");
        assert_eq!(
            headers["permissions-policy"],
            "camera=(), microphone=(), geolocation=()"
        );
    }
}
