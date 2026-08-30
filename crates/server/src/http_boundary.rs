//! Shared HTTP transport boundaries for correlation, protocol errors, browser
//! origins, caching, retry hints, and API credentials.

use super::*;
use std::net::IpAddr;
use tower_http::{
    cors::{Any, CorsLayer},
    request_id::{MakeRequestId as _, MakeRequestUuid},
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct ValidatedBrowserOrigin {
    pub(super) serialized: String,
    pub(super) header: HeaderValue,
    pub(super) scheme: String,
    pub(super) authority: String,
    pub(super) host: String,
}

impl ValidatedBrowserOrigin {
    fn parse(value: &str) -> std::result::Result<Self, String> {
        let value = value.trim();
        if value.is_empty() || value.len() > MAX_BROWSER_ORIGIN_CHARS {
            return Err(format!(
                "browser origin must contain between 1 and {MAX_BROWSER_ORIGIN_CHARS} characters"
            ));
        }
        if value.eq_ignore_ascii_case("null") {
            return Err("opaque browser origins are not allowed".to_string());
        }
        let uri = value
            .parse::<axum::http::Uri>()
            .map_err(|_| "browser origin must be an absolute HTTP(S) origin".to_string())?;
        let scheme = uri
            .scheme_str()
            .filter(|scheme| matches!(*scheme, "http" | "https"))
            .ok_or_else(|| "browser origin scheme must be http or https".to_string())?
            .to_ascii_lowercase();
        let authority = uri
            .authority()
            .ok_or_else(|| "browser origin must include a host".to_string())?;
        if authority.as_str().contains('@') {
            return Err("browser origin must not include user information".to_string());
        }
        if uri
            .path_and_query()
            .is_some_and(|path| path.as_str() != "/")
        {
            return Err("browser origin must not include a path, query, or fragment".to_string());
        }
        let authority = authority.as_str().to_ascii_lowercase();
        let host = uri
            .authority()
            .map(|value| value.host().to_ascii_lowercase())
            .filter(|value| !value.is_empty())
            .ok_or_else(|| "browser origin must include a host".to_string())?;
        let serialized = format!("{scheme}://{authority}");
        let header = HeaderValue::from_str(&serialized)
            .map_err(|_| "browser origin is not a valid HTTP header value".to_string())?;
        Ok(Self {
            serialized,
            header,
            scheme,
            authority,
            host,
        })
    }

    pub(super) fn has_loopback_host(&self) -> bool {
        let unbracketed = self.host.trim_start_matches('[').trim_end_matches(']');
        unbracketed.eq_ignore_ascii_case("localhost")
            || unbracketed
                .parse::<IpAddr>()
                .is_ok_and(|address| address.is_loopback())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) enum BrowserOriginPolicy {
    SameOrigin,
    Exact(ValidatedBrowserOrigin),
    Any,
}

pub(super) fn parse_browser_origin_policy(
    value: &str,
) -> std::result::Result<BrowserOriginPolicy, String> {
    let value = value.trim();
    if value.eq_ignore_ascii_case(DEFAULT_BROWSER_ORIGIN_POLICY) {
        Ok(BrowserOriginPolicy::SameOrigin)
    } else if value == "*" {
        Ok(BrowserOriginPolicy::Any)
    } else {
        ValidatedBrowserOrigin::parse(value).map(BrowserOriginPolicy::Exact)
    }
}

#[derive(Clone, Debug)]
pub(super) struct BrowserOriginGuard {
    pub(super) policy: BrowserOriginPolicy,
    pub(super) loopback_listener: bool,
}

impl BrowserOriginGuard {
    fn permits(&self, request: &AxumRequest) -> bool {
        let mut values = request.headers().get_all(header::ORIGIN).iter();
        let Some(value) = values.next() else {
            return true;
        };
        if values.next().is_some() {
            return false;
        }
        let Ok(value) = value.to_str() else {
            return false;
        };
        let Ok(origin) = ValidatedBrowserOrigin::parse(value) else {
            return false;
        };
        if self.policy == BrowserOriginPolicy::Any {
            return true;
        }
        if let BrowserOriginPolicy::Exact(allowed) = &self.policy
            && origin.serialized == allowed.serialized
        {
            return true;
        }
        if origin.scheme != "http" {
            return false;
        }
        if self.loopback_listener && !origin.has_loopback_host() {
            return false;
        }
        request
            .headers()
            .get(header::HOST)
            .and_then(|host| host.to_str().ok())
            .is_some_and(|host| host.eq_ignore_ascii_case(&origin.authority))
    }
}

pub(super) fn error_response(
    status: axum::http::StatusCode,
    error_type: &str,
    message: impl std::fmt::Display,
) -> axum::response::Response {
    (
        status,
        Json(json!({
            "error": {
                "message": message.to_string(),
                "type": error_type
            }
        })),
    )
        .into_response()
}

pub(super) fn api_error(
    err: ApiError,
    message: impl std::fmt::Display,
) -> axum::response::Response {
    error_response(err.status(), err.error_type(), message)
}

pub(super) fn constant_time_eq(left: &[u8], right: &[u8]) -> bool {
    let max_len = left.len().max(right.len());
    let mut diff = left.len() ^ right.len();
    for idx in 0..max_len {
        let l = left.get(idx).copied().unwrap_or(0);
        let r = right.get(idx).copied().unwrap_or(0);
        diff |= (l ^ r) as usize;
    }
    diff == 0
}

pub(super) fn valid_http_request_id(value: &HeaderValue) -> bool {
    let bytes = value.as_bytes();
    !bytes.is_empty()
        && bytes.len() <= MAX_HTTP_REQUEST_ID_CHARS
        && bytes
            .iter()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.' | b':'))
}

fn normalized_http_request_id(request: &AxumRequest) -> RequestId {
    if let Some(value) = request
        .headers()
        .get(HTTP_REQUEST_ID_HEADER)
        .filter(|value| valid_http_request_id(value))
    {
        return RequestId::new(value.clone());
    }

    MakeRequestUuid
        .make_request_id(request)
        .expect("MakeRequestUuid always returns a request ID")
}

pub(super) async fn correlate_http_request(mut request: AxumRequest, next: Next) -> Response {
    let request_id = normalized_http_request_id(&request);
    request.headers_mut().insert(
        header::HeaderName::from_static(HTTP_REQUEST_ID_HEADER),
        request_id.header_value().clone(),
    );
    request.extensions_mut().insert(request_id.clone());

    let mut response = next.run(request).await;
    response.headers_mut().insert(
        header::HeaderName::from_static(HTTP_REQUEST_ID_HEADER),
        request_id.header_value().clone(),
    );
    response.extensions_mut().insert(request_id);
    response
}

pub(super) fn requires_no_store(path: &str) -> bool {
    matches!(path, "/health" | "/ready" | "/metrics" | "/v1" | "/api")
        || path.starts_with("/v1/")
        || path.starts_with("/api/")
}

#[derive(Clone, Copy)]
enum ApiProtocolFamily {
    OpenAi,
    Ollama,
}

fn api_protocol_family(path: &str) -> Option<ApiProtocolFamily> {
    if path == "/v1" || path.starts_with("/v1/") {
        Some(ApiProtocolFamily::OpenAi)
    } else if path == "/api" || path.starts_with("/api/") {
        Some(ApiProtocolFamily::Ollama)
    } else {
        None
    }
}

pub(super) fn has_protocol_error_content_type(response: &Response) -> bool {
    let Some(content_type) = response
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
    else {
        return false;
    };
    let media_type = content_type
        .split(';')
        .next()
        .unwrap_or_default()
        .trim()
        .to_ascii_lowercase();
    media_type == "application/json"
        || media_type.ends_with("+json")
        || media_type == "text/event-stream"
        || media_type == "application/x-ndjson"
}

fn openai_framework_error(status: axum::http::StatusCode) -> (&'static str, &'static str) {
    match status {
        axum::http::StatusCode::REQUEST_TIMEOUT => (
            ApiError::Timeout.error_type(),
            "The request timed out before it could be completed.",
        ),
        axum::http::StatusCode::PAYLOAD_TOO_LARGE => (
            ApiError::InvalidRequest.error_type(),
            "The request body exceeds the configured size limit.",
        ),
        axum::http::StatusCode::UNSUPPORTED_MEDIA_TYPE => (
            ApiError::InvalidRequest.error_type(),
            "The request Content-Type is not supported for this API route.",
        ),
        axum::http::StatusCode::UNPROCESSABLE_ENTITY => (
            ApiError::InvalidRequest.error_type(),
            "The request body does not match the endpoint schema.",
        ),
        axum::http::StatusCode::TOO_MANY_REQUESTS => (
            ApiError::RateLimitExceeded.error_type(),
            "The server is temporarily at request capacity.",
        ),
        axum::http::StatusCode::NOT_FOUND => (
            ApiError::NotFound.error_type(),
            "The requested OpenAI-compatible API resource does not exist.",
        ),
        axum::http::StatusCode::UNAUTHORIZED => (
            ApiError::AuthenticationError.error_type(),
            "Authentication is required for this API route.",
        ),
        axum::http::StatusCode::SERVICE_UNAVAILABLE => (
            ApiError::ServiceUnavailable.error_type(),
            "The service is temporarily unavailable.",
        ),
        status if status.is_server_error() => (
            ApiError::InternalError.error_type(),
            "The server could not process the request.",
        ),
        _ => (
            ApiError::InvalidRequest.error_type(),
            "The request is malformed or unsupported.",
        ),
    }
}

fn ollama_framework_error(status: axum::http::StatusCode) -> &'static str {
    match status {
        axum::http::StatusCode::REQUEST_TIMEOUT => "request timed out before it could be completed",
        axum::http::StatusCode::PAYLOAD_TOO_LARGE => {
            "request body exceeds the configured size limit"
        }
        axum::http::StatusCode::UNSUPPORTED_MEDIA_TYPE => {
            "request Content-Type is not supported for this API route"
        }
        axum::http::StatusCode::UNPROCESSABLE_ENTITY => {
            "request body does not match the endpoint schema"
        }
        axum::http::StatusCode::TOO_MANY_REQUESTS => "server is temporarily at request capacity",
        axum::http::StatusCode::NOT_FOUND => "requested Ollama-compatible API resource not found",
        axum::http::StatusCode::UNAUTHORIZED => "authentication is required for this API route",
        axum::http::StatusCode::SERVICE_UNAVAILABLE => "service is temporarily unavailable",
        status if status.is_server_error() => "server could not process the request",
        _ => "request is malformed or unsupported",
    }
}

pub(super) async fn normalize_protocol_error_response(
    request: AxumRequest,
    next: Next,
) -> Response {
    let family = api_protocol_family(request.uri().path());
    let response = next.run(request).await;
    let Some(family) = family else {
        return response;
    };
    let status = response.status();
    if !(status.is_client_error() || status.is_server_error())
        || has_protocol_error_content_type(&response)
    {
        return response;
    }

    let shaped = match family {
        ApiProtocolFamily::OpenAi => {
            let (error_type, message) = openai_framework_error(status);
            error_response(status, error_type, message)
        }
        ApiProtocolFamily::Ollama => ollama_error_response(status, ollama_framework_error(status)),
    };
    let (_, shaped_body) = shaped.into_parts();
    let (mut parts, _) = response.into_parts();
    parts.headers.remove(header::CONTENT_LENGTH);
    parts.headers.insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("application/json"),
    );
    Response::from_parts(parts, shaped_body)
}

pub(super) async fn prevent_dynamic_response_caching(request: AxumRequest, next: Next) -> Response {
    let no_store = requires_no_store(request.uri().path());
    let mut response = next.run(request).await;
    if no_store {
        response
            .headers_mut()
            .insert(header::CACHE_CONTROL, HeaderValue::from_static("no-store"));
    }
    response
}

pub(super) async fn publish_transient_retry_after(request: AxumRequest, next: Next) -> Response {
    let mut response = next.run(request).await;
    if response.status() == axum::http::StatusCode::TOO_MANY_REQUESTS {
        response
            .headers_mut()
            .entry(header::RETRY_AFTER)
            .or_insert(HeaderValue::from_static(
                DEFAULT_CAPACITY_RETRY_AFTER_SECONDS,
            ));
    }
    response
}

pub(super) async fn publish_authentication_challenge(request: AxumRequest, next: Next) -> Response {
    let mut response = next.run(request).await;
    if response.status() == axum::http::StatusCode::UNAUTHORIZED {
        response
            .headers_mut()
            .entry(header::WWW_AUTHENTICATE)
            .or_insert(HeaderValue::from_static(
                DEFAULT_BEARER_AUTHENTICATION_CHALLENGE,
            ));
    }
    response
}

pub(super) fn configured_cors_layer(policy: &BrowserOriginPolicy) -> CorsLayer {
    let layer = match policy {
        BrowserOriginPolicy::SameOrigin => CorsLayer::new(),
        BrowserOriginPolicy::Exact(origin) => CorsLayer::new().allow_origin(origin.header.clone()),
        BrowserOriginPolicy::Any => CorsLayer::new().allow_origin(Any),
    };
    layer.allow_methods(Any).allow_headers(Any).expose_headers([
        header::HeaderName::from_static(HTTP_REQUEST_ID_HEADER),
        header::RETRY_AFTER,
        header::WWW_AUTHENTICATE,
    ])
}

pub(super) async fn enforce_browser_origin(
    State(guard): State<BrowserOriginGuard>,
    request: AxumRequest,
    next: Next,
) -> Response {
    if guard.permits(&request) {
        next.run(request).await
    } else {
        (
            axum::http::StatusCode::FORBIDDEN,
            "The browser origin is not allowed by the Bloom server policy.",
        )
            .into_response()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum CredentialScope {
    Inference,
    Operator,
}

fn request_matches_api_key(req: &AxumRequest, expected: &str) -> bool {
    let bearer = format!("Bearer {expected}");
    let authorization_ok = req
        .headers()
        .get(header::AUTHORIZATION)
        .and_then(|value| value.to_str().ok())
        .is_some_and(|value| constant_time_eq(value.as_bytes(), bearer.as_bytes()));
    let x_api_key_ok = req
        .headers()
        .get("x-api-key")
        .and_then(|value| value.to_str().ok())
        .is_some_and(|value| constant_time_eq(value.as_bytes(), expected.as_bytes()));
    authorization_ok || x_api_key_ok
}

fn request_credential_scope(state: &ServerState, req: &AxumRequest) -> Option<CredentialScope> {
    if state
        .operator_api_key
        .as_deref()
        .is_some_and(|key| request_matches_api_key(req, key))
    {
        return Some(CredentialScope::Operator);
    }
    if state
        .api_key
        .as_deref()
        .is_some_and(|key| request_matches_api_key(req, key))
    {
        return Some(if state.operator_api_key.is_some() {
            CredentialScope::Inference
        } else {
            CredentialScope::Operator
        });
    }
    None
}

fn authentication_disabled(state: &ServerState) -> bool {
    state.api_key.is_none() && state.operator_api_key.is_none()
}

async fn continue_with_scope(mut req: AxumRequest, next: Next, scope: CredentialScope) -> Response {
    req.extensions_mut().insert(scope);
    next.run(req).await
}

pub(super) async fn require_api_key(
    State(state): State<Arc<ServerState>>,
    req: AxumRequest,
    next: Next,
) -> Response {
    if authentication_disabled(&state) {
        return continue_with_scope(req, next, CredentialScope::Operator).await;
    }

    match request_credential_scope(&state, &req) {
        Some(scope) => continue_with_scope(req, next, scope).await,
        None => api_error(
            ApiError::AuthenticationError,
            "Missing or invalid API key for protected API endpoint.",
        ),
    }
}

pub(super) async fn require_operator_api_key(
    State(state): State<Arc<ServerState>>,
    req: AxumRequest,
    next: Next,
) -> Response {
    if authentication_disabled(&state) {
        return continue_with_scope(req, next, CredentialScope::Operator).await;
    }

    match request_credential_scope(&state, &req) {
        Some(CredentialScope::Operator) => {
            continue_with_scope(req, next, CredentialScope::Operator).await
        }
        Some(CredentialScope::Inference) => api_error(
            ApiError::PermissionDenied,
            "The inference API key cannot access operator model-management endpoints.",
        ),
        None => api_error(
            ApiError::AuthenticationError,
            "Missing or invalid operator API key for model-management endpoint.",
        ),
    }
}

pub(super) async fn require_ollama_api_key(
    State(state): State<Arc<ServerState>>,
    req: AxumRequest,
    next: Next,
) -> Response {
    if authentication_disabled(&state) {
        return continue_with_scope(req, next, CredentialScope::Operator).await;
    }

    match request_credential_scope(&state, &req) {
        Some(scope) => continue_with_scope(req, next, scope).await,
        None => ollama_error_response(
            axum::http::StatusCode::UNAUTHORIZED,
            "missing or invalid API key for protected API endpoint",
        ),
    }
}

pub(super) async fn require_ollama_operator_api_key(
    State(state): State<Arc<ServerState>>,
    req: AxumRequest,
    next: Next,
) -> Response {
    if authentication_disabled(&state) {
        return continue_with_scope(req, next, CredentialScope::Operator).await;
    }

    match request_credential_scope(&state, &req) {
        Some(CredentialScope::Operator) => {
            continue_with_scope(req, next, CredentialScope::Operator).await
        }
        Some(CredentialScope::Inference) => ollama_error_response(
            axum::http::StatusCode::FORBIDDEN,
            "the inference API key cannot access operator model-management endpoints",
        ),
        None => ollama_error_response(
            axum::http::StatusCode::UNAUTHORIZED,
            "missing or invalid operator API key for model-management endpoint",
        ),
    }
}

pub(super) async fn handle_openai_route_not_found() -> Response {
    api_error(
        ApiError::NotFound,
        "The requested OpenAI-compatible API route does not exist.",
    )
}

pub(super) async fn handle_openai_method_not_allowed() -> Response {
    error_response(
        axum::http::StatusCode::METHOD_NOT_ALLOWED,
        ApiError::InvalidRequest.error_type(),
        "The HTTP method is not supported for this OpenAI-compatible API route.",
    )
}

pub(super) async fn handle_ollama_route_not_found() -> Response {
    ollama_error_response(
        axum::http::StatusCode::NOT_FOUND,
        "Ollama-compatible API route not found",
    )
}

pub(super) async fn handle_ollama_method_not_allowed() -> Response {
    ollama_error_response(
        axum::http::StatusCode::METHOD_NOT_ALLOWED,
        "HTTP method not allowed for this Ollama-compatible API route",
    )
}
