//! Bounded, process-local storage for explicitly retained Responses API state.

use super::*;
use serde::Serialize;
use std::collections::{HashMap, VecDeque};
use std::fmt;
use std::sync::{Arc, Mutex};

pub(crate) const MAX_STORED_RESPONSES: usize = 256;
pub(crate) const MAX_STORED_RESPONSE_BYTES: usize = 40 * MIB as usize;
pub(crate) const MAX_RESPONSE_STORE_BYTES: usize = 64 * MIB as usize;
pub(crate) const RESPONSE_STORE_TTL: Duration = Duration::from_secs(24 * 60 * 60);

#[derive(Debug, Clone)]
pub(crate) struct StoredResponse {
    pub(crate) response: serde_json::Value,
    pub(crate) history: Vec<ChatCompletionMessage>,
    pub(crate) input_items: Vec<serde_json::Value>,
    pub(crate) model: String,
}

#[derive(Debug, Clone, Copy)]
struct ResponseStoreLimits {
    max_responses: usize,
    max_record_bytes: usize,
    max_total_bytes: usize,
    ttl: Duration,
}

impl Default for ResponseStoreLimits {
    fn default() -> Self {
        Self {
            max_responses: MAX_STORED_RESPONSES,
            max_record_bytes: MAX_STORED_RESPONSE_BYTES,
            max_total_bytes: MAX_RESPONSE_STORE_BYTES,
            ttl: RESPONSE_STORE_TTL,
        }
    }
}

#[derive(Debug, Clone)]
struct StoredRecord {
    value: StoredResponse,
    stored_at: Instant,
    bytes: usize,
}

#[derive(Debug, Default)]
struct ResponseStoreInner {
    records: HashMap<String, StoredRecord>,
    insertion_order: VecDeque<String>,
    total_bytes: usize,
}

#[derive(Debug, Clone)]
pub(crate) struct ResponseStore {
    inner: Arc<Mutex<ResponseStoreInner>>,
    limits: ResponseStoreLimits,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ResponseStoreError {
    InvalidResponse,
    TooLarge { bytes: usize, limit: usize },
}

#[derive(Debug)]
pub(crate) struct PendingResponseStorage {
    store: ResponseStore,
    history: Vec<ChatCompletionMessage>,
    input_items: Vec<serde_json::Value>,
}

impl PendingResponseStorage {
    pub(crate) fn new(
        store: ResponseStore,
        history: Vec<ChatCompletionMessage>,
        input_items: Vec<serde_json::Value>,
    ) -> Self {
        Self {
            store,
            history,
            input_items,
        }
    }

    pub(crate) fn commit(
        mut self,
        response: &serde_json::Value,
    ) -> std::result::Result<(), ResponseStoreError> {
        append_responses_output_to_history(&mut self.history, response)
            .map_err(|_| ResponseStoreError::InvalidResponse)?;
        self.store
            .insert(response.clone(), self.history, self.input_items)
    }
}

impl fmt::Display for ResponseStoreError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidResponse => {
                formatter.write_str("The generated response could not be retained safely.")
            }
            Self::TooLarge { bytes, limit } => write!(
                formatter,
                "The generated response requires {bytes} bytes of local state, exceeding the {limit}-byte per-response retention limit."
            ),
        }
    }
}

impl Default for ResponseStore {
    fn default() -> Self {
        Self::new(ResponseStoreLimits::default())
    }
}

impl ResponseStore {
    fn new(limits: ResponseStoreLimits) -> Self {
        Self {
            inner: Arc::new(Mutex::new(ResponseStoreInner::default())),
            limits,
        }
    }

    pub(crate) fn get(&self, response_id: &str) -> Option<StoredResponse> {
        self.get_at(response_id, Instant::now())
    }

    fn get_at(&self, response_id: &str, now: Instant) -> Option<StoredResponse> {
        let mut inner = self.lock();
        self.purge_expired(&mut inner, now);
        inner
            .records
            .get(response_id)
            .map(|record| record.value.clone())
    }

    pub(crate) fn delete(&self, response_id: &str) -> bool {
        let mut inner = self.lock();
        self.purge_expired(&mut inner, Instant::now());
        remove_record(&mut inner, response_id).is_some()
    }

    pub(crate) fn insert(
        &self,
        response: serde_json::Value,
        history: Vec<ChatCompletionMessage>,
        input_items: Vec<serde_json::Value>,
    ) -> std::result::Result<(), ResponseStoreError> {
        self.insert_at(response, history, input_items, Instant::now())
    }

    fn insert_at(
        &self,
        response: serde_json::Value,
        history: Vec<ChatCompletionMessage>,
        input_items: Vec<serde_json::Value>,
        now: Instant,
    ) -> std::result::Result<(), ResponseStoreError> {
        let response_id = response
            .get("id")
            .and_then(serde_json::Value::as_str)
            .filter(|response_id| !response_id.is_empty())
            .ok_or(ResponseStoreError::InvalidResponse)?
            .to_string();
        let model = response
            .get("model")
            .and_then(serde_json::Value::as_str)
            .filter(|model| !model.is_empty())
            .ok_or(ResponseStoreError::InvalidResponse)?
            .to_string();
        let bytes = retained_size(&response, &history, &input_items)?;
        if bytes > self.limits.max_record_bytes || bytes > self.limits.max_total_bytes {
            return Err(ResponseStoreError::TooLarge {
                bytes,
                limit: self
                    .limits
                    .max_record_bytes
                    .min(self.limits.max_total_bytes),
            });
        }

        let value = StoredResponse {
            response,
            history,
            input_items,
            model,
        };
        let mut inner = self.lock();
        self.purge_expired(&mut inner, now);
        remove_record(&mut inner, &response_id);
        while inner.records.len() >= self.limits.max_responses
            || inner.total_bytes.saturating_add(bytes) > self.limits.max_total_bytes
        {
            let Some(oldest) = inner.insertion_order.front().cloned() else {
                break;
            };
            remove_record(&mut inner, &oldest);
        }
        inner.total_bytes = inner.total_bytes.saturating_add(bytes);
        inner.insertion_order.push_back(response_id.clone());
        inner.records.insert(
            response_id,
            StoredRecord {
                value,
                stored_at: now,
                bytes,
            },
        );
        Ok(())
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, ResponseStoreInner> {
        self.inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    fn purge_expired(&self, inner: &mut ResponseStoreInner, now: Instant) {
        let expired = inner
            .insertion_order
            .iter()
            .filter(|response_id| {
                inner.records.get(*response_id).is_some_and(|record| {
                    now.saturating_duration_since(record.stored_at) >= self.limits.ttl
                })
            })
            .cloned()
            .collect::<Vec<_>>();
        for response_id in expired {
            remove_record(inner, &response_id);
        }
    }
}

fn retained_size(
    response: &serde_json::Value,
    history: &[ChatCompletionMessage],
    input_items: &[serde_json::Value],
) -> std::result::Result<usize, ResponseStoreError> {
    #[derive(Serialize)]
    struct RetainedSize<'a> {
        response: &'a serde_json::Value,
        history: &'a [ChatCompletionMessage],
        input_items: &'a [serde_json::Value],
    }

    serde_json::to_vec(&RetainedSize {
        response,
        history,
        input_items,
    })
    .map(|bytes| bytes.len())
    .map_err(|_| ResponseStoreError::InvalidResponse)
}

fn remove_record(inner: &mut ResponseStoreInner, response_id: &str) -> Option<StoredRecord> {
    let removed = inner.records.remove(response_id)?;
    inner.total_bytes = inner.total_bytes.saturating_sub(removed.bytes);
    if let Some(index) = inner
        .insertion_order
        .iter()
        .position(|candidate| candidate == response_id)
    {
        inner.insertion_order.remove(index);
    }
    Some(removed)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn response(response_id: &str, text: &str) -> serde_json::Value {
        json!({
            "id": response_id,
            "object": "response",
            "model": "tiny.gguf",
            "output": [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": text}]}]
        })
    }

    fn history(text: &str) -> Vec<ChatCompletionMessage> {
        vec![ChatCompletionMessage {
            role: "user".to_string(),
            content: json!(text),
            extensions: BTreeMap::new(),
        }]
    }

    fn store_with_limits(max_responses: usize, max_total_bytes: usize) -> ResponseStore {
        ResponseStore::new(ResponseStoreLimits {
            max_responses,
            max_record_bytes: max_total_bytes,
            max_total_bytes,
            ttl: Duration::from_secs(60),
        })
    }

    #[test]
    fn stores_cloned_state_and_deletes_it() {
        let store = store_with_limits(2, 16 * 1024);
        let input_items = vec![json!({"id": "msg-input", "type": "message"})];
        store
            .insert(
                response("resp-one", "hello"),
                history("hello"),
                input_items.clone(),
            )
            .unwrap();

        let stored = store.get("resp-one").unwrap();
        assert_eq!(stored.model, "tiny.gguf");
        assert_eq!(stored.history[0].content, json!("hello"));
        assert_eq!(stored.input_items, input_items);
        assert!(store.delete("resp-one"));
        assert!(!store.delete("resp-one"));
        assert!(store.get("resp-one").is_none());
    }

    #[test]
    fn evicts_the_oldest_record_at_the_count_limit() {
        let store = store_with_limits(2, 64 * 1024);
        for response_id in ["resp-one", "resp-two", "resp-three"] {
            store
                .insert(response(response_id, "ok"), history("input"), Vec::new())
                .unwrap();
        }
        assert!(store.get("resp-one").is_none());
        assert!(store.get("resp-two").is_some());
        assert!(store.get("resp-three").is_some());
    }

    #[test]
    fn rejects_one_oversized_record_without_evicting_existing_state() {
        let store = store_with_limits(2, 1_024);
        store
            .insert(response("resp-one", "ok"), history("input"), Vec::new())
            .unwrap();
        let error = store
            .insert(
                response("resp-two", &"x".repeat(2_000)),
                history("input"),
                Vec::new(),
            )
            .unwrap_err();
        assert!(matches!(error, ResponseStoreError::TooLarge { .. }));
        assert!(store.get("resp-one").is_some());
        assert!(store.get("resp-two").is_none());
    }

    #[test]
    fn expires_records_lazily() {
        let store = ResponseStore::new(ResponseStoreLimits {
            max_responses: 2,
            max_record_bytes: 16 * 1024,
            max_total_bytes: 16 * 1024,
            ttl: Duration::from_secs(5),
        });
        let inserted_at = Instant::now();
        store
            .insert_at(
                response("resp-one", "ok"),
                history("input"),
                Vec::new(),
                inserted_at,
            )
            .unwrap();
        assert!(
            store
                .get_at("resp-one", inserted_at + Duration::from_secs(4))
                .is_some()
        );
        assert!(
            store
                .get_at("resp-one", inserted_at + Duration::from_secs(5))
                .is_none()
        );
    }
}
