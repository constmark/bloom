//! Small browser-only file helpers kept outside the API protocol client.

use wasm_bindgen::JsCast;
use wasm_bindgen_futures::JsFuture;

const MODAL_FOCUSABLE_SELECTOR: &str = "button:not([disabled]):not([tabindex=\"-1\"]),a[href]:not([tabindex=\"-1\"]),input:not([disabled]):not([tabindex=\"-1\"]),select:not([disabled]):not([tabindex=\"-1\"]),textarea:not([disabled]):not([tabindex=\"-1\"]),[tabindex]:not([tabindex=\"-1\"])";

pub fn capture_active_element() -> Option<web_sys::HtmlElement> {
    web_sys::window()?
        .document()?
        .active_element()?
        .dyn_into::<web_sys::HtmlElement>()
        .ok()
}

pub fn restore_focus(element: Option<web_sys::HtmlElement>) {
    if let Some(element) = element {
        let _ = element.focus();
    }
}

pub fn focus_first_in_modal(modal_id: &str) -> bool {
    modal_focusable_elements(modal_id)
        .first()
        .is_some_and(|element| element.focus().is_ok())
}

pub fn trap_modal_tab(modal_id: &str, backwards: bool) -> bool {
    let Some(document) = web_sys::window().and_then(|window| window.document()) else {
        return false;
    };
    let elements = modal_focusable_elements(modal_id);
    let active = document.active_element();
    let active_index = active.as_ref().and_then(|active| {
        elements
            .iter()
            .position(|element| js_sys::Object::is(active.as_ref(), element.as_ref()))
    });
    modal_focus_target(elements.len(), active_index, backwards)
        .and_then(|index| elements.get(index))
        .is_some_and(|element| element.focus().is_ok())
}

fn modal_focus_target(
    element_count: usize,
    active_index: Option<usize>,
    backwards: bool,
) -> Option<usize> {
    if element_count == 0 {
        return None;
    }
    let active_index = active_index.filter(|index| *index < element_count);
    match (backwards, active_index) {
        (true, Some(0)) => Some(element_count - 1),
        (false, Some(index)) if index + 1 == element_count => Some(0),
        (true, None) => Some(element_count - 1),
        (false, None) => Some(0),
        _ => None,
    }
}

fn modal_focusable_elements(modal_id: &str) -> Vec<web_sys::HtmlElement> {
    let Some(modal) = web_sys::window()
        .and_then(|window| window.document())
        .and_then(|document| document.get_element_by_id(modal_id))
    else {
        return Vec::new();
    };
    let Ok(nodes) = modal.query_selector_all(MODAL_FOCUSABLE_SELECTOR) else {
        return Vec::new();
    };
    let window = web_sys::window();
    (0..nodes.length())
        .filter_map(|index| nodes.item(index))
        .filter_map(|node| node.dyn_into::<web_sys::HtmlElement>().ok())
        .filter(|element| {
            let hidden_by_ancestor = element
                .closest("[hidden],[aria-hidden=\"true\"]")
                .ok()
                .flatten()
                .is_some();
            let hidden_by_style = window
                .as_ref()
                .and_then(|window| window.get_computed_style(element).ok().flatten())
                .is_some_and(|style| {
                    style.get_property_value("display").as_deref() == Ok("none")
                        || style.get_property_value("visibility").as_deref() == Ok("hidden")
                });
            !hidden_by_ancestor
                && !hidden_by_style
                && (element.offset_width() > 0 || element.offset_height() > 0)
        })
        .collect()
}

pub fn monotonic_now_ms() -> f64 {
    web_sys::window()
        .and_then(|window| window.performance())
        .map(|performance| performance.now())
        .unwrap_or_else(js_sys::Date::now)
}

pub async fn copy_text_to_clipboard(text: &str) -> Result<(), String> {
    let window = web_sys::window().ok_or_else(|| "browser window is unavailable".to_string())?;
    let navigator = window.navigator();
    let clipboard = js_sys::Reflect::get(
        navigator.as_ref(),
        &wasm_bindgen::JsValue::from_str("clipboard"),
    )
    .map_err(|error| format!("failed to access the browser clipboard: {error:?}"))?;
    if clipboard.is_null() || clipboard.is_undefined() {
        return Err(
            "Clipboard access is unavailable. Use HTTPS or localhost and allow clipboard permission."
                .to_string(),
        );
    }
    let write_text =
        js_sys::Reflect::get(&clipboard, &wasm_bindgen::JsValue::from_str("writeText"))
            .map_err(|error| format!("failed to access clipboard writing: {error:?}"))?
            .dyn_into::<js_sys::Function>()
            .map_err(|_| "This browser does not support clipboard text writing.".to_string())?;
    let promise = write_text
        .call1(&clipboard, &wasm_bindgen::JsValue::from_str(text))
        .map_err(|error| format!("clipboard write was rejected: {error:?}"))?
        .dyn_into::<js_sys::Promise>()
        .map_err(|_| "The browser returned an invalid clipboard operation.".to_string())?;
    JsFuture::from(promise).await.map_err(|error| {
        format!(
            "Clipboard write was rejected. Check browser permission and secure-context requirements: {error:?}"
        )
    })?;
    Ok(())
}

pub async fn read_bounded_text_file(
    file: &web_sys::File,
    description: &str,
    max_bytes: u64,
) -> Result<String, String> {
    let size = file.size();
    if !size.is_finite() || size <= 0.0 || size > max_bytes as f64 {
        return Err(format!(
            "{description} must be between 1 byte and {max_bytes} bytes"
        ));
    }
    let text = JsFuture::from(file.text())
        .await
        .map_err(|error| format!("failed to read {description}: {error:?}"))?
        .as_string()
        .ok_or_else(|| format!("{description} is not valid text"))?;
    if text.len() as u64 > max_bytes {
        return Err(format!("{description} exceeds the supported size limit"));
    }
    Ok(text)
}

pub fn download_text_file(filename: &str, mime: &str, text: &str) -> Result<(), String> {
    if filename.is_empty()
        || filename.len() > 128
        || filename.contains('/')
        || filename.contains('\\')
        || filename.chars().any(char::is_control)
    {
        return Err("download filename is invalid".to_string());
    }
    let parts = js_sys::Array::new();
    parts.push(&wasm_bindgen::JsValue::from_str(text));
    let options = web_sys::BlobPropertyBag::new();
    options.set_type(mime);
    let blob = web_sys::Blob::new_with_str_sequence_and_options(parts.as_ref(), &options)
        .map_err(|error| format!("failed to create text download: {error:?}"))?;
    let object_url = web_sys::Url::create_object_url_with_blob(&blob)
        .map_err(|error| format!("failed to create text download URL: {error:?}"))?;
    let result = (|| {
        let document = web_sys::window()
            .and_then(|window| window.document())
            .ok_or_else(|| "browser document is unavailable".to_string())?;
        let anchor = document
            .create_element("a")
            .map_err(|error| format!("failed to create download link: {error:?}"))?
            .dyn_into::<web_sys::HtmlAnchorElement>()
            .map_err(|_| "failed to create an HTML download link".to_string())?;
        anchor.set_href(&object_url);
        anchor.set_download(filename);
        let body = document
            .body()
            .ok_or_else(|| "browser document body is unavailable".to_string())?;
        body.append_child(&anchor)
            .map_err(|error| format!("failed to attach download link: {error:?}"))?;
        anchor.click();
        body.remove_child(&anchor)
            .map_err(|error| format!("failed to remove download link: {error:?}"))?;
        Ok(())
    })();
    let _ = web_sys::Url::revoke_object_url(&object_url);
    result
}

#[cfg(test)]
mod tests {
    use super::modal_focus_target;

    #[test]
    fn modal_focus_target_cycles_only_at_boundaries() {
        assert_eq!(modal_focus_target(0, None, false), None);
        assert_eq!(modal_focus_target(3, Some(0), true), Some(2));
        assert_eq!(modal_focus_target(3, Some(2), false), Some(0));
        assert_eq!(modal_focus_target(3, Some(1), false), None);
        assert_eq!(modal_focus_target(3, Some(1), true), None);
    }

    #[test]
    fn modal_focus_target_recovers_focus_from_outside_the_boundary() {
        assert_eq!(modal_focus_target(3, None, false), Some(0));
        assert_eq!(modal_focus_target(3, None, true), Some(2));
        assert_eq!(modal_focus_target(3, Some(9), false), Some(0));
    }
}
