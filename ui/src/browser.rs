//! Small browser-only file helpers kept outside the API protocol client.

use std::cell::RefCell;
use std::rc::Rc;

use wasm_bindgen::closure::Closure;
use wasm_bindgen::JsCast;
use wasm_bindgen_futures::JsFuture;

const MODAL_FOCUSABLE_SELECTOR: &str = "button:not([disabled]):not([tabindex=\"-1\"]),a[href]:not([tabindex=\"-1\"]),input:not([disabled]):not([tabindex=\"-1\"]),select:not([disabled]):not([tabindex=\"-1\"]),textarea:not([disabled]):not([tabindex=\"-1\"]),[tabindex]:not([tabindex=\"-1\"])";
const MAX_BUFFERED_MICROPHONE_SECONDS: usize = 30;

/// A live browser microphone capture that exposes bounded mono PCM windows.
///
/// The deprecated ScriptProcessor API is intentionally used here because it
/// remains the smallest broadly supported bridge from Web Audio into WASM.
/// The node writes silence to its output, so connecting it to the destination
/// keeps callbacks active without playing the microphone through the speakers.
pub struct MicrophoneCapture {
    context: web_sys::AudioContext,
    stream: web_sys::MediaStream,
    source: web_sys::MediaStreamAudioSourceNode,
    processor: web_sys::ScriptProcessorNode,
    samples: Rc<RefCell<Vec<f32>>>,
    sample_rate: u32,
    stopped: bool,
    _on_audio: Closure<dyn FnMut(web_sys::AudioProcessingEvent)>,
}

impl MicrophoneCapture {
    pub async fn start() -> Result<Self, String> {
        let window =
            web_sys::window().ok_or_else(|| "browser window is unavailable".to_string())?;
        let media_devices = window
            .navigator()
            .media_devices()
            .map_err(|error| microphone_error("Microphone access is unavailable", error))?;
        let constraints = web_sys::MediaStreamConstraints::new();
        constraints.set_audio_bool(true);
        constraints.set_video_bool(false);
        let stream = JsFuture::from(
            media_devices
                .get_user_media_with_constraints(&constraints)
                .map_err(|error| microphone_error("Could not request microphone access", error))?,
        )
        .await
        .map_err(|error| {
            microphone_error(
                "Microphone permission was denied or no input device is available",
                error,
            )
        })?
        .dyn_into::<web_sys::MediaStream>()
        .map_err(|_| "The browser returned an invalid microphone stream.".to_string())?;

        let context = match web_sys::AudioContext::new() {
            Ok(context) => context,
            Err(error) => {
                stop_media_stream(&stream);
                return Err(microphone_error(
                    "Could not start browser audio capture",
                    error,
                ));
            }
        };
        let sample_rate = context.sample_rate().round() as u32;
        if !(8_000..=192_000).contains(&sample_rate) {
            stop_media_stream(&stream);
            let _ = context.close();
            return Err("The browser reported an unsupported microphone sample rate.".to_string());
        }
        let source = match context.create_media_stream_source(&stream) {
            Ok(source) => source,
            Err(error) => {
                stop_media_stream(&stream);
                let _ = context.close();
                return Err(microphone_error("Could not connect the microphone", error));
            }
        };
        let processor = match context
            .create_script_processor_with_buffer_size_and_number_of_input_channels_and_number_of_output_channels(
                4_096, 1, 1,
            )
        {
            Ok(processor) => processor,
            Err(error) => {
                stop_media_stream(&stream);
                let _ = context.close();
                return Err(microphone_error(
                    "Could not create the audio capture processor",
                    error,
                ));
            }
        };
        let samples = Rc::new(RefCell::new(Vec::<f32>::new()));
        let callback_samples = Rc::clone(&samples);
        let max_samples = sample_rate as usize * MAX_BUFFERED_MICROPHONE_SECONDS;
        let on_audio = Closure::wrap(Box::new(move |event: web_sys::AudioProcessingEvent| {
            let Ok(input) = event.input_buffer() else {
                return;
            };
            let Ok(mut chunk) = input.get_channel_data(0) else {
                return;
            };
            chunk.retain(|sample| sample.is_finite());
            let mut buffered = callback_samples.borrow_mut();
            let remaining = max_samples.saturating_sub(buffered.len());
            buffered.extend(chunk.into_iter().take(remaining));
        }) as Box<dyn FnMut(_)>);
        processor.set_onaudioprocess(Some(on_audio.as_ref().unchecked_ref()));
        if let Err(error) = source.connect_with_audio_node(&processor) {
            processor.set_onaudioprocess(None);
            stop_media_stream(&stream);
            let _ = context.close();
            return Err(microphone_error(
                "Could not activate microphone capture",
                error,
            ));
        }
        if let Err(error) = processor.connect_with_audio_node(&context.destination()) {
            processor.set_onaudioprocess(None);
            let _ = source.disconnect();
            stop_media_stream(&stream);
            let _ = context.close();
            return Err(microphone_error(
                "Could not activate the audio processor",
                error,
            ));
        }
        if let Ok(resume) = context.resume() {
            let _ = JsFuture::from(resume).await;
        }

        Ok(Self {
            context,
            stream,
            source,
            processor,
            samples,
            sample_rate,
            stopped: false,
            _on_audio: on_audio,
        })
    }

    /// Stop acquiring new samples while retaining the final buffered window.
    pub fn stop_input(&mut self) {
        if self.stopped {
            return;
        }
        self.stopped = true;
        stop_media_stream(&self.stream);
        self.processor.set_onaudioprocess(None);
        let _ = self.source.disconnect();
        let _ = self.processor.disconnect();
    }

    /// Drain captured audio and resample it to the model-facing sample rate.
    pub fn take_resampled(&mut self, target_rate: u32) -> Vec<f32> {
        let captured = std::mem::take(&mut *self.samples.borrow_mut());
        resample_mono_pcm(&captured, self.sample_rate, target_rate)
    }
}

impl Drop for MicrophoneCapture {
    fn drop(&mut self) {
        self.stop_input();
        let _ = self.context.close();
    }
}

fn stop_media_stream(stream: &web_sys::MediaStream) {
    for track in stream.get_tracks().iter() {
        if let Ok(track) = track.dyn_into::<web_sys::MediaStreamTrack>() {
            track.stop();
        }
    }
}

fn microphone_error(context: &str, error: wasm_bindgen::JsValue) -> String {
    let detail = error
        .dyn_ref::<js_sys::Error>()
        .map(js_sys::Error::message)
        .map(String::from)
        .filter(|message| !message.is_empty())
        .unwrap_or_else(|| format!("{error:?}"));
    format!("{context}: {detail}. Use HTTPS or localhost and allow microphone permission.")
}

fn resample_mono_pcm(samples: &[f32], source_rate: u32, target_rate: u32) -> Vec<f32> {
    if samples.is_empty() || source_rate == 0 || target_rate == 0 {
        return Vec::new();
    }
    if source_rate == target_rate {
        return samples
            .iter()
            .copied()
            .map(|sample| sample.clamp(-1.0, 1.0))
            .collect();
    }
    let output_len =
        ((samples.len() as u64 * target_rate as u64) / source_rate as u64).max(1) as usize;
    let source_per_output = source_rate as f64 / target_rate as f64;
    (0..output_len)
        .map(|index| {
            let position = index as f64 * source_per_output;
            let left = position.floor() as usize;
            let right = (left + 1).min(samples.len() - 1);
            let fraction = (position - left as f64) as f32;
            (samples[left] + (samples[right] - samples[left]) * fraction).clamp(-1.0, 1.0)
        })
        .collect()
}

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
    use super::{modal_focus_target, resample_mono_pcm};

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

    #[test]
    fn microphone_resampling_is_bounded_and_keeps_endpoints_stable() {
        let source = vec![-1.0, -0.5, 0.0, 0.5, 1.0, 2.0];
        let downsampled = resample_mono_pcm(&source, 48_000, 16_000);
        assert_eq!(downsampled.len(), 2);
        assert_eq!(downsampled, vec![-1.0, 0.5]);

        let unchanged = resample_mono_pcm(&source, 16_000, 16_000);
        assert_eq!(unchanged.last(), Some(&1.0));
        assert!(resample_mono_pcm(&source, 0, 16_000).is_empty());
    }
}
