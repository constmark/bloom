//! Safe Markdown rendering for model-generated assistant messages.

use pulldown_cmark::{html, CowStr, Event, Options, Parser, Tag, TagEnd};

/// Render assistant Markdown into a constrained HTML fragment.
///
/// Raw HTML is escaped, remote images are reduced to their alt text, and links
/// using scriptable or local protocols are rendered as plain text. The caller
/// may therefore place the returned fragment at Dioxus's explicit HTML boundary.
pub fn render_assistant_markdown(input: &str) -> String {
    let options = Options::ENABLE_STRIKETHROUGH
        | Options::ENABLE_TABLES
        | Options::ENABLE_TASKLISTS
        | Options::ENABLE_FOOTNOTES;
    let parser = Parser::new_ext(input, options);
    let mut events = Vec::new();
    let mut suppressed_links = 0_u32;
    let mut image_depth = 0_u32;

    for event in parser {
        match event {
            Event::Html(value) | Event::InlineHtml(value) => {
                events.push(Event::Text(value));
            }
            Event::Start(Tag::Link { dest_url, .. })
                if !safe_link_destination(dest_url.as_ref()) =>
            {
                suppressed_links = suppressed_links.saturating_add(1);
            }
            Event::End(TagEnd::Link) if suppressed_links > 0 => {
                suppressed_links -= 1;
            }
            Event::Start(Tag::Image { .. }) => {
                image_depth = image_depth.saturating_add(1);
                events.push(Event::Text(CowStr::Borrowed("[Image: ")));
            }
            Event::End(TagEnd::Image) if image_depth > 0 => {
                image_depth -= 1;
                events.push(Event::Text(CowStr::Borrowed("]")));
            }
            event => events.push(event),
        }
    }

    let mut output = String::new();
    html::push_html(&mut output, events.into_iter());
    output.replace(
        "<a href=",
        "<a target=\"_blank\" rel=\"noopener noreferrer\" href=",
    )
}

fn safe_link_destination(destination: &str) -> bool {
    let destination = destination.trim();
    if destination.is_empty()
        || destination.len() > 2_048
        || destination.chars().any(char::is_control)
    {
        return false;
    }
    if destination.starts_with('#') {
        return true;
    }
    let lowercase = destination.to_ascii_lowercase();
    lowercase.starts_with("https://")
        || lowercase.starts_with("http://")
        || lowercase.starts_with("mailto:")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn raw_html_is_escaped_instead_of_entering_the_dom() {
        let rendered = render_assistant_markdown(
            "<script>alert('unsafe')</script>\n<img src=x onerror=alert(1)>",
        );

        assert!(!rendered.contains("<script"));
        assert!(!rendered.contains("<img"));
        assert!(rendered.contains("&lt;script&gt;"));
        assert!(rendered.contains("&lt;img src=x onerror=alert(1)&gt;"));
    }

    #[test]
    fn unsafe_links_become_text_and_safe_links_open_in_isolation() {
        let rendered = render_assistant_markdown(
            "[script](<javascript:alert(1)>) [data](<data:text/html,unsafe>) [safe](https://example.com/docs)",
        );

        assert!(!rendered.contains("javascript:"));
        assert!(!rendered.contains("data:text"));
        assert!(rendered.contains("script"));
        assert!(rendered.contains("data"));
        assert!(rendered.contains("href=\"https://example.com/docs\""));
        assert!(rendered.contains("target=\"_blank\""));
        assert!(rendered.contains("rel=\"noopener noreferrer\""));
    }

    #[test]
    fn remote_images_are_reduced_to_alt_text() {
        let rendered = render_assistant_markdown(
            "Before ![architecture diagram](https://tracker.invalid/pixel.png) after",
        );

        assert!(!rendered.contains("tracker.invalid"));
        assert!(!rendered.contains("<img"));
        assert!(rendered.contains("[Image: architecture diagram]"));
    }

    #[test]
    fn common_assistant_formatting_is_preserved() {
        let rendered = render_assistant_markdown(
            "## Result\n\n- first\n- second\n\n| Name | Value |\n| --- | --- |\n| CPU | ready |\n\n```rust\nfn main() {}\n```\n\n~~old~~",
        );

        assert!(rendered.contains("<h2>Result</h2>"));
        assert!(rendered.contains("<ul>"));
        assert!(rendered.contains("<table>"));
        assert!(rendered.contains("<pre><code class=\"language-rust\">"));
        assert!(rendered.contains("fn main() {}"));
        assert!(rendered.contains("<del>old</del>"));
    }

    #[test]
    fn link_policy_rejects_local_and_scriptable_protocols() {
        assert!(safe_link_destination("https://example.com"));
        assert!(safe_link_destination("HTTP://localhost:3000/path"));
        assert!(safe_link_destination("mailto:maintainer@example.com"));
        assert!(safe_link_destination("#section"));
        assert!(!safe_link_destination("javascript:alert(1)"));
        assert!(!safe_link_destination("file:///etc/passwd"));
        assert!(!safe_link_destination("data:text/html,unsafe"));
        assert!(!safe_link_destination("https://example.com\nunsafe"));
    }

    #[test]
    fn encoded_or_autolinked_script_urls_cannot_create_anchors() {
        let rendered =
            render_assistant_markdown("[encoded](jav&#x61;script:alert(1)) <javascript:alert(2)>");

        assert!(!rendered.contains("<a "));
        assert!(!rendered.contains("href="));
        assert!(!rendered.contains("<javascript"));
    }
}
