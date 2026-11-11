//! Bounded incremental framing for browser streaming responses.
//!
//! Scan each input byte once. Neither a fragmented frame nor a transport chunk
//! containing many events should repeatedly scan or copy its accumulated tail.

const MAX_SSE_FRAME_BYTES: usize = 1024 * 1024;

#[derive(Debug, Default)]
pub(crate) struct BrowserSseDecoder {
    line: Vec<u8>,
    data: String,
    has_data: bool,
    frame_bytes: usize,
    // The last line ending is part of the frame only if another nonempty line
    // follows. Otherwise it belongs to the blank-line frame separator.
    line_ending_bytes: usize,
    skip_lf: bool,
    has_non_whitespace: bool,
    first_line_seen: bool,
}

impl BrowserSseDecoder {
    pub(crate) fn push(&mut self, bytes: &[u8]) -> Result<Vec<String>, String> {
        let mut events = Vec::new();
        for &byte in bytes {
            if std::mem::take(&mut self.skip_lf) && byte == b'\n' {
                if self.line_ending_bytes != 0 {
                    self.line_ending_bytes += 1;
                }
                continue;
            }
            if matches!(byte, b'\r' | b'\n') {
                self.skip_lf = byte == b'\r';
                if self.line.is_empty() {
                    if self.has_data {
                        events.push(std::mem::take(&mut self.data));
                    }
                    self.has_data = false;
                    self.frame_bytes = 0;
                    self.line_ending_bytes = 0;
                    self.has_non_whitespace = false;
                } else {
                    self.complete_line()?;
                    self.line.clear();
                    self.line_ending_bytes = 1;
                }
            } else {
                self.frame_bytes += self.line_ending_bytes + 1;
                self.line_ending_bytes = 0;
                if self.frame_bytes > MAX_SSE_FRAME_BYTES {
                    return Err(format!(
                        "The browser stream emitted an SSE frame larger than {MAX_SSE_FRAME_BYTES} bytes."
                    ));
                }
                self.has_non_whitespace |= !byte.is_ascii_whitespace();
                self.line.push(byte);
            }
        }
        Ok(events)
    }

    fn complete_line(&mut self) -> Result<(), String> {
        let mut line = std::str::from_utf8(&self.line)
            .map_err(|_| "The browser stream emitted invalid UTF-8 SSE data.".to_string())?;
        if !self.first_line_seen {
            self.first_line_seen = true;
            line = line.strip_prefix('\u{feff}').unwrap_or(line);
        }
        // A field without a colon has an empty value, including `data`.
        let (field, value) = line.split_once(':').unwrap_or((line, ""));
        if field == "data" {
            if self.has_data {
                self.data.push('\n');
            }
            self.data.push_str(value.strip_prefix(' ').unwrap_or(value));
            self.has_data = true;
        }
        Ok(())
    }

    pub(crate) fn finish(&self) -> Result<(), String> {
        if self.has_non_whitespace {
            Err("The browser stream ended with an incomplete SSE frame.".to_string())
        } else {
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_all_line_endings_at_every_transport_boundary() {
        for ending in ["\n", "\r", "\r\n"] {
            for separator in ["\n", "\r", "\r\n"] {
                // A CR followed by LF is one ending, not an empty line.
                if ending == "\r" && separator == "\n" {
                    continue;
                }
                let stream = format!(
                    ": heartbeat{ending}event: ignored{ending}data: 🌸{ending}data: next{ending}{separator}data: [DONE]{ending}{separator}"
                );
                for split in 0..=stream.len() {
                    let mut decoder = BrowserSseDecoder::default();
                    let mut events = decoder.push(&stream.as_bytes()[..split]).unwrap();
                    events.extend(decoder.push(&stream.as_bytes()[split..]).unwrap());
                    assert_eq!(events, ["🌸\nnext", "[DONE]"], "split {split}: {stream:?}");
                    decoder.finish().unwrap();
                }
            }
        }
    }

    #[test]
    fn rejects_unterminated_done_and_preserves_only_one_optional_space() {
        for input in [b"data: [DONE]".as_slice(), b"data: [DONE]\n"] {
            let mut decoder = BrowserSseDecoder::default();
            assert!(decoder.push(input).unwrap().is_empty());
            assert!(decoder.finish().is_err());
        }
        let mut decoder = BrowserSseDecoder::default();
        assert_eq!(
            decoder
                .push("\u{feff}data:  spaced\n\n".as_bytes())
                .unwrap(),
            [" spaced"]
        );
    }

    #[test]
    fn preserves_empty_data_fields_and_ignores_comments() {
        let mut decoder = BrowserSseDecoder::default();
        assert_eq!(
            decoder
                .push(b": ignored\n\ndata:\ndata: value\ndata\n\ndata\n\n")
                .unwrap(),
            ["\nvalue\n", ""]
        );
        decoder.finish().unwrap();
    }

    #[test]
    fn fragmented_large_frames_and_many_small_frames_are_incremental() {
        let mut decoder = BrowserSseDecoder::default();
        decoder.push(b"data: ").unwrap();
        for _ in 0..16_384 {
            assert!(decoder.push(b"x").unwrap().is_empty());
        }
        assert_eq!(decoder.push(b"\n\n").unwrap(), ["x".repeat(16_384)]);
        let events = decoder
            .push("data: x\n\n".repeat(16_384).as_bytes())
            .unwrap();
        assert_eq!(events.len(), 16_384);
        assert!(events.iter().all(|event| event == "x"));
        decoder.finish().unwrap();
    }

    #[test]
    fn enforces_the_frame_limit_across_chunks_including_ignored_fields() {
        let mut decoder = BrowserSseDecoder::default();
        decoder.push(&vec![b':'; MAX_SSE_FRAME_BYTES]).unwrap();
        assert!(decoder.push(b"x").is_err());

        let mut decoder = BrowserSseDecoder::default();
        decoder.push(&vec![b':'; MAX_SSE_FRAME_BYTES]).unwrap();
        assert!(decoder.push(b"\r\n\r\n").unwrap().is_empty());
        decoder.finish().unwrap();
        assert_eq!(decoder.push(b"data: ok\n\n").unwrap(), ["ok"]);

        let mut decoder = BrowserSseDecoder::default();
        decoder.push(&vec![b':'; MAX_SSE_FRAME_BYTES - 1]).unwrap();
        assert!(decoder.push(b"\r\nx").is_err());
    }

    #[test]
    fn rejects_invalid_utf8_and_incomplete_events() {
        let mut decoder = BrowserSseDecoder::default();
        assert!(decoder.push(b"data: \xff\n\n").is_err());
        for bytes in [
            b"data: unfinished".as_slice(),
            b"data: unfinished\n",
            b": unfinished\r",
        ] {
            let mut decoder = BrowserSseDecoder::default();
            decoder.push(bytes).unwrap();
            assert!(decoder.finish().is_err());
        }
    }
}
