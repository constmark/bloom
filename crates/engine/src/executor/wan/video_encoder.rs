//! Video encoding utilities for Wan2.1 output.
//!
//! Provides frame-to-file encoding:
//! - GIF encoding (pure Rust, no external deps)
//! - PNG/JPEG single frame encoding
//! - Optional MP4 via ffmpeg CLI subprocess

use std::path::Path;

use anyhow::{Result, anyhow};

use crate::io::VideoOutput;

const MAX_FFMPEG_STDERR_BYTES: usize = 64 * 1024;

/// Encode video output to a file.
///
/// The format is determined by the file extension:
/// - `.gif`: GIF animation (pure Rust)
/// - `.mp4`: MP4 video via ffmpeg subprocess
/// - `.png`: Save first frame as PNG
/// - `.jpg`/`.jpeg`: Save first frame as JPEG
pub fn encode_video(video: &VideoOutput, output_path: &Path) -> Result<()> {
    validate_video(video)?;
    let ext = output_path
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or("gif")
        .to_lowercase();

    match ext.as_str() {
        "gif" => encode_gif(video, output_path),
        "mp4" | "mov" | "avi" | "mkv" => encode_video_ffmpeg(video, output_path),
        "png" => encode_frame_png(&video.frames[0], video.width, video.height, output_path),
        "jpg" | "jpeg" => {
            encode_frame_jpeg(&video.frames[0], video.width, video.height, output_path)
        }
        other => Err(anyhow!("unsupported output format: .{}", other)),
    }
}

fn expected_frame_len(width: u32, height: u32) -> Result<usize> {
    let pixels = u64::from(width)
        .checked_mul(u64::from(height))
        .ok_or_else(|| anyhow!("video dimensions overflow the pixel count"))?;
    let bytes = pixels
        .checked_mul(3)
        .ok_or_else(|| anyhow!("video dimensions overflow the RGB byte count"))?;
    usize::try_from(bytes).map_err(|_| anyhow!("video frame is too large for this platform"))
}

fn validate_video(video: &VideoOutput) -> Result<usize> {
    if video.width == 0 || video.height == 0 {
        return Err(anyhow!("video dimensions must be greater than zero"));
    }
    if !video.fps.is_finite() || video.fps <= 0.0 {
        return Err(anyhow!("video fps must be finite and greater than zero"));
    }
    if video.frames.is_empty() {
        return Err(anyhow!("video must contain at least one frame"));
    }
    let actual_frame_count = u32::try_from(video.frames.len())
        .map_err(|_| anyhow!("video contains more frames than the metadata can represent"))?;
    if video.frame_count != actual_frame_count {
        return Err(anyhow!(
            "video frame_count metadata ({}) does not match {} encoded frames",
            video.frame_count,
            actual_frame_count
        ));
    }
    let expected_len = expected_frame_len(video.width, video.height)?;
    for (index, frame) in video.frames.iter().enumerate() {
        if frame.len() != expected_len {
            return Err(anyhow!(
                "video frame {index} has {} bytes; expected {expected_len} for {}x{} RGB",
                frame.len(),
                video.width,
                video.height
            ));
        }
    }
    Ok(expected_len)
}

/// Encode video as animated GIF.
///
/// Uses the in-tree pure-Rust GIF encoder.
pub fn encode_gif(video: &VideoOutput, output_path: &Path) -> Result<()> {
    use std::fs::File;
    use std::io::BufWriter;

    validate_video(video)?;
    if video.width > u32::from(u16::MAX) || video.height > u32::from(u16::MAX) {
        return Err(anyhow!("GIF dimensions cannot exceed 65535 pixels"));
    }

    let file = File::create(output_path)
        .map_err(|e| anyhow!("failed to create {}: {}", output_path.display(), e))?;
    let mut writer = BufWriter::new(file);

    // GIF encoding using raw frame data
    // Frame delay in centiseconds (1/100th of a second)
    let delay = (100.0 / video.fps).max(1.0) as u16;

    let mut gif_encoder =
        SimpleGifEncoder::new(&mut writer, video.width as u16, video.height as u16)?;

    for frame in &video.frames {
        gif_encoder.write_frame(frame, delay)?;
    }

    gif_encoder.finish()?;

    tracing::info!(
        "GIF saved to {} ({} frames, {}x{})",
        output_path.display(),
        video.frames.len(),
        video.width,
        video.height
    );

    Ok(())
}

/// Minimal GIF89a encoder (uncompressed).
///
/// This is a simplified GIF writer that produces valid GIF89a output.
/// For production quality, replace with the `gif` crate.
struct SimpleGifEncoder<'a> {
    writer: &'a mut dyn std::io::Write,
    width: u16,
    height: u16,
    frame_count: usize,
}

impl<'a> SimpleGifEncoder<'a> {
    fn new(writer: &'a mut dyn std::io::Write, width: u16, height: u16) -> Result<Self> {
        // GIF89a header
        writer.write_all(b"GIF89a")?;

        // Logical Screen Descriptor
        writer.write_all(&width.to_le_bytes())?;
        writer.write_all(&height.to_le_bytes())?;
        // Global Color Table: 256 colors, 8 bits per primary color
        writer.write_all(&[0xF7, 0x00, 0x00])?; // packed: GCT flag=1, color res=7, sort=0, GCT size=7 (256)

        // Global Color Table (256 * 3 = 768 bytes)
        // Generate a 6x6x6 color cube + padding
        let mut color_table = vec![0u8; 768];
        for i in 0..256 {
            let r = ((i / 36) % 6) * 51;
            let g = ((i / 6) % 6) * 51;
            let b = (i % 6) * 51;
            color_table[i * 3] = r as u8;
            color_table[i * 3 + 1] = g as u8;
            color_table[i * 3 + 2] = b as u8;
        }
        writer.write_all(&color_table)?;

        // Netscape extension for animation loop
        writer.write_all(&[
            0x21, 0xFF, 0x0B, // Extension, App Extension, Block size
        ])?;
        writer.write_all(b"NETSCAPE2.0")?;
        writer.write_all(&[0x03, 0x01, 0x00, 0x00, 0x00])?; // Sub-block: loop count = 0 (infinite)

        Ok(Self {
            writer,
            width,
            height,
            frame_count: 0,
        })
    }

    fn write_frame(&mut self, rgb_data: &[u8], delay_cs: u16) -> Result<()> {
        // Graphic Control Extension
        self.writer.write_all(&[
            0x21, 0xF9, 0x04, // Extension, GCE, block size
            0x00, // No transparency, no disposal
        ])?;
        self.writer.write_all(&delay_cs.to_le_bytes())?;
        self.writer.write_all(&[0x00, 0x00])?; // Transparent color index, terminator

        // Image Descriptor
        self.writer.write_all(&[0x2C])?; // Image separator
        self.writer.write_all(&[0x00, 0x00])?; // Left
        self.writer.write_all(&[0x00, 0x00])?; // Top
        self.writer.write_all(&self.width.to_le_bytes())?;
        self.writer.write_all(&self.height.to_le_bytes())?;
        self.writer.write_all(&[0x00])?; // No local color table

        // Image Data (LZW compressed)
        let min_code_size: u8 = 8;
        self.writer.write_all(&[min_code_size])?;

        // Quantize RGB to color indices
        let pixel_count = (self.width as usize) * (self.height as usize);
        let mut indices = Vec::with_capacity(pixel_count);
        for i in 0..pixel_count {
            let r = rgb_data[i * 3] as f32;
            let g = rgb_data[i * 3 + 1] as f32;
            let b = rgb_data[i * 3 + 2] as f32;
            // Map to 6x6x6 color cube index
            let ri = (r / 51.0).min(5.0) as u8;
            let gi = (g / 51.0).min(5.0) as u8;
            let bi = (b / 51.0).min(5.0) as u8;
            indices.push(ri * 36 + gi * 6 + bi);
        }

        // Simple LZW compression
        let compressed = lzw_compress(&indices, min_code_size)?;

        // Write in sub-blocks (max 255 bytes each)
        let mut offset = 0;
        while offset < compressed.len() {
            let chunk_size = (compressed.len() - offset).min(255);
            self.writer.write_all(&[chunk_size as u8])?;
            self.writer
                .write_all(&compressed[offset..offset + chunk_size])?;
            offset += chunk_size;
        }
        self.writer.write_all(&[0x00])?; // Block terminator

        self.frame_count += 1;
        Ok(())
    }

    fn finish(&mut self) -> Result<()> {
        self.writer.write_all(&[0x3B])?; // GIF trailer
        self.writer.flush()?;
        Ok(())
    }
}

/// Minimal LZW compression for GIF.
fn lzw_compress(indices: &[u8], min_code_size: u8) -> Result<Vec<u8>> {
    // Simplified LZW: use clear codes frequently to keep it simple
    let clear_code = 1u16 << min_code_size;
    let eoi_code = clear_code + 1;

    let mut output_bits: Vec<bool> = Vec::new();
    let mut code_size = (min_code_size as usize) + 1;
    let mut next_code = eoi_code + 1;

    // Write clear code
    write_bits(&mut output_bits, clear_code, code_size);

    if indices.is_empty() {
        write_bits(&mut output_bits, eoi_code, code_size);
        return Ok(bits_to_bytes(&output_bits));
    }

    // Simple table-based LZW
    let mut table: std::collections::HashMap<Vec<u8>, u16> = std::collections::HashMap::new();
    for i in 0..clear_code {
        table.insert(vec![i as u8], i);
    }

    let mut current: Vec<u8> = vec![indices[0]];

    for &byte in &indices[1..] {
        let mut extended = current.clone();
        extended.push(byte);

        if table.contains_key(&extended) {
            current = extended;
        } else {
            // Output code for current
            let code = *table
                .get(&current)
                .ok_or_else(|| anyhow!("GIF LZW dictionary lost its current sequence"))?;
            write_bits(&mut output_bits, code, code_size);

            // Add new entry
            if next_code < 4096 {
                table.insert(extended, next_code);
                if next_code >= (1 << code_size) && code_size < 12 {
                    code_size += 1;
                }
                next_code += 1;
            } else {
                // Table full, emit clear code and reset
                write_bits(&mut output_bits, clear_code, code_size);
                code_size = (min_code_size as usize) + 1;
                next_code = eoi_code + 1;
                table.clear();
                for i in 0..clear_code {
                    table.insert(vec![i as u8], i);
                }
            }

            current = vec![byte];
        }
    }

    // Output remaining
    let code = *table
        .get(&current)
        .ok_or_else(|| anyhow!("GIF LZW dictionary lost its final sequence"))?;
    write_bits(&mut output_bits, code, code_size);
    write_bits(&mut output_bits, eoi_code, code_size);

    Ok(bits_to_bytes(&output_bits))
}

fn write_bits(bits: &mut Vec<bool>, value: u16, num_bits: usize) {
    for i in 0..num_bits {
        bits.push((value >> i) & 1 == 1);
    }
}

fn bits_to_bytes(bits: &[bool]) -> Vec<u8> {
    let mut bytes = Vec::new();
    let mut byte = 0u8;
    let mut bit_idx = 0;
    for &bit in bits {
        if bit {
            byte |= 1 << bit_idx;
        }
        bit_idx += 1;
        if bit_idx == 8 {
            bytes.push(byte);
            byte = 0;
            bit_idx = 0;
        }
    }
    if bit_idx > 0 {
        bytes.push(byte);
    }
    bytes
}

/// Encode video via ffmpeg subprocess.
///
/// Writes raw frames to a pipe and lets ffmpeg encode to MP4 (H.264).
pub fn encode_video_ffmpeg(video: &VideoOutput, output_path: &Path) -> Result<()> {
    use std::io::Write;
    use std::process::{Command, Stdio};
    use std::thread;

    use crate::core::process::{ChildGuard, read_bounded};

    validate_video(video)?;

    let mut command = Command::new("ffmpeg");
    command
        .args([
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            &format!("{}x{}", video.width, video.height),
            "-r",
            &format!("{}", video.fps),
            "-i",
            "pipe:0",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-preset",
            "medium",
            "-crf",
            "23",
            &output_path.to_string_lossy(),
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::piped());
    let mut ffmpeg = ChildGuard::spawn(&mut command).map_err(|e| {
        anyhow!(
            "failed to spawn ffmpeg: {}. Install ffmpeg or use .gif output.",
            e
        )
    })?;

    // Drain diagnostics concurrently with frame writes. ffmpeg can emit
    // enough progress/error output to fill a pipe before it consumes all
    // frames; waiting until stdin is closed would otherwise deadlock.
    let stderr = ffmpeg
        .stderr
        .take()
        .ok_or_else(|| anyhow!("ffmpeg stderr is unavailable"))?;
    let stderr_reader = thread::spawn(move || read_bounded(stderr, MAX_FFMPEG_STDERR_BYTES));

    let write_result = (|| -> Result<()> {
        let stdin = ffmpeg
            .stdin
            .as_mut()
            .ok_or_else(|| anyhow!("failed to open ffmpeg stdin"))?;
        for frame in &video.frames {
            stdin.write_all(frame)?;
        }
        Ok(())
    })();
    drop(ffmpeg.stdin.take());

    if let Err(error) = write_result {
        // Stop the worker before joining the reader: a failed stdin write can
        // leave ffmpeg alive indefinitely while its stderr remains open.
        let _ = ffmpeg.kill();
        let _ = ffmpeg.wait();
        let _ = stderr_reader.join();
        return Err(error);
    }
    let (stderr_bytes, stderr_exceeded) = stderr_reader
        .join()
        .map_err(|_| anyhow!("ffmpeg stderr reader panicked"))??;
    let status = ffmpeg.wait()?;
    if !status.success() {
        let stderr = String::from_utf8_lossy(&stderr_bytes);
        let suffix = if stderr_exceeded {
            format!(" (diagnostics truncated at {MAX_FFMPEG_STDERR_BYTES} bytes)")
        } else {
            String::new()
        };
        return Err(anyhow!("ffmpeg encoding failed: {stderr}{suffix}"));
    }

    tracing::info!(
        "Video saved to {} ({} frames, {}x{})",
        output_path.display(),
        video.frames.len(),
        video.width,
        video.height
    );

    Ok(())
}

/// Save a single frame as PNG.
fn encode_frame_png(rgb_data: &[u8], width: u32, height: u32, output_path: &Path) -> Result<()> {
    let expected_len = expected_frame_len(width, height)?;
    if rgb_data.len() != expected_len {
        return Err(anyhow!(
            "RGB frame has {} bytes; expected {expected_len} for {width}x{height}",
            rgb_data.len()
        ));
    }
    let image = image::RgbImage::from_raw(width, height, rgb_data.to_vec())
        .ok_or_else(|| anyhow!("failed to construct RGB image from frame data"))?;
    image.save_with_format(output_path, image::ImageFormat::Png)?;
    tracing::info!("Frame saved as PNG to {}", output_path.display());
    Ok(())
}

/// Save a single frame as JPEG.
fn encode_frame_jpeg(rgb_data: &[u8], width: u32, height: u32, output_path: &Path) -> Result<()> {
    let expected_len = expected_frame_len(width, height)?;
    if rgb_data.len() != expected_len {
        return Err(anyhow!(
            "RGB frame has {} bytes; expected {expected_len} for {width}x{height}",
            rgb_data.len()
        ));
    }
    let image = image::RgbImage::from_raw(width, height, rgb_data.to_vec())
        .ok_or_else(|| anyhow!("failed to construct RGB image from frame data"))?;
    image.save_with_format(output_path, image::ImageFormat::Jpeg)?;
    tracing::info!("Frame saved as JPEG to {}", output_path.display());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::GenericImageView;

    #[test]
    fn test_lzw_compress_empty() {
        let result = lzw_compress(&[], 8).expect("empty input should produce a valid LZW stream");
        assert!(!result.is_empty()); // At least clear + EOI codes
    }

    #[test]
    fn test_lzw_compress_simple() {
        let data = vec![0u8, 0, 0, 0, 1, 1, 1, 1];
        let result = lzw_compress(&data, 8).expect("sample input should compress");
        assert!(!result.is_empty());
    }

    #[test]
    fn test_bits_to_bytes() {
        let bits = vec![true, false, true, false, true, false, true, false];
        let bytes = bits_to_bytes(&bits);
        assert_eq!(bytes, vec![0b01010101]);
    }

    fn test_video() -> VideoOutput {
        VideoOutput {
            width: 2,
            height: 1,
            fps: 24.0,
            frame_count: 1,
            frames: vec![vec![255, 0, 0, 0, 255, 0]],
        }
    }

    #[test]
    fn png_and_jpeg_outputs_are_real_image_files() {
        let dir = tempfile::tempdir().unwrap();
        let png = dir.path().join("frame.png");
        let jpeg = dir.path().join("frame.jpg");
        let video = test_video();

        encode_video(&video, &png).unwrap();
        encode_video(&video, &jpeg).unwrap();

        assert_eq!(image::open(&png).unwrap().dimensions(), (2, 1));
        assert_eq!(image::open(&jpeg).unwrap().dimensions(), (2, 1));
        assert!(png.exists());
        assert!(jpeg.exists());
        assert!(!dir.path().join("frame.ppm").exists());
    }

    #[test]
    fn invalid_video_inputs_fail_without_panicking() {
        let dir = tempfile::tempdir().unwrap();
        let output = dir.path().join("frame.png");
        let mut video = test_video();

        video.frames.clear();
        assert!(encode_video(&video, &output).is_err());

        video = test_video();
        video.frames[0].pop();
        assert!(encode_video(&video, &output).is_err());

        video = test_video();
        video.fps = f32::NAN;
        assert!(encode_video(&video, &output).is_err());

        video = test_video();
        video.frame_count = 2;
        assert!(encode_video(&video, &output).is_err());
    }
}
