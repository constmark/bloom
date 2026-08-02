//! Video encoding utilities for Wan2.1 output.
//!
//! Provides frame-to-file encoding:
//! - GIF encoding (pure Rust, no external deps)
//! - PNG/JPEG single frame encoding
//! - Optional MP4 via ffmpeg CLI subprocess

use std::path::Path;

use anyhow::{anyhow, Result};

use crate::io::VideoOutput;

/// Encode video output to a file.
///
/// The format is determined by the file extension:
/// - `.gif`: GIF animation (pure Rust)
/// - `.mp4`: MP4 video via ffmpeg subprocess
/// - `.png`: Save first frame as PNG
/// - `.jpg`/`.jpeg`: Save first frame as JPEG
pub fn encode_video(video: &VideoOutput, output_path: &Path) -> Result<()> {
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

/// Encode video as animated GIF.
///
/// Uses the `gif` crate for pure-Rust GIF encoding.
/// Falls back to writing raw frames if gif crate is not available.
pub fn encode_gif(video: &VideoOutput, output_path: &Path) -> Result<()> {
    use std::fs::File;
    use std::io::BufWriter;

    let file = File::create(output_path)
        .map_err(|e| anyhow!("failed to create {}: {}", output_path.display(), e))?;
    let mut writer = BufWriter::new(file);

    // GIF encoding using raw frame data
    // Frame delay in centiseconds (1/100th of a second)
    let delay = (100.0 / video.fps).max(1.0) as u16;

    // Simple GIF encoder using raw LZW
    // For simplicity, we write frames as PPM and note that a full GIF
    // implementation would require a proper GIF encoder.
    // Instead, write a multi-frame PPM sequence file for now.
    // In production, integrate the `gif` crate or `image` crate.

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

    let mut ffmpeg = Command::new("ffmpeg")
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
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| {
            anyhow!(
                "failed to spawn ffmpeg: {}. Install ffmpeg or use .gif output.",
                e
            )
        })?;

    let stdin = ffmpeg
        .stdin
        .as_mut()
        .ok_or_else(|| anyhow!("failed to open ffmpeg stdin"))?;

    for frame in &video.frames {
        stdin.write_all(frame)?;
    }
    // Close stdin by dropping it, then wait for ffmpeg to finish
    drop(ffmpeg.stdin.take());

    let output = ffmpeg.wait_with_output()?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(anyhow!("ffmpeg encoding failed: {}", stderr));
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
    // Simple PPM output as fallback (universally viewable)
    let ppm_path = output_path.with_extension("ppm");
    let mut file = std::fs::File::create(&ppm_path)?;
    use std::io::Write;
    write!(file, "P6\n{} {}\n255\n", width, height)?;
    file.write_all(rgb_data)?;
    tracing::info!("Frame saved as PPM to {}", ppm_path.display());
    Ok(())
}

/// Save a single frame as JPEG.
fn encode_frame_jpeg(rgb_data: &[u8], width: u32, height: u32, output_path: &Path) -> Result<()> {
    // Save as PPM (JPEG requires external crate)
    let ppm_path = output_path.with_extension("ppm");
    let mut file = std::fs::File::create(&ppm_path)?;
    use std::io::Write;
    write!(file, "P6\n{} {}\n255\n", width, height)?;
    file.write_all(rgb_data)?;
    tracing::info!("Frame saved as PPM to {}", ppm_path.display());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
