//! Memory telemetry for GPU/RAM monitoring.
//!
//! Provides `MemoryTelemetry` to track VRAM and RAM usage across the inference
//! lifecycle, useful for monitoring, OOM prevention, and metrics reporting.

use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicU64, Ordering};

static PEAK_RAM_BYTES: AtomicU64 = AtomicU64::new(0);

/// Snapshot of memory usage at a point in time.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MemoryTelemetry {
    /// Total VRAM available on the device (bytes).
    pub total_vram: u64,
    /// Currently used VRAM (bytes).
    pub used_vram: u64,
    /// Total system RAM available (bytes).
    pub total_ram: u64,
    /// Currently used system RAM (bytes, RSS).
    pub used_ram: u64,
    /// Peak VRAM observed during this session (bytes).
    pub peak_vram: u64,
    /// Peak RAM observed during this session (bytes).
    pub peak_ram: u64,
    /// Device name (e.g. "Apple M1 Pro", "NVIDIA A100").
    pub device_name: String,
}

impl MemoryTelemetry {
    /// Create a new empty telemetry snapshot.
    pub fn new() -> Self {
        Self::default()
    }

    /// Refresh RAM usage from the current process.
    ///
    /// On macOS/Linux, reads from `/proc/self/status` or uses `mach_task_basic_info`.
    /// Falls back to 0 if the platform is not supported.
    pub fn refresh_ram(&mut self) {
        self.used_ram = current_rss_bytes();
        self.total_ram = total_system_ram();
        self.peak_ram = update_peak_ram(self.used_ram);
    }

    /// Set VRAM values from an external source (e.g., CUDA API).
    pub fn set_vram(&mut self, total: u64, used: u64) {
        self.total_vram = total;
        self.used_vram = used;
        if used > self.peak_vram {
            self.peak_vram = used;
        }
    }

    /// VRAM utilization ratio (0.0 to 1.0).
    pub fn vram_utilization(&self) -> f64 {
        if self.total_vram > 0 {
            self.used_vram as f64 / self.total_vram as f64
        } else {
            0.0
        }
    }

    /// RAM utilization ratio (0.0 to 1.0).
    pub fn ram_utilization(&self) -> f64 {
        if self.total_ram > 0 {
            self.used_ram as f64 / self.total_ram as f64
        } else {
            0.0
        }
    }

    /// Available VRAM headroom in bytes.
    pub fn vram_headroom(&self) -> u64 {
        self.total_vram.saturating_sub(self.used_vram)
    }

    /// Available RAM headroom in bytes.
    pub fn ram_headroom(&self) -> u64 {
        self.total_ram.saturating_sub(self.used_ram)
    }

    /// Whether memory pressure is high (>85% utilization on either VRAM or RAM).
    pub fn is_high_pressure(&self) -> bool {
        self.vram_utilization() > 0.85 || self.ram_utilization() > 0.85
    }

    /// Human-readable summary.
    pub fn summary(&self) -> String {
        format!(
            "VRAM: {}/{} ({:.1}%), RAM: {}/{} ({:.1}%), device: {}",
            format_bytes_short(self.used_vram),
            format_bytes_short(self.total_vram),
            self.vram_utilization() * 100.0,
            format_bytes_short(self.used_ram),
            format_bytes_short(self.total_ram),
            self.ram_utilization() * 100.0,
            if self.device_name.is_empty() {
                "unknown"
            } else {
                &self.device_name
            }
        )
    }
}

fn format_bytes_short(bytes: u64) -> String {
    const KB: f64 = 1024.0;
    const MB: f64 = KB * 1024.0;
    const GB: f64 = MB * 1024.0;
    let b = bytes as f64;
    if b >= GB {
        format!("{:.1}GB", b / GB)
    } else if b >= MB {
        format!("{:.0}MB", b / MB)
    } else if b >= KB {
        format!("{:.0}KB", b / KB)
    } else {
        format!("{}B", bytes)
    }
}

/// Get the current process's RSS (Resident Set Size) in bytes.
fn current_rss_bytes() -> u64 {
    // Linux: read from /proc/self/status
    #[cfg(target_os = "linux")]
    {
        if let Ok(contents) = std::fs::read_to_string("/proc/self/status") {
            for line in contents.lines() {
                if line.starts_with("VmRSS:") {
                    let parts: Vec<&str> = line.split_whitespace().collect();
                    if parts.len() >= 2 {
                        if let Ok(kb) = parts[1].parse::<u64>() {
                            return kb * 1024;
                        }
                    }
                }
            }
        }
    }

    #[cfg(target_os = "macos")]
    {
        let pid = std::process::id().to_string();
        if let Ok(output) = std::process::Command::new("ps")
            .args(["-o", "rss=", "-p", &pid])
            .output()
        {
            if output.status.success() {
                if let Ok(s) = String::from_utf8(output.stdout) {
                    if let Ok(kb) = s.trim().parse::<u64>() {
                        return kb * 1024;
                    }
                }
            }
        }
    }

    0
}

fn update_peak_ram(used_ram: u64) -> u64 {
    let mut current = PEAK_RAM_BYTES.load(Ordering::Relaxed);
    while used_ram > current {
        match PEAK_RAM_BYTES.compare_exchange_weak(
            current,
            used_ram,
            Ordering::Relaxed,
            Ordering::Relaxed,
        ) {
            Ok(_) => return used_ram,
            Err(observed) => current = observed,
        }
    }
    current
}

/// Get total system RAM in bytes.
fn total_system_ram() -> u64 {
    #[cfg(target_os = "linux")]
    {
        if let Ok(contents) = std::fs::read_to_string("/proc/meminfo") {
            for line in contents.lines() {
                if line.starts_with("MemTotal:") {
                    let parts: Vec<&str> = line.split_whitespace().collect();
                    if parts.len() >= 2 {
                        if let Ok(kb) = parts[1].parse::<u64>() {
                            return kb * 1024;
                        }
                    }
                }
            }
        }
    }

    #[cfg(target_os = "macos")]
    {
        if let Ok(output) = std::process::Command::new("sysctl")
            .arg("-n")
            .arg("hw.memsize")
            .output()
        {
            if let Ok(s) = String::from_utf8(output.stdout) {
                if let Ok(total) = s.trim().parse::<u64>() {
                    return total;
                }
            }
        }
    }

    0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_telemetry_defaults() {
        let t = MemoryTelemetry::new();
        assert_eq!(t.total_vram, 0);
        assert_eq!(t.used_vram, 0);
        assert!(!t.is_high_pressure());
        assert_eq!(t.vram_utilization(), 0.0);
    }

    #[test]
    fn test_memory_telemetry_vram() {
        let mut t = MemoryTelemetry::new();
        t.set_vram(16 * 1024 * 1024 * 1024, 8 * 1024 * 1024 * 1024);
        assert!((t.vram_utilization() - 0.5).abs() < f64::EPSILON);
        assert_eq!(t.vram_headroom(), 8 * 1024 * 1024 * 1024);
        assert!(!t.is_high_pressure());
        assert_eq!(t.peak_vram, 8 * 1024 * 1024 * 1024);

        // Update with higher usage
        t.set_vram(16 * 1024 * 1024 * 1024, 15 * 1024 * 1024 * 1024);
        assert!(t.is_high_pressure());
        assert_eq!(t.peak_vram, 15 * 1024 * 1024 * 1024);
    }

    #[test]
    fn test_format_bytes_short() {
        assert_eq!(format_bytes_short(512), "512B");
        assert_eq!(format_bytes_short(2048), "2KB");
        assert_eq!(format_bytes_short(128 * 1024 * 1024), "128MB");
        assert_eq!(format_bytes_short(4 * 1024 * 1024 * 1024), "4.0GB");
    }

    #[test]
    fn test_summary() {
        let mut t = MemoryTelemetry::new();
        t.device_name = "Test GPU".to_string();
        t.set_vram(8_000_000_000, 4_000_000_000);
        let s = t.summary();
        assert!(s.contains("Test GPU"));
        assert!(s.contains("50.0%"));
    }

    #[test]
    fn test_ram_peak_is_process_wide() {
        let baseline = update_peak_ram(1024);
        assert!(baseline >= 1024);
        let higher = update_peak_ram(baseline + 2048);
        assert_eq!(higher, baseline + 2048);
        let lower = update_peak_ram(512);
        assert_eq!(lower, higher);
    }
}
