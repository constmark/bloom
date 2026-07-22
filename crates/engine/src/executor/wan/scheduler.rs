//! Flow Matching scheduler for Wan2.1 diffusion sampling.
//!
//! Implements UniPC multi-step scheduler with flow-matching sigma schedule.

use candle_core::{Result, Tensor};

/// Generate sampling sigmas for flow matching.
///
/// Creates a linearly-spaced sigma schedule shifted by `shift`.
/// sigmas = shift * t / (1 + (shift - 1) * t)
/// where t is linearly spaced from 1 to 1/num_steps.
pub fn get_sampling_sigmas(num_steps: u32, shift: f64) -> Vec<f64> {
    let sigmas: Vec<f64> = (0..=num_steps)
        .map(|i| {
            let t = 1.0 - (i as f64) / (num_steps as f64);
            if shift == 1.0 {
                t
            } else {
                shift * t / (1.0 + (shift - 1.0) * t)
            }
        })
        .collect();
    sigmas
}

/// Retrieve timesteps from sigmas (scale by num_train_timesteps).
pub fn retrieve_timesteps(sigmas: &[f64], num_train_timesteps: u32) -> Vec<f64> {
    sigmas
        .iter()
        .map(|s| s * num_train_timesteps as f64)
        .collect()
}

/// Flow matching UniPC multi-step scheduler.
///
/// Simplified implementation that uses Euler steps with flow matching.
/// For production quality, the full UniPC polynomial update should be used.
pub struct FlowUniPCScheduler {
    pub num_train_timesteps: u32,
    pub timesteps: Vec<f64>,
    pub sigmas: Vec<f64>,
    /// Previous model outputs for multi-step (stores last few predictions).
    model_outputs: Vec<Tensor>,
}

impl FlowUniPCScheduler {
    pub fn new(num_train_timesteps: u32, _shift: f64, use_dynamic_shifting: bool) -> Self {
        let _ = use_dynamic_shifting;
        Self {
            num_train_timesteps,
            timesteps: Vec::new(),
            sigmas: Vec::new(),
            model_outputs: Vec::new(),
        }
    }

    /// Set timesteps for sampling.
    pub fn set_timesteps(&mut self, num_steps: u32, shift: f64) {
        self.sigmas = get_sampling_sigmas(num_steps, shift);
        self.timesteps = retrieve_timesteps(&self.sigmas, self.num_train_timesteps);
        self.model_outputs.clear();
    }

    /// Single denoising step.
    ///
    /// model_output: predicted noise [batch, channels, ...]
    /// timestep: current timestep value
    /// sample: current noisy sample [batch, channels, ...]
    ///
    /// Returns: denoised sample.
    ///
    /// This implements a simplified Euler step for flow matching:
    /// x_{t-1} = x_t + (sigma_{t-1} - sigma_t) * v_pred
    /// where v_pred is the predicted velocity (noise - clean).
    ///
    /// For higher quality, replace with full UniPC polynomial update.
    pub fn step(
        &mut self,
        model_output: &Tensor,
        step_index: usize,
        sample: &Tensor,
    ) -> Result<Tensor> {
        if step_index + 1 >= self.sigmas.len() {
            return Ok(sample.clone());
        }

        let sigma_t = self.sigmas[step_index];
        let sigma_next = self.sigmas[step_index + 1];

        // Flow matching step: x_{t-1} = x_t + (sigma_next - sigma_t) * v_pred
        // In flow matching, the model predicts v = x_0 - noise
        // The step is: x_{t-1} = x_t + dt * v where dt = sigma_next - sigma_t
        let dt = sigma_next - sigma_t;
        sample.affine(1.0, dt)?.add(model_output)
    }

    /// Number of timesteps.
    pub fn num_steps(&self) -> usize {
        self.timesteps.len().saturating_sub(1)
    }
}

/// Flow matching DPM++ multi-step scheduler.
pub struct FlowDPMPlusPlusScheduler {
    pub num_train_timesteps: u32,
    pub timesteps: Vec<f64>,
    pub sigmas: Vec<f64>,
    model_outputs: Vec<Tensor>,
}

impl FlowDPMPlusPlusScheduler {
    pub fn new(num_train_timesteps: u32, _shift: f64, use_dynamic_shifting: bool) -> Self {
        let _ = use_dynamic_shifting;
        Self {
            num_train_timesteps,
            timesteps: Vec::new(),
            sigmas: Vec::new(),
            model_outputs: Vec::new(),
        }
    }

    pub fn set_timesteps(&mut self, num_steps: u32, shift: f64) {
        self.sigmas = get_sampling_sigmas(num_steps, shift);
        self.timesteps = retrieve_timesteps(&self.sigmas, self.num_train_timesteps);
        self.model_outputs.clear();
    }

    /// DPM++ single step (simplified as Euler for flow matching).
    pub fn step(
        &mut self,
        model_output: &Tensor,
        step_index: usize,
        sample: &Tensor,
    ) -> Result<Tensor> {
        if step_index + 1 >= self.sigmas.len() {
            return Ok(sample.clone());
        }

        let sigma_t = self.sigmas[step_index];
        let sigma_next = self.sigmas[step_index + 1];
        let dt = sigma_next - sigma_t;
        sample.affine(1.0, dt)?.add(model_output)
    }

    pub fn num_steps(&self) -> usize {
        self.timesteps.len().saturating_sub(1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_sampling_sigmas() {
        let sigmas = get_sampling_sigmas(50, 5.0);
        assert_eq!(sigmas.len(), 51); // num_steps + 1
        assert!(sigmas[0] > 0.9); // first sigma close to 1.0
        assert!(sigmas.last().unwrap() < &0.01); // last sigma close to 0
    }

    #[test]
    fn test_sigmas_monotonic_decrease() {
        let sigmas = get_sampling_sigmas(50, 5.0);
        for i in 0..sigmas.len() - 1 {
            assert!(
                sigmas[i] > sigmas[i + 1],
                "sigmas should be monotonically decreasing"
            );
        }
    }

    #[test]
    fn test_retrieve_timesteps() {
        let sigmas = get_sampling_sigmas(10, 1.0);
        let timesteps = retrieve_timesteps(&sigmas, 1000);
        assert_eq!(timesteps.len(), 11);
        assert!((timesteps[0] - 1000.0).abs() < 1e-6);
    }

    #[test]
    fn test_unipc_scheduler_step() {
        let device = Device::Cpu;
        let mut scheduler = FlowUniPCScheduler::new(1000, 5.0, false);
        scheduler.set_timesteps(10, 5.0);

        let sample = Tensor::randn(0f32, 1.0, (1, 16, 4, 4, 4), &device).unwrap();
        let model_output = Tensor::zeros_like(&sample).unwrap();

        let result = scheduler.step(&model_output, 0, &sample).unwrap();
        assert_eq!(result.dims(), sample.dims());
    }

    #[test]
    fn test_scheduler_shift_effect() {
        let sigmas_no_shift = get_sampling_sigmas(50, 1.0);
        let sigmas_shifted = get_sampling_sigmas(50, 5.0);

        // With shift > 1, middle sigmas should be larger (slower initial denoising)
        let mid = 25;
        assert!(sigmas_shifted[mid] > sigmas_no_shift[mid]);
    }
}
