use std::f64::consts::PI;
use tch::Tensor;
use tch::{kind::Kind, Device};

#[derive(Debug)]
pub struct Fade {
    fade_in_len: i64,
    fade_out_len: i64,
    fade_shape: String,
}

impl Fade {
    pub fn new(fade_in_len: i64, fade_out_len: i64, fade_shape: &str) -> Self {
        Self {
            fade_in_len,
            fade_out_len,
            fade_shape: fade_shape.to_string(),
        }
    }

    pub fn forward(&self, waveform: &Tensor) -> Tensor {
        // The time dimension is assumed to be the last one
        let wave_len = waveform.size()[waveform.dim() - 1];
        let device = waveform.device();
        let fade_in = self.fade_in(wave_len, device);
        let fade_out = self.fade_out(wave_len, device);
        waveform * &fade_in * &fade_out
    }

    fn fade_in(&self, waveform_length: i64, device: Device) -> Tensor {
        let fade_in_len = self.fade_in_len.min(waveform_length);
        let fade = Tensor::linspace(0., 1., fade_in_len, (Kind::Float, device));
        let ones = Tensor::ones(&[waveform_length - fade_in_len], (Kind::Float, device));

        // Apply the requested fade shape to the 0..1 ramp
        let shaped_fade = match self.fade_shape.as_str() {
            "exponential" => {
                // fade = 2^(fade - 1) * fade
                let two = Tensor::from(2.).to_device(device);
                two.pow(&(fade.shallow_clone() - 1.)) * &fade
            }
            "logarithmic" => {
                // fade = log10(0.1 + fade) + 1
                (&fade + 0.1).log10() + 1.
            }
            "quarter_sine" => {
                // fade = sin(fade * pi/2)
                (&fade * (PI / 2.)).sin()
            }
            "half_sine" => {
                // fade = sin(fade * pi - pi/2) / 2 + 0.5
                ((&fade * PI - (PI / 2.)).sin() / 2.) + 0.5
            }
            // "linear" or any unrecognised shape
            _ => fade,
        };

        // Concatenate and clamp
        Tensor::cat(&[&shaped_fade, &ones], 0).clamp(0., 1.)
    }

    fn fade_out(&self, waveform_length: i64, device: Device) -> Tensor {
        let fade_out_len = self.fade_out_len.min(waveform_length);
        let fade = Tensor::linspace(0., 1., fade_out_len, (Kind::Float, device));
        let ones = Tensor::ones(&[waveform_length - fade_out_len], (Kind::Float, device));

        // Apply the requested fade shape to the 0..1 ramp, then reverse it as needed
        let shaped_fade = match self.fade_shape.as_str() {
            "exponential" => {
                // fade = 2^(-fade) * (1 - fade)
                let two = Tensor::from(2.).to_device(device);
                two.pow(&(-&fade)) * (1. - &fade)
            }
            "logarithmic" => {
                // fade = log10(1.1 - fade) + 1
                (Tensor::from(1.1).to_device(device) - &fade).log10() + 1.
            }
            "quarter_sine" => {
                // fade = sin(fade * pi/2 + pi/2)
                ((&fade * (PI / 2.)) + (PI / 2.)).sin()
            }
            "half_sine" => {
                // fade = sin(fade * pi + pi/2)/2 + 0.5
                ((&fade * PI + (PI / 2.)).sin() / 2.) + 0.5
            }
            "linear" => {
                // fade = 1 - fade
                1. - fade
            }
            _ => 1. - fade,
        };

        // Concatenate [ones, shaped_fade] so that the fade is at the end
        let out_tensor = Tensor::cat(&[&ones, &shaped_fade], 0).clamp(0., 1.);
        out_tensor
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fade_linear() {
        let fade = Fade::new(10, 10, "linear");
        let waveform = Tensor::ones(&[1, 20], (Kind::Float, Device::Cpu));
        let result = fade.forward(&waveform);
        assert_eq!(result.size(), vec![1, 20]);
    }

    #[test]
    fn test_fade_exponential() {
        let fade = Fade::new(10, 10, "exponential");
        let waveform = Tensor::ones(&[1, 20], (Kind::Float, Device::Cpu));
        let result = fade.forward(&waveform);
        assert_eq!(result.size(), vec![1, 20]);
    }

    #[test]
    fn test_fade_logarithmic() {
        let fade = Fade::new(10, 10, "logarithmic");
        let waveform = Tensor::ones(&[1, 20], (Kind::Float, Device::Cpu));
        let result = fade.forward(&waveform);
        assert_eq!(result.size(), vec![1, 20]);
    }

    #[test]
    fn test_fade_quarter_sine() {
        let fade = Fade::new(10, 10, "quarter_sine");
        let waveform = Tensor::ones(&[1, 20], (Kind::Float, Device::Cpu));
        let result = fade.forward(&waveform);
        assert_eq!(result.size(), vec![1, 20]);
    }

    #[test]
    fn test_fade_half_sine() {
        let fade = Fade::new(10, 10, "half_sine");
        let waveform = Tensor::ones(&[1, 20], (Kind::Float, Device::Cpu));
        let result = fade.forward(&waveform);
        assert_eq!(result.size(), vec![1, 20]);
    }

    #[test]
    fn test_fade_in_length_greater_than_waveform_length() {
        let fade = Fade::new(30, 10, "linear");
        let waveform = Tensor::ones(&[1, 20], (Kind::Float, Device::Cpu));
        let result = fade.forward(&waveform);
        assert_eq!(result.size(), vec![1, 20]);
    }

    #[test]
    fn test_fade_out_length_greater_than_waveform_length() {
        let fade = Fade::new(10, 30, "linear");
        let waveform = Tensor::ones(&[1, 20], (Kind::Float, Device::Cpu));
        let result = fade.forward(&waveform);
        assert_eq!(result.size(), vec![1, 20]);
    }
}
