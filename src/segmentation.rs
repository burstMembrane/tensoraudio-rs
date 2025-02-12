use crate::transforms::Fade;
use anyhow::Result;
use log::{debug, info};
use std::time::Instant;
use tch::kind::Kind;
use tch::Tensor;

// //FIX: this errors out sometimes with smaller segments and long audio files
pub fn segment_audio(
    audio: &Tensor,
    segment_length: i64,
    overlap_seconds: f32,
    fade_type: &str,
    sample_rate: u32,
) -> Result<(Tensor, i64)> {
    let start = Instant::now();
    info!("Segmenting audio with overlap");

    // Ensure float32 dtype and [-1, 1] range
    let audio = audio.to_kind(Kind::Float);

    let total_samples = audio.size()[1];

    // Assuming 44100Hz sample rate - you'll need to add sample_rate as a parameter
    let overlap_length = (sample_rate as f32 * overlap_seconds) as i64;
    let segment_samples = segment_length * sample_rate as i64;
    let step_size = segment_samples - overlap_length;

    let num_segments = (total_samples + step_size - 1) / step_size;

    // Calculate padding for last segment
    let last_start = (num_segments - 1) * step_size;
    let needed_samples = last_start + segment_length;
    let mut padding = if needed_samples > total_samples {
        needed_samples - total_samples
    } else {
        0
    };

    debug!(
        "Padding amount: {}, total samples: {}",
        padding, total_samples
    );

    // Use linear crossfade instead of complex windowing
    let fade = Fade::new(overlap_length, 0, fade_type);

    // Simplify segment creation
    let mut segments = Vec::with_capacity(num_segments as usize);
    for i in 0..num_segments {
        let start_idx = i * step_size;
        let end_idx = (start_idx + segment_samples).min(total_samples);
        let mut segment = audio.slice(1, start_idx, end_idx, 1);

        if end_idx - start_idx < segment_samples {
            padding = segment_samples - (end_idx - start_idx);
            let pad = Tensor::zeros(&[segment.size()[0], padding], (Kind::Float, audio.device()));
            segment = Tensor::cat(&[segment, pad], 1);
        }

        segments.push(segment); // Remove fade.forward here
    }
    // apply fade to each segment
    for segment in &mut segments {
        *segment = fade.forward(segment);
    }
    // Stack segments and return with padding amount
    let output = Tensor::stack(&segments, 0);
    debug!("Output tensor shape: {:?}", output.size());
    info!("Segmentation took {:.2?}", start.elapsed());
    Ok((output, padding))
}

pub fn desegment_audio(
    segments: &Tensor,
    segment_length: i64,
    overlap_seconds: f32,
    padding: i64,
    fade_type: &str,
    sample_rate: u32,
) -> Result<Tensor> {
    debug!("Desegmenting audio");
    let num_segments = segments.size()[0];
    let channels = segments.size()[1];
    let overlap_length = (overlap_seconds * sample_rate as f32) as i64;

    let segment_samples = segment_length * sample_rate as i64;
    let step_size = segment_samples - overlap_length;

    let total_padded = (num_segments - 1) * step_size + segment_samples;
    let total_length = total_padded - padding;

    let output = Tensor::zeros(&[channels, total_padded], (Kind::Float, segments.device()));
    let weights = Tensor::zeros(&[total_padded], (Kind::Float, segments.device()));

    let fade = Fade::new(overlap_length / 2, overlap_length / 2, fade_type);
    // debug all our variables
    debug!(
        "num_segments: {}, channels: {}, overlap_length: {}, segment_samples: {}, step_size: {}, total_padded: {}, total_length: {}",
        num_segments, channels, overlap_length, segment_samples, step_size, total_padded, total_length
    );
    // Precompute the faded weights (tensor of ones after fade)
    let weights_fade = fade
        .forward(&Tensor::ones(
            &[1, segment_samples],
            (Kind::Float, segments.device()),
        ))
        .squeeze();

    for i in 0..num_segments {
        let start_idx = i * step_size;
        let end_idx = start_idx + segment_samples;

        let segment = segments.select(0, i);
        let segment = if segment.dim() == 1 {
            segment.unsqueeze(0)
        } else {
            segment
        };
        let mut faded_segment = fade.forward(&segment);

        let mut output_segment = output.slice(1, start_idx, end_idx, 1);
        if output_segment.dim() == 1 {
            output_segment = output_segment.unsqueeze(0);
        }

        let faded_len = faded_segment.size()[1];
        let output_len = output_segment.size()[1];
        if faded_len < output_len {
            let pad_len = output_len - faded_len;

            faded_segment = faded_segment.unsqueeze(0);
            faded_segment = faded_segment.zero_pad1d(0, pad_len);
            faded_segment = faded_segment.squeeze();
        }

        let _ = output_segment.f_add_(&faded_segment)?;

        let mut weights_segment = weights.slice(0, start_idx, end_idx, 1);
        let _ = weights_segment.f_add_(&weights_fade)?;
    }

    // Ensure output is float32 and in [-1, 1] range
    let normalized = (output / weights.unsqueeze(0)).to_kind(Kind::Float);

    let final_output = if padding > 0 {
        normalized.slice(1, 0, total_length, 1)
    } else {
        normalized
    };

    let final_output = if final_output.dim() == 1 {
        final_output.unsqueeze(0)
    } else {
        final_output
    };

    Ok(final_output)
}

#[allow(unused_imports)]
mod tests {
    use crate::audio::{
        generate_random_noise, get_audio_channels, get_audio_duration, get_audio_sample_rate,
        read_audio_file_tensor, write_audio_file_tensor, NoiseColor,
    };
    use crate::segmentation::{desegment_audio, segment_audio};

    use std::path::PathBuf;
    use tch::{Device, Tensor};

    #[test]
    fn test_segment_audio_with_overlap() {
        let test_path = PathBuf::from("testdata/test.wav");
        let test_path_str = test_path.to_str().unwrap();

        // Read the test audio file
        let tensor = read_audio_file_tensor(test_path_str, Device::Cpu);
        let sample_rate = get_audio_sample_rate(test_path_str) as i64; // Convert to i64

        let segment_length = 1;
        let overlap_seconds = 0.1;
        let fade_type = "linear";

        let (result, _) = segment_audio(
            &tensor,
            segment_length,
            overlap_seconds,
            fade_type,
            sample_rate as u32,
        )
        .unwrap();

        // Calculate expected segments based on actual file duration
        let duration = tensor.size()[1] as f32 / sample_rate as f32;
        let expected_segments = ((duration - overlap_seconds)
            / (segment_length as f32 - overlap_seconds))
            .ceil() as i64;
        let expected_channels = tensor.size()[0];

        assert_eq!(
            result.size(),
            vec![
                expected_segments,
                expected_channels,
                segment_length * sample_rate
            ]
        );
    }

    #[test]
    fn test_desegment_audio() {
        let test_path = PathBuf::from("testdata/test.wav");
        let test_path_str = test_path.to_str().unwrap();

        // Read the test audio file
        let tensor = read_audio_file_tensor(test_path_str, Device::Cpu);
        let sample_rate = get_audio_sample_rate(test_path_str);

        let segment_length = 1;
        let overlap_seconds = 0.1;
        let fade_type = "linear";

        let (segments, padding) = segment_audio(
            &tensor,
            segment_length,
            overlap_seconds,
            fade_type,
            sample_rate,
        )
        .unwrap();
        let result = desegment_audio(
            &segments,
            segment_length,
            overlap_seconds,
            padding,
            fade_type,
            sample_rate,
        )
        .unwrap();

        // Get expected dimensions from original audio
        let expected_channels = tensor.size()[0];
        let expected_samples = tensor.size()[1];
        assert!(result.size() == vec![expected_channels, expected_samples]);
    }

    #[test]
    fn test_segment_reconstruct_audio() {
        let test_path = PathBuf::from("testdata/test.wav");
        let test_path_str = test_path.to_str().unwrap();
        let audio = read_audio_file_tensor(test_path_str, Device::Cpu);
        let sample_rate = get_audio_sample_rate(test_path_str);

        let segment_length = 1;
        let overlap_seconds = 0.1;
        let fade_type = "linear";
        let (segments, padding) = segment_audio(
            &audio,
            segment_length,
            overlap_seconds,
            fade_type,
            sample_rate,
        )
        .unwrap();

        let reconstructed_audio = desegment_audio(
            &segments,
            segment_length,
            overlap_seconds,
            padding,
            fade_type,
            sample_rate,
        )
        .unwrap();

        let output_path = PathBuf::from("testdata/reconstructed.wav");
        assert!(reconstructed_audio.size() == audio.size());

        let _ = write_audio_file_tensor(
            &reconstructed_audio,
            output_path.to_str().unwrap(),
            sample_rate,
            2,
            "int",
        );
    }
}
