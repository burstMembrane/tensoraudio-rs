use hound::{SampleFormat, WavReader, WavSpec, WavWriter};
use std::convert::TryFrom;
use tch::{Device, Kind, Tensor};

pub fn load(path: &str) -> Vec<f32> {
    let mut reader = WavReader::open(path).unwrap();
    let spec = reader.spec();

    match spec.sample_format {
        SampleFormat::Float => reader.samples::<f32>().map(|s| s.unwrap()).collect(),
        SampleFormat::Int => {
            if spec.bits_per_sample == 16 {
                reader
                    .samples::<i16>()
                    .map(|s| s.unwrap() as f32 / i16::MAX as f32)
                    .collect()
            } else {
                reader
                    .samples::<i32>()
                    .map(|s| s.unwrap() as f32 / i32::MAX as f32)
                    .collect()
            }
        }
    }
}

fn tensor_to_vec_f32(tensor: Tensor) -> Result<Vec<f32>, tch::TchError> {
    Vec::<f32>::try_from(tensor)
}

pub fn save(path: &str, data: &[f32], sample_rate: u32, channels: u16, sample_format: &str) {
    let sample_format = match sample_format {
        "float" => SampleFormat::Float,
        "int" => SampleFormat::Int,
        _ => panic!("Invalid sample format"),
    };

    let bits_per_sample = match sample_format {
        SampleFormat::Float => 32, // Float format requires 32 bits
        SampleFormat::Int => 16,   // Int format uses 16 bits
    };

    let spec = WavSpec {
        sample_rate,
        channels,
        bits_per_sample,
        sample_format,
    };

    let mut writer = WavWriter::create(path, spec).unwrap();

    match sample_format {
        SampleFormat::Float => {
            for &sample in data {
                writer.write_sample(sample).unwrap();
            }
        }
        SampleFormat::Int => {
            for &sample in data {
                // Scale the float sample (-1.0 to 1.0) to i16 range
                let scaled = (sample * i16::MAX as f32) as i16;
                writer.write_sample(scaled).unwrap();
            }
        }
    }

    writer.finalize().unwrap();
}

pub fn load_tensor(path: &str, device: Device) -> Tensor {
    let audio_data = load(path);
    let channels = get_channels(path) as usize;
    let samples_per_channel = audio_data.len() / channels;

    // Create a vector of vectors, one for each channel
    let mut deinterleaved: Vec<Vec<f32>> = vec![Vec::with_capacity(samples_per_channel); channels];

    // Deinterleave the samples
    for (i, &sample) in audio_data.iter().enumerate() {
        let channel = i % channels;
        deinterleaved[channel].push(sample);
    }

    // Flatten the deinterleaved data into a single vec and create tensor
    let flat: Vec<f32> = deinterleaved.into_iter().flatten().collect();
    Tensor::from_slice(&flat)
        .reshape(&[channels as i64, samples_per_channel as i64])
        .to(device)
}

pub fn save_tensor(
    tensor: &Tensor,
    path: &str,
    sample_rate: u32,
    channels: u16,
    sample_format: &str,
) -> Result<(), tch::TchError> {
    // Get dimensions

    // assert that the channels are the same
    assert_eq!(channels, tensor.size()[0] as u16);
    let channels = tensor.size()[0] as u16;

    // First transpose to get [duration, channels], then flatten
    let audio_data = tensor_to_vec_f32(tensor.transpose(0, 1).flatten(0, 1))?;

    save(path, &audio_data, sample_rate, channels, sample_format);
    Ok(())
}

pub fn get_duration(path: &str) -> u32 {
    // get the duration in samples of the audio file
    let reader = WavReader::open(path).unwrap();
    reader.duration()
}

pub fn get_sample_rate(path: &str) -> u32 {
    let reader = WavReader::open(path).unwrap();
    reader.spec().sample_rate
}

pub fn get_channels(path: &str) -> u16 {
    let reader = WavReader::open(path).unwrap();
    reader.spec().channels
}

pub fn read_chunk(path: &str, start: usize, end: usize) -> Vec<f32> {
    let mut reader = WavReader::open(path).unwrap();
    let spec = reader.spec();
    let channels = spec.channels as usize;
    let duration = get_duration(path);
    if start > duration as usize {
        panic!("Start is greater than the duration of the audio file");
    }
    if end > duration as usize {
        panic!("End is greater than the duration of the audio file");
    }
    // Adjust start and end to account for channels
    let start_sample = start * channels;
    let end_sample = end * channels;

    let interleaved = match spec.sample_format {
        SampleFormat::Float => reader
            .samples::<f32>()
            .skip(start_sample)
            .take(end_sample - start_sample)
            .map(|s| s.unwrap())
            .collect(),
        SampleFormat::Int => {
            if spec.bits_per_sample == 16 {
                reader
                    .samples::<i16>()
                    .skip(start_sample)
                    .take(end_sample - start_sample)
                    .map(|s| s.unwrap() as f32 / i16::MAX as f32)
                    .collect()
            } else {
                reader
                    .samples::<i32>()
                    .skip(start_sample)
                    .take(end_sample - start_sample)
                    .map(|s| s.unwrap() as f32 / i32::MAX as f32)
                    .collect()
            }
        }
    };
    interleaved
}

pub fn load_chunk_tensor(path: &str, start: usize, end: usize, device: Device) -> Tensor {
    let audio_data = read_chunk(path, start, end);
    let channels = get_channels(path) as usize;
    let samples_per_channel = audio_data.len() / channels;
    let mut deinterleaved: Vec<Vec<f32>> = vec![Vec::with_capacity(samples_per_channel); channels];
    for (i, &sample) in audio_data.iter().enumerate() {
        let channel = i % channels;
        deinterleaved[channel].push(sample);
    }
    let flat: Vec<f32> = deinterleaved.into_iter().flatten().collect();
    Tensor::from_slice(&flat)
        .reshape(&[channels as i64, samples_per_channel as i64])
        .to(device)
}

pub fn generate_sine_wave(
    frequency: f32,
    duration: u32,
    sample_rate: u32,
    channels: u16,
) -> Vec<f32> {
    let mut audio_data = Vec::new();
    let samples_per_channel = sample_rate * duration;

    // Generate the base sine wave for one channel
    let single_channel: Vec<f32> = (0..samples_per_channel)
        .map(|x| x as f32 / sample_rate as f32)
        .map(|t| {
            let sample = (t * frequency * 2.0 * std::f32::consts::PI).sin();
            sample * i16::MAX as f32
        })
        .collect();

    // Duplicate the sine wave for each channel
    for i in 0..samples_per_channel {
        for _ in 0..channels {
            audio_data.push(single_channel[i as usize]);
        }
    }
    audio_data
}

#[derive(Debug, Clone, Copy)]
pub enum NoiseColor {
    White,
    Pink,
}

pub fn generate_random_noise(
    duration: i64,
    sample_rate: i64,
    channels: i64,
    color: NoiseColor,
) -> Vec<f32> {
    let samples = duration * sample_rate;
    let mut random_tensor = Tensor::randn(&[channels, samples], (Kind::Float, Device::Cpu));

    match color {
        NoiseColor::White => {} // White noise needs no filtering
        NoiseColor::Pink => {
            // Apply 1/f (pink noise) filtering in frequency domain
            // For each channel
            for c in 0..channels {
                let channel = random_tensor.select(0, c);

                // Create frequency domain filter
                let mut filter = Vec::with_capacity(samples as usize);
                for i in 0..samples {
                    let freq = (i as f32 + 1.0).sqrt().recip(); // 1/sqrt(f) for power spectrum 1/f
                    filter.push(freq);
                }

                let filter_tensor = Tensor::from_slice(&filter);

                // Simple frequency-domain filtering by multiplication
                let filtered = channel * filter_tensor;

                // Update the channel
                random_tensor.get(c).copy_(&filtered);
            }
        }
    }

    // Normalize the output
    let std = random_tensor.std_mean(false).0;
    random_tensor /= std;

    // First transpose to get [duration, channels], then flatten
    let audio_data = tensor_to_vec_f32(random_tensor.transpose(0, 1).flatten(0, 1)).unwrap();
    audio_data
}

pub fn to_mono(data: &[f32], channels: u16) -> Vec<f32> {
    let channels = channels as usize;
    let samples_per_channel = data.len() / channels;
    let mut mono_data = Vec::with_capacity(samples_per_channel);

    // Process each sample across all channels
    for frame in 0..samples_per_channel {
        let mut sample_sum = 0.0;
        // Sum all channels for this sample
        for channel in 0..channels {
            sample_sum += data[frame * channels + channel];
        }
        // Average the sum and store
        mono_data.push(sample_sum / channels as f32);
    }

    mono_data
}

pub fn interleave(channels: &[Vec<f32>]) -> Vec<f32> {
    let samples_per_channel = channels[0].len();
    let num_channels = channels.len();
    let mut interleaved = Vec::with_capacity(samples_per_channel * num_channels);

    for sample_idx in 0..samples_per_channel {
        for channel in channels {
            interleaved.push(channel[sample_idx]);
        }
    }

    interleaved
}

pub fn deinterleave(data: &[f32], num_channels: usize) -> Vec<Vec<f32>> {
    let samples_per_channel = data.len() / num_channels;
    let mut channels: Vec<Vec<f32>> = vec![Vec::with_capacity(samples_per_channel); num_channels];

    for (i, &sample) in data.iter().enumerate() {
        let channel = i % num_channels;
        channels[channel].push(sample);
    }

    channels
}

#[cfg(test)]
mod tests {

    use crate::audio::*;

    use tch::Device;

    #[test]
    fn test_load() {
        let audio_data = load("testdata/test.wav");
        let duration = get_duration("testdata/test.wav");
        let channels = get_channels("testdata/test.wav");

        // audio data should be the duration * channels
        assert_eq!(audio_data.len() / channels as usize, duration as usize);
    }

    #[test]
    fn test_get_sample_rate() {
        let sample_rate = get_sample_rate("testdata/test.wav");
        assert_eq!(sample_rate, 44100);
    }
    #[test]
    fn test_get_channels() {
        let channels = get_channels("testdata/test.wav");
        assert_eq!(channels, 2);
    }
    #[test]
    fn test_read_chunk() {
        let audio_data = read_chunk("testdata/test.wav", 0, 1000);
        assert_eq!(audio_data.len(), 2000);
    }
    #[test]
    fn test_save() {
        let audio_data = load("testdata/test.wav");

        save("testdata/test_write.wav", &audio_data, 44100, 2, "int");
    }

    #[test]
    fn test_read_chunk_int() {
        let audio_data = read_chunk("testdata/test.wav", 0, 1000);
        assert_eq!(audio_data.len(), 2000);
    }

    #[test]
    fn test_write_chunk() {
        let audio_data = read_chunk("testdata/test.wav", 0, 441000);
        save(
            "testdata/test_chunk_write.wav",
            &audio_data,
            44100,
            2,
            "int",
        );
    }

    #[test]
    fn test_load_tensor() {
        let tensor = load_tensor("testdata/test.wav", Device::Cpu);
        let duration = get_duration("testdata/test.wav");
        let channels = get_channels("testdata/test.wav");
        assert_eq!(tensor.size()[0], channels as i64);
        assert_eq!(tensor.size()[1], duration as i64);
    }

    #[test]
    fn test_tensor_to_audio_file() -> Result<(), tch::TchError> {
        let tensor = load_tensor("testdata/test.wav", Device::Cpu);
        save_tensor(&tensor, "testdata/tensor.wav", 44100, 2 as u16, "int")?;

        // Verify the output
        let original = load("testdata/test.wav");
        let written = load("testdata/tensor.wav");

        assert_eq!(original.len(), written.len());

        // Compare samples (allowing for small floating-point differences)
        for (orig, written) in original.iter().zip(written.iter()) {
            assert!((orig - written).abs() < 1e-6);
        }
        Ok(())
    }

    #[test]
    fn test_generate_sine_wave() {
        let audio_data = generate_sine_wave(60.0, 1, 44100, 2);
        save("testdata/sine.wav", &audio_data, 44100, 2, "int");
    }

    #[test]
    fn test_generate_random_noise() {
        // Test white noise
        let white_noise = generate_random_noise(1, 44100, 2, NoiseColor::White);
        save("testdata/white_noise.wav", &white_noise, 44100, 2, "int");

        // Test pink noise
        let pink_noise = generate_random_noise(1, 44100, 2, NoiseColor::Pink);
        save("testdata/pink_noise.wav", &pink_noise, 44100, 2, "int");
    }

    #[test]
    fn test_to_mono() {
        let audio_data = load("testdata/test.wav");
        let mono_data = to_mono(&audio_data, 2);
        save("testdata/mono.wav", &mono_data, 44100, 1, "int");
    }

    #[test]
    fn test_interleave_deinterleave() {
        // Create test data
        let channel1 = vec![1.0, 2.0, 3.0];
        let channel2 = vec![4.0, 5.0, 6.0];
        let channels = vec![channel1.clone(), channel2.clone()];

        // Test interleaving
        let interleaved = interleave(&channels);
        assert_eq!(interleaved, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);

        // Test deinterleaving
        let deinterleaved = deinterleave(&interleaved, 2);
        assert_eq!(deinterleaved[0], channel1);
        assert_eq!(deinterleaved[1], channel2);

        // Test round trip
        let round_trip = interleave(&deinterleave(&interleaved, 2));
        assert_eq!(round_trip, interleaved);
    }

    #[test]
    fn test_load_chunk_tensor() {
        let tensor = load_chunk_tensor("testdata/test.wav", 0, 1000, Device::Cpu);
        assert_eq!(tensor.size()[0], 2);
        assert_eq!(tensor.size()[1], 1000);
    }
}
