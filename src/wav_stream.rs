use hound::{self, Sample};
use std::io::{self, BufReader, Write};

/// Reads a WAV file from standard input, streaming in chunks, and returns a tuple with:
/// - a vector of `i16` samples (the entire audio data), and
/// - the duration of the audio in seconds.
///
/// The function:
/// 1. Locks stdin and wraps it in a buffered reader.
/// 2. Uses `hound::WavReader` to read the WAV header (which provides the sample rate,
///    number of channels, and total sample count).
/// 3. Computes the duration as (number of frames)/(sample rate), where a frame is a set of samples
///    for all channels.
/// 4. Streams in the audio samples in chunks (here using a chunk size of 1024 samples).
pub fn read_wav_from_stdin_chunked() -> Result<(Vec<i16>, f32), hound::Error> {
    // Lock stdin and wrap in a buffered reader for efficiency.
    let stdin = io::stdin();
    let mut reader = BufReader::new(stdin.lock());

    // Create a WAV reader from the buffered stdin.
    let mut wav_reader = hound::WavReader::new(&mut reader)?;

    // Extract header information.
    let spec = wav_reader.spec();
    let num_channels = spec.channels as u32;
    let sample_rate = spec.sample_rate;

    // Total number of samples across all channels.
    let total_samples = wav_reader.duration(); // or wav_reader.len()

    // Number of frames is total_samples divided by number of channels.
    let num_frames = total_samples / num_channels;
    // Compute duration in seconds.
    let duration_secs = num_frames as f32 / sample_rate as f32 * num_channels as f32;

    // Read audio samples in chunks.
    let chunk_size = 1024;
    let mut samples = Vec::with_capacity(total_samples as usize);
    let mut chunk = Vec::with_capacity(chunk_size);

    // Iterate over each sample (here assuming samples are i16).
    for sample in wav_reader.samples::<i16>() {
        let s = sample?;
        chunk.push(s);

        // When the chunk is full, move the data into the samples vector.
        if chunk.len() == chunk_size {
            samples.extend(chunk.drain(..));
        }
    }
    // Append any remaining samples.
    if !chunk.is_empty() {
        samples.extend(chunk.drain(..));
    }

    Ok((samples, duration_secs))
}

pub fn write_wav_to_stdout(samples: &[i16], sample_rate: u32) -> Result<(), hound::Error> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };

    // Get header for streaming WAV file
    let header = spec.into_header_for_infinite_file();

    // Lock stdout for writing
    let stdout = io::stdout();
    let mut stdout = stdout.lock();

    // Write WAV header
    stdout.write_all(&header[..])?;

    // Write samples directly to stdout
    for &sample in samples {
        if sample.write(&mut stdout, 16).is_err() {
            break;
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_read_wav_from_stdin_chunked() {
        let (samples, duration) = read_wav_from_stdin_chunked().unwrap();
        assert!(samples.len() > 0);
        assert!(duration > 0.0);
    }

    #[test]
    fn test_write_wav_to_stdout() {
        let samples = vec![10000, 10000];
        let sample_rate = 44100;
        write_wav_to_stdout(&samples, sample_rate).unwrap();
    }
}
