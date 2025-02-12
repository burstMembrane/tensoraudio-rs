use clap::Parser;
use clio::Input;
use tensoraudio::read_wav_from_stdin_chunked;
use tensoraudio::save;

#[derive(Parser)]
struct Args {
    #[clap(value_parser)]
    input: Input,
}

fn main() {
    let args = Args::parse();
    let input = args.input;
    if !input.is_std() {
        eprintln!("Input must be a WAV file from stdin");
        return;
    }
    let (samples, duration) = read_wav_from_stdin_chunked().unwrap();

    // Convert i16 samples to f32
    let samples_f32: Vec<f32> = samples.iter().map(|&x| x as f32 / 32768.0).collect();

    // write the samples to a file
    save("output.wav", &samples_f32, 44100, 2, "float");
    println!("Duration: {:?}", duration);
}
