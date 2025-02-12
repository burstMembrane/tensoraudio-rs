use clap::Parser;
use tch::Device;

use tensoraudio::get_sample_rate;
use tensoraudio::load_chunk_tensor;

#[derive(Parser)]
struct Args {
    #[clap(short, long)]
    path: String,
}

fn main() {
    let args = Args::parse();
    // load 1000 samples from the start of the audio
    let audio = load_chunk_tensor(&args.path, 0, 1000, Device::Cpu);
    // get the sample rate
    let sample_rate = get_sample_rate(&args.path);
    // size will be [channels, samples]
    // print the size, sample rate, and path
    println!("Path: {}", args.path);
    println!("Audio Size: {:?}", audio.size());
    println!("Sample Rate: {}", sample_rate);
}
