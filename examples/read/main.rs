use clap::Parser;
use tch::Device;

use tensoraudio::get_sample_rate;
use tensoraudio::load_tensor;

#[derive(Parser)]
struct Args {
    #[clap(short, long)]
    path: String,
}

fn main() {
    let args = Args::parse();
    // load audio to a tensor
    let audio = load_tensor(&args.path, Device::Cpu);
    // get the sample rate
    let sample_rate = get_sample_rate(&args.path);
    // size will be [channels, samples]
    // print the size, sample rate, and path
    println!("Path: {}", args.path);
    println!("Audio Size: {:?}", audio.size());
    println!("Sample Rate: {}", sample_rate);
}
