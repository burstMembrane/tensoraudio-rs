# Audio Processing Library

A Rust library for audio file manipulation and processing, with support for WAV files, audio segmentation, and various audio transformations.

## Features

- **Audio File Operations**
  - Read/write WAV files
  - Support for different sample formats (float32, int16)
  - Channel operations (mono/stereo)

- **Audio Processing**
  - Segmentation with customizable overlap
  - Various fade types
  - Audio format conversion
  - Channel manipulation

- **Audio Generation**
  - Sine wave generation
  - White noise generation
  - Pink noise generation

- **Tensor Operations**
  - PyTorch integration via tch-rs
  - Tensor-based audio processing
  - GPU support (via Device selection)

## Requirements

- Rust 1.56 or higher
- libtorch (PyTorch C++ library)



## Dependencies

- [tch-rs](https://github.com/mli/tch-rs)
- [hound](https://github.com/kornelski/rust-hound)
- [log](https://github.com/rust-lang/log)
- [rand](https://github.com/rust-lang/rand)
