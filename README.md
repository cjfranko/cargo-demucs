# cargo-demucs

A Rust library and CLI wrapper for the [Demucs](https://github.com/adefossez/demucs) music source separation tool.

Demucs is a state-of-the-art deep-learning model that separates a mixed audio track into individual stems (vocals, drums, bass, and other instruments). This crate lets you invoke Demucs from Rust code or from the command line without writing Python or shell scripts.

---

## Prerequisites

Demucs itself is a Python package. Install it before using this crate:

```bash
pip install demucs
```

> **Note:** A GPU (CUDA or Apple MPS) is recommended for fast processing, but the CPU backend works everywhere.

---

## Library usage

Add the crate to your `Cargo.toml`:

```toml
[dependencies]
cargo-demucs = "0.1"
```

### Separate a track into all stems (WAV, default model)

```rust
use cargo_demucs::Demucs;

let output = Demucs::builder()
    .input("/path/to/song.mp3")
    .output_dir("/tmp/separated")
    .run()
    .expect("Demucs failed");

println!("{}", output.stderr); // Demucs progress is on stderr
```

### Extract only vocals as MP3

```rust
use cargo_demucs::{Demucs, OutputFormat, Stem};

Demucs::builder()
    .model("htdemucs_ft")        // fine-tuned model
    .stem(Stem::Vocals)          // extract vocals only
    .format(OutputFormat::Mp3)
    .mp3_bitrate(320)
    .output_dir("/tmp/out")
    .input("track.flac")
    .run()
    .unwrap();
```

### Two-stems mode (vocals vs. everything else)

```rust
use cargo_demucs::Demucs;

Demucs::builder()
    .two_stems("vocals")         // produces vocals + no_vocals
    .output_dir("/tmp/out")
    .input("track.wav")
    .run()
    .unwrap();
```

### Check availability and query version

```rust
use cargo_demucs::Demucs;

if Demucs::is_available() {
    println!("Demucs version: {}", Demucs::version().unwrap());
} else {
    eprintln!("Demucs is not installed.");
}
```

---

## Builder options

| Method | Description |
|---|---|
| `.model(name)` | Model name: `htdemucs` (default), `htdemucs_ft`, `mdx`, `mdx_extra`, … |
| `.device(device)` | `Device::Cpu` (default), `Device::Cuda`, `Device::Mps`, `Device::Custom("…")` |
| `.output_dir(path)` | Output directory (default: `separated/`) |
| `.jobs(n)` | Number of parallel workers |
| `.stem(stem)` | Single stem to extract: `Stem::Vocals`, `Stem::Drums`, `Stem::Bass`, `Stem::Other`, `Stem::Guitar`, `Stem::Piano`, `Stem::All` (default) |
| `.two_stems(name)` | Two-stems mode – e.g. `"vocals"` → `vocals` + `no_vocals` |
| `.format(fmt)` | `OutputFormat::Wav` (default), `OutputFormat::Mp3`, `OutputFormat::Flac` |
| `.mp3_bitrate(kbps)` | MP3 bitrate in kbps (used with `OutputFormat::Mp3`) |
| `.shifts(n)` | Random shifts for equivariant stabilisation |
| `.overlap(f)` | Overlap between windows (0–1) |
| `.no_split()` | Process the whole track as one chunk |
| `.segment(seconds)` | Segment length in seconds |
| `.clip_mode(mode)` | `"rescale"` or `"clamp"` |
| `.verbose()` | Enable verbose output from Demucs |
| `.input(path)` | Add an input audio file |
| `.inputs(paths)` | Add multiple input files at once |

---

## CLI usage

After installing the crate you can call it directly:

```bash
cargo install cargo-demucs
```

```
cargo-demucs – Rust wrapper for the Demucs music source separation tool

USAGE:
    cargo-demucs [OPTIONS] <INPUT>...

OPTIONS:
    -n, --name <MODEL>         Model name (default: htdemucs)
    -d, --device <DEVICE>      Compute device: cpu, cuda, mps (default: cpu)
    -o, --out <DIR>            Output directory (default: separated/)
    -j, --jobs <N>             Number of parallel workers
        --stem <STEM>          Extract a single stem: vocals, drums, bass, other
        --two-stems <STEM>     Two-stems mode (stem + no_<stem>)
        --mp3                  Save output as MP3
        --flac                 Save output as FLAC
        --mp3-bitrate <KBPS>   MP3 bitrate in kbps (default: 320)
        --shifts <N>           Number of random shifts
        --overlap <FLOAT>      Overlap between windows, 0–1
        --no-split             Process the whole track as one segment
        --segment <SECONDS>    Segment length in seconds
        --clip-mode <MODE>     Clipping mode: rescale or clamp
    -v, --verbose              Enable verbose output
        --check                Check if Demucs is installed and exit
    -V, --version              Print Demucs version and exit
    -h, --help                 Print this help message and exit
```

### Examples

```bash
# Separate a track using the default model:
cargo-demucs song.mp3

# Fine-tuned model, MP3 output, custom directory:
cargo-demucs -n htdemucs_ft --mp3 -o /tmp/out song.flac

# Extract vocals only:
cargo-demucs --stem vocals song.wav

# Check if Demucs is installed:
cargo-demucs --check
```

---

## Output layout

Stems are written to `<output_dir>/<model>/<track_name>/`:

```
separated/
└── htdemucs/
    └── song/
        ├── vocals.wav
        ├── drums.wav
        ├── bass.wav
        └── other.wav
```

The helper function `cargo_demucs::stem_output_path` can compute this path programmatically.

---

## License

MIT – see [LICENSE](LICENSE).
