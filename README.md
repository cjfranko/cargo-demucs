# cargo-demucs

A Rust library and CLI wrapper for the [Demucs](https://github.com/adefossez/demucs) music source separation tool.

Demucs is a state-of-the-art deep-learning model that separates a mixed audio track into individual stems (vocals, drums, bass, and other instruments).

**No Python installation is required.** On first run, `cargo-demucs` automatically downloads a self-contained Python interpreter ([python-build-standalone](https://github.com/indygreg/python-build-standalone)) and installs Demucs and all its dependencies into a private environment (`~/.cargo-demucs/`). Subsequent runs use the cached environment and start immediately.

---

## Installation

```bash
cargo install cargo-demucs
```

That's it. Python, PyTorch, and Demucs are downloaded and managed automatically.

> **Note:** The first run downloads several GB of dependencies (PyTorch is large). An internet connection is required on first use. A GPU (CUDA or Apple MPS) is recommended for fast processing, but the CPU backend works on every machine.

---

## CLI usage

### First-time setup (optional – also triggered automatically)

```bash
cargo-demucs --setup
```

### Separate a track

```bash
# Separate a track using the default model (htdemucs):
cargo-demucs song.mp3

# Fine-tuned model, MP3 output, custom directory:
cargo-demucs -n htdemucs_ft --mp3 -o /tmp/out song.flac

# Extract vocals only:
cargo-demucs --stem vocals song.wav

# Check if the managed environment is ready:
cargo-demucs --check
```

### All options

```
USAGE:
    cargo-demucs [OPTIONS] <INPUT>...

OPTIONS:
    -n, --name <MODEL>         Model name (default: htdemucs)
                               Examples: htdemucs, htdemucs_ft, mdx, mdx_extra
    -d, --device <DEVICE>      Compute device: cpu, cuda, mps (default: cpu)
    -o, --out <DIR>            Output directory (default: separated/)
    -j, --jobs <N>             Number of parallel workers
        --stem <STEM>          Extract a single stem: vocals, drums, bass, other
        --two-stems <STEM>     Two-stems mode (stem + no_<stem>)
        --mp3                  Save output as MP3
        --flac                 Save output as FLAC
        --mp3-bitrate <KBPS>   MP3 bitrate in kbps (default: 320)
        --shifts <N>           Number of random shifts (equivariant stabilisation)
        --overlap <FLOAT>      Overlap between windows, 0–1 (default: 0.25)
        --no-split             Process the whole track as one segment
        --segment <SECONDS>    Segment length in seconds
        --clip-mode <MODE>     Clipping mode: rescale or clamp
    -v, --verbose              Enable verbose output
        --setup                Download and install the managed Python environment
        --check                Check if the managed environment is ready and exit
    -V, --version              Print Demucs version and exit
    -h, --help                 Print this help message and exit

ENVIRONMENT:
    CARGO_DEMUCS_HOME   Override the managed environment directory
                        (default: ~/.cargo-demucs/)
```

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

// The first call triggers automatic setup if needed.
let output = Demucs::builder()
    .input("/path/to/song.mp3")
    .output_dir("/tmp/separated")
    .run()
    .expect("Demucs failed");

println!("{}", output.stderr); // Demucs progress is on stderr
```

### Pre-warm the environment before processing

```rust
use cargo_demucs::Demucs;

// Show a "loading" screen while setup runs, then process.
Demucs::setup().expect("setup failed");

Demucs::builder()
    .input("track.flac")
    .output_dir("/tmp/out")
    .run()
    .unwrap();
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

### Check whether the managed environment is ready

```rust
use cargo_demucs::Demucs;

if Demucs::is_available() {
    println!("Ready!");
} else {
    // Trigger setup explicitly, or let builder.run() do it automatically.
    Demucs::setup().unwrap();
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

## How the managed environment works

On first use, `cargo-demucs`:

1. Downloads the appropriate [python-build-standalone](https://github.com/indygreg/python-build-standalone) archive for the current OS and CPU architecture (~30 MB compressed).
2. Extracts it to `~/.cargo-demucs/python/`.
3. Runs `pip install demucs` inside that isolated environment (downloads PyTorch, torchaudio, etc. – several GB).
4. On subsequent runs it detects the cached environment and starts immediately.

The environment location can be overridden with the `CARGO_DEMUCS_HOME` environment variable.

### Supported platforms

| OS | Architecture |
|---|---|
| Linux | x86_64 |
| Linux | aarch64 (ARM64) |
| macOS | x86_64 |
| macOS | aarch64 (Apple Silicon) |
| Windows | x86_64 |

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
