//! cargo-demucs – CLI front-end for the Demucs music source separation tool.
//!
//! Run `cargo-demucs --help` for usage information.

use cargo_demucs::{Demucs, DemucsError, Device, OutputFormat, Stem};
use std::process;

fn main() {
    let args: Vec<String> = std::env::args().collect();

    // Minimal argument parser – does not pull in external crates.
    if args.iter().any(|a| a == "--help" || a == "-h") {
        print_help();
        return;
    }

    if args.iter().any(|a| a == "--version" || a == "-V") {
        match Demucs::version() {
            Ok(v) => println!("{v}"),
            Err(e) => {
                eprintln!("error: {e}");
                process::exit(1);
            }
        }
        return;
    }

    if args.iter().any(|a| a == "--check") {
        if Demucs::is_available() {
            println!("Demucs is available (managed environment ready).");
        } else {
            eprintln!(
                "Demucs is not yet set up. Run `cargo-demucs --setup` to \
                 download and install the managed Python environment."
            );
            process::exit(1);
        }
        return;
    }

    if args.iter().any(|a| a == "--setup") {
        match Demucs::setup() {
            Ok(()) => {
                println!("Managed Python environment is ready.");
            }
            Err(e) => {
                eprintln!("error: {e}");
                process::exit(1);
            }
        }
        return;
    }

    // Parse flags and positional arguments.
    let mut model: Option<String> = None;
    let mut device: Option<Device> = None;
    let mut output_dir: Option<String> = None;
    let mut jobs: Option<u32> = None;
    let mut stem: Option<Stem> = None;
    let mut format: Option<OutputFormat> = None;
    let mut mp3_bitrate: Option<u32> = None;
    let mut two_stems: Option<String> = None;
    let mut shifts: Option<u32> = None;
    let mut overlap: Option<f32> = None;
    let mut no_split = false;
    let mut segment: Option<u32> = None;
    let mut clip_mode: Option<String> = None;
    let mut verbose = false;
    let mut inputs: Vec<String> = Vec::new();

    let mut iter = args.iter().skip(1).peekable();

    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "-n" | "--name" => {
                model = iter.next().cloned();
            }
            "-d" | "--device" => {
                device = iter.next().map(|s| match s.as_str() {
                    "cpu" => Device::Cpu,
                    "cuda" => Device::Cuda,
                    "mps" => Device::Mps,
                    other => Device::Custom(other.to_owned()),
                });
            }
            "-o" | "--out" => {
                output_dir = iter.next().cloned();
            }
            "-j" | "--jobs" => {
                jobs = iter.next().and_then(|s| s.parse().ok());
            }
            "--stem" => {
                stem = match iter.next().map(|s| s.as_str()) {
                    Some("vocals") => Some(Stem::Vocals),
                    Some("drums") => Some(Stem::Drums),
                    Some("bass") => Some(Stem::Bass),
                    Some("other") => Some(Stem::Other),
                    Some("guitar") => Some(Stem::Guitar),
                    Some("piano") => Some(Stem::Piano),
                    Some(unknown) => {
                        eprintln!(
                            "error: unknown stem '{}'. Valid stems: vocals, drums, bass, other, guitar, piano",
                            unknown
                        );
                        process::exit(2);
                    }
                    None => {
                        eprintln!("error: --stem requires an argument");
                        process::exit(2);
                    }
                };
            }
            "--mp3" => {
                format = Some(OutputFormat::Mp3);
            }
            "--flac" => {
                format = Some(OutputFormat::Flac);
            }
            "--mp3-bitrate" => {
                mp3_bitrate = iter.next().and_then(|s| s.parse().ok());
            }
            "--two-stems" => {
                two_stems = iter.next().cloned();
            }
            "--shifts" => {
                shifts = iter.next().and_then(|s| s.parse().ok());
            }
            "--overlap" => {
                overlap = iter.next().and_then(|s| s.parse().ok());
            }
            "--no-split" => {
                no_split = true;
            }
            "--segment" => {
                segment = iter.next().and_then(|s| s.parse().ok());
            }
            "--clip-mode" => {
                clip_mode = iter.next().cloned();
            }
            "-v" | "--verbose" => {
                verbose = true;
            }
            other if !other.starts_with('-') => {
                inputs.push(other.to_owned());
            }
            other => {
                eprintln!("Unknown option: {other}");
                eprintln!("Run with --help for usage.");
                process::exit(2);
            }
        }
    }

    if inputs.is_empty() {
        eprintln!("error: at least one input audio file is required.");
        eprintln!("Run with --help for usage.");
        process::exit(2);
    }

    // Build the Demucs invocation.
    let mut builder = Demucs::builder();

    if let Some(m) = model {
        builder = builder.model(m);
    }
    if let Some(d) = device {
        builder = builder.device(d);
    }
    if let Some(o) = output_dir {
        builder = builder.output_dir(o);
    }
    if let Some(j) = jobs {
        builder = builder.jobs(j);
    }
    if let Some(s) = stem {
        builder = builder.stem(s);
    }
    if let Some(f) = format {
        builder = builder.format(f);
    }
    if let Some(b) = mp3_bitrate {
        builder = builder.mp3_bitrate(b);
    }
    if let Some(ts) = two_stems {
        builder = builder.two_stems(ts);
    }
    if let Some(sh) = shifts {
        builder = builder.shifts(sh);
    }
    if let Some(ov) = overlap {
        builder = builder.overlap(ov);
    }
    if no_split {
        builder = builder.no_split();
    }
    if let Some(seg) = segment {
        builder = builder.segment(seg);
    }
    if let Some(cm) = clip_mode {
        builder = builder.clip_mode(cm);
    }
    if verbose {
        builder = builder.verbose();
    }
    builder = builder.inputs(inputs);

    match builder.run() {
        Ok(out) => {
            if !out.stdout.is_empty() {
                print!("{}", out.stdout);
            }
            if !out.stderr.is_empty() {
                eprint!("{}", out.stderr);
            }
        }
        Err(DemucsError::NotInstalled) => {
            eprintln!(
                "error: Demucs could not be started.\n\
                 Try running `cargo-demucs --setup` to set up the managed \
                 Python environment, or ensure you have an internet connection \
                 on first run."
            );
            process::exit(1);
        }
        Err(e) => {
            eprintln!("error: {e}");
            process::exit(1);
        }
    }
}

fn print_help() {
    println!(
        "cargo-demucs – Rust wrapper for the Demucs music source separation tool

cargo-demucs bundles its own Python environment. On the first run it
automatically downloads Python and installs Demucs – no system Python required.

USAGE:
    cargo-demucs [OPTIONS] <INPUT>...

ARGS:
    <INPUT>...    One or more audio files to process

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

EXAMPLES:
    # First-time setup (also happens automatically on first use):
    cargo-demucs --setup

    # Separate a track using the default model (htdemucs):
    cargo-demucs song.mp3

    # Use the fine-tuned model and write MP3 output to /tmp/out:
    cargo-demucs -n htdemucs_ft --mp3 -o /tmp/out song.flac

    # Extract only vocals:
    cargo-demucs --stem vocals song.wav

    # Check if the managed environment is ready:
    cargo-demucs --check
"
    );
}
