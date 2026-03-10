//! # cargo-demucs
//!
//! A Rust library (and CLI) for invoking the [Demucs](https://github.com/adefossez/demucs)
//! music source separation tool.
//!
//! Demucs is a Python-based model that can split a mixed audio track into its
//! individual stems (vocals, drums, bass, and other instruments). This crate
//! wraps the `demucs` command-line interface and **bundles its own Python
//! environment** so that end users do not need Python installed on their
//! machine.
//!
//! On the first run, `cargo-demucs` automatically downloads a
//! [python-build-standalone](https://github.com/indygreg/python-build-standalone)
//! distribution into `~/.cargo-demucs/` and installs Demucs (and its
//! dependencies) via `pip`.  Subsequent runs use the cached environment and
//! start immediately.
//!
//! ## Quick start
//!
//! ```no_run
//! use cargo_demucs::Demucs;
//!
//! let output = Demucs::builder()
//!     .model("htdemucs")
//!     .output_dir("/tmp/separated")
//!     .input("/path/to/song.mp3")
//!     .run()
//!     .expect("Demucs failed");
//!
//! println!("stdout: {}", output.stdout);
//! ```

pub mod error;
pub mod python_env;

use std::ffi::OsStr;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

pub use error::DemucsError;

/// The result type used throughout this crate.
pub type Result<T> = std::result::Result<T, DemucsError>;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// The audio format to write the separated stems in.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum OutputFormat {
    /// WAV (lossless) – this is the Demucs default.
    #[default]
    Wav,
    /// MP3 (lossy).
    Mp3,
    /// FLAC (lossless compressed).
    Flac,
}

/// Which individual stem(s) to extract.
///
/// When `All` is selected (the default) every available stem is written to
/// disk. The others extract only the named stem, which is faster.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum Stem {
    /// Extract all available stems (default).
    #[default]
    All,
    /// Vocal track only.
    Vocals,
    /// Drum track only.
    Drums,
    /// Bass track only.
    Bass,
    /// Remainder after vocals/drums/bass are removed.
    Other,
    /// Guitar track (available in some models).
    Guitar,
    /// Piano track (available in some models).
    Piano,
}

impl Stem {
    fn as_str(&self) -> Option<&'static str> {
        match self {
            Stem::All => None,
            Stem::Vocals => Some("vocals"),
            Stem::Drums => Some("drums"),
            Stem::Bass => Some("bass"),
            Stem::Other => Some("other"),
            Stem::Guitar => Some("guitar"),
            Stem::Piano => Some("piano"),
        }
    }
}

/// The compute device to run inference on.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum Device {
    /// Use the CPU (always available, but slower).
    #[default]
    Cpu,
    /// Use an NVIDIA GPU via CUDA.
    Cuda,
    /// Use Apple Silicon via Metal Performance Shaders.
    Mps,
    /// A custom device string that is passed through verbatim.
    Custom(String),
}

impl Device {
    fn as_str(&self) -> &str {
        match self {
            Device::Cpu => "cpu",
            Device::Cuda => "cuda",
            Device::Mps => "mps",
            Device::Custom(s) => s.as_str(),
        }
    }
}

// ---------------------------------------------------------------------------
// Output from a successful run
// ---------------------------------------------------------------------------

/// The output produced by a successful Demucs run.
#[derive(Debug, Clone)]
pub struct DemucsOutput {
    /// The standard output captured from the Demucs process.
    pub stdout: String,
    /// The standard error captured from the Demucs process.
    pub stderr: String,
}

// ---------------------------------------------------------------------------
// Builder
// ---------------------------------------------------------------------------

/// Builder for a Demucs invocation.
///
/// Obtain one via [`Demucs::builder`].
///
/// # Example
///
/// ```no_run
/// use cargo_demucs::Demucs;
///
/// Demucs::builder()
///     .model("htdemucs_ft")
///     .device(cargo_demucs::Device::Cpu)
///     .output_dir("/tmp/out")
///     .jobs(2)
///     .input("/music/track.flac")
///     .run()
///     .unwrap();
/// ```
#[derive(Debug, Default)]
pub struct DemucsBuilder {
    model: Option<String>,
    device: Option<Device>,
    output_dir: Option<PathBuf>,
    jobs: Option<u32>,
    stem: Option<Stem>,
    format: Option<OutputFormat>,
    mp3_bitrate: Option<u32>,
    two_stems: Option<String>,
    shifts: Option<u32>,
    overlap: Option<f32>,
    no_split: bool,
    segment: Option<u32>,
    clip_mode: Option<String>,
    verbose: bool,
    inputs: Vec<PathBuf>,
}

impl DemucsBuilder {
    /// Set the Demucs model to use (e.g. `"htdemucs"`, `"htdemucs_ft"`,
    /// `"mdx"`, `"mdx_extra"`).
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    /// Choose the compute device.
    pub fn device(mut self, device: Device) -> Self {
        self.device = Some(device);
        self
    }

    /// Set the output directory for separated stems.
    pub fn output_dir(mut self, path: impl Into<PathBuf>) -> Self {
        self.output_dir = Some(path.into());
        self
    }

    /// Number of parallel jobs (workers).
    pub fn jobs(mut self, n: u32) -> Self {
        self.jobs = Some(n);
        self
    }

    /// Extract only the given stem.
    pub fn stem(mut self, stem: Stem) -> Self {
        self.stem = Some(stem);
        self
    }

    /// Set the output audio format.
    pub fn format(mut self, format: OutputFormat) -> Self {
        self.format = Some(format);
        self
    }

    /// MP3 bitrate in kbps (only used when [`OutputFormat::Mp3`] is selected).
    pub fn mp3_bitrate(mut self, kbps: u32) -> Self {
        self.mp3_bitrate = Some(kbps);
        self
    }

    /// Use two-stems mode, separating the track into the named stem and
    /// everything else (e.g. `"vocals"` → `vocals` + `no_vocals`).
    pub fn two_stems(mut self, stem: impl Into<String>) -> Self {
        self.two_stems = Some(stem.into());
        self
    }

    /// Number of random shifts for equivariant stabilisation (default 0).
    pub fn shifts(mut self, n: u32) -> Self {
        self.shifts = Some(n);
        self
    }

    /// Overlap between prediction windows, between 0 and 1.
    pub fn overlap(mut self, overlap: f32) -> Self {
        self.overlap = Some(overlap);
        self
    }

    /// Disable track splitting (process the whole file as one chunk).
    pub fn no_split(mut self) -> Self {
        self.no_split = true;
        self
    }

    /// Set the segment length (in seconds) for split processing.
    pub fn segment(mut self, seconds: u32) -> Self {
        self.segment = Some(seconds);
        self
    }

    /// Clipping prevention mode (`"rescale"` or `"clamp"`).
    pub fn clip_mode(mut self, mode: impl Into<String>) -> Self {
        self.clip_mode = Some(mode.into());
        self
    }

    /// Enable verbose output from Demucs.
    pub fn verbose(mut self) -> Self {
        self.verbose = true;
        self
    }

    /// Add an input audio file.
    pub fn input(mut self, path: impl Into<PathBuf>) -> Self {
        self.inputs.push(path.into());
        self
    }

    /// Add multiple input audio files at once.
    pub fn inputs<I, P>(mut self, paths: I) -> Self
    where
        I: IntoIterator<Item = P>,
        P: Into<PathBuf>,
    {
        self.inputs.extend(paths.into_iter().map(Into::into));
        self
    }

    /// Validate the builder state and run Demucs.
    ///
    /// Returns [`DemucsOutput`] on success or a [`DemucsError`] on failure.
    pub fn run(self) -> Result<DemucsOutput> {
        if self.inputs.is_empty() {
            return Err(DemucsError::InvalidOption {
                message: "at least one input file must be provided".into(),
            });
        }

        // Validate that each input file exists.
        for input in &self.inputs {
            if !input.exists() {
                return Err(DemucsError::InputNotFound {
                    path: input.display().to_string(),
                });
            }
        }

        // Build the command.
        let mut cmd = build_demucs_command(&self)?;

        let output: Output = cmd.output()?;

        let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
        let stderr = String::from_utf8_lossy(&output.stderr).into_owned();

        if !output.status.success() {
            let exit_code = output.status.code().unwrap_or(-1);
            return Err(DemucsError::ProcessFailed { exit_code, stderr });
        }

        Ok(DemucsOutput { stdout, stderr })
    }
}

// ---------------------------------------------------------------------------
// Main entry-point struct
// ---------------------------------------------------------------------------

/// The primary interface for running Demucs from Rust.
///
/// Use [`Demucs::builder`] to obtain a [`DemucsBuilder`] that lets you
/// configure every aspect of the run before executing it.
pub struct Demucs;

impl Demucs {
    /// Create a new [`DemucsBuilder`].
    pub fn builder() -> DemucsBuilder {
        DemucsBuilder::default()
    }

    /// Pre-warm the managed Python environment.
    ///
    /// On the first call this downloads python-build-standalone and runs
    /// `pip install demucs`, which may take several minutes (PyTorch is
    /// large).  On subsequent calls it returns immediately.
    ///
    /// Calling this method explicitly is optional – the builder's [`run`]
    /// method will trigger setup automatically – but it allows applications
    /// to show a loading screen or progress bar before starting a separation
    /// job.
    ///
    /// [`run`]: DemucsBuilder::run
    pub fn setup() -> Result<()> {
        python_env::ensure_managed_python()?;
        Ok(())
    }

    /// Check whether Demucs is available without triggering an automatic
    /// download.
    ///
    /// Returns `true` when any of the following is true:
    /// * The managed Python environment exists and has `demucs` installed.
    /// * A standalone `demucs` binary is on `PATH`.
    /// * A system Python interpreter with `demucs` is on `PATH`.
    ///
    /// This method **never** downloads anything; use [`setup`] or the
    /// builder's [`run`] for that.
    ///
    /// [`setup`]: Demucs::setup
    /// [`run`]: DemucsBuilder::run
    pub fn is_available() -> bool {
        // 1. Check the managed environment (fast – just tests file existence
        //    and then `python -c "import demucs"`).
        let env_dir = python_env::managed_env_dir();
        let managed_python = python_env::managed_python_exe(&env_dir);
        if managed_python.exists() && python_env::is_demucs_installed(&managed_python) {
            return true;
        }

        // 2. Standalone `demucs` binary on PATH.
        if Command::new("demucs")
            .arg("--help")
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
        {
            return true;
        }

        // 3. System Python with demucs installed.
        for python in &["python3", "python"] {
            if Command::new(python)
                .args(["-m", "demucs", "--help"])
                .output()
                .map(|o| o.status.success())
                .unwrap_or(false)
            {
                return true;
            }
        }

        false
    }

    /// Return the version string reported by Demucs, or an error if it is not
    /// available.
    ///
    /// This will trigger the managed environment setup if it has not been done
    /// yet.
    pub fn version() -> Result<String> {
        let output = run_demucs_args(["--version"])?;
        // Demucs prints the version to stderr; check both.
        let text = if output.stdout.trim().is_empty() {
            output.stderr
        } else {
            output.stdout
        };
        Ok(text.trim().to_owned())
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Try to find a working Demucs command.
///
/// Priority order:
/// 1. **Managed Python environment** – always tried first. If the environment
///    is not set up yet, `ensure_managed_python` will download it now.
/// 2. Standalone `demucs` binary on `PATH` (system installation fallback).
/// 3. System Python with `demucs` installed (system installation fallback).
///
/// If none of the above succeeds, [`DemucsError::NotInstalled`] is returned.
fn demucs_command() -> Result<Command> {
    // 1. Try the managed environment.  This triggers setup on first run.
    match python_env::ensure_managed_python() {
        Ok(python) => {
            let mut cmd = Command::new(python);
            cmd.args(["-m", "demucs"]);
            return Ok(cmd);
        }
        Err(e) => {
            // Log the setup failure and fall back to system installations.
            eprintln!(
                "cargo-demucs: managed Python setup failed ({e}); \
                 falling back to system installation..."
            );
        }
    }

    // 2. Standalone `demucs` binary.
    if Command::new("demucs")
        .arg("--help")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
    {
        return Ok(Command::new("demucs"));
    }

    // 3. System Python module.
    for python in &["python3", "python"] {
        let probed = Command::new(python)
            .args(["-m", "demucs", "--help"])
            .output();

        if probed.map(|o| o.status.success()).unwrap_or(false) {
            let mut cmd = Command::new(python);
            cmd.args(["-m", "demucs"]);
            return Ok(cmd);
        }
    }

    Err(DemucsError::NotInstalled)
}

/// Run Demucs with the given raw arguments and return its output.
fn run_demucs_args<I, S>(args: I) -> Result<DemucsOutput>
where
    I: IntoIterator<Item = S>,
    S: AsRef<OsStr>,
{
    let mut cmd = demucs_command()?;
    cmd.args(args);
    let out = cmd.output()?;
    Ok(DemucsOutput {
        stdout: String::from_utf8_lossy(&out.stdout).into_owned(),
        stderr: String::from_utf8_lossy(&out.stderr).into_owned(),
    })
}

/// Translate a [`DemucsBuilder`] into a ready-to-execute [`Command`].
fn build_demucs_command(builder: &DemucsBuilder) -> Result<Command> {
    let mut cmd = demucs_command()?;

    if let Some(model) = &builder.model {
        cmd.args(["-n", model]);
    }

    if let Some(device) = &builder.device {
        cmd.args(["-d", device.as_str()]);
    }

    if let Some(output_dir) = &builder.output_dir {
        // Ensure the output directory can be created.
        if !output_dir.exists() {
            std::fs::create_dir_all(output_dir).map_err(|_| {
                DemucsError::OutputDirectoryError {
                    path: output_dir.display().to_string(),
                }
            })?;
        }
        cmd.arg("-o").arg(output_dir);
    }

    if let Some(jobs) = builder.jobs {
        cmd.args(["-j", &jobs.to_string()]);
    }

    if let Some(stem) = &builder.stem
        && let Some(s) = stem.as_str()
    {
        cmd.args(["--stem", s]);
    }

    if let Some(two_stems) = &builder.two_stems {
        cmd.args(["--two-stems", two_stems]);
    }

    match builder.format.as_ref().unwrap_or(&OutputFormat::Wav) {
        OutputFormat::Mp3 => {
            cmd.arg("--mp3");
            if let Some(kbps) = builder.mp3_bitrate {
                cmd.args(["--mp3-bitrate", &kbps.to_string()]);
            }
        }
        OutputFormat::Flac => {
            cmd.arg("--flac");
        }
        OutputFormat::Wav => {} // default; no flag needed
    }

    if let Some(shifts) = builder.shifts {
        cmd.args(["--shifts", &shifts.to_string()]);
    }

    if let Some(overlap) = builder.overlap {
        cmd.args(["--overlap", &overlap.to_string()]);
    }

    if builder.no_split {
        cmd.arg("--no-split");
    }

    if let Some(segment) = builder.segment {
        cmd.args(["--segment", &segment.to_string()]);
    }

    if let Some(clip_mode) = &builder.clip_mode {
        cmd.args(["--clip-mode", clip_mode]);
    }

    if builder.verbose {
        cmd.arg("-v");
    }

    // Input files must come last.
    for input in &builder.inputs {
        cmd.arg(input);
    }

    Ok(cmd)
}

/// Return the path that Demucs writes stems to for a given track.
///
/// Demucs writes its output to `<output_dir>/<model>/<track_stem>/`.
/// If `output_dir` is `None` the default `separated/` directory is assumed.
pub fn stem_output_path(
    output_dir: Option<&Path>,
    model: &str,
    track: &Path,
) -> PathBuf {
    let base = output_dir.unwrap_or_else(|| Path::new("separated"));
    let track_name = track
        .file_stem()
        .unwrap_or_else(|| OsStr::new("track"))
        .to_string_lossy();
    base.join(model).join(track_name.as_ref())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    // ------------------------------------------------------------------
    // stem_output_path
    // ------------------------------------------------------------------

    #[test]
    fn stem_output_path_with_output_dir() {
        let path =
            stem_output_path(Some(Path::new("/tmp/out")), "htdemucs", Path::new("/music/song.mp3"));
        assert_eq!(path, PathBuf::from("/tmp/out/htdemucs/song"));
    }

    #[test]
    fn stem_output_path_default_dir() {
        let path = stem_output_path(None, "mdx", Path::new("track.flac"));
        assert_eq!(path, PathBuf::from("separated/mdx/track"));
    }

    // ------------------------------------------------------------------
    // Stem::as_str
    // ------------------------------------------------------------------

    #[test]
    fn stem_as_str_all_returns_none() {
        assert!(Stem::All.as_str().is_none());
    }

    #[test]
    fn stem_as_str_specific_stems() {
        assert_eq!(Stem::Vocals.as_str(), Some("vocals"));
        assert_eq!(Stem::Drums.as_str(), Some("drums"));
        assert_eq!(Stem::Bass.as_str(), Some("bass"));
        assert_eq!(Stem::Other.as_str(), Some("other"));
        assert_eq!(Stem::Guitar.as_str(), Some("guitar"));
        assert_eq!(Stem::Piano.as_str(), Some("piano"));
    }

    // ------------------------------------------------------------------
    // Device::as_str
    // ------------------------------------------------------------------

    #[test]
    fn device_as_str() {
        assert_eq!(Device::Cpu.as_str(), "cpu");
        assert_eq!(Device::Cuda.as_str(), "cuda");
        assert_eq!(Device::Mps.as_str(), "mps");
        assert_eq!(Device::Custom("xpu".into()).as_str(), "xpu");
    }

    // ------------------------------------------------------------------
    // Builder validation – no input files
    // ------------------------------------------------------------------

    #[test]
    fn builder_run_without_inputs_returns_error() {
        let result = Demucs::builder().model("htdemucs").run();
        assert!(matches!(result, Err(DemucsError::InvalidOption { .. })));
    }

    // ------------------------------------------------------------------
    // Builder validation – missing input file
    // ------------------------------------------------------------------

    #[test]
    fn builder_run_with_nonexistent_input_returns_error() {
        let result = Demucs::builder()
            .input("/nonexistent/path/to/audio.mp3")
            .run();
        assert!(matches!(result, Err(DemucsError::InputNotFound { .. })));
    }

    // ------------------------------------------------------------------
    // Builder – two_stems flag
    // ------------------------------------------------------------------

    #[test]
    fn builder_two_stems_stored() {
        let builder = Demucs::builder().two_stems("vocals");
        assert_eq!(builder.two_stems.as_deref(), Some("vocals"));
    }

    // ------------------------------------------------------------------
    // Builder – chaining
    // ------------------------------------------------------------------

    #[test]
    fn builder_chaining_stores_values() {
        let builder = Demucs::builder()
            .model("htdemucs_ft")
            .device(Device::Cpu)
            .output_dir("/tmp/out")
            .jobs(4)
            .stem(Stem::Vocals)
            .format(OutputFormat::Mp3)
            .mp3_bitrate(320)
            .shifts(2)
            .overlap(0.25)
            .no_split()
            .segment(60)
            .clip_mode("rescale")
            .verbose();

        assert_eq!(builder.model.as_deref(), Some("htdemucs_ft"));
        assert_eq!(builder.device, Some(Device::Cpu));
        assert_eq!(builder.output_dir, Some(PathBuf::from("/tmp/out")));
        assert_eq!(builder.jobs, Some(4));
        assert_eq!(builder.stem, Some(Stem::Vocals));
        assert_eq!(builder.format, Some(OutputFormat::Mp3));
        assert_eq!(builder.mp3_bitrate, Some(320));
        assert_eq!(builder.shifts, Some(2));
        assert_eq!(builder.overlap, Some(0.25));
        assert!(builder.no_split);
        assert_eq!(builder.segment, Some(60));
        assert_eq!(builder.clip_mode.as_deref(), Some("rescale"));
        assert!(builder.verbose);
    }

    // ------------------------------------------------------------------
    // Builder – inputs helper
    // ------------------------------------------------------------------

    #[test]
    fn builder_inputs_bulk() {
        let files = vec!["/a/one.mp3", "/b/two.wav"];
        let builder = Demucs::builder().inputs(files.iter());
        assert_eq!(builder.inputs.len(), 2);
    }
}
