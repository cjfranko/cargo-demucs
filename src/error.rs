use thiserror::Error;

/// Errors that can occur while using cargo-demucs.
#[derive(Debug, Error)]
pub enum DemucsError {
    /// Demucs (or the required Python interpreter) was not found on the system.
    #[error("Demucs is not installed or not reachable. Install it with: pip install demucs")]
    NotInstalled,

    /// The requested audio input file does not exist.
    #[error("Input file not found: {path}")]
    InputNotFound { path: String },

    /// The output directory could not be created or written to.
    #[error("Cannot create or write to output directory: {path}")]
    OutputDirectoryError { path: String },

    /// An invalid or unrecognised option was passed to the builder.
    #[error("Invalid option: {message}")]
    InvalidOption { message: String },

    /// The Demucs process exited with a non-zero status code.
    #[error("Demucs process failed (exit code {exit_code}): {stderr}")]
    ProcessFailed { exit_code: i32, stderr: String },

    /// An I/O error occurred while spawning or communicating with the process.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}
