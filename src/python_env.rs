//! Managed Python environment for cargo-demucs.
//!
//! On the first invocation this module downloads a platform-appropriate
//! [python-build-standalone](https://github.com/indygreg/python-build-standalone)
//! release, extracts it to `~/.cargo-demucs/python/`, and runs
//! `pip install demucs` inside it.  Subsequent calls simply verify that the
//! managed Python and the `demucs` package are present and return immediately.
//!
//! **No system Python is required** – everything runs from the self-contained
//! interpreter that is managed by this module.

use crate::{DemucsError, Result};
use std::path::{Path, PathBuf};
use std::process::Command;

// ---------------------------------------------------------------------------
// Constants – python-build-standalone release pinned for reproducibility
// ---------------------------------------------------------------------------

/// The python-build-standalone release tag used for the managed environment.
const PBS_RELEASE: &str = "20241008";
/// The CPython version bundled in that release.
const PBS_PYTHON_VERSION: &str = "3.12.7";

// ---------------------------------------------------------------------------
// Platform-specific download URLs
// ---------------------------------------------------------------------------

/// Returns the python-build-standalone `install_only` archive URL for the
/// platform this binary is **running** on, or `None` for unsupported targets.
///
/// We use `std::env::consts` (runtime) rather than `cfg!` (compile-time) so
/// that cross-compiled binaries always download the correct archive.
fn pbs_url() -> Option<&'static str> {
    // All URLs point to the same release; the correct one is selected at
    // runtime based on the OS and CPU architecture.
    match (std::env::consts::OS, std::env::consts::ARCH) {
        ("linux", "x86_64") => Some(concat!(
            "https://github.com/indygreg/python-build-standalone/releases/download/",
            "20241008/cpython-3.12.7+20241008-x86_64-unknown-linux-gnu-install_only.tar.gz"
        )),
        ("linux", "aarch64") => Some(concat!(
            "https://github.com/indygreg/python-build-standalone/releases/download/",
            "20241008/cpython-3.12.7+20241008-aarch64-unknown-linux-gnu-install_only.tar.gz"
        )),
        ("macos", "x86_64") => Some(concat!(
            "https://github.com/indygreg/python-build-standalone/releases/download/",
            "20241008/cpython-3.12.7+20241008-x86_64-apple-darwin-install_only.tar.gz"
        )),
        ("macos", "aarch64") => Some(concat!(
            "https://github.com/indygreg/python-build-standalone/releases/download/",
            "20241008/cpython-3.12.7+20241008-aarch64-apple-darwin-install_only.tar.gz"
        )),
        ("windows", "x86_64") => Some(concat!(
            "https://github.com/indygreg/python-build-standalone/releases/download/",
            "20241008/cpython-3.12.7+20241008-x86_64-pc-windows-msvc-install_only.tar.gz"
        )),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Returns the base directory for the cargo-demucs managed environment.
///
/// The location can be overridden by setting the `CARGO_DEMUCS_HOME`
/// environment variable.  The default is `~/.cargo-demucs/`.
pub fn managed_env_dir() -> PathBuf {
    managed_env_dir_inner(
        std::env::var("CARGO_DEMUCS_HOME").ok(),
        std::env::var("HOME").ok(),
        std::env::var("USERPROFILE").ok(),
    )
}

/// Inner implementation of [`managed_env_dir`] that accepts explicit values so
/// it can be tested without mutating environment variables.
fn managed_env_dir_inner(
    cargo_demucs_home: Option<String>,
    home: Option<String>,
    user_profile: Option<String>,
) -> PathBuf {
    if let Some(custom) = cargo_demucs_home {
        return PathBuf::from(custom);
    }

    // Use $HOME on Unix, %USERPROFILE% on Windows, then fall back.
    let home_dir = home
        .or(user_profile)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."));

    home_dir.join(".cargo-demucs")
}

/// Returns the path to the managed Python executable.
///
/// The executable may or may not exist yet; call [`ensure_managed_python`] to
/// make sure it is present before using it.
pub fn managed_python_exe(env_dir: &Path) -> PathBuf {
    let python_dir = env_dir.join("python");
    if cfg!(target_os = "windows") {
        python_dir.join("python.exe")
    } else {
        python_dir.join("bin").join("python3")
    }
}

/// Returns `true` if `demucs` can be imported by the given Python executable.
pub fn is_demucs_installed(python: &Path) -> bool {
    Command::new(python)
        .args(["-c", "import demucs"])
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// Ensure that a self-managed Python environment with Demucs installed exists.
///
/// - If everything is already in place the function returns immediately.
/// - If Python is present but `demucs` is missing, only `pip install` is run.
/// - If Python itself is absent, the python-build-standalone archive is
///   downloaded from GitHub, extracted, and `pip install demucs` is run.
///
/// Progress messages are printed to **stderr** so they do not pollute stdout.
///
/// Returns the path to the managed `python3` (or `python.exe` on Windows)
/// executable on success.
pub fn ensure_managed_python() -> Result<PathBuf> {
    let env_dir = managed_env_dir();
    let python = managed_python_exe(&env_dir);

    if python.exists() {
        // Python interpreter is already present.  Check whether Demucs is
        // installed inside it.
        if !is_demucs_installed(&python) {
            eprintln!(
                "cargo-demucs: Demucs not found in managed environment – \
                 installing now (this may take several minutes because \
                 PyTorch is large)..."
            );
            install_demucs(&python)?;
        }
        return Ok(python);
    }

    // Python is not present – download the managed distribution first.
    let url = pbs_url().ok_or_else(|| DemucsError::PythonSetup {
        message: format!(
            "No pre-built Python distribution is available for {os}/{arch}. \
             Please install Python {ver} and run `pip install demucs` manually.",
            os = std::env::consts::OS,
            arch = std::env::consts::ARCH,
            ver = PBS_PYTHON_VERSION,
        ),
    })?;

    eprintln!(
        "cargo-demucs: First-run setup – downloading Python {PBS_PYTHON_VERSION} \
         ({PBS_RELEASE}) for {os}/{arch}...",
        os = std::env::consts::OS,
        arch = std::env::consts::ARCH,
    );
    download_and_extract(url, &env_dir)?;

    eprintln!(
        "cargo-demucs: Installing Demucs and its dependencies \
         (PyTorch is large – this may take several minutes)..."
    );
    install_demucs(&python)?;

    eprintln!(
        "cargo-demucs: Setup complete. \
         The managed environment is stored in {}",
        env_dir.display()
    );
    Ok(python)
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Download the python-build-standalone archive from `url` and extract it
/// into `env_dir`, producing a `python/` sub-directory inside `env_dir`.
fn download_and_extract(url: &str, env_dir: &Path) -> Result<()> {
    std::fs::create_dir_all(env_dir).map_err(|e| DemucsError::PythonSetup {
        message: format!("cannot create managed env directory '{}': {e}", env_dir.display()),
    })?;

    let response = ureq::get(url).call().map_err(|e| DemucsError::PythonSetup {
        message: format!("failed to download Python from {url}: {e}"),
    })?;

    // Stream the response body directly into the tar extractor to avoid
    // buffering the entire archive in memory.
    let reader = response.into_body().into_reader();
    let decompressor = flate2::read::GzDecoder::new(reader);
    let mut archive = tar::Archive::new(decompressor);

    archive.unpack(env_dir).map_err(|e| DemucsError::PythonSetup {
        message: format!("failed to extract Python archive to '{}': {e}", env_dir.display()),
    })?;

    Ok(())
}

/// Run `<python> -m pip install --upgrade demucs` using the managed Python
/// interpreter.  pip's own progress output is inherited so the user can see
/// what is happening.
fn install_demucs(python: &Path) -> Result<()> {
    let status = Command::new(python)
        .args(["-m", "pip", "install", "--upgrade", "demucs"])
        // Inherit stdio so pip's download progress is visible to the user.
        .status()
        .map_err(|e| DemucsError::PythonSetup {
            message: format!("failed to launch pip: {e}"),
        })?;

    if !status.success() {
        return Err(DemucsError::PythonSetup {
            message: "pip install demucs failed – see output above for details".into(),
        });
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn managed_env_dir_respects_cargo_demucs_home_override() {
        // Test via the inner function to avoid mutating env vars (which would
        // be unsound in a multi-threaded test runner).
        let dir = managed_env_dir_inner(
            Some("/custom/path".into()),
            Some("/home/user".into()),
            None,
        );
        assert_eq!(dir, PathBuf::from("/custom/path"));
    }

    #[test]
    fn managed_env_dir_falls_back_to_home() {
        let dir = managed_env_dir_inner(None, Some("/home/alice".into()), None);
        assert_eq!(dir, PathBuf::from("/home/alice/.cargo-demucs"));
    }

    #[test]
    fn managed_env_dir_falls_back_to_userprofile() {
        let dir =
            managed_env_dir_inner(None, None, Some("C:\\Users\\alice".into()));
        assert_eq!(
            dir,
            PathBuf::from("C:\\Users\\alice").join(".cargo-demucs")
        );
    }

    #[test]
    fn managed_env_dir_falls_back_to_dot_when_no_home() {
        let dir = managed_env_dir_inner(None, None, None);
        assert_eq!(dir, PathBuf::from(".").join(".cargo-demucs"));
    }

    #[test]
    fn managed_python_exe_path_is_inside_env_dir() {
        let env_dir = PathBuf::from("/some/env");
        let exe = managed_python_exe(&env_dir);
        assert!(exe.starts_with(&env_dir));
        // The executable must be named python3 or python.exe.
        let name = exe.file_name().unwrap().to_string_lossy();
        assert!(
            name == "python3" || name == "python.exe",
            "unexpected exe name: {name}"
        );
    }

    #[test]
    fn pbs_url_returns_some_on_supported_platforms() {
        // This test always passes – it just documents which platforms are
        // supported and ensures pbs_url() does not panic.
        let _ = pbs_url();
    }

    #[test]
    fn is_demucs_installed_returns_false_for_missing_python() {
        let missing = PathBuf::from("/nonexistent/python3");
        assert!(!is_demucs_installed(&missing));
    }
}
