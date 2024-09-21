use std::io;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("backend failed with error: {0}")]
    Backend(#[from] tinytrace_backend::Error),
    #[error("failed to read scene: {0}")]
    Io(#[from] io::Error),
}
