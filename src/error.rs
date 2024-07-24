use crate::backend;
use std::{
    backtrace::{self, Backtrace},
    fmt, io,
};

#[derive(Debug, thiserror::Error)]
pub enum ErrorKind {
    #[error("backend failed with error: {0}")]
    Backend(#[from] backend::Error),
    #[error("failed to read scene: {0}")]
    Io(#[from] io::Error),
}

#[derive(Debug)]
pub struct Error {
    pub kind: ErrorKind,
    pub backtrace: backtrace::Backtrace,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.kind.fmt(f)
    }
}

impl<T: Into<ErrorKind>> From<T> for Error {
    fn from(value: T) -> Self {
        Self {
            kind: value.into(),
            backtrace: Backtrace::capture(),
        }
    }
}

pub type Result<T> = std::result::Result<T, Error>;
