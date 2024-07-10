use ash::vk;
use std::{
    backtrace::{self, Backtrace},
    fmt, io,
};

#[derive(Debug, thiserror::Error)]
pub enum ErrorKind {
    #[error("vulkan failed with error: {0}")]
    Backend(#[from] vk::Result),
    #[error("failed loading with error: {0}")]
    Loading(#[from] ash::LoadingError),
    #[error("no suitable device found")]
    NoDevice,
    #[error("failed to compile shader: {0}")]
    Compilation(#[from] shaderc::Error),
    #[error("no suitable surface found")]
    NoSuitableSurface,
    #[error("failed to read scene: {0}")]
    Io(#[from] io::Error),
    #[error("missing surface")]
    MissingSurface,
    #[error("no swapchain image acquired")]
    NoSwapchainImage,
    #[error("wrong swapchain image, has index {index:?} expected {expected}")]
    WrongSwapchainImage { index: Option<u32>, expected: u32 },
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
