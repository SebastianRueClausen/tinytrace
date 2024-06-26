use ash::vk;
use std::io;

#[derive(Debug, thiserror::Error)]
pub enum Error {
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
}

pub type Result<T> = std::result::Result<T, Error>;
