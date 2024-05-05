use ash::vk;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("vulkan failed with error: {0}")]
    Backend(#[from] vk::Result),
    #[error("failed loading with error: {0}")]
    Loading(#[from] ash::LoadingError),
    #[error("no suitable device found")]
    NoDevice,
}
