use ash::vk;

/// Possible errors resulting from the backend. This does not cover errors related to the use
/// of the backend module, but rather issues related to the hardware, driver or internal issues
/// with the backend. Shader compile errors are included to support hot reloading of shaders.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// Occurs when an unexpected result is returned by a Vulkan driver call.
    #[error("unexpected vulkan result: {0}")]
    VulkanResult(#[from] vk::Result),
    /// Occurs when the backend is unable to load the vulkan driver. This can mean that no Vulkan
    /// driver is installed, there is an issue with the driver or the driver is outdated.
    #[error("failed loading vulkan driver: {0}")]
    VulkanLoad(#[from] ash::LoadingError),
    /// Occurs when the backend is unable to find a device that supports all required extensions.
    /// This can also mean that a feature isn't available in the installed version of the driver.
    #[error("no suitable device found")]
    NoSuitableDevice,
    /// Occurs when an unsupported display server is used.
    #[error("no suitable surface found")]
    NoSuitableSurface,
    /// Issues encountered when compiling shaders.
    #[error("failed to compile shader: {0}")]
    Shader(#[from] shaderc::Error),
    /// Usually occurs when the window is resized.
    #[error("the used surface is outdated")]
    SurfaceOutdated,
}
