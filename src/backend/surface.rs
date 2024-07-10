use std::slice;

use ash::{khr, vk};
use raw_window_handle::{RawDisplayHandle, RawWindowHandle};

use crate::error::ErrorKind;

use super::resource;
use super::sync::Access;
use super::{Device, Error, Image, Instance, Result};

pub struct Swapchain {
    surface_loader: khr::surface::Instance,
    swapchain_loader: khr::swapchain::Device,
    surface: vk::SurfaceKHR,
    swapchain: vk::SwapchainKHR,
    pub format: vk::Format,
}

impl Swapchain {
    pub fn new(
        instance: &Instance,
        device: &Device,
        window: RawWindowHandle,
        display: RawDisplayHandle,
        extent: vk::Extent2D,
    ) -> Result<(Self, Vec<Image>)> {
        let (surface_loader, surface) = create_surface(instance, window, display)?;
        let surface_capabilities = unsafe {
            surface_loader
                .get_physical_device_surface_capabilities(device.physical_device, surface)?
        };
        let format = swapchain_format(&surface_loader, surface, device.physical_device)?;
        let usages = vk::ImageUsageFlags::COLOR_ATTACHMENT
            | vk::ImageUsageFlags::TRANSFER_DST
            | vk::ImageUsageFlags::STORAGE;
        let swapchain_info = vk::SwapchainCreateInfoKHR::default()
            .surface(surface)
            .min_image_count(surface_capabilities.min_image_count.max(2))
            .image_color_space(vk::ColorSpaceKHR::SRGB_NONLINEAR)
            .image_array_layers(1)
            .image_usage(usages)
            .image_format(format)
            .queue_family_indices(slice::from_ref(&device.queue_family_index))
            .pre_transform(vk::SurfaceTransformFlagsKHR::IDENTITY)
            .image_extent(extent)
            .composite_alpha({
                let composite_modes = [
                    vk::CompositeAlphaFlagsKHR::OPAQUE,
                    vk::CompositeAlphaFlagsKHR::PRE_MULTIPLIED,
                    vk::CompositeAlphaFlagsKHR::POST_MULTIPLIED,
                ];
                composite_modes
                    .into_iter()
                    .find(|mode| {
                        surface_capabilities
                            .supported_composite_alpha
                            .contains(*mode)
                    })
                    .unwrap_or(vk::CompositeAlphaFlagsKHR::INHERIT)
            })
            .present_mode(vk::PresentModeKHR::FIFO);
        let swapchain_loader = khr::swapchain::Device::new(instance, device);
        let swapchain = unsafe { swapchain_loader.create_swapchain(&swapchain_info, None)? };
        let images = unsafe {
            swapchain_loader
                .get_swapchain_images(swapchain)
                .map_err(Error::from)?
        };
        let images: Vec<_> = images
            .into_iter()
            .enumerate()
            .map(|(index, image)| {
                Ok(Image {
                    view: resource::create_image_view(device, image, format, 1)?,
                    layout: vk::ImageLayout::UNDEFINED,
                    aspect: vk::ImageAspectFlags::COLOR,
                    extent: extent.into(),
                    access: Access::default(),
                    timestamp: 0,
                    swapchain_index: Some(index as u32),
                    usage_flags: usages,
                    mip_level_count: 1,
                    format,
                    image,
                })
            })
            .collect::<Result<_>>()?;
        let swapchain = Self {
            surface_loader,
            surface,
            swapchain_loader,
            swapchain,
            format,
        };
        Ok((swapchain, images))
    }

    pub fn destroy(&self, _device: &Device) {
        unsafe {
            self.swapchain_loader
                .destroy_swapchain(self.swapchain, None);
            self.surface_loader.destroy_surface(self.surface, None);
        }
    }

    pub fn image_index(&self, semaphore: vk::Semaphore) -> Result<u32> {
        let (index, _outdated) = unsafe {
            self.swapchain_loader
                .acquire_next_image(self.swapchain, u64::MAX, semaphore, vk::Fence::null())
                .map_err(Error::from)?
        };
        // TODO: Handle outdated.
        Ok(index)
    }

    pub fn present(&self, device: &Device, wait: vk::Semaphore, index: u32) -> Result<()> {
        let present_info = vk::PresentInfoKHR::default()
            .image_indices(slice::from_ref(&index))
            .swapchains(slice::from_ref(&self.swapchain))
            .wait_semaphores(slice::from_ref(&wait));
        // TODO: Handle results.
        let _result = unsafe {
            self.swapchain_loader
                .queue_present(device.queue, &present_info)
        };
        Ok(())
    }
}

pub fn create_surface(
    instance: &Instance,
    window: RawWindowHandle,
    display: RawDisplayHandle,
) -> Result<(khr::surface::Instance, vk::SurfaceKHR)> {
    let loader = khr::surface::Instance::new(&instance.entry, instance);
    let surface = match (display, window) {
        (RawDisplayHandle::Windows(_), RawWindowHandle::Win32(handle)) => {
            let info = vk::Win32SurfaceCreateInfoKHR::default()
                .hinstance(
                    handle
                        .hinstance
                        .ok_or(Error::from(ErrorKind::NoSuitableSurface))?
                        .get(),
                )
                .hwnd(handle.hwnd.get());
            let loader = khr::win32_surface::Instance::new(&instance.entry, &instance.instance);
            unsafe { loader.create_win32_surface(&info, None) }
        }
        (RawDisplayHandle::Wayland(display), RawWindowHandle::Wayland(window)) => {
            let info = vk::WaylandSurfaceCreateInfoKHR::default()
                .display(display.display.as_ptr())
                .surface(window.surface.as_ptr());
            let loader = khr::wayland_surface::Instance::new(&instance.entry, instance);
            unsafe { loader.create_wayland_surface(&info, None) }
        }
        (RawDisplayHandle::Xlib(display), RawWindowHandle::Xlib(window)) => {
            let info = vk::XlibSurfaceCreateInfoKHR::default()
                .dpy(
                    display
                        .display
                        .ok_or(Error::from(ErrorKind::NoSuitableSurface))?
                        .as_ptr(),
                )
                .window(window.window);
            let loader = khr::xlib_surface::Instance::new(&instance.entry, instance);
            unsafe { loader.create_xlib_surface(&info, None) }
        }
        (RawDisplayHandle::Xcb(display), RawWindowHandle::Xcb(window)) => {
            let info = vk::XcbSurfaceCreateInfoKHR::default()
                .connection(
                    display
                        .connection
                        .ok_or(Error::from(ErrorKind::NoSuitableSurface))?
                        .as_ptr(),
                )
                .window(window.window.get());
            let loader = khr::xcb_surface::Instance::new(&instance.entry, instance);
            unsafe { loader.create_xcb_surface(&info, None) }
        }
        _ => {
            return Err(Error::from(ErrorKind::NoSuitableSurface));
        }
    };
    surface
        .map_err(Error::from)
        .map(|surface| (loader, surface))
}

fn swapchain_format(
    loader: &khr::surface::Instance,
    surface: vk::SurfaceKHR,
    physical_device: vk::PhysicalDevice,
) -> Result<vk::Format> {
    let formats = unsafe {
        loader
            .get_physical_device_surface_formats(physical_device, surface)
            .expect("no suitable swapchain formats")
    };
    if formats.len() == 1
        && formats
            .first()
            .is_some_and(|format| format.format == vk::Format::UNDEFINED)
    {
        return Ok(vk::Format::R8G8B8A8_UNORM);
    }
    let format = formats
        .into_iter()
        .find(|format| {
            format.format == vk::Format::R8G8B8A8_UNORM
                || format.format == vk::Format::B8G8R8A8_UNORM
        })
        .expect("no suitable swapchain formats");
    Ok(format.format)
}
