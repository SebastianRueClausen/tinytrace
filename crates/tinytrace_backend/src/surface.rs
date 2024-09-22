use std::slice;

use ash::vk::SurfaceKHR;
use ash::{khr, vk};
use raw_window_handle::{RawDisplayHandle, RawWindowHandle};

use super::sync::Access;
use super::{resource, ImageFormat};
use super::{Device, Error, Image, Instance};
use crate::Extent;

pub struct Swapchain {
    surface_loader: khr::surface::Instance,
    swapchain_loader: khr::swapchain::Device,
    surface: vk::SurfaceKHR,
    swapchain: vk::SwapchainKHR,
    pub format: ImageFormat,
}

impl Swapchain {
    pub fn new(
        instance: &Instance,
        device: &Device,
        window: RawWindowHandle,
        display: RawDisplayHandle,
    ) -> Result<(Self, Vec<Image>), Error> {
        let (surface_loader, surface) = create_surface(instance, window, display)?;
        let swapchain_loader = khr::swapchain::Device::new(instance, device);
        let extent = current_extent(device, surface, &surface_loader)?;
        let (swapchain, format) = create_swapchain(
            device,
            &swapchain_loader,
            surface,
            &surface_loader,
            extent,
            vk::SwapchainKHR::null(),
        )?;
        let images = create_swapchain_images(device, &swapchain_loader, swapchain, format, extent)?;
        let swapchain = Self {
            surface_loader,
            surface,
            swapchain_loader,
            swapchain,
            format,
        };
        Ok((swapchain, images))
    }

    pub fn recreate(&mut self, device: &Device) -> Result<Vec<Image>, Error> {
        let extent = current_extent(device, self.surface, &self.surface_loader)?;
        let old_swapchain = self.swapchain;
        (self.swapchain, self.format) = create_swapchain(
            device,
            &self.swapchain_loader,
            self.surface,
            &self.surface_loader,
            extent,
            self.swapchain,
        )?;
        unsafe {
            self.swapchain_loader.destroy_swapchain(old_swapchain, None);
        }
        create_swapchain_images(
            device,
            &self.swapchain_loader,
            self.swapchain,
            self.format,
            extent,
        )
    }

    pub fn destroy(&self) {
        unsafe {
            self.swapchain_loader
                .destroy_swapchain(self.swapchain, None);
            self.surface_loader.destroy_surface(self.surface, None);
        }
    }

    pub fn image_index(&self, semaphore: vk::Semaphore) -> Result<usize, Error> {
        handle_present_result(unsafe {
            self.swapchain_loader.acquire_next_image(
                self.swapchain,
                u64::MAX,
                semaphore,
                vk::Fence::null(),
            )
        })
        .map(|(index, _)| index as usize)
    }

    pub fn present(&self, device: &Device, wait: vk::Semaphore, index: u32) -> Result<(), Error> {
        let present_info = vk::PresentInfoKHR::default()
            .image_indices(slice::from_ref(&index))
            .swapchains(slice::from_ref(&self.swapchain))
            .wait_semaphores(slice::from_ref(&wait));
        handle_present_result(unsafe {
            self.swapchain_loader
                .queue_present(device.queue, &present_info)
        })
        .map(|_| ())
    }
}

fn current_extent(
    device: &Device,
    surface: SurfaceKHR,
    surface_loader: &khr::surface::Instance,
) -> Result<Extent, Error> {
    let vk::Extent2D { width, height } = unsafe {
        surface_loader
            .get_physical_device_surface_capabilities(device.physical_device, surface)?
            .current_extent
    };
    Ok(Extent::new(width, height))
}

fn handle_present_result<T>(result: Result<T, vk::Result>) -> Result<T, Error> {
    match result {
        Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => Err(Error::SurfaceOutdated),
        Err(error) => Err(Error::VulkanResult(error)),
        Ok(value) => Ok(value),
    }
}

fn swapchain_image_usages() -> vk::ImageUsageFlags {
    vk::ImageUsageFlags::COLOR_ATTACHMENT
        | vk::ImageUsageFlags::TRANSFER_DST
        | vk::ImageUsageFlags::STORAGE
}

fn create_swapchain_images(
    device: &Device,
    loader: &khr::swapchain::Device,
    swapchain: vk::SwapchainKHR,
    format: ImageFormat,
    extent: Extent,
) -> Result<Vec<Image>, Error> {
    let images = unsafe { loader.get_swapchain_images(swapchain)? };
    images
        .into_iter()
        .enumerate()
        .map(|(index, image)| {
            Ok(Image {
                view: resource::create_image_view(device, image, format, 1)?,
                layout: vk::ImageLayout::UNDEFINED,
                access: Access::default(),
                swapchain_index: Some(index as u32),
                mip_level_count: 1,
                timestamp: 0,
                extent,
                format,
                image,
            })
        })
        .collect()
}

fn create_swapchain(
    device: &Device,
    loader: &khr::swapchain::Device,
    surface: vk::SurfaceKHR,
    surface_loader: &khr::surface::Instance,
    extent: Extent,
    old: vk::SwapchainKHR,
) -> Result<(vk::SwapchainKHR, ImageFormat), Error> {
    let surface_capabilities = unsafe {
        surface_loader.get_physical_device_surface_capabilities(device.physical_device, surface)?
    };
    let format = swapchain_format(surface_loader, surface, device.physical_device)?;
    let swapchain_info = vk::SwapchainCreateInfoKHR::default()
        .surface(surface)
        .min_image_count(surface_capabilities.min_image_count.max(2))
        .image_color_space(vk::ColorSpaceKHR::SRGB_NONLINEAR)
        .image_array_layers(1)
        .image_usage(swapchain_image_usages())
        .image_format(format.into())
        .queue_family_indices(slice::from_ref(&device.queue_family_index))
        .pre_transform(vk::SurfaceTransformFlagsKHR::IDENTITY)
        .image_extent(extent.into())
        .old_swapchain(old)
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
        .present_mode(vk::PresentModeKHR::MAILBOX);
    let swapchain = unsafe { loader.create_swapchain(&swapchain_info, None)? };
    Ok((swapchain, format))
}

fn create_surface(
    instance: &Instance,
    window: RawWindowHandle,
    display: RawDisplayHandle,
) -> Result<(khr::surface::Instance, vk::SurfaceKHR), Error> {
    let loader = khr::surface::Instance::new(&instance.entry, instance);
    let surface = match (display, window) {
        (RawDisplayHandle::Windows(_), RawWindowHandle::Win32(handle)) => {
            let info = vk::Win32SurfaceCreateInfoKHR::default()
                .hinstance(handle.hinstance.ok_or(Error::NoSuitableSurface)?.get())
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
                .dpy(display.display.ok_or(Error::NoSuitableSurface)?.as_ptr())
                .window(window.window);
            let loader = khr::xlib_surface::Instance::new(&instance.entry, instance);
            unsafe { loader.create_xlib_surface(&info, None) }
        }
        (RawDisplayHandle::Xcb(display), RawWindowHandle::Xcb(window)) => {
            let info = vk::XcbSurfaceCreateInfoKHR::default()
                .connection(display.connection.ok_or(Error::NoSuitableSurface)?.as_ptr())
                .window(window.window.get());
            let loader = khr::xcb_surface::Instance::new(&instance.entry, instance);
            unsafe { loader.create_xcb_surface(&info, None) }
        }
        _ => {
            return Err(Error::NoSuitableSurface);
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
) -> Result<ImageFormat, Error> {
    let formats = unsafe { loader.get_physical_device_surface_formats(physical_device, surface)? };
    if formats.len() == 1
        && formats
            .first()
            .is_some_and(|format| format.format == vk::Format::UNDEFINED)
    {
        return Ok(ImageFormat::Rgba8Unorm);
    }
    formats
        .into_iter()
        .find_map(|format| {
            (format.format == vk::Format::R8G8B8A8_UNORM)
                .then_some(ImageFormat::Rgba8Unorm)
                .or((format.format == vk::Format::B8G8R8A8_UNORM)
                    .then_some(ImageFormat::Bgra8Unorm))
        })
        .ok_or(Error::NoSuitableSurface)
}
