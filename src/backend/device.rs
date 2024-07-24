use std::ffi::CStr;
use std::{ffi, ops, slice};

use super::instance::Instance;
use super::Error;
use ash::{ext, khr, vk};

pub struct Device {
    pub device: ash::Device,
    pub physical_device: vk::PhysicalDevice,
    pub memory_properties: vk::PhysicalDeviceMemoryProperties,
    pub queue_family_index: u32,
    pub queue: vk::Queue,
    pub descriptor_buffer: ext::descriptor_buffer::Device,
    pub acceleration_structure: khr::acceleration_structure::Device,
    pub descriptor_buffer_properties: vk::PhysicalDeviceDescriptorBufferPropertiesEXT<'static>,
}

impl ops::Deref for Device {
    type Target = ash::Device;

    fn deref(&self) -> &Self::Target {
        &self.device
    }
}

impl Device {
    pub fn new(instance: &Instance) -> Result<Self, Error> {
        let (physical_device, queue_family_index) = select_physical_device(instance)?;
        let memory_properties =
            unsafe { instance.get_physical_device_memory_properties(physical_device) };
        let queue_info = vk::DeviceQueueCreateInfo::default()
            .queue_family_index(queue_family_index)
            .queue_priorities(&[1.0]);
        let mut features = vk::PhysicalDeviceFeatures2::default().features({
            vk::PhysicalDeviceFeatures::default()
                .sampler_anisotropy(true)
                .shader_int16(true)
        });
        let mut vulkan_1_1_features = vk::PhysicalDeviceVulkan11Features::default()
            .storage_buffer16_bit_access(true)
            .uniform_and_storage_buffer16_bit_access(true);
        let mut vulkan_1_2_features = vk::PhysicalDeviceVulkan12Features::default()
            .buffer_device_address(true)
            .descriptor_binding_variable_descriptor_count(true)
            .runtime_descriptor_array(true)
            .shader_float16(true)
            .timeline_semaphore(true);
        let mut vulkan_1_3_features = vk::PhysicalDeviceVulkan13Features::default()
            .synchronization2(true)
            .maintenance4(true);
        let mut descriptor_buffer_features =
            vk::PhysicalDeviceDescriptorBufferFeaturesEXT::default()
                .descriptor_buffer(true)
                .descriptor_buffer_image_layout_ignored(true);
        let mut acceleration_structure_features =
            vk::PhysicalDeviceAccelerationStructureFeaturesKHR::default()
                .acceleration_structure(true);
        let mut ray_query_features =
            vk::PhysicalDeviceRayQueryFeaturesKHR::default().ray_query(true);
        let mut position_fetch_features =
            vk::PhysicalDeviceRayTracingPositionFetchFeaturesKHR::default()
                .ray_tracing_position_fetch(true);
        let device_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(slice::from_ref(&queue_info))
            .enabled_extension_names(EXTENSIONS)
            .push_next(&mut features)
            .push_next(&mut vulkan_1_1_features)
            .push_next(&mut vulkan_1_2_features)
            .push_next(&mut vulkan_1_3_features)
            .push_next(&mut descriptor_buffer_features)
            .push_next(&mut acceleration_structure_features)
            .push_next(&mut ray_query_features)
            .push_next(&mut position_fetch_features);
        let device = unsafe { instance.create_device(physical_device, &device_info, None)? };
        let queue = unsafe { device.get_device_queue(queue_family_index, 0) };
        let descriptor_buffer_properties =
            get_descriptor_buffer_properties(instance, physical_device);
        Ok(Self {
            acceleration_structure: khr::acceleration_structure::Device::new(instance, &device),
            descriptor_buffer: ext::descriptor_buffer::Device::new(instance, &device),
            descriptor_buffer_properties,
            queue_family_index,
            memory_properties,
            physical_device,
            queue,
            device,
        })
    }

    pub fn wait_until_idle(&self) -> Result<(), Error> {
        unsafe { self.device_wait_idle().map_err(Error::from) }
    }

    pub fn destroy(&self) {
        unsafe {
            self.device.destroy_device(None);
        }
    }
}

fn get_graphics_queue_family_index(
    instance: &Instance,
    physical_device: vk::PhysicalDevice,
) -> Option<u32> {
    unsafe { instance.get_physical_device_queue_family_properties(physical_device) }
        .into_iter()
        .enumerate()
        .find_map(|(queue_index, queue)| {
            queue
                .queue_flags
                .contains(vk::QueueFlags::GRAPHICS)
                .then_some(queue_index as u32)
        })
}

fn get_descriptor_buffer_properties(
    instance: &Instance,
    physical_device: vk::PhysicalDevice,
) -> vk::PhysicalDeviceDescriptorBufferPropertiesEXT<'static> {
    unsafe {
        let mut descriptor_buffer_properties =
            vk::PhysicalDeviceDescriptorBufferPropertiesEXT::default();
        let mut props =
            vk::PhysicalDeviceProperties2::default().push_next(&mut descriptor_buffer_properties);
        instance.get_physical_device_properties2(physical_device, &mut props);
        descriptor_buffer_properties
    }
}

fn select_physical_device(instance: &Instance) -> Result<(vk::PhysicalDevice, u32), Error> {
    let (mut fallback, mut preferred) = (None, None);
    unsafe {
        for physical_device in instance.enumerate_physical_devices()? {
            let Some(queue_index) = get_graphics_queue_family_index(instance, physical_device)
            else {
                continue;
            };
            let properties = instance.get_physical_device_properties(physical_device);
            if properties.api_version < vk::make_api_version(0, 1, 3, 0) {
                continue;
            }
            let Ok(extensions) = instance.enumerate_device_extension_properties(physical_device)
            else {
                continue;
            };
            let has_all_extensions = EXTENSIONS.iter().all(|extension| {
                extensions.iter().any(|properties| {
                    properties
                        .extension_name_as_c_str()
                        .map(|name| name == CStr::from_ptr(*extension as *const ffi::c_char))
                        .unwrap_or(false)
                })
            });
            if !has_all_extensions {
                continue;
            }
            if properties.device_type == vk::PhysicalDeviceType::DISCRETE_GPU {
                preferred.get_or_insert((physical_device, queue_index));
            }
            fallback.get_or_insert((physical_device, queue_index));
        }
    }
    preferred.or(fallback).ok_or(Error::NoSuitableDevice)
}

const EXTENSIONS: &[*const ffi::c_char] = &[
    ext::descriptor_buffer::NAME.as_ptr(),
    khr::swapchain::NAME.as_ptr(),
    khr::acceleration_structure::NAME.as_ptr(),
    khr::deferred_host_operations::NAME.as_ptr(),
    khr::ray_query::NAME.as_ptr(),
    khr::ray_tracing_position_fetch::NAME.as_ptr(),
    khr::spirv_1_4::NAME.as_ptr(),
];
