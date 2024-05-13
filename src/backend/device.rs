use crate::error::Error;
use std::ops;
use std::slice;

use super::instance::Instance;
use ash::{ext, khr, vk};

pub struct Device {
    pub device: ash::Device,
    pub physical_device: vk::PhysicalDevice,
    pub memory_properties: vk::PhysicalDeviceMemoryProperties,
    pub queue_family_index: u32,
    pub queue: vk::Queue,
    pub command_pool: vk::CommandPool,
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
        let extensions = [
            ext::descriptor_buffer::NAME.as_ptr(),
            khr::swapchain::NAME.as_ptr(),
            khr::deferred_host_operations::NAME.as_ptr(),
            khr::acceleration_structure::NAME.as_ptr(),
            khr::ray_tracing_pipeline::NAME.as_ptr(),
        ];
        let mut features = vk::PhysicalDeviceFeatures2::default().features({
            vk::PhysicalDeviceFeatures::default()
                .multi_draw_indirect(true)
                .pipeline_statistics_query(true)
                .sampler_anisotropy(true)
                .shader_int16(true)
                .shader_int64(true)
        });
        let mut features_1_1 = vk::PhysicalDeviceVulkan11Features::default()
            .storage_buffer16_bit_access(true)
            .uniform_and_storage_buffer16_bit_access(true)
            .shader_draw_parameters(true);
        let mut features_1_2 = vk::PhysicalDeviceVulkan12Features::default()
            .buffer_device_address(true)
            .descriptor_binding_variable_descriptor_count(true)
            .runtime_descriptor_array(true)
            .shader_float16(true)
            .scalar_block_layout(true);
        let mut features_1_3 = vk::PhysicalDeviceVulkan13Features::default()
            .dynamic_rendering(true)
            .synchronization2(true)
            .maintenance4(true);
        let mut features_descriptor_buffer =
            vk::PhysicalDeviceDescriptorBufferFeaturesEXT::default()
                .descriptor_buffer(true)
                .descriptor_buffer_image_layout_ignored(true);
        let mut features_acc_struct = vk::PhysicalDeviceAccelerationStructureFeaturesKHR::default()
            .acceleration_structure(true);
        let device_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(slice::from_ref(&queue_info))
            .enabled_extension_names(&extensions)
            .push_next(&mut features)
            .push_next(&mut features_1_1)
            .push_next(&mut features_1_2)
            .push_next(&mut features_1_3)
            .push_next(&mut features_descriptor_buffer)
            .push_next(&mut features_acc_struct);
        let device = unsafe { instance.create_device(physical_device, &device_info, None)? };
        let queue = unsafe { device.get_device_queue(queue_family_index, 0) };
        let command_pool = unsafe {
            let info = vk::CommandPoolCreateInfo::default()
                .flags(vk::CommandPoolCreateFlags::empty())
                .queue_family_index(queue_family_index);
            device.create_command_pool(&info, None)?
        };
        let descriptor_buffer_properties =
            get_descriptor_buffer_properties(instance, physical_device);
        Ok(Self {
            acceleration_structure: khr::acceleration_structure::Device::new(instance, &device),
            descriptor_buffer: ext::descriptor_buffer::Device::new(instance, &device),
            descriptor_buffer_properties,
            queue_family_index,
            memory_properties,
            physical_device,
            command_pool,
            queue,
            device,
        })
    }

    pub fn wait_until_idle(&self) -> Result<(), Error> {
        unsafe { self.device_wait_idle().map_err(Error::from) }
    }

    pub fn clear_command_pool(&self) -> Result<(), Error> {
        unsafe {
            let flags = vk::CommandPoolResetFlags::RELEASE_RESOURCES;
            self.device
                .reset_command_pool(self.command_pool, flags)
                .map_err(Error::from)
        }
    }

    pub fn destroy(&self) {
        unsafe {
            self.device.destroy_command_pool(self.command_pool, None);
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
            if properties.device_type == vk::PhysicalDeviceType::DISCRETE_GPU {
                preferred.get_or_insert((physical_device, queue_index));
            }
            fallback.get_or_insert((physical_device, queue_index));
        }
    }
    preferred.or(fallback).ok_or(Error::NoDevice)
}
