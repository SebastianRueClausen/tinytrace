use ash::vk;
use std::ops;

use super::{device::Device, Buffer};
use crate::error::Error;

pub struct CommandBuffer {
    pub buffer: vk::CommandBuffer,
}

impl ops::Deref for CommandBuffer {
    type Target = vk::CommandBuffer;

    fn deref(&self) -> &Self::Target {
        &self.buffer
    }
}

impl CommandBuffer {
    pub fn new(device: &Device) -> Result<Self, Error> {
        let allocate_info = vk::CommandBufferAllocateInfo::default()
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_pool(device.command_pool)
            .command_buffer_count(1);
        let buffers = unsafe { device.allocate_command_buffers(&allocate_info)? };
        Ok(Self { buffer: buffers[0] })
    }

    pub fn begin(&self, device: &Device, descriptor_buffer: &Buffer) -> Result<(), Error> {
        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        unsafe {
            device
                .begin_command_buffer(self.buffer, &begin_info)
                .map_err(Error::from)?;
            let binding_infos = [vk::DescriptorBufferBindingInfoEXT::default()
                .address(descriptor_buffer.device_address(device))
                .usage(descriptor_buffer.usage_flags)];
            device
                .descriptor_buffer
                .cmd_bind_descriptor_buffers(**self, &binding_infos);
        }
        Ok(())
    }

    pub fn end(&self, device: &Device) -> Result<(), Error> {
        unsafe { device.end_command_buffer(self.buffer).map_err(Error::from) }
    }
}
