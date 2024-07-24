use ash::vk;

use super::{
    device::Device, shader::DescriptorBuffer, sync::Sync, Allocator, Buffer, BufferRequest,
    BufferType, Error, MemoryLocation,
};

#[derive(Debug)]
pub struct CommandBuffer {
    pub pool: vk::CommandPool,
    pub buffer: vk::CommandBuffer,
    pub descriptor_buffer: DescriptorBuffer,
    /// The timestamp signaled when all commands have executed.
    pub timestamp: u64,
}

impl CommandBuffer {
    pub fn new(device: &Device, allocator: &mut Allocator) -> Result<Self, Error> {
        let pool = unsafe {
            let info =
                vk::CommandPoolCreateInfo::default().queue_family_index(device.queue_family_index);
            device.create_command_pool(&info, None)?
        };
        let allocate_info = vk::CommandBufferAllocateInfo::default()
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_pool(pool)
            .command_buffer_count(1);
        let buffers = unsafe { device.allocate_command_buffers(&allocate_info)? };
        let request = BufferRequest {
            size: 1024 * 1024 * 4,
            ty: BufferType::Descriptor,
            memory_location: MemoryLocation::Host,
        };
        let buffer = Buffer::new(device, allocator, &request)?;
        let descriptor_buffer = DescriptorBuffer {
            data: allocator.map(device, buffer.memory_index)?,
            size: request.size as usize,
            bound_range: Default::default(),
            buffer,
        };
        Ok(Self {
            descriptor_buffer,
            buffer: buffers[0],
            timestamp: 0,
            pool,
        })
    }

    pub fn begin(&self, device: &Device) -> Result<(), Error> {
        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        unsafe {
            device
                .begin_command_buffer(self.buffer, &begin_info)
                .map_err(Error::from)?;
            let binding_infos = [vk::DescriptorBufferBindingInfoEXT::default()
                .address(self.descriptor_buffer.buffer.device_address(device))
                .usage(self.descriptor_buffer.buffer.usage_flags)];
            device
                .descriptor_buffer
                .cmd_bind_descriptor_buffers(self.buffer, &binding_infos);
        }
        Ok(())
    }

    pub fn end(&mut self, device: &Device, timestamp: u64) -> Result<(), Error> {
        self.descriptor_buffer.bound_range = 0..0;
        self.timestamp = timestamp;
        unsafe { device.end_command_buffer(self.buffer).map_err(Error::from) }
    }

    pub fn clear(&self, semaphores: &Sync, device: &Device) -> Result<(), Error> {
        semaphores.wait_for_timestamp(device, self.timestamp)?;
        unsafe {
            let flags = vk::CommandPoolResetFlags::RELEASE_RESOURCES;
            device
                .reset_command_pool(self.pool, flags)
                .map_err(Error::from)
        }
    }

    pub fn destroy(&self, device: &Device) {
        unsafe {
            self.descriptor_buffer.buffer.destroy(device);
            device.destroy_command_pool(self.pool, None);
        }
    }
}
