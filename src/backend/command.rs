use ash::vk;
use std::{mem, ops, slice};

use super::device::Device;
use crate::error::Error;

pub struct CommandBuffer {
    pub buffer: vk::CommandBuffer,
    is_recording: bool,
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
        Ok(Self {
            buffer: buffers[0],
            is_recording: false,
        })
    }

    pub fn destroy(&self, device: &Device) {
        unsafe {
            device.free_command_buffers(device.command_pool, &[self.buffer]);
        }
    }

    pub fn begin(&mut self, device: &Device) -> Result<(), Error> {
        if mem::replace(&mut self.is_recording, true) {
            return Ok(());
        }
        let begin_info = vk::CommandBufferBeginInfo::default();
        unsafe {
            device
                .begin_command_buffer(self.buffer, &begin_info)
                .map_err(Error::from)
        }
    }

    pub fn end(&mut self, device: &Device) -> Result<(), Error> {
        if !mem::replace(&mut self.is_recording, false) {
            return Ok(());
        }
        unsafe { device.end_command_buffer(self.buffer).map_err(Error::from) }
    }

    pub fn full_barrier(&self, device: &Device) {
        let barrier = vk::MemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .src_access_mask(vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE);
        let dependency_info =
            vk::DependencyInfo::default().memory_barriers(slice::from_ref(&barrier));
        unsafe {
            device.cmd_pipeline_barrier2(self.buffer, &dependency_info);
        }
    }
}
