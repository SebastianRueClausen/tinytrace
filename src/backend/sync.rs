use std::ops;

use super::{Buffer, Context, Handle, Image};
use ash::vk;

#[derive(Debug, Clone, Copy, Default)]
pub struct Access {
    pub stage: vk::PipelineStageFlags2,
    pub access: vk::AccessFlags2,
}

impl Access {
    fn writes(&self) -> bool {
        let write_set = vk::AccessFlags2::MEMORY_WRITE
            | vk::AccessFlags2::SHADER_STORAGE_WRITE
            | vk::AccessFlags2::TRANSFER_WRITE;
        write_set.contains(self.access)
    }
}

impl ops::BitOrAssign for Access {
    fn bitor_assign(&mut self, rhs: Self) {
        self.access |= rhs.access;
        self.stage |= rhs.stage;
    }
}

#[derive(Debug, Clone)]
pub struct ImageAccess {
    pub image: Handle<Image>,
    pub access: Access,
    pub layout: vk::ImageLayout,
}

#[derive(Debug, Clone)]
pub struct BufferAccess {
    pub buffer: Handle<Buffer>,
    pub access: Access,
}

impl Context {
    fn pipeline_barriers(
        &mut self,
        image_barriers: &[ImageAccess],
        buffer_barriers: &[BufferAccess],
    ) {
        let image_barriers: Vec<_> = image_barriers
            .iter()
            .map(|access| {
                let image = self.image_mut(&access.image);
                let subresource_range = vk::ImageSubresourceRange::default()
                    .aspect_mask(image.aspect)
                    .base_mip_level(0)
                    .level_count(image.mip_level_count)
                    .base_array_layer(0)
                    .layer_count(1);
                let barrier = vk::ImageMemoryBarrier2::default()
                    .src_access_mask(image.access.access)
                    .dst_access_mask(access.access.access)
                    .src_stage_mask(image.access.stage)
                    .dst_stage_mask(access.access.stage)
                    .old_layout(image.layout)
                    .new_layout(access.layout)
                    .image(**image)
                    .subresource_range(subresource_range);
                image.access = access.access;
                image.layout = access.layout;
                barrier
            })
            .collect();
        let buffer_barriers: Vec<_> = buffer_barriers
            .iter()
            .map(|access| {
                let buffer = self.buffer_mut(&access.buffer);
                let barrier = vk::BufferMemoryBarrier2::default()
                    .src_access_mask(buffer.access.access)
                    .dst_access_mask(access.access.access)
                    .src_stage_mask(buffer.access.stage)
                    .dst_stage_mask(access.access.stage)
                    .buffer(**buffer)
                    .offset(0)
                    .size(vk::WHOLE_SIZE);
                buffer.access = access.access;
                barrier
            })
            .collect();
        let dependency_info = vk::DependencyInfo::default()
            .image_memory_barriers(&image_barriers)
            .buffer_memory_barriers(&buffer_barriers);
        unsafe {
            self.device
                .cmd_pipeline_barrier2(self.command_buffer.buffer, &dependency_info);
        }
    }

    pub fn access_resources(&mut self, images: &[ImageAccess], buffers: &[BufferAccess]) {
        let images: Vec<_> = images
            .iter()
            .filter(|access| {
                let image = self.image_mut(&access.image);
                let has_dependency = access.access.writes() || image.access.writes();
                if !has_dependency || access.layout == image.layout {
                    image.access |= access.access;
                    false
                } else {
                    true
                }
            })
            .cloned()
            .collect();
        let buffers: Vec<_> = buffers
            .iter()
            .filter(|access| {
                let buffer = self.buffer_mut(&access.buffer);
                let has_dependency = buffer.access.writes() || access.access.writes();
                if !has_dependency {
                    buffer.access |= access.access;
                }
                has_dependency
            })
            .cloned()
            .collect();
        self.pipeline_barriers(&images, &buffers);
    }
}
