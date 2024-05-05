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
        let update_access = |access: Access, resource_access: &mut Access| {
            let has_dependency = resource_access.writes() || access.writes();
            if !has_dependency {
                resource_access.access |= access.access;
            }
            has_dependency
        };
        let images: Vec<_> = images
            .iter()
            .filter(|access| {
                update_access(access.access, &mut self.image_mut(&access.image).access)
            })
            .cloned()
            .collect();
        let buffers: Vec<_> = buffers
            .iter()
            .filter(|access| {
                update_access(access.access, &mut self.buffer_mut(&access.buffer).access)
            })
            .cloned()
            .collect();
        self.pipeline_barriers(&images, &buffers);
    }
}
