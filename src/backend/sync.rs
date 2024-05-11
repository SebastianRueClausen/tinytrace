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

    fn image_layout(&self) -> vk::ImageLayout {
        if self.access.contains(vk::AccessFlags2::TRANSFER_READ) {
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL
        } else if self.access.contains(vk::AccessFlags2::TRANSFER_WRITE) {
            vk::ImageLayout::TRANSFER_DST_OPTIMAL
        } else if !self.access.contains(vk::AccessFlags2::SHADER_WRITE) {
            vk::ImageLayout::READ_ONLY_OPTIMAL
        } else {
            vk::ImageLayout::GENERAL
        }
    }
}

impl ops::BitOrAssign for Access {
    fn bitor_assign(&mut self, rhs: Self) {
        self.access |= rhs.access;
        self.stage |= rhs.stage;
    }
}

impl Context {
    fn pipeline_barriers(
        &mut self,
        image_barriers: &[(Handle<Image>, Access)],
        buffer_barriers: &[(Handle<Buffer>, Access)],
    ) {
        let image_barriers: Vec<_> = image_barriers
            .iter()
            .map(|(handle, access)| {
                let image = self.image_mut(handle);
                let subresource_range = vk::ImageSubresourceRange::default()
                    .aspect_mask(image.aspect)
                    .base_mip_level(0)
                    .level_count(image.mip_level_count)
                    .base_array_layer(0)
                    .layer_count(1);
                let barrier = vk::ImageMemoryBarrier2::default()
                    .src_access_mask(image.access.access)
                    .dst_access_mask(access.access)
                    .src_stage_mask(image.access.stage)
                    .dst_stage_mask(access.stage)
                    .old_layout(image.layout)
                    .new_layout(access.image_layout())
                    .image(**image)
                    .subresource_range(subresource_range);
                image.access = *access;
                image.layout = access.image_layout();
                barrier
            })
            .collect();
        let buffer_barriers: Vec<_> = buffer_barriers
            .iter()
            .map(|(handle, access)| {
                let buffer = self.buffer_mut(handle);
                let barrier = vk::BufferMemoryBarrier2::default()
                    .src_access_mask(buffer.access.access)
                    .dst_access_mask(access.access)
                    .src_stage_mask(buffer.access.stage)
                    .dst_stage_mask(access.stage)
                    .buffer(**buffer)
                    .offset(0)
                    .size(vk::WHOLE_SIZE);
                buffer.access = *access;
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

    pub fn access_resources(
        &mut self,
        images: &[(Handle<Image>, Access)],
        buffers: &[(Handle<Buffer>, Access)],
    ) {
        let images: Vec<_> = images
            .iter()
            .filter(|(handle, access)| {
                let image = self.image_mut(handle);
                let has_dependency = access.writes()
                    || image.access.writes()
                    || access.image_layout() != image.layout;
                if has_dependency {
                    image.access |= *access;
                }
                has_dependency
            })
            .cloned()
            .collect();
        let buffers: Vec<_> = buffers
            .iter()
            .filter(|(handle, access)| {
                let buffer = self.buffer_mut(handle);
                let has_dependency = access.writes() || buffer.access.writes();
                if !has_dependency {
                    buffer.access |= *access;
                }
                has_dependency
            })
            .cloned()
            .collect();
        self.pipeline_barriers(&images, &buffers);
    }
}
