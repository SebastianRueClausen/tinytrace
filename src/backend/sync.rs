use std::ops;

use super::{Blas, Buffer, Context, Device, Error, Handle, Image, Result, Tlas};
use ash::vk;

#[derive(Debug, Clone, Copy, Default, PartialEq, PartialOrd, Eq, Ord)]
pub struct Access {
    pub stage: vk::PipelineStageFlags2,
    pub access: vk::AccessFlags2,
}

impl Access {
    fn writes(&self) -> bool {
        let write_set = vk::AccessFlags2::MEMORY_WRITE
            | vk::AccessFlags2::SHADER_STORAGE_WRITE
            | vk::AccessFlags2::TRANSFER_WRITE
            | vk::AccessFlags2::ACCELERATION_STRUCTURE_WRITE_KHR;
        write_set.intersects(self.access)
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
        memory_barriers: &[(Access, Access)],
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
                    .image(image.image)
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
                    .buffer(buffer.buffer)
                    .offset(0)
                    .size(vk::WHOLE_SIZE);
                buffer.access = *access;
                barrier
            })
            .collect();
        let memory_barriers: Vec<_> = memory_barriers
            .iter()
            .map(|(src, dst)| {
                vk::MemoryBarrier2::default()
                    .src_access_mask(src.access)
                    .dst_access_mask(dst.access)
                    .src_stage_mask(src.stage)
                    .dst_stage_mask(dst.stage)
            })
            .collect();
        let dependency_info = vk::DependencyInfo::default()
            .image_memory_barriers(&image_barriers)
            .buffer_memory_barriers(&buffer_barriers)
            .memory_barriers(&memory_barriers);
        unsafe {
            self.device
                .cmd_pipeline_barrier2(self.command_buffer.buffer, &dependency_info);
        }
    }

    pub fn access_resources(
        &mut self,
        images: &[(Handle<Image>, Access)],
        buffers: &[(Handle<Buffer>, Access)],
        blases: &[(Handle<Blas>, Access)],
        tlases: &[(Handle<Tlas>, Access)],
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
        let structure_access = |structure: &mut Access, access: Access| {
            let has_dependency = structure.writes() || access.writes();
            if has_dependency {
                *structure = Access::default();
            }
            *structure |= access;
            has_dependency.then_some((access, *structure))
        };
        let mut memory: Vec<_> = blases
            .iter()
            .filter_map(|(handle, access)| {
                structure_access(&mut self.blas_mut(handle).access, *access)
            })
            .collect();
        memory.extend(tlases.iter().filter_map(|(handle, access)| {
            structure_access(&mut self.tlas_mut(handle).access, *access)
        }));
        memory.sort();
        memory.dedup();
        self.pipeline_barriers(&images, &buffers, &memory);
    }
}

pub struct Semaphores {
    pub acquire: vk::Semaphore,
    pub release: vk::Semaphore,
    /// If a swapchain image was acquired, we have to wait for it.
    pub image_acquired: bool,
}

impl Semaphores {
    pub fn new(device: &Device) -> Result<Self> {
        Ok(Self {
            acquire: create_semaphore(device)?,
            release: create_semaphore(device)?,
            image_acquired: false,
        })
    }

    pub fn destroy(&self, device: &Device) {
        unsafe {
            device.destroy_semaphore(self.acquire, None);
            device.destroy_semaphore(self.release, None);
        }
    }
}

fn create_semaphore(device: &Device) -> Result<vk::Semaphore> {
    let semaphore_info = vk::SemaphoreCreateInfo::default();
    unsafe {
        device
            .create_semaphore(&semaphore_info, None)
            .map_err(Error::from)
    }
}
