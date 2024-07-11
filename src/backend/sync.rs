use std::{
    ops, slice,
    time::{Duration, Instant},
};

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

    /// Determine the image layout based on the access flags.
    /// Assume that no flags means presenting the image, which is a bit of a hack.
    fn image_layout(&self) -> vk::ImageLayout {
        match self.access {
            vk::AccessFlags2::TRANSFER_READ => vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            vk::AccessFlags2::TRANSFER_WRITE => vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::AccessFlags2::SHADER_READ => vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            _ if self.access.is_empty() => vk::ImageLayout::PRESENT_SRC_KHR,
            _ => vk::ImageLayout::GENERAL,
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
                .cmd_pipeline_barrier2(self.command_buffer().buffer, &dependency_info);
        }
    }

    pub fn access_resources(
        &mut self,
        images: &[(Handle<Image>, Access)],
        buffers: &[(Handle<Buffer>, Access)],
        blases: &[(Handle<Blas>, Access)],
        tlases: &[(Handle<Tlas>, Access)],
    ) -> Result<()> {
        let next_timestamp = self.sync.timestamp + 1;
        let images: Vec<_> = images
            .iter()
            .filter(|(handle, access)| {
                let image = self.image_mut(handle);
                let has_write_dependency =
                    (access.writes() || image.access.writes()) && !image.access.access.is_empty();
                let has_dependency = has_write_dependency || access.image_layout() != image.layout;
                if has_dependency {
                    image.access = Access::default();
                }
                image.timestamp = next_timestamp;
                image.access |= *access;
                has_dependency
            })
            .cloned()
            .collect();
        let handle_access = |src: &mut Access, dst: Access| {
            let has_dependency = (src.writes() || dst.writes()) && !src.access.is_empty();
            if has_dependency {
                *src = Access::default();
            }
            *src |= dst;
            has_dependency.then_some((dst, *src))
        };
        let buffers: Vec<_> = buffers
            .iter()
            .filter(|(handle, access)| {
                let buffer = self.buffer_mut(handle);
                buffer.timestamp = next_timestamp;
                handle_access(&mut buffer.access, *access).is_some()
            })
            .cloned()
            .collect();
        let mut memory: Vec<_> = blases
            .iter()
            .filter_map(|(handle, access)| {
                let blas = self.blas_mut(handle);
                blas.timestamp = next_timestamp;
                handle_access(&mut blas.access, *access)
            })
            .collect();
        memory.extend(tlases.iter().filter_map(|(handle, access)| {
            let tlas = self.tlas_mut(handle);
            tlas.timestamp = next_timestamp;
            handle_access(&mut tlas.access, *access)
        }));

        // Avoid the driver overhead of multiple of the same barriers.
        memory.sort();
        memory.dedup();

        self.pipeline_barriers(&images, &buffers, &memory);
        Ok(())
    }
}

/// The synchronization related to a frame.
pub struct Frame {
    /// Signaled when the swapchain is acquired.
    pub acquired: vk::Semaphore,
    /// Signaled when the swapchain image ready to be presented.
    pub released: vk::Semaphore,
    /// The timestamp of when the commands of the previous frame with this index is done executing.
    pub present_timestamp: Option<u64>,
}

/// All state used for synchronization.
pub struct Sync {
    /// The "Frames in Flight".
    pub frames: Vec<Frame>,
    /// The timeline semaphore.
    pub timeline: vk::Semaphore,
    /// The current timestamp, i.e. the one signaled when all previously submitted commands are
    /// done executing.
    pub timestamp: u64,
    /// The current frame index.
    pub frame_index: usize,
}

impl Sync {
    pub fn new(device: &Device, frame_count: usize) -> Result<Self> {
        let frames: Vec<_> = (0..frame_count)
            .map(|_| {
                Ok(Frame {
                    acquired: create_semaphore(device, vk::SemaphoreType::BINARY)?,
                    released: create_semaphore(device, vk::SemaphoreType::BINARY)?,
                    present_timestamp: None,
                })
            })
            .collect::<Result<_>>()?;
        Ok(Self {
            timeline: create_semaphore(device, vk::SemaphoreType::TIMELINE)?,
            timestamp: 0,
            frame_index: 0,
            frames,
        })
    }

    /// Wait until `timestamp` is signaled.
    pub fn wait_for_timestamp(&self, device: &Device, timestamp: u64) -> Result<Duration> {
        let before = Instant::now();
        unsafe {
            let wait_info = vk::SemaphoreWaitInfo::default()
                .semaphores(slice::from_ref(&self.timeline))
                .values(slice::from_ref(&timestamp));
            device
                .wait_semaphores(&wait_info, u64::MAX)
                .map_err(Error::from)
                .map(|_| before.elapsed())
        }
    }

    /// Get the `Frame` of `back` frames ago.
    pub fn frame(&self, back: usize) -> &Frame {
        &self.frames[self.frame_index.wrapping_sub(back) % self.frames.len()]
    }

    /// Signal that the current frame is done.
    pub fn advance_frame(&mut self, timestamp: u64) {
        let frame_count = self.frames.len();
        self.frames[self.frame_index % frame_count].present_timestamp = Some(timestamp);
        self.frame_index += 1;
    }

    pub fn destroy(&self, device: &Device) {
        for frame in &self.frames {
            unsafe {
                device.destroy_semaphore(frame.acquired, None);
                device.destroy_semaphore(frame.released, None);
            }
        }
        unsafe {
            device.destroy_semaphore(self.timeline, None);
        }
    }
}

pub fn create_semaphore(device: &Device, ty: vk::SemaphoreType) -> Result<vk::Semaphore> {
    let mut type_info = vk::SemaphoreTypeCreateInfo::default().semaphore_type(ty);
    let semaphore_info = vk::SemaphoreCreateInfo::default().push_next(&mut type_info);
    unsafe {
        device
            .create_semaphore(&semaphore_info, None)
            .map_err(Error::from)
    }
}
