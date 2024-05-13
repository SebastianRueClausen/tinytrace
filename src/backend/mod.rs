//! # The backend tinytrace.
//!
//! The Vulkan backend of tinytrace. It is written to both have a simple implementation and be
//! simple to use. It makes a few compromises to make this possible:
//! * Only use compute shaders.
//! * Only use a single command buffer.
//! * Somewhat naive synchronization.
//! * Limited use of buffers and images.
//!
//! In order to not be painful to use, it does a lot of stuff automatically:
//! * Synchronization.
//! * Uploading and downloading of data.
//! * Descriptor management.
//! * Resource cleanup.
//!
//! It's not really a safe abtraction, but does catch a foot guns.

pub mod command;
pub mod copy;
mod device;
mod glsl;
mod handle;
mod instance;
pub mod resource;
mod shader;
mod sync;

#[cfg(test)]
mod test;

use std::collections::HashMap;
use std::{ops, slice};

use super::error::{Error, Result};
use ash::vk;
use command::CommandBuffer;
pub use copy::{BufferWrite, Download, ImageWrite};
use device::Device;
pub use handle::Handle;
use instance::Instance;
pub use resource::{
    Allocator, Blas, Buffer, BufferRequest, BufferType, Image, ImageRequest, ImageType, Sampler,
    SamplerRequest, Tlas,
};
pub use shader::{Binding, BindingType, Shader};
use shader::{BoundShader, DescriptorBuffer};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Lifetime {
    Static,
    Surface,
    Scene,
    Frame,
}

#[derive(Default, Debug)]
struct Pool {
    buffers: Vec<Buffer>,
    images: Vec<Image>,
    samplers: Vec<Sampler>,
    shaders: Vec<Shader>,
    blases: Vec<Blas>,
    tlases: Vec<Tlas>,
    allocator: Allocator,
    epoch: usize,
}

impl Pool {
    fn clear(&mut self, device: &Device) {
        macro_rules! drain {
            ($($item:ident), *) => {
                $(self.$item.drain(..).for_each(|b| b.destroy(device));)*
            };
        }
        drain!(tlases, blases, buffers, images, shaders);
        self.allocator.destroy(device);
        self.epoch += 1;
    }
}

pub struct Context {
    pools: HashMap<Lifetime, Pool>,
    /// The only command buffer used. It is always recording.
    command_buffer: CommandBuffer,
    device: Device,
    instance: Instance,
    bound_shader: Option<BoundShader>,
    descriptor_buffer: DescriptorBuffer,
}

impl Drop for Context {
    fn drop(&mut self) {
        for pool in self.pools.values_mut() {
            pool.clear(&self.device);
        }
        self.device.destroy();
        self.instance.destroy();
    }
}

fn create_descriptor_buffer(device: &Device, pool: &mut Pool) -> Result<DescriptorBuffer> {
    let request = BufferRequest {
        size: 1024 * 1024 * 4,
        ty: BufferType::Descriptor,
        memory_flags: vk::MemoryPropertyFlags::HOST_VISIBLE
            | vk::MemoryPropertyFlags::HOST_COHERENT,
    };
    let buffer = Buffer::new(device, &mut pool.allocator, &request)?;
    Ok(DescriptorBuffer {
        data: pool.allocator.map(device, buffer.memory_index)?,
        buffer: Handle::new(Lifetime::Static, 0, &mut pool.buffers, buffer),
        bound_range: ops::Range::default(),
        size: request.size as usize,
    })
}

impl Context {
    pub fn new() -> Result<Self> {
        let instance = Instance::new(true)?;
        let device = Device::new(&instance)?;

        let mut static_pool = Pool::default();
        let descriptor_buffer = create_descriptor_buffer(&device, &mut static_pool)?;

        let context = Self {
            pools: HashMap::from([(Lifetime::Static, static_pool)]),
            command_buffer: CommandBuffer::new(&device)?,
            descriptor_buffer,
            bound_shader: None,
            device,
            instance,
        };

        context.begind_command_buffer()?;
        Ok(context)
    }

    fn begind_command_buffer(&self) -> Result<()> {
        self.command_buffer
            .begin(&self.device, self.buffer(&self.descriptor_buffer.buffer))
    }

    fn check_handle<T>(&self, handle: &Handle<T>) {
        debug_assert_eq!(self.pools[&handle.lifetime].epoch, handle.epoch)
    }

    fn pool_mut(&mut self, lifetime: Lifetime) -> &mut Pool {
        self.pools.entry(lifetime).or_default()
    }

    pub fn advance_lifetime(&mut self, lifetime: Lifetime) {
        if let Some(pool) = self.pools.get_mut(&lifetime) {
            pool.clear(&self.device);
        }
    }
}

macro_rules! accessor {
    ($ty:ty, $name:ident, $field:ident) => {
        pub fn $name(&self, handle: &Handle<$ty>) -> &$ty {
            self.check_handle(handle);
            &self.pools[&handle.lifetime].$field[handle.index]
        }
    };
    ($ty:ty, $name:ident, $field:ident, mut) => {
        pub fn $name(&mut self, handle: &Handle<$ty>) -> &mut $ty {
            self.check_handle(handle);
            &mut self.pools.get_mut(&handle.lifetime).unwrap().$field[handle.index]
        }
    };
}

impl Context {
    accessor!(Buffer, buffer, buffers);
    accessor!(Buffer, buffer_mut, buffers, mut);
    accessor!(Image, image, images);
    accessor!(Image, image_mut, images, mut);
    accessor!(Shader, shader, shaders);
    accessor!(Sampler, sampler, samplers);
    accessor!(Blas, blas, blases);
    accessor!(Blas, blas_mut, blases, mut);
    accessor!(Tlas, tlas, tlases);
    accessor!(Tlas, tlas_mut, tlases, mut);
}

impl Context {
    pub fn create_buffer(
        &mut self,
        lifetime: Lifetime,
        request: &BufferRequest,
    ) -> Result<Handle<Buffer>> {
        let pool = self.pools.entry(lifetime).or_default();
        Buffer::new(&self.device, &mut pool.allocator, request)
            .map(|buffer| Handle::new(lifetime, pool.epoch, &mut pool.buffers, buffer))
    }

    pub fn create_image(
        &mut self,
        lifetime: Lifetime,
        request: &ImageRequest,
    ) -> Result<Handle<Image>> {
        let pool = self.pools.entry(lifetime).or_default();
        Image::new(&self.device, &mut pool.allocator, request)
            .map(|image| Handle::new(lifetime, pool.epoch, &mut pool.images, image))
    }

    pub fn create_sampler(
        &mut self,
        lifetime: Lifetime,
        request: &SamplerRequest,
    ) -> Result<Handle<Sampler>> {
        let pool = self.pools.entry(lifetime).or_default();
        Sampler::new(&self.device, request)
            .map(|sampler| Handle::new(lifetime, pool.epoch, &mut pool.samplers, sampler))
    }

    pub fn execute_commands(&mut self) -> Result<()> {
        self.command_buffer.end(&self.device)?;
        let submit_info = vk::SubmitInfo::default()
            .wait_dst_stage_mask(slice::from_ref(&vk::PipelineStageFlags::ALL_COMMANDS))
            .wait_semaphores(&[])
            .signal_semaphores(&[])
            .command_buffers(slice::from_ref(&self.command_buffer.buffer));
        unsafe {
            self.device
                .queue_submit(self.device.queue, &[submit_info], vk::Fence::null())?;
        }
        self.device.wait_until_idle()?;
        self.device.clear_command_pool()?;
        self.bound_shader = None;
        self.descriptor_buffer.bound_range = 0..0;
        self.begind_command_buffer()
    }

    fn get_scratch(&mut self, size: vk::DeviceSize) -> Result<(Handle<Buffer>, *mut u8)> {
        let pool = self.pools.entry(Lifetime::Frame).or_default();
        let scratch = Buffer::new(
            &self.device,
            &mut pool.allocator,
            &BufferRequest {
                ty: BufferType::Scratch,
                memory_flags: vk::MemoryPropertyFlags::HOST_VISIBLE
                    | vk::MemoryPropertyFlags::HOST_COHERENT,
                size,
            },
        )?;
        let mapping = pool.allocator.map(&self.device, scratch.memory_index)?;
        let handle = Handle::new(Lifetime::Frame, pool.epoch, &mut pool.buffers, scratch);
        Ok((handle, mapping))
    }
}
