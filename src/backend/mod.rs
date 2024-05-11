pub mod command;
pub mod copy;
mod device;
mod glsl;
mod instance;
pub mod resource;
mod shader;
mod sync;

#[cfg(test)]
mod test;

use std::{
    collections::HashMap,
    hash::{self, Hash},
    marker::PhantomData,
    ops, slice,
};

use super::error::{Error, Result};
use ash::vk;
use command::CommandBuffer;
pub use copy::{BufferWrite, Download, ImageWrite};
use device::Device;
use instance::Instance;
pub use resource::{
    Allocator, Buffer, BufferRequest, BufferType, Image, ImageRequest, ImageType, Sampler,
    SamplerRequest,
};
pub use shader::{Binding, BindingType, Shader};
use shader::{BoundShader, DescriptorBuffer};

/// The lifetime of a resource.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Lifetime {
    Static,
    Surface,
    Scene,
    Frame,
}

/// The handle of a resource.
#[derive(Debug, Copy)]
pub struct Handle<T> {
    lifetime: Lifetime,
    index: usize,
    epoch: usize,
    _marker: std::marker::PhantomData<T>,
}

impl<T> Handle<T> {
    fn new(lifetime: Lifetime, epoch: usize, vec: &mut Vec<T>, value: T) -> Self {
        vec.push(value);
        Self {
            index: vec.len() - 1,
            epoch,
            lifetime,
            _marker: PhantomData,
        }
    }
}

impl<T> PartialEq for Handle<T> {
    fn eq(&self, other: &Self) -> bool {
        self.index == other.index && self.epoch == other.epoch && self.lifetime == other.lifetime
    }
}

impl<T> Eq for Handle<T> {}

impl<T> Clone for Handle<T> {
    fn clone(&self) -> Self {
        Self {
            lifetime: self.lifetime,
            index: self.index,
            epoch: self.epoch,
            _marker: PhantomData,
        }
    }
}

impl<T> hash::Hash for Handle<T> {
    fn hash<H: hash::Hasher>(&self, state: &mut H) {
        self.lifetime.hash(state);
        self.index.hash(state);
        self.epoch.hash(state);
    }
}

#[derive(Default, Debug)]
struct Pool {
    buffers: Vec<Buffer>,
    images: Vec<Image>,
    samplers: Vec<Sampler>,
    shaders: Vec<Shader>,
    allocator: Allocator,
    epoch: usize,
}

impl Pool {
    fn clear(&mut self, device: &Device) {
        self.buffers.drain(..).for_each(|b| b.destroy(device));
        self.images.drain(..).for_each(|i| i.destroy(device));
        self.shaders.drain(..).for_each(|s| s.destroy(device));
        self.allocator.destroy(device);
        self.epoch += 1;
    }
}

pub struct Context {
    pools: HashMap<Lifetime, Pool>,
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
            pools: [(Lifetime::Static, static_pool)].into_iter().collect(),
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

    fn clear_pool(&mut self, lifetime: Lifetime) {
        if let Some(pool) = self.pools.get_mut(&lifetime) {
            pool.clear(&self.device);
        }
    }

    pub fn buffer(&self, handle: &Handle<Buffer>) -> &Buffer {
        self.check_handle(handle);
        &self.pools[&handle.lifetime].buffers[handle.index]
    }

    pub fn image(&self, handle: &Handle<Image>) -> &Image {
        self.check_handle(handle);
        &self.pools[&handle.lifetime].images[handle.index]
    }

    pub fn shader(&self, handle: &Handle<Shader>) -> &Shader {
        self.check_handle(handle);
        &self.pools[&handle.lifetime].shaders[handle.index]
    }

    pub fn sampler(&self, handle: &Handle<Sampler>) -> &Sampler {
        self.check_handle(handle);
        &self.pools[&handle.lifetime].samplers[handle.index]
    }

    fn buffer_mut(&mut self, handle: &Handle<Buffer>) -> &mut Buffer {
        self.check_handle(handle);
        &mut self.pools.get_mut(&handle.lifetime).unwrap().buffers[handle.index]
    }

    fn image_mut(&mut self, handle: &Handle<Image>) -> &mut Image {
        self.check_handle(handle);
        &mut self.pools.get_mut(&handle.lifetime).unwrap().images[handle.index]
    }

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
        self.begind_command_buffer()
    }
}
