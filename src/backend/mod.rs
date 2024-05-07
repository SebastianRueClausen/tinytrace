pub mod command;
pub mod copy;
mod descriptor;
mod device;
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
    slice,
};

use super::error::Error;
use ash::vk;
use command::CommandBuffer;
pub use copy::{BufferWrite, Download, ImageWrite};
pub use descriptor::{Binding, BindingType, DescriptorLayout};
use device::Device;
use instance::Instance;
pub use resource::{Allocator, Buffer, BufferRequest, Image, ImageRange, ImageRequest, ImageView};
pub use shader::Shader;

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

/// A pool of resources. This is meant
#[derive(Default, Debug)]
struct Pool {
    buffers: Vec<Buffer>,
    images: Vec<Image>,
    image_views: Vec<ImageView>,
    descriptor_layouts: Vec<DescriptorLayout>,
    pipelines: Vec<Shader>,
    allocator: Allocator,
    epoch: usize,
}

impl Pool {
    fn clear(&mut self, device: &Device) {
        self.buffers.drain(..).for_each(|b| b.destroy(device));
        self.images.drain(..).for_each(|i| i.destroy(device));
        self.image_views.drain(..).for_each(|v| v.destroy(device));
        self.descriptor_layouts
            .drain(..)
            .for_each(|l| l.destroy(device));
        self.pipelines.drain(..).for_each(|p| p.destroy(device));
        self.allocator.destroy(device);
        self.epoch += 1;
    }
}

pub struct Context {
    pools: HashMap<Lifetime, Pool>,
    command_buffer: CommandBuffer,
    device: Device,
    instance: Instance,
}

impl Drop for Context {
    fn drop(&mut self) {
        for pool in self.pools.values_mut() {
            pool.clear(&self.device);
        }
        self.command_buffer.destroy(&self.device);
        self.device.destroy();
        self.instance.destroy();
    }
}

impl Context {
    pub fn new() -> Result<Self, Error> {
        let instance = Instance::new(true)?;
        let device = Device::new(&instance)?;
        Ok(Self {
            pools: HashMap::default(),
            command_buffer: CommandBuffer::new(&device)?,
            device,
            instance,
        })
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

    pub fn image_view(&self, handle: &Handle<ImageView>) -> &ImageView {
        self.check_handle(handle);
        &self.pools[&handle.lifetime].image_views[handle.index]
    }

    pub fn descriptor_layout(&self, handle: &Handle<DescriptorLayout>) -> &DescriptorLayout {
        self.check_handle(handle);
        &self.pools[&handle.lifetime].descriptor_layouts[handle.index]
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
    ) -> Result<Handle<Buffer>, Error> {
        let pool = self.pools.entry(lifetime).or_default();
        Buffer::new(&self.device, &mut pool.allocator, request)
            .map(|buffer| Handle::new(lifetime, pool.epoch, &mut pool.buffers, buffer))
    }

    pub fn create_image(
        &mut self,
        lifetime: Lifetime,
        request: &ImageRequest,
    ) -> Result<Handle<Image>, Error> {
        let pool = self.pools.entry(lifetime).or_default();
        Image::new(&self.device, &mut pool.allocator, request)
            .map(|image| Handle::new(lifetime, pool.epoch, &mut pool.images, image))
    }

    pub fn create_image_view(
        &mut self,
        image: &Handle<Image>,
        range: ImageRange,
    ) -> Result<Handle<ImageView>, Error> {
        self.check_handle(image);
        let pool = self.pools.entry(image.lifetime).or_default();
        ImageView::new(&self.device, &pool.images[image.index], range)
            .map(|view| Handle::new(image.lifetime, pool.epoch, &mut pool.image_views, view))
    }

    pub fn execute_commands(&mut self) -> Result<(), Error> {
        self.command_buffer.end(&self.device)?;
        let submit_info = vk::SubmitInfo::default()
            .wait_dst_stage_mask(slice::from_ref(&vk::PipelineStageFlags::ALL_COMMANDS))
            .wait_semaphores(&[])
            .signal_semaphores(&[])
            .command_buffers(slice::from_ref(&self.command_buffer.buffer));
        unsafe {
            self.device.queue_submit(
                self.device.queue,
                slice::from_ref(&submit_info),
                vk::Fence::null(),
            )?;
        }
        self.device.wait_until_idle()?;
        self.device.clear_command_pool()
    }
}
