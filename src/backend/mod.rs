pub mod command;
pub mod copy;
mod device;
mod instance;
pub mod resource;
mod sync;

#[cfg(test)]
mod test;

use std::{collections::HashMap, hash::Hash, marker::PhantomData, slice};

use super::error::Error;
use ash::vk;
use command::CommandBuffer;
pub use copy::{BufferWrite, Download, ImageWrite};
use device::Device;
use instance::Instance;
pub use resource::{Allocator, Buffer, BufferRequest, Image, ImageRange, ImageRequest, ImageView};

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
    fn new(lifetime: Lifetime, index: usize, epoch: usize) -> Self {
        Self {
            lifetime,
            index,
            epoch,
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

impl<T> std::hash::Hash for Handle<T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
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
    allocator: Allocator,
    epoch: usize,
}

impl Pool {
    fn clear(&mut self, device: &Device) {
        for buffer in self.buffers.drain(..) {
            buffer.destroy(device);
        }
        for image in self.images.drain(..) {
            image.destroy(device);
        }
        for image_view in self.image_views.drain(..) {
            image_view.destroy(device);
        }
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
            instance,
            device,
        })
    }

    fn check_handle<T>(&self, handle: &Handle<T>) {
        debug_assert_eq!(
            self.pools
                .get(&handle.lifetime)
                .expect("invalid lifetime")
                .epoch,
            handle.epoch
        )
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
        pool.buffers
            .push(Buffer::new(&self.device, &mut pool.allocator, request)?);
        Ok(Handle::new(lifetime, pool.buffers.len() - 1, pool.epoch))
    }

    pub fn create_image(
        &mut self,
        lifetime: Lifetime,
        request: &ImageRequest,
    ) -> Result<Handle<Image>, Error> {
        let pool = self.pools.entry(lifetime).or_default();
        pool.images
            .push(Image::new(&self.device, &mut pool.allocator, request)?);
        Ok(Handle::new(lifetime, pool.images.len() - 1, pool.epoch))
    }

    pub fn create_image_view(
        &mut self,
        lifetime: Lifetime,
        image: &Handle<Image>,
        range: ImageRange,
    ) -> Result<Handle<Image>, Error> {
        self.check_handle(image);
        let pool = self.pools.entry(lifetime).or_default();
        pool.image_views.push(ImageView::new(
            &self.device,
            &pool.images[image.index],
            range,
        )?);
        Ok(Handle::new(
            lifetime,
            pool.image_views.len() - 1,
            pool.epoch,
        ))
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
