#![warn(clippy::all)]
//! # The backend tinytrace
//!
//! The Vulkan backend of tinytrace. It is written to both be a simple implementation and be
//! simple to use. It makes a few compromises to make this possible:
//! * Only use compute shaders.
//! * Somewhat naive synchronization.
//! * Limited use of buffers and images.
//!
//! In order to not be painful to use, it does a lot of stuff automatically:
//! * Synchronize between commands.
//! * Upload and download buffers and images.
//! * Manage descriptors.
//! * Cleanup resources.

mod command;
mod copy;
mod device;
mod error;
mod glsl;
mod handle;
mod instance;
mod resource;
mod shader;
mod surface;
mod sync;
mod timing;

#[cfg(test)]
mod test;

use std::collections::HashMap;
use std::slice;
use std::time::Duration;

use self::sync::Sync;
use self::{surface::Swapchain, sync::Access};

use ash::vk;
use command::CommandBuffer;
pub use copy::{BufferWrite, Download, ImageWrite};
use device::Device;
pub use error::Error;
pub use handle::Handle;
use instance::Instance;
use raw_window_handle::{RawDisplayHandle, RawWindowHandle};
use resource::Allocator;
pub use resource::{
    Blas, BlasBuild, BlasRequest, Buffer, BufferRange, BufferRequest, BufferType, Extent, Filter,
    Image, ImageFormat, ImageRequest, MemoryLocation, Offset, Sampler, SamplerRequest, Tlas,
    TlasInstance,
};
use shader::BoundShader;
pub use shader::{Binding, BindingType, Shader, ShaderRequest};

/// The lifetime of a resource.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Lifetime {
    /// The resource is never cleared.
    Static,
    /// The resource lives as long as the surface, usually meaning that it is
    /// tied to the surface size.
    Surface,
    /// The resource lives as long as the scene.
    Scene,
    /// The resource only lives for the current frame.
    Frame,
    /// Data related to the renderer, meaning it is cleared when the renderer is reconfigured.
    Renderer,
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
    fn clear_accesses(&mut self) {
        for buffer in &mut self.buffers {
            buffer.access = Access::default();
        }
        for image in &mut self.images {
            image.access = Access::default();
        }
        for blas in &mut self.blases {
            blas.access = Access::default();
        }
        for tlas in &mut self.tlases {
            tlas.access = Access::default();
        }
    }

    fn clear(&mut self, sync: &Sync, device: &Device) -> Result<(), Error> {
        macro_rules! timestamp {
            ($item:ident) => {
                self.$item.iter().map(|b| b.timestamp).max().unwrap_or(0)
            };
        }
        let timestamps = [
            timestamp!(buffers),
            timestamp!(images),
            timestamp!(blases),
            timestamp!(tlases),
        ];
        sync.wait_for_timestamp(device, timestamps.into_iter().max().unwrap_or(0))?;
        macro_rules! drain {
            ($($item:ident), *) => {
                $(self.$item.drain(..).for_each(|b| b.destroy(device));)*
            };
        }
        drain!(tlases, blases, samplers, buffers, images, shaders);
        self.allocator.destroy(device);
        self.epoch += 1;
        Ok(())
    }
}

pub struct Context {
    instance: Instance,
    device: Device,
    /// Swapchain and swapchain images.
    swapchain: Option<(Swapchain, Vec<Handle<Image>>)>,
    pools: HashMap<Lifetime, Pool>,
    /// Ring-buffer of command buffers. The first command buffer is always active.
    command_buffers: Vec<CommandBuffer>,
    bound_shader: Option<BoundShader>,
    /// The index of the last acquired swapchain swapchain image.
    acquired_swapchain_image: Option<Handle<Image>>,
    /// Map from path to source of shader includes.
    includes: HashMap<&'static str, String>,
    timestamps: HashMap<String, Duration>,
    sync: Sync,
}

impl Drop for Context {
    fn drop(&mut self) {
        self.execute_commands(false)
            .expect("failed to execute commands");
        self.device
            .wait_until_idle()
            .expect("failed to wait to idle");
        for pool in self.pools.values_mut() {
            pool.clear(&self.sync, &self.device)
                .expect("failed to destroy resources");
        }
        for command_buffer in self.command_buffers.drain(..) {
            command_buffer.destroy(&self.device);
        }
        self.sync.destroy(&self.device);
        if let Some((swapchain, _)) = &self.swapchain {
            swapchain.destroy();
        }
        self.device.destroy();
        self.instance.destroy();
    }
}

impl Context {
    pub fn new(window: Option<(RawWindowHandle, RawDisplayHandle)>) -> Result<Self, Error> {
        let instance = Instance::new(true)?;
        let device = Device::new(&instance)?;

        let (mut static_pool, mut surface_pool) = (Pool::default(), Pool::default());

        let swapchain = if let Some((window, display)) = window {
            let (swapchain, images) = Swapchain::new(&instance, &device, window, display)?;
            let images = images
                .into_iter()
                .map(|image| Handle::new(Lifetime::Surface, 0, &mut surface_pool.images, image))
                .collect();
            Some((swapchain, images))
        } else {
            None
        };

        let command_buffers: Vec<_> = (0..COMMAND_BUFFER_COUNT)
            .map(|_| CommandBuffer::new(&device, &mut static_pool.allocator))
            .collect::<Result<_, _>>()?;

        let pools = HashMap::from([
            (Lifetime::Static, static_pool),
            (Lifetime::Surface, surface_pool),
        ]);

        let mut context = Self {
            sync: Sync::new(&device, FRAMES_IN_FLIGHT)?,
            includes: HashMap::default(),
            timestamps: HashMap::default(),
            acquired_swapchain_image: None,
            command_buffers,
            bound_shader: None,
            swapchain,
            device,
            pools,
            instance,
        };

        context.begin_next_command_buffer()?;

        Ok(context)
    }

    fn command_buffer(&self) -> &CommandBuffer {
        self.command_buffers.first().unwrap()
    }

    fn command_buffer_mut(&mut self) -> &mut CommandBuffer {
        self.command_buffers.first_mut().unwrap()
    }

    fn begin_next_command_buffer(&mut self) -> Result<(), Error> {
        self.command_buffers.rotate_right(1);
        let buffer = self.command_buffers.first_mut().unwrap();
        self.timestamps
            .extend(buffer.clear(&self.sync, &self.device)?);
        buffer.begin(&self.device)
    }

    fn check_handle<T>(&self, handle: &Handle<T>) {
        assert_eq!(self.pools[&handle.lifetime].epoch, handle.epoch)
    }

    fn pool_mut(&mut self, lifetime: Lifetime) -> &mut Pool {
        self.pools.entry(lifetime).or_default()
    }

    /// Advance `lifetime`. This destroys all resources with `lifetime` and
    /// invalidates their handles. Returns the timestamp when all pending commands have been
    /// executed.
    fn advance_lifetime(&mut self, lifetime: Lifetime, present: bool) -> Result<u64, Error> {
        // TODO: Maybe check if commands have been recorded before doing this.
        let timestamp = self.execute_commands(present)?;
        if let Some(pool) = self.pools.get_mut(&lifetime) {
            pool.clear(&self.sync, &self.device)?;
        }
        Ok(timestamp)
    }

    pub fn clear_resources_with_lifetime(&mut self, lifetime: Lifetime) -> Result<(), Error> {
        self.advance_lifetime(lifetime, false).map(|_| ())
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
    fn swapchain(&self) -> &(Swapchain, Vec<Handle<Image>>) {
        self.swapchain.as_ref().expect("no swapchain present")
    }

    pub fn surface_format(&self) -> ImageFormat {
        let (swapchain, _) = self.swapchain();
        swapchain.format
    }

    pub fn frame_index(&self) -> usize {
        self.sync.frame_index
    }

    /// Add shader include. This allows shaders to include this using `path`.
    pub fn add_include(&mut self, path: &'static str, source: String) {
        self.includes.insert(path, source);
    }

    pub fn timestamp(&self, name: &str) -> Option<Duration> {
        self.timestamps.get(name).copied()
    }

    pub fn wait_until_idle(&mut self) -> Result<(), Error> {
        self.execute_commands(false)?;
        for command_buffer in self.command_buffers[1..].iter_mut().rev() {
            self.timestamps
                .extend(command_buffer.clear(&self.sync, &self.device)?);
        }
        Ok(())
    }

    pub fn create_sampler(
        &mut self,
        lifetime: Lifetime,
        request: &SamplerRequest,
    ) -> Result<Handle<Sampler>, Error> {
        let pool = self.pools.entry(lifetime).or_default();
        Sampler::new(&self.device, request)
            .map(|sampler| Handle::new(lifetime, pool.epoch, &mut pool.samplers, sampler))
    }

    /// Executes all recorded commands. Returns the timestamp signaled when when done.
    fn execute_commands(&mut self, present: bool) -> Result<u64, Error> {
        let command_buffer = self.command_buffers.first_mut().unwrap();
        command_buffer.end(&self.device, self.sync.timestamp + 1)?;

        let mut wait = vec![semaphore_submit_info(
            self.sync.timeline,
            self.sync.timestamp,
        )];
        let mut signal = vec![semaphore_submit_info(
            self.sync.timeline,
            self.sync.timestamp + 1,
        )];

        // If we are about to present, we wait for the swapchain image to be
        // acquired and signal when it is "released", meaning all commands are
        // done executing. This is required as present can't wait for timeline
        // semaphores.
        if present {
            wait.push(semaphore_submit_info(self.sync.frame(0).acquired, 0));
            signal.push(semaphore_submit_info(self.sync.frame(0).released, 0));
        }

        let command_buffer_info =
            vk::CommandBufferSubmitInfo::default().command_buffer(self.command_buffer().buffer);
        let submit_info = vk::SubmitInfo2::default()
            .command_buffer_infos(slice::from_ref(&command_buffer_info))
            .wait_semaphore_infos(&wait)
            .signal_semaphore_infos(&signal);
        unsafe {
            self.device.queue_submit2(
                self.device.queue,
                slice::from_ref(&submit_info),
                vk::Fence::null(),
            )?;
        }

        // The following commands are going to be recorded to another command
        // buffer, so all command buffer local data most be reset.
        self.pools.values_mut().for_each(Pool::clear_accesses);
        self.bound_shader = None;
        self.sync.timestamp += 1;

        self.begin_next_command_buffer()?;

        Ok(self.sync.timestamp)
    }

    /// Acquire the next swapchain image. This must be done exactly once before calling `present`.
    /// The error `Error::SurfaceOutdated` may be returned, which signals that
    pub fn swapchain_image(&mut self) -> Result<Handle<Image>, Error> {
        // Wait here to avoid the CPU being to far ahead of the GPU. Also ensure that
        // the `acquired` semaphore is available.
        if let Some(timestamp) = self.sync.frame(0).present_timestamp {
            self.sync.wait_for_timestamp(&self.device, timestamp)?;
        }
        let (swapchain, images) = self.swapchain();
        let image = images[swapchain.image_index(self.sync.frame(0).acquired)?].clone();
        self.acquired_swapchain_image = Some(image.clone());
        Ok(image)
    }

    pub fn resize_surface(&mut self) -> Result<(), Error> {
        // We have to wait until idle here because we don't use fences when presenting.
        self.device.wait_until_idle()?;
        let pool = self.pools.entry(Lifetime::Surface).or_default();
        pool.clear(&self.sync, &self.device)?;
        if let Some((swapchain, images)) = &mut self.swapchain {
            *images = swapchain
                .recreate(&self.device)?
                .into_iter()
                .map(|image| Handle::new(Lifetime::Surface, pool.epoch, &mut pool.images, image))
                .collect();
        }
        Ok(())
    }

    pub fn present(&mut self) -> Result<(), Error> {
        if let Some(image) = self.acquired_swapchain_image.take() {
            let access = Access {
                stage: vk::PipelineStageFlags2::BOTTOM_OF_PIPE,
                access: vk::AccessFlags2::empty(),
            };
            self.access_resources(&[(image.clone(), access)], &[], &[], &[])?;
            let timestamp = self.advance_lifetime(Lifetime::Frame, true)?;
            let image_index = self.image(&image).swapchain_index.unwrap();
            let (swapchain, _) = self.swapchain();
            let present_result =
                swapchain.present(&self.device, self.sync.frame(0).released, image_index);
            self.sync.advance_frame(timestamp);
            present_result
        } else {
            panic!("no swapchain image has been acquired");
        }
    }

    fn get_scratch(&mut self, size: vk::DeviceSize) -> Result<(Handle<Buffer>, *mut u8), Error> {
        let pool = self.pools.entry(Lifetime::Frame).or_default();
        let scratch = Buffer::new(
            &self.device,
            &mut pool.allocator,
            &BufferRequest {
                memory_location: MemoryLocation::Host,
                ty: BufferType::Scratch,
                size,
            },
        )?;
        let mapping = pool.allocator.map(&self.device, scratch.memory_index)?;
        let handle = Handle::new(Lifetime::Frame, pool.epoch, &mut pool.buffers, scratch);
        Ok((handle, mapping))
    }
}

fn semaphore_submit_info(semaphore: vk::Semaphore, value: u64) -> vk::SemaphoreSubmitInfo<'static> {
    vk::SemaphoreSubmitInfo::default()
        .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
        .semaphore(semaphore)
        .value(value)
}

const COMMAND_BUFFER_COUNT: usize = 4;
const FRAMES_IN_FLIGHT: usize = 2;
