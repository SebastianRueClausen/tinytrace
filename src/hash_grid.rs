use std::mem;

use ash::vk;

use tinytrace_backend::{
    Buffer, BufferRequest, BufferType, Context, Error, Handle, Lifetime, MemoryLocation,
};

pub(super) struct HashGrid {
    pub keys: Handle<Buffer>,
    pub layout: HashGridLayout,
}

impl HashGrid {
    pub fn new(context: &mut Context, layout: HashGridLayout) -> Result<Self, Error> {
        let hashes = context.create_buffer(
            Lifetime::Renderer,
            &BufferRequest {
                size: (mem::size_of::<u64>() * layout.capacity as usize) as vk::DeviceSize,
                ty: BufferType::Storage,
                memory_location: MemoryLocation::Device,
            },
        )?;
        let hash_grid = Self {
            keys: hashes,
            layout,
        };
        hash_grid.clear(context)?;
        Ok(hash_grid)
    }

    pub fn clear(&self, context: &mut Context) -> Result<(), Error> {
        context.fill_buffer(&self.keys, u32::MAX)?;
        Ok(())
    }
}

#[repr(C)]
#[derive(Default, Debug, Clone, Copy, bytemuck::NoUninit, bytemuck::AnyBitPattern)]
pub struct HashGridLayout {
    pub scene_scale: f32,
    pub capacity: u32,
    pub bucket_size: u32,
    padding: u32,
}

impl HashGridLayout {
    pub fn new(capacity: u32, bucket_size: u32, scene_scale: f32) -> Self {
        Self {
            scene_scale,
            capacity,
            bucket_size,
            padding: 0,
        }
    }
}
