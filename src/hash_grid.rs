use std::mem;

use tinytrace_backend::{
    Buffer, BufferRequest, BufferType, Context, Error, Handle, Lifetime, MemoryLocation,
};

pub(super) struct HashGrid {
    pub keys: Handle<Buffer>,
    pub capacity: u32,
    pub bucket_size: u32,
    pub scene_scale: f32,
}

impl HashGrid {
    pub fn new(
        context: &mut Context,
        scene_scale: f32,
        capacity: u32,
        bucket_size: u32,
    ) -> Result<Self, Error> {
        let keys = context.create_buffer(
            Lifetime::Renderer,
            &BufferRequest {
                memory_location: MemoryLocation::Device,
                size: (mem::size_of::<u64>() * capacity as usize) as u64,
                ty: BufferType::Storage,
            },
        )?;
        let hash_grid = Self {
            scene_scale,
            bucket_size,
            keys,
            capacity,
        };
        hash_grid.clear(context)?;
        Ok(hash_grid)
    }

    pub fn descriptor(&self, context: &Context) -> HashGridDescriptor {
        HashGridDescriptor {
            keys: context.buffer_device_address(&self.keys),
            scene_scale: self.scene_scale,
            capacity: self.capacity,
            bucket_size: self.bucket_size,
            padding: 0,
        }
    }

    pub fn clear(&self, context: &mut Context) -> Result<(), Error> {
        context.fill_buffer(&self.keys, u32::MAX)?;
        Ok(())
    }
}

#[repr(C)]
#[derive(Default, Debug, Clone, Copy, bytemuck::NoUninit, bytemuck::AnyBitPattern)]
pub struct HashGridDescriptor {
    keys: u64,
    scene_scale: f32,
    bucket_size: u32,
    capacity: u32,
    padding: u32,
}
