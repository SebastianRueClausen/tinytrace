use std::{borrow::Cow, mem};

use tinytrace_backend::{
    Buffer, BufferRequest, BufferType, BufferWrite, Context, Error, Handle, Lifetime,
    MemoryLocation,
};

pub(super) struct HashGrid {
    pub keys: Handle<Buffer>,
    pub data: Handle<Buffer>,
    pub capacity: u32,
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
        let data = context.create_buffer(
            Lifetime::Renderer,
            &BufferRequest {
                memory_location: MemoryLocation::Device,
                size: mem::size_of::<HashGridData>() as u64,
                ty: BufferType::Uniform,
            },
        )?;
        context.write_buffers(&[BufferWrite {
            buffer: data.clone(),
            data: Cow::from(bytemuck::bytes_of(&HashGridData {
                keys: context.buffer_device_address(&keys),
                scene_scale,
                capacity,
                bucket_size,
                padding: 0,
            })),
        }])?;
        let hash_grid = Self {
            keys,
            data,
            capacity,
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
pub struct HashGridData {
    keys: u64,
    scene_scale: f32,
    capacity: u32,
    bucket_size: u32,
    padding: u32,
}
