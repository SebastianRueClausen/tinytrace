use super::{
    resource, sync::Access, Buffer, BufferRequest, Context, Error, Handle, Image, Lifetime,
};
use ash::vk;

use std::collections::HashMap;

#[derive(Debug)]
pub struct BufferWrite<'a> {
    pub buffer: Handle<Buffer>,
    pub data: &'a [u8],
}

#[derive(Debug)]
pub struct ImageWrite<'a> {
    pub image: Handle<Image>,
    pub offset: vk::Offset3D,
    pub extent: vk::Extent3D,
    pub mips: &'a [Box<[u8]>],
}

#[derive(Debug)]
pub struct Download {
    pub buffers: HashMap<Handle<Buffer>, Box<[u8]>>,
    pub images: HashMap<Handle<Image>, Box<[u8]>>,
}

const CHUNK_ALIGNMENT: usize = 16;

const WRITE_ACCESS: Access = Access {
    stage: vk::PipelineStageFlags2::TRANSFER,
    access: vk::AccessFlags2::TRANSFER_WRITE,
};

const READ_ACCESS: Access = Access {
    stage: vk::PipelineStageFlags2::TRANSFER,
    access: vk::AccessFlags2::TRANSFER_READ,
};

fn buffer_image_copy(
    aspect: vk::ImageAspectFlags,
    level: u32,
    buffer_offset: vk::DeviceSize,
    offset: vk::Offset3D,
    extent: vk::Extent3D,
) -> vk::BufferImageCopy {
    let subresource = vk::ImageSubresourceLayers::default()
        .aspect_mask(aspect)
        .base_array_layer(0)
        .layer_count(1)
        .mip_level(level);
    vk::BufferImageCopy::default()
        .buffer_offset(buffer_offset)
        .image_extent(extent)
        .image_offset(offset)
        .buffer_image_height(extent.height)
        .buffer_row_length(extent.width)
        .image_subresource(subresource)
}

impl Context {
    fn get_scratch(&mut self, size: vk::DeviceSize) -> Result<(Buffer, *mut u8), Error> {
        let pool = self.pools.entry(Lifetime::Frame).or_default();
        let scratch = Buffer::new(
            &self.device,
            &mut pool.allocator,
            &BufferRequest {
                ty: super::BufferType::Scratch,
                memory_flags: vk::MemoryPropertyFlags::HOST_VISIBLE
                    | vk::MemoryPropertyFlags::HOST_COHERENT,
                size,
            },
        )?;
        let mapping = pool.allocator.map(&self.device, scratch.memory_index)?;
        Ok((scratch, mapping))
    }

    pub fn write_buffers(&mut self, writes: &[BufferWrite]) -> Result<(), Error> {
        let buffer_accesses: Vec<_> = writes
            .iter()
            .map(|write| (write.buffer.clone(), WRITE_ACCESS))
            .collect();
        self.access_resources(&[], &buffer_accesses);

        // Create and write to scratch.
        let scratch_size: usize = writes
            .iter()
            .map(|write| write.data.len().next_multiple_of(CHUNK_ALIGNMENT))
            .sum();
        let (scratch, mapping) = self.get_scratch(scratch_size as vk::DeviceSize)?;
        writes.iter().fold(mapping, |ptr, write| unsafe {
            ptr.copy_from_nonoverlapping(write.data.as_ptr(), write.data.len());
            ptr.add(write.data.len().next_multiple_of(CHUNK_ALIGNMENT))
        });

        // Upload scratch to buffers.
        writes.iter().fold(0, |offset, write| unsafe {
            let byte_count = write.data.len() as u64;
            if byte_count != 0 {
                let buffer_copy = vk::BufferCopy::default()
                    .src_offset(offset)
                    .dst_offset(0)
                    .size(byte_count);
                self.device.cmd_copy_buffer(
                    *self.command_buffer,
                    scratch.buffer,
                    **self.buffer(&write.buffer),
                    &[buffer_copy],
                );
            }
            offset + byte_count
        });
        self.pool_mut(Lifetime::Frame).buffers.push(scratch);
        Ok(())
    }

    pub fn write_images(&mut self, writes: &[ImageWrite]) -> Result<(), Error> {
        let image_accesses: Vec<_> = writes
            .iter()
            .map(|write| (write.image.clone(), WRITE_ACCESS))
            .collect();
        self.access_resources(&image_accesses, &[]);

        // Create and write to scratch.
        let scratch_size = writes
            .iter()
            .flat_map(|write| {
                write
                    .mips
                    .iter()
                    .map(|mip| mip.len().next_multiple_of(CHUNK_ALIGNMENT) as u64)
            })
            .sum();
        let (scratch, mapping) = self.get_scratch(scratch_size)?;
        writes
            .iter()
            .flat_map(|write| write.mips.iter())
            .fold(mapping, |ptr, mip| unsafe {
                ptr.copy_from_nonoverlapping(mip.as_ptr(), mip.len());
                ptr.add(mip.len().next_multiple_of(CHUNK_ALIGNMENT))
            });

        // Upload scratch to images.
        writes
            .iter()
            .flat_map(|write| {
                write.mips.iter().enumerate().map(move |(level, data)| {
                    let offset = resource::mip_level_offset(write.offset, level as u32);
                    let extent = resource::mip_level_extent(write.extent, level as u32);
                    (write.image.clone(), extent, offset, data, level as u32)
                })
            })
            .fold(
                0,
                |buffer_offset, (image, extent, offset, data, level)| unsafe {
                    let image = self.image(&image);
                    let image_copy =
                        buffer_image_copy(image.aspect, level, buffer_offset, offset, extent);
                    let layout = vk::ImageLayout::TRANSFER_DST_OPTIMAL;
                    self.device.cmd_copy_buffer_to_image(
                        *self.command_buffer,
                        *scratch,
                        **image,
                        layout,
                        &[image_copy],
                    );
                    buffer_offset + data.len().next_multiple_of(CHUNK_ALIGNMENT) as u64
                },
            );
        self.pool_mut(Lifetime::Frame).buffers.push(scratch);
        Ok(())
    }

    /// Download data from buffers and images.
    ///
    /// The memory of images with multiple mips is densely packed.
    pub fn download(
        &mut self,
        buffers: &[Handle<Buffer>],
        images: &[Handle<Image>],
    ) -> Result<Download, Error> {
        let image_accesses: Vec<_> = images
            .iter()
            .map(|image| (image.clone(), READ_ACCESS))
            .collect();
        let buffer_accesses: Vec<_> = buffers
            .iter()
            .map(|buffer| (buffer.clone(), READ_ACCESS))
            .collect();
        self.access_resources(&image_accesses, &buffer_accesses);

        // Figure out scratch size.
        let buffer_scratch_size: vk::DeviceSize = buffers
            .iter()
            .map(|buffer| {
                self.buffer(buffer)
                    .size
                    .next_multiple_of(CHUNK_ALIGNMENT as vk::DeviceSize)
            })
            .sum();
        let image_scratch_size: vk::DeviceSize = images
            .iter()
            .map(|image| {
                self.image(image)
                    .size()
                    .next_multiple_of(CHUNK_ALIGNMENT as vk::DeviceSize)
            })
            .sum();
        let (scratch, mut mapping) = self.get_scratch(buffer_scratch_size + image_scratch_size)?;

        // Copy images into scratch.
        let scratch_offset = images.iter().fold(0, |scratch_offset, image| unsafe {
            let image = self.image(image);
            let mut image_offset = 0;
            let copies: Vec<_> = (0..image.mip_level_count)
                .map(|level| {
                    let buffer_offset = scratch_offset + image_offset;
                    image_offset += image.mip_byte_size(level);
                    buffer_image_copy(
                        image.aspect,
                        level,
                        buffer_offset,
                        vk::Offset3D::default(),
                        resource::mip_level_extent(image.extent, level),
                    )
                })
                .collect();
            let layout = vk::ImageLayout::TRANSFER_SRC_OPTIMAL;
            self.device.cmd_copy_image_to_buffer(
                *self.command_buffer,
                image.image,
                layout,
                *scratch,
                &copies,
            );
            scratch_offset
                + image
                    .size()
                    .next_multiple_of(CHUNK_ALIGNMENT as vk::DeviceSize)
        });

        // Copy buffers into scratch.
        buffers
            .iter()
            .fold(scratch_offset, |scratch_offset, buffer| unsafe {
                let byte_count = self.buffer(buffer).size;
                if byte_count != 0 {
                    let buffer_copy = vk::BufferCopy::default()
                        .src_offset(0)
                        .dst_offset(scratch_offset)
                        .size(byte_count);
                    self.device.cmd_copy_buffer(
                        *self.command_buffer,
                        self.buffer(buffer).buffer,
                        scratch.buffer,
                        &[buffer_copy],
                    );
                }
                scratch_offset + byte_count.next_multiple_of(CHUNK_ALIGNMENT as vk::DeviceSize)
            });

        self.pool_mut(Lifetime::Frame).buffers.push(scratch);
        self.execute_commands()?;

        // Copy data from scratch memory.
        let mut copy_from_mapping = |size| unsafe {
            let mut memory = vec![0x0u8; size];
            memory.as_mut_ptr().copy_from_nonoverlapping(mapping, size);
            mapping = mapping.add(size.next_multiple_of(CHUNK_ALIGNMENT));
            memory.into_boxed_slice()
        };
        let images: HashMap<_, _> = images
            .iter()
            .map(|image| {
                let data = copy_from_mapping(self.image(image).size() as usize);
                (image.clone(), data)
            })
            .collect();
        let buffers: HashMap<_, _> = buffers
            .iter()
            .map(|buffer| {
                let data = copy_from_mapping(self.buffer(buffer).size as usize);
                (buffer.clone(), data)
            })
            .collect();

        self.clear_pool(Lifetime::Frame);
        Ok(Download { buffers, images })
    }
}
