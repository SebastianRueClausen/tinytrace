use ash::vk;
use std::collections::HashMap;
use std::ops;

use super::device::Device;
use super::sync::Access;
use crate::error::Error;

#[derive(Debug, Clone, Copy)]
pub struct BufferRequest {
    pub size: vk::DeviceSize,
    pub usage_flags: vk::BufferUsageFlags,
    pub memory_flags: vk::MemoryPropertyFlags,
}

impl BufferRequest {
    pub fn scratch(size: vk::DeviceSize) -> Self {
        Self {
            size,
            usage_flags: vk::BufferUsageFlags::TRANSFER_SRC
                | vk::BufferUsageFlags::TRANSFER_DST
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            memory_flags: vk::MemoryPropertyFlags::HOST_VISIBLE
                | vk::MemoryPropertyFlags::HOST_COHERENT,
        }
    }
}

#[derive(Debug)]
pub struct Buffer {
    pub buffer: vk::Buffer,
    pub memory_index: MemoryIndex,
    pub size: vk::DeviceSize,
    pub access: Access,
    pub usage_flags: vk::BufferUsageFlags,
}

impl ops::Deref for Buffer {
    type Target = vk::Buffer;

    fn deref(&self) -> &Self::Target {
        &self.buffer
    }
}

fn buffer_address(device: &Device, buffer: vk::Buffer) -> vk::DeviceAddress {
    let address_info = vk::BufferDeviceAddressInfo::default().buffer(buffer);
    unsafe { device.get_buffer_device_address(&address_info) }
}

impl Buffer {
    pub fn new(
        device: &Device,
        allocator: &mut Allocator,
        request: &BufferRequest,
    ) -> Result<Self, Error> {
        let size = request.size.max(4);
        let buffer_info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(request.usage_flags);
        let buffer = unsafe { device.create_buffer(&buffer_info, None)? };
        let memory_requirements = unsafe { device.get_buffer_memory_requirements(buffer) };
        let (memory, memory_index) =
            allocator.allocate(device, request.memory_flags, memory_requirements)?;
        unsafe {
            device.bind_buffer_memory(buffer, memory, memory_index.offset)?;
        }
        Ok(Self {
            access: Access::default(),
            usage_flags: request.usage_flags,
            size,
            buffer,
            memory_index,
        })
    }

    pub fn device_address(&self, device: &Device) -> vk::DeviceAddress {
        buffer_address(device, self.buffer)
    }

    pub fn destroy(&self, device: &Device) {
        unsafe {
            device.destroy_buffer(self.buffer, None);
        }
    }
}

fn format_aspect(format: vk::Format) -> vk::ImageAspectFlags {
    match format {
        vk::Format::D32_SFLOAT => vk::ImageAspectFlags::DEPTH,
        _ => vk::ImageAspectFlags::COLOR,
    }
}

#[derive(Debug, Clone)]
pub struct ImageRequest {
    pub extent: vk::Extent3D,
    pub format: vk::Format,
    pub mip_level_count: u32,
    pub usage_flags: vk::ImageUsageFlags,
    pub memory_flags: vk::MemoryPropertyFlags,
}

#[derive(Debug)]
pub struct Image {
    pub image: vk::Image,
    pub extent: vk::Extent3D,
    pub format: vk::Format,
    pub aspect: vk::ImageAspectFlags,
    pub mip_level_count: u32,
    pub swapchain: bool,
    pub layout: vk::ImageLayout,
    pub access: Access,
    pub usage_flags: vk::ImageUsageFlags,
}

impl ops::Deref for Image {
    type Target = vk::Image;

    fn deref(&self) -> &Self::Target {
        &self.image
    }
}

impl Image {
    pub fn new(
        device: &Device,
        allocator: &mut Allocator,
        request: &ImageRequest,
    ) -> Result<Self, Error> {
        let layout = vk::ImageLayout::UNDEFINED;
        let image_info = vk::ImageCreateInfo::default()
            .format(request.format)
            .extent(request.extent)
            .mip_levels(request.mip_level_count)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(request.usage_flags)
            .image_type(vk::ImageType::TYPE_2D)
            .initial_layout(layout);
        let image = unsafe { device.create_image(&image_info, None)? };
        let memory_requirements = unsafe { device.get_image_memory_requirements(image) };
        let (memory, index) =
            allocator.allocate(device, request.memory_flags, memory_requirements)?;
        unsafe {
            device.bind_image_memory(image, memory, index.offset)?;
        }
        Ok(Self {
            access: Access::default(),
            aspect: format_aspect(request.format),
            extent: request.extent,
            format: request.format,
            mip_level_count: request.mip_level_count,
            layout: vk::ImageLayout::UNDEFINED,
            usage_flags: request.usage_flags,
            swapchain: false,
            image,
        })
    }

    pub fn mip_byte_size(&self, level: u32) -> vk::DeviceSize {
        let extent = mip_level_extent(self.extent, level);
        let FormatInfo {
            block_extent,
            bytes_per_block,
        } = format_info(self.format);
        let block_count =
            (extent.width / block_extent.width) * (extent.height / block_extent.height);
        block_count as vk::DeviceSize * bytes_per_block
    }

    pub fn size(&self) -> vk::DeviceSize {
        (0..self.mip_level_count)
            .map(|level| self.mip_byte_size(level))
            .sum()
    }

    pub fn destroy(&self, device: &Device) {
        if !self.swapchain {
            unsafe {
                device.destroy_image(self.image, None);
            }
        }
    }
}

pub fn mip_level_extent(extent: vk::Extent3D, level: u32) -> vk::Extent3D {
    vk::Extent3D {
        width: extent.width >> level,
        height: extent.height >> level,
        depth: extent.depth,
    }
}

pub fn mip_level_offset(offset: vk::Offset3D, level: u32) -> vk::Offset3D {
    vk::Offset3D {
        x: offset.x >> level,
        y: offset.y >> level,
        z: offset.z,
    }
}

pub struct FormatInfo {
    pub block_extent: vk::Extent2D,
    pub bytes_per_block: vk::DeviceSize,
}

pub fn format_info(format: vk::Format) -> FormatInfo {
    match format {
        vk::Format::R8G8B8A8_SRGB => FormatInfo {
            block_extent: vk::Extent2D::default().width(1).height(1),
            bytes_per_block: 4,
        },
        _ => {
            panic!("unsupported format");
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImageRange {
    pub mip_level_count: u32,
    pub base_mip_level: u32,
}

impl ImageRange {
    pub const BASE: Self = Self {
        mip_level_count: 1,
        base_mip_level: 0,
    };
}

#[derive(Debug, Clone)]
pub struct ImageView {
    view: vk::ImageView,
    #[allow(dead_code)]
    range: ImageRange,
}

impl ops::Deref for ImageView {
    type Target = vk::ImageView;

    fn deref(&self) -> &Self::Target {
        &self.view
    }
}

impl ImageView {
    pub fn new(device: &Device, image: &Image, range: ImageRange) -> Result<Self, Error> {
        let aspect_mask = format_aspect(image.format);
        let image_view_info = vk::ImageViewCreateInfo::default()
            .image(image.image)
            .format(image.format)
            .view_type(vk::ImageViewType::TYPE_2D)
            .subresource_range(vk::ImageSubresourceRange {
                base_mip_level: range.base_mip_level,
                level_count: range.mip_level_count,
                base_array_layer: 0,
                layer_count: 1,
                aspect_mask,
            });
        let view = unsafe { device.create_image_view(&image_view_info, None)? };
        Ok(ImageView { view, range })
    }

    pub fn destroy(&self, device: &Device) {
        unsafe {
            device.destroy_image_view(self.view, None);
        }
    }
}

fn memory_type_index(device: &Device, type_bits: u32, flags: vk::MemoryPropertyFlags) -> u32 {
    for (index, memory_type) in device.memory_properties.memory_types.iter().enumerate() {
        if type_bits & (1 << index) != 0 && memory_type.property_flags.contains(flags) {
            return index as u32;
        }
    }
    panic!("invalid memory type")
}

#[derive(Debug)]
pub struct MemoryBlock {
    memory: vk::DeviceMemory,
    size: vk::DeviceSize,
    offset: vk::DeviceSize,
    mapping: Option<*mut u8>,
}

impl ops::Deref for MemoryBlock {
    type Target = vk::DeviceMemory;

    fn deref(&self) -> &Self::Target {
        &self.memory
    }
}

impl MemoryBlock {
    pub fn new(
        device: &Device,
        size: vk::DeviceSize,
        memory_type_index: u32,
    ) -> Result<Self, Error> {
        let mut allocate_flags_info =
            vk::MemoryAllocateFlagsInfo::default().flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);
        let allocation_info = vk::MemoryAllocateInfo::default()
            .allocation_size(size)
            .memory_type_index(memory_type_index)
            .push_next(&mut allocate_flags_info);
        debug_assert!(size > 0, "trying to allocate 0 bytes");
        let memory = unsafe { device.allocate_memory(&allocation_info, None)? };
        Ok(Self {
            mapping: None,
            offset: 0,
            memory,
            size,
        })
    }

    fn allocate(
        &mut self,
        size: vk::DeviceSize,
        alignment: vk::DeviceSize,
    ) -> Option<vk::DeviceSize> {
        let start = self.offset.next_multiple_of(alignment);
        let end = start + size;
        if end > self.size {
            return None;
        }
        self.offset = end;
        Some(start)
    }

    pub fn destroy(&self, device: &Device) {
        unsafe {
            device.free_memory(self.memory, None);
        }
    }

    pub fn map(&self, device: &Device) -> Result<*mut u8, Error> {
        let flags = vk::MemoryMapFlags::empty();
        Ok(unsafe { device.map_memory(self.memory, 0, vk::WHOLE_SIZE, flags)? as *mut u8 })
    }
}

#[derive(Clone, Copy, Debug)]
pub struct MemoryIndex {
    block_index: usize,
    memory_type_index: u32,
    offset: vk::DeviceSize,
}

const DEFAULT_BLOCK_SIZE: vk::DeviceSize = 1024 * 1024 * 20;

#[derive(Default, Debug)]
pub struct Allocator {
    blocks: HashMap<u32, Vec<MemoryBlock>>,
}

impl Allocator {
    pub fn allocate(
        &mut self,
        device: &Device,
        memory_flags: vk::MemoryPropertyFlags,
        requirements: vk::MemoryRequirements,
    ) -> Result<(vk::DeviceMemory, MemoryIndex), Error> {
        let memory_type_index =
            memory_type_index(device, requirements.memory_type_bits, memory_flags);
        let blocks = self.blocks.entry(memory_type_index).or_default();
        let create_block = || {
            MemoryBlock::new(
                device,
                requirements.size.next_multiple_of(DEFAULT_BLOCK_SIZE),
                memory_type_index,
            )
        };
        if blocks.is_empty() {
            blocks.push(create_block()?);
        };
        let block = blocks.last_mut().unwrap();
        if let Some(offset) = block.allocate(requirements.size, requirements.alignment) {
            return Ok((
                block.memory,
                MemoryIndex {
                    block_index: blocks.len() - 1,
                    memory_type_index,
                    offset,
                },
            ));
        }
        let block = create_block()?;
        let output = (
            block.memory,
            MemoryIndex {
                block_index: blocks.len(),
                memory_type_index,
                offset: 0,
            },
        );
        blocks.push(block);
        Ok(output)
    }

    pub fn map(&mut self, device: &Device, index: MemoryIndex) -> Result<*mut u8, Error> {
        let block = &mut self.blocks.get_mut(&index.memory_type_index).unwrap()[index.block_index];
        let mapping = block.mapping.get_or_insert_with(|| unsafe {
            device
                .map_memory(block.memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())
                .expect("failed to map memory") as *mut u8
        });
        Ok(unsafe { mapping.offset(index.offset as isize) })
    }

    pub fn destroy(&mut self, device: &Device) {
        for block in self.blocks.drain().flat_map(|(_, blocks)| blocks) {
            block.destroy(device);
        }
    }
}
