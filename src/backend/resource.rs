use ash::vk;
use std::collections::HashMap;
use std::ops;

use super::device::Device;
use super::sync::Access;
use crate::error::{Error, Result};

#[derive(Debug, Clone, Copy)]
pub enum BufferType {
    Uniform,
    Storage,
    Scratch,
    Descriptor,
}

impl BufferType {
    fn usage_flags(&self) -> vk::BufferUsageFlags {
        let base = vk::BufferUsageFlags::TRANSFER_SRC
            | vk::BufferUsageFlags::TRANSFER_DST
            | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS;
        let flags = match self {
            BufferType::Uniform => vk::BufferUsageFlags::UNIFORM_BUFFER,
            BufferType::Storage => vk::BufferUsageFlags::STORAGE_BUFFER,
            BufferType::Descriptor => {
                vk::BufferUsageFlags::SAMPLER_DESCRIPTOR_BUFFER_EXT
                    | vk::BufferUsageFlags::RESOURCE_DESCRIPTOR_BUFFER_EXT
            }
            _ => vk::BufferUsageFlags::empty(),
        };
        base | flags
    }
}

#[derive(Debug, Clone, Copy)]
pub struct BufferRequest {
    pub size: vk::DeviceSize,
    pub ty: BufferType,
    pub memory_flags: vk::MemoryPropertyFlags,
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

impl Buffer {
    pub fn new(
        device: &Device,
        allocator: &mut Allocator,
        request: &BufferRequest,
    ) -> Result<Self> {
        let size = request.size.max(4);
        let usage_flags = request.ty.usage_flags();
        let buffer_info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(usage_flags);
        let buffer = unsafe { device.create_buffer(&buffer_info, None)? };
        let memory_requirements = unsafe { device.get_buffer_memory_requirements(buffer) };
        let (memory, memory_index) =
            allocator.allocate(device, request.memory_flags, memory_requirements)?;
        unsafe {
            device.bind_buffer_memory(buffer, memory, memory_index.offset)?;
        }
        Ok(Self {
            access: Access::default(),
            usage_flags,
            size,
            buffer,
            memory_index,
        })
    }

    pub fn device_address(&self, device: &Device) -> vk::DeviceAddress {
        let address_info = vk::BufferDeviceAddressInfo::default().buffer(**self);
        unsafe { device.get_buffer_device_address(&address_info) }
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

#[derive(Debug, Clone, Copy)]
pub enum ImageType {
    Texture,
    Storage,
}

impl ImageType {
    fn usage_flags(&self) -> vk::ImageUsageFlags {
        let base = vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST;
        let flags = match self {
            ImageType::Texture => vk::ImageUsageFlags::SAMPLED,
            ImageType::Storage => vk::ImageUsageFlags::STORAGE,
        };
        base | flags
    }
}

#[derive(Debug, Clone)]
pub struct ImageRequest {
    pub extent: vk::Extent3D,
    pub format: vk::Format,
    pub mip_level_count: u32,
    pub ty: ImageType,
    pub memory_flags: vk::MemoryPropertyFlags,
}

#[derive(Debug)]
pub struct Image {
    pub image: vk::Image,
    pub view: vk::ImageView,
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
    pub fn new(device: &Device, allocator: &mut Allocator, request: &ImageRequest) -> Result<Self> {
        let layout = vk::ImageLayout::UNDEFINED;
        let usage_flags = request.ty.usage_flags();
        let image_info = vk::ImageCreateInfo::default()
            .format(request.format)
            .extent(request.extent)
            .mip_levels(request.mip_level_count)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(usage_flags)
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
            view: create_image_view(device, image, request.format, request.mip_level_count)?,
            access: Access::default(),
            aspect: format_aspect(request.format),
            extent: request.extent,
            format: request.format,
            mip_level_count: request.mip_level_count,
            layout: vk::ImageLayout::UNDEFINED,
            swapchain: false,
            usage_flags,
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
        unsafe {
            device.destroy_image_view(self.view, None);
        };
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

pub fn create_image_view(
    device: &Device,
    image: vk::Image,
    format: vk::Format,
    mip_level_count: u32,
) -> Result<vk::ImageView> {
    let aspect_mask = format_aspect(format);
    let image_view_info = vk::ImageViewCreateInfo::default()
        .image(image)
        .format(format)
        .view_type(vk::ImageViewType::TYPE_2D)
        .subresource_range(vk::ImageSubresourceRange {
            base_mip_level: 0,
            level_count: mip_level_count,
            base_array_layer: 0,
            layer_count: 1,
            aspect_mask,
        });
    unsafe {
        device
            .create_image_view(&image_view_info, None)
            .map_err(Error::from)
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

impl MemoryBlock {
    pub fn new(device: &Device, size: vk::DeviceSize, memory_type_index: u32) -> Result<Self> {
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
        if start + size > self.size {
            return None;
        }
        self.offset = start + size;
        Some(start)
    }

    pub fn destroy(&self, device: &Device) {
        unsafe {
            device.free_memory(self.memory, None);
        }
    }

    pub fn map(&self, device: &Device) -> Result<*mut u8> {
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
    ) -> Result<(vk::DeviceMemory, MemoryIndex)> {
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

    pub fn map(&mut self, device: &Device, index: MemoryIndex) -> Result<*mut u8> {
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

#[derive(Default)]
pub struct SamplerRequest {
    pub filter: vk::Filter,
    pub max_anisotropy: Option<f32>,
    pub address_mode: vk::SamplerAddressMode,
}

#[derive(Debug)]
pub struct Sampler {
    pub sampler: vk::Sampler,
}

impl ops::Deref for Sampler {
    type Target = vk::Sampler;

    fn deref(&self) -> &Self::Target {
        &self.sampler
    }
}

impl Sampler {
    pub fn new(device: &Device, request: &SamplerRequest) -> Result<Self> {
        let create_info = vk::SamplerCreateInfo::default()
            .mag_filter(request.filter)
            .min_filter(request.filter)
            .address_mode_u(request.address_mode)
            .address_mode_v(request.address_mode)
            .address_mode_w(request.address_mode)
            .max_anisotropy(request.max_anisotropy.unwrap_or_default())
            .border_color(vk::BorderColor::INT_OPAQUE_BLACK)
            .compare_op(vk::CompareOp::ALWAYS)
            .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
            .unnormalized_coordinates(false)
            .anisotropy_enable(request.max_anisotropy.is_some())
            .max_lod(vk::LOD_CLAMP_NONE);
        let sampler = unsafe {
            device
                .create_sampler(&create_info, None)
                .map_err(Error::from)?
        };
        Ok(Self { sampler })
    }

    pub fn destroy(&self, device: &Device) {
        unsafe {
            device.destroy_sampler(self.sampler, None);
        }
    }
}
