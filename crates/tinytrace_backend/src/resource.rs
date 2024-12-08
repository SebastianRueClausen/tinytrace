use ash::vk;
use glam::Mat4;
use std::borrow::Cow;
use std::{array, collections::HashMap, mem, slice};

use super::device::Device;
use super::instance::Instance;
use super::sync::Access;
use super::Error;
use super::{BufferWrite, Context, Handle, Lifetime};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Extent {
    pub width: u32,
    pub height: u32,
}

impl Extent {
    pub fn new(width: u32, height: u32) -> Self {
        Self { width, height }
    }

    pub fn mip_level(self, level: u32) -> Self {
        Self {
            width: self.width >> level,
            height: self.height >> level,
        }
    }
}

impl From<Extent> for vk::Extent2D {
    fn from(Extent { width, height }: Extent) -> Self {
        Self { width, height }
    }
}

impl From<Extent> for vk::Extent3D {
    fn from(Extent { width, height }: Extent) -> Self {
        Self {
            width,
            height,
            depth: 1,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Default)]
pub struct Offset {
    pub x: i32,
    pub y: i32,
}

impl Offset {
    pub fn new(x: i32, y: i32) -> Self {
        Self { x, y }
    }

    pub fn mip_level(self, level: u32) -> Self {
        Self {
            x: self.x >> level,
            y: self.y >> level,
        }
    }
}

impl From<Offset> for vk::Offset2D {
    fn from(Offset { x, y }: Offset) -> Self {
        Self { x, y }
    }
}

impl From<Offset> for vk::Offset3D {
    fn from(Offset { x, y }: Offset) -> Self {
        Self { x, y, z: 0 }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum BufferType {
    Uniform,
    Storage,
    Scratch,
    Descriptor,
    AccelerationStructure,
}

impl BufferType {
    fn usage_flags(&self) -> vk::BufferUsageFlags {
        let base = vk::BufferUsageFlags::TRANSFER_SRC
            | vk::BufferUsageFlags::TRANSFER_DST
            | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS;
        let flags = match self {
            BufferType::Uniform => vk::BufferUsageFlags::UNIFORM_BUFFER,
            BufferType::Storage => {
                vk::BufferUsageFlags::STORAGE_BUFFER
                    | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
            }
            BufferType::Descriptor => {
                vk::BufferUsageFlags::SAMPLER_DESCRIPTOR_BUFFER_EXT
                    | vk::BufferUsageFlags::RESOURCE_DESCRIPTOR_BUFFER_EXT
            }
            BufferType::AccelerationStructure => {
                vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR
            }
            BufferType::Scratch => vk::BufferUsageFlags::empty(),
        };
        base | flags
    }
}

#[derive(Debug, Clone, Copy)]
pub struct BufferRequest {
    pub size: vk::DeviceSize,
    pub ty: BufferType,
    pub memory_location: MemoryLocation,
}

#[derive(Debug, Clone, Copy)]
pub struct TemporaryBufferRequest<'a> {
    pub ty: BufferType,
    pub data: &'a [u8],
}

#[derive(Debug)]
pub struct Buffer {
    pub(crate) buffer: vk::Buffer,
    pub(crate) memory_index: MemoryIndex,
    pub(crate) size: vk::DeviceSize,
    pub(crate) access: Access,
    pub(crate) usage_flags: vk::BufferUsageFlags,
    pub(crate) timestamp: u64,
}

impl Context {
    pub fn create_buffer(
        &mut self,
        lifetime: Lifetime,
        request: &BufferRequest,
    ) -> Result<Handle<Buffer>, Error> {
        let pool = self.pools.entry(lifetime).or_default();
        let buffer = Buffer::new(&self.device, &mut pool.allocator, request)?;
        Ok(Handle::new(lifetime, pool.epoch, &mut pool.buffers, buffer))
    }

    pub fn create_temporary_buffer(
        &mut self,
        request: &TemporaryBufferRequest,
    ) -> Result<Handle<Buffer>, Error> {
        let buffer = self.create_buffer(
            Lifetime::Frame,
            &BufferRequest {
                memory_location: MemoryLocation::Device,
                size: request.data.len() as u64,
                ty: request.ty,
            },
        )?;
        self.write_buffers(&[BufferWrite {
            buffer: buffer.clone(),
            data: Cow::from(request.data),
        }])?;
        Ok(buffer)
    }

    pub fn buffer_device_address(&self, buffer: &Handle<Buffer>) -> u64 {
        self.buffer(buffer).device_address(&self.device)
    }
}

impl Buffer {
    pub(crate) fn new(
        device: &Device,
        allocator: &mut Allocator,
        request: &BufferRequest,
    ) -> Result<Self, Error> {
        let size = request.size.max(4);
        let usage_flags = request.ty.usage_flags();
        let buffer_info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(usage_flags);
        let buffer = unsafe { device.create_buffer(&buffer_info, None)? };
        let memory_requirements = unsafe { device.get_buffer_memory_requirements(buffer) };
        let memory_flags = request.memory_location.memory_property_flags();
        let (memory, memory_index) =
            allocator.allocate(device, memory_flags, memory_requirements)?;
        unsafe {
            device.bind_buffer_memory(buffer, memory, memory_index.offset)?;
        }
        Ok(Self {
            access: Access::default(),
            timestamp: 0,
            usage_flags,
            size,
            buffer,
            memory_index,
        })
    }

    pub fn size(&self) -> vk::DeviceSize {
        self.size
    }

    pub(crate) fn device_address(&self, device: &Device) -> vk::DeviceAddress {
        let address_info = vk::BufferDeviceAddressInfo::default().buffer(self.buffer);
        unsafe { device.get_buffer_device_address(&address_info) }
    }

    pub(crate) fn destroy(&self, device: &Device) {
        unsafe {
            device.destroy_buffer(self.buffer, None);
        }
    }
}

fn format_features(
    instance: &Instance,
    device: &Device,
    format: vk::Format,
) -> vk::FormatFeatureFlags {
    let mut properties = vk::FormatProperties2::default();
    unsafe {
        let physical_device = device.physical_device;
        instance.get_physical_device_format_properties2(physical_device, format, &mut properties);
    }
    properties.format_properties.optimal_tiling_features
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ImageFormat {
    Rgba8Srgb = vk::Format::R8G8B8A8_SRGB.as_raw() as isize,
    R8Unorm = vk::Format::R8_UNORM.as_raw() as isize,
    R8Snorm = vk::Format::R8_SNORM.as_raw() as isize,
    R16Float = vk::Format::R16_SFLOAT.as_raw() as isize,
    Rgba8Unorm = vk::Format::R8G8B8A8_UNORM.as_raw() as isize,
    Bgra8Unorm = vk::Format::B8G8R8A8_UNORM.as_raw() as isize,
    Rgba32Float = vk::Format::R32G32B32A32_SFLOAT.as_raw() as isize,
    RgBc5Unorm = vk::Format::BC5_UNORM_BLOCK.as_raw() as isize,
    RgbBc1Srgb = vk::Format::BC1_RGB_SRGB_BLOCK.as_raw() as isize,
    RgbaBc1Srgb = vk::Format::BC1_RGBA_SRGB_BLOCK.as_raw() as isize,
    RgbBc1Unorm = vk::Format::BC1_RGB_UNORM_BLOCK.as_raw() as isize,
}

pub struct FormatInfo {
    pub block_extent: Extent,
    pub bytes_per_block: vk::DeviceSize,
}

impl FormatInfo {
    fn new(block_extent: Extent, bytes_per_block: vk::DeviceSize) -> Self {
        Self {
            block_extent,
            bytes_per_block,
        }
    }
}

impl From<ImageFormat> for vk::Format {
    fn from(format: ImageFormat) -> Self {
        vk::Format::from_raw(format as i32)
    }
}

impl ImageFormat {
    pub fn info(self) -> FormatInfo {
        match self {
            Self::Rgba8Srgb | Self::Rgba8Unorm | Self::Bgra8Unorm => {
                FormatInfo::new(Extent::new(1, 1), 4)
            }
            Self::R8Unorm | Self::R8Snorm => FormatInfo::new(Extent::new(1, 1), 1),
            Self::R16Float => FormatInfo::new(Extent::new(1, 1), 2),
            Self::Rgba32Float => FormatInfo::new(Extent::new(1, 1), 16),
            Self::RgBc5Unorm => FormatInfo::new(Extent::new(4, 4), 16),
            Self::RgbBc1Srgb | Self::RgbaBc1Srgb | Self::RgbBc1Unorm => {
                FormatInfo::new(Extent::new(4, 4), 8)
            }
        }
    }

    pub fn aspect(self) -> vk::ImageAspectFlags {
        vk::ImageAspectFlags::COLOR
    }
}

#[derive(Debug, Clone)]
pub struct ImageRequest {
    pub extent: Extent,
    pub format: ImageFormat,
    pub mip_level_count: u32,
    pub memory_location: MemoryLocation,
}

#[derive(Debug)]
pub struct Image {
    pub(crate) image: vk::Image,
    pub(crate) view: vk::ImageView,
    pub(crate) extent: Extent,
    pub(crate) format: ImageFormat,
    pub(crate) mip_level_count: u32,
    pub(crate) swapchain_index: Option<u32>,
    pub(crate) layout: vk::ImageLayout,
    pub(crate) access: Access,
    pub(crate) timestamp: u64,
}

impl Context {
    pub fn create_image(
        &mut self,
        lifetime: Lifetime,
        request: &ImageRequest,
    ) -> Result<Handle<Image>, Error> {
        let pool = self.pools.entry(lifetime).or_default();
        let mut usage_flags = vk::ImageUsageFlags::TRANSFER_SRC
            | vk::ImageUsageFlags::TRANSFER_DST
            | vk::ImageUsageFlags::SAMPLED;
        let format_features = format_features(&self.instance, &self.device, request.format.into());
        if format_features.contains(vk::FormatFeatureFlags::STORAGE_IMAGE) {
            usage_flags |= vk::ImageUsageFlags::STORAGE;
        }
        let image_info = vk::ImageCreateInfo::default()
            .format(request.format.into())
            .extent(request.extent.into())
            .mip_levels(request.mip_level_count)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(usage_flags)
            .image_type(vk::ImageType::TYPE_2D)
            .initial_layout(vk::ImageLayout::UNDEFINED);
        let image = unsafe { self.device.create_image(&image_info, None)? };
        let memory_requirements = unsafe { self.device.get_image_memory_requirements(image) };
        let memory_flags = request.memory_location.memory_property_flags();
        let (memory, index) =
            pool.allocator
                .allocate(&self.device, memory_flags, memory_requirements)?;
        unsafe {
            self.device.bind_image_memory(image, memory, index.offset)?;
        }
        let image = Image {
            view: create_image_view(&self.device, image, request.format, request.mip_level_count)?,
            extent: request.extent,
            format: request.format,
            mip_level_count: request.mip_level_count,
            layout: vk::ImageLayout::UNDEFINED,
            swapchain_index: None,
            access: Access::default(),
            timestamp: 0,
            image,
        };
        Ok(Handle::new(lifetime, pool.epoch, &mut pool.images, image))
    }
}

impl Image {
    pub fn mip_byte_size(&self, level: u32) -> vk::DeviceSize {
        let extent = self.extent.mip_level(level);
        let FormatInfo {
            block_extent,
            bytes_per_block,
        } = self.format.info();
        let block_count =
            (extent.width / block_extent.width) * (extent.height / block_extent.height);
        block_count as vk::DeviceSize * bytes_per_block
    }

    pub fn mip_level_count(&self) -> u32 {
        self.mip_level_count
    }

    pub fn extent(&self) -> Extent {
        self.extent
    }

    pub fn format(&self) -> ImageFormat {
        self.format
    }

    pub fn size(&self) -> vk::DeviceSize {
        (0..self.mip_level_count)
            .map(|level| self.mip_byte_size(level))
            .sum()
    }

    pub(crate) fn destroy(&self, device: &Device) {
        if self.swapchain_index.is_none() {
            unsafe {
                device.destroy_image(self.image, None);
            }
        }
        unsafe {
            device.destroy_image_view(self.view, None);
        };
    }
}

pub fn create_image_view(
    device: &Device,
    image: vk::Image,
    format: ImageFormat,
    mip_level_count: u32,
) -> Result<vk::ImageView, Error> {
    let image_view_info = vk::ImageViewCreateInfo::default()
        .image(image)
        .format(format.into())
        .view_type(vk::ImageViewType::TYPE_2D)
        .subresource_range(vk::ImageSubresourceRange {
            aspect_mask: format.aspect(),
            base_mip_level: 0,
            level_count: mip_level_count,
            base_array_layer: 0,
            layer_count: 1,
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

#[derive(Debug, Clone, Copy)]
pub enum MemoryLocation {
    Host,
    Device,
}

impl MemoryLocation {
    pub fn memory_property_flags(&self) -> vk::MemoryPropertyFlags {
        match self {
            MemoryLocation::Host => {
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT
            }
            MemoryLocation::Device => vk::MemoryPropertyFlags::DEVICE_LOCAL,
        }
    }
}

// A linearly allocated contiguous block of memory.
#[derive(Debug)]
struct MemoryBlock {
    memory: vk::DeviceMemory,
    size: vk::DeviceSize,
    offset: vk::DeviceSize,
    mapping: Option<*mut u8>,
}

impl MemoryBlock {
    fn new(device: &Device, size: vk::DeviceSize, memory_type_index: u32) -> Result<Self, Error> {
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

    fn destroy(&self, device: &Device) {
        unsafe {
            device.free_memory(self.memory, None);
        }
    }
}

// A reference to a piece of memory from an allocator.
#[derive(Clone, Copy, Debug)]
pub(crate) struct MemoryIndex {
    block_index: usize,
    memory_type_index: u32,
    offset: vk::DeviceSize,
}

const DEFAULT_BLOCK_SIZE: vk::DeviceSize = 1024 * 1024 * 20;

// A linear block allocator.
#[derive(Default, Debug)]
pub(crate) struct Allocator {
    // The block allocated for each memory type index.
    blocks: HashMap<u32, Vec<MemoryBlock>>,
}

impl Allocator {
    // TODO: It might be worth checking previous blocks for free space.
    fn allocate(
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

        // Either allocate memory from the current block or allocate a new block.
        if let Some(offset) = block.allocate(requirements.size, requirements.alignment) {
            Ok((
                block.memory,
                MemoryIndex {
                    block_index: blocks.len() - 1,
                    memory_type_index,
                    offset,
                },
            ))
        } else {
            let mut block = create_block()?;
            block.offset = requirements.size;
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

#[derive(Debug, PartialEq, Eq, Clone, Copy, Default)]
pub enum Filter {
    #[default]
    Linear = vk::Filter::LINEAR.as_raw() as isize,
    Nearest = vk::Filter::NEAREST.as_raw() as isize,
}

impl From<Filter> for vk::Filter {
    fn from(filter: Filter) -> Self {
        Self::from_raw(filter as i32)
    }
}

#[derive(Default)]
pub struct SamplerRequest {
    pub filter: Filter,
    pub max_anisotropy: Option<f32>,
    pub clamp_to_edge: bool,
}

#[derive(Debug)]
pub struct Sampler {
    pub(crate) sampler: vk::Sampler,
}

impl Sampler {
    pub(crate) fn new(device: &Device, request: &SamplerRequest) -> Result<Self, Error> {
        let address_mode = match request.clamp_to_edge {
            true => vk::SamplerAddressMode::CLAMP_TO_EDGE,
            false => vk::SamplerAddressMode::REPEAT,
        };
        let create_info = vk::SamplerCreateInfo::default()
            .mag_filter(request.filter.into())
            .min_filter(request.filter.into())
            .address_mode_u(address_mode)
            .address_mode_v(address_mode)
            .address_mode_w(address_mode)
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

    pub(crate) fn destroy(&self, device: &Device) {
        unsafe {
            device.destroy_sampler(self.sampler, None);
        }
    }
}

fn acceleration_structure_device_address(
    device: &Device,
    acc_struct: vk::AccelerationStructureKHR,
) -> vk::DeviceAddress {
    let address_info =
        vk::AccelerationStructureDeviceAddressInfoKHR::default().acceleration_structure(acc_struct);
    unsafe {
        device
            .acceleration_structure
            .get_acceleration_structure_device_address(&address_info)
    }
}

#[derive(Debug, PartialEq, Eq, Clone, Copy, Default)]
pub enum VertexFormat {
    #[default]
    Snorm16 = vk::Format::R16G16B16_SNORM.as_raw() as isize,
    Float32 = vk::Format::R32G32B32_SFLOAT.as_raw() as isize,
}

impl From<VertexFormat> for vk::Format {
    fn from(vertex_format: VertexFormat) -> Self {
        Self::from_raw(vertex_format as i32)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct BlasRequest {
    pub vertex_format: VertexFormat,
    pub vertex_stride: u64,
    pub triangle_count: u32,
    pub vertex_count: u32,
    pub first_vertex: u32,
}

#[derive(Debug)]
pub struct Blas {
    acceleration_structure: vk::AccelerationStructureKHR,
    build_scratch_size: vk::DeviceSize,
    request: BlasRequest,
    buffer: Handle<Buffer>,
    pub(crate) access: Access,
    pub(crate) timestamp: u64,
}

impl Context {
    fn blas_geometry(
        &self,
        request: &BlasRequest,
        vertices: Option<BufferRange>,
        indices: Option<BufferRange>,
    ) -> vk::AccelerationStructureGeometryKHR<'static> {
        let device_address = |range: BufferRange| vk::DeviceOrHostAddressConstKHR {
            device_address: self.buffer(&range.buffer).device_address(&self.device) + range.offset,
        };
        let geometry_data = vk::AccelerationStructureGeometryDataKHR {
            triangles: vk::AccelerationStructureGeometryTrianglesDataKHR::default()
                .vertex_stride(request.vertex_stride)
                .vertex_format(request.vertex_format.into())
                .max_vertex(request.vertex_count - 1)
                .index_data(indices.map(device_address).unwrap_or_default())
                .vertex_data(vertices.map(device_address).unwrap_or_default())
                .index_type(vk::IndexType::UINT32),
        };
        vk::AccelerationStructureGeometryKHR::default()
            .geometry_type(vk::GeometryTypeKHR::TRIANGLES)
            .geometry(geometry_data)
    }

    fn blas_build_info<'geometry>(
        &self,
        scratch: Option<&Handle<Buffer>>,
        blas: Option<&Handle<Blas>>,
        geometry: &'geometry vk::AccelerationStructureGeometryKHR,
    ) -> vk::AccelerationStructureBuildGeometryInfoKHR<'geometry> {
        vk::AccelerationStructureBuildGeometryInfoKHR::default()
            .dst_acceleration_structure(
                blas.map(|blas| self.blas(blas).acceleration_structure)
                    .unwrap_or_default(),
            )
            .ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL)
            .flags(
                vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE
                    | vk::BuildAccelerationStructureFlagsKHR::ALLOW_DATA_ACCESS,
            )
            .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
            .geometries(slice::from_ref(geometry))
            .scratch_data(vk::DeviceOrHostAddressKHR {
                device_address: scratch
                    .map(|buffer| self.buffer(buffer).device_address(&self.device))
                    .unwrap_or_default(),
            })
    }

    pub fn create_blas(
        &mut self,
        lifetime: Lifetime,
        request: &BlasRequest,
    ) -> Result<Handle<Blas>, Error> {
        let geometry = self.blas_geometry(request, None, None);
        let mut size_info = vk::AccelerationStructureBuildSizesInfoKHR::default();
        unsafe {
            self.device
                .acceleration_structure
                .get_acceleration_structure_build_sizes(
                    vk::AccelerationStructureBuildTypeKHR::DEVICE,
                    &self.blas_build_info(None, None, &geometry),
                    &[request.triangle_count],
                    &mut size_info,
                );
        };
        let buffer = self.create_buffer(
            lifetime,
            &BufferRequest {
                size: size_info.acceleration_structure_size,
                ty: BufferType::AccelerationStructure,
                memory_location: MemoryLocation::Device,
            },
        )?;
        let create_info = vk::AccelerationStructureCreateInfoKHR::default()
            .ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL)
            .size(size_info.acceleration_structure_size)
            .buffer(self.buffer(&buffer).buffer)
            .offset(0);
        let acceleration_structure = unsafe {
            self.device
                .acceleration_structure
                .create_acceleration_structure(&create_info, None)
                .map_err(Error::from)?
        };
        let blas = Blas {
            access: Access::default(),
            timestamp: 0,
            build_scratch_size: size_info.build_scratch_size,
            acceleration_structure,
            request: *request,
            buffer,
        };
        let pool = self.pools.entry(lifetime).or_default();
        Ok(Handle::new(lifetime, pool.epoch, &mut pool.blases, blas))
    }
}

impl Blas {
    pub(crate) fn destroy(&self, device: &Device) {
        let loader = &device.acceleration_structure;
        unsafe {
            loader.destroy_acceleration_structure(self.acceleration_structure, None);
        }
    }

    pub(crate) fn device_address(&self, device: &Device) -> vk::DeviceAddress {
        acceleration_structure_device_address(device, self.acceleration_structure)
    }
}

#[derive(Debug, Clone)]
pub struct BufferRange {
    pub buffer: Handle<Buffer>,
    pub offset: vk::DeviceSize,
}

#[derive(Debug, Clone)]
pub struct BlasBuild {
    pub blas: Handle<Blas>,
    pub vertices: BufferRange,
    pub indices: BufferRange,
}

impl Context {
    pub fn build_blases(&mut self, builds: &[BlasBuild]) -> Result<&mut Self, Error> {
        if builds.is_empty() {
            return Ok(self);
        }
        let scratch_buffers: Vec<Handle<Buffer>> = builds
            .iter()
            .map(|build| {
                let size = self.blas(&build.blas).build_scratch_size;
                self.create_buffer(
                    Lifetime::Frame,
                    &BufferRequest {
                        memory_location: MemoryLocation::Device,
                        ty: BufferType::Storage,
                        size,
                    },
                )
            })
            .collect::<Result<_, _>>()?;
        let geometries: Vec<vk::AccelerationStructureGeometryKHR> = builds
            .iter()
            .map(|build| {
                self.blas_geometry(
                    &self.blas(&build.blas).request,
                    Some(build.vertices.clone()),
                    // Indices are offset in `AccelerationStructureBuildRangeInfoKHR` instead.
                    Some(BufferRange {
                        buffer: build.indices.buffer.clone(),
                        offset: 0,
                    }),
                )
            })
            .collect();
        let build_infos: Vec<vk::AccelerationStructureBuildGeometryInfoKHR> = builds
            .iter()
            .zip(geometries.iter())
            .zip(scratch_buffers.iter())
            .map(|((build, geometry), scratch_buffer)| {
                self.blas_build_info(Some(scratch_buffer), Some(&build.blas), geometry)
            })
            .collect();
        let range_infos: Vec<_> = builds
            .iter()
            .map(|build| {
                let blas = self.blas(&build.blas);
                vk::AccelerationStructureBuildRangeInfoKHR::default()
                    .primitive_count(blas.request.triangle_count)
                    .primitive_offset(build.indices.offset as u32)
                    .first_vertex(blas.request.first_vertex)
            })
            .collect();
        let range_infos_refs: Vec<_> = range_infos.iter().map(slice::from_ref).collect();
        let buffer_accesses: Vec<_> = builds
            .iter()
            .map(|build| (self.blas(&build.blas).buffer.clone(), STRUCTURE_DST_ACCESS))
            .chain(builds.iter().flat_map(|build| {
                [&build.indices, &build.vertices]
                    .map(|range| (range.buffer.clone(), STRUCTURE_SRC_ACCESS))
            }))
            .collect();
        let blas_accesses: Vec<_> = builds
            .iter()
            .map(|build| (build.blas.clone(), STRUCTURE_DST_ACCESS))
            .collect();
        self.access_resources(&[], &buffer_accesses, &blas_accesses, &[])?;
        unsafe {
            self.device
                .acceleration_structure
                .cmd_build_acceleration_structures(
                    self.command_buffer().buffer,
                    &build_infos,
                    &range_infos_refs,
                );
        }
        Ok(self)
    }
}

#[derive(Debug)]
pub struct Tlas {
    pub(crate) acceleration_structure: vk::AccelerationStructureKHR,
    scratch: Handle<Buffer>,
    instances: Handle<Buffer>,
    pub(crate) access: Access,
    pub(crate) timestamp: u64,
}

impl Context {
    pub fn create_tlas(
        &mut self,
        lifetime: Lifetime,
        instance_count: u32,
    ) -> Result<Handle<Tlas>, Error> {
        let modes = [
            vk::BuildAccelerationStructureModeKHR::BUILD,
            vk::BuildAccelerationStructureModeKHR::UPDATE,
        ];
        let [build_size, update_size] = modes.map(|mode| {
            let geometry_data = vk::AccelerationStructureGeometryDataKHR {
                instances: vk::AccelerationStructureGeometryInstancesDataKHR::default(),
            };
            let geometry = vk::AccelerationStructureGeometryKHR::default()
                .geometry_type(vk::GeometryTypeKHR::INSTANCES)
                .geometry(geometry_data);
            let build_info = vk::AccelerationStructureBuildGeometryInfoKHR::default()
                .ty(vk::AccelerationStructureTypeKHR::TOP_LEVEL)
                .geometries(slice::from_ref(&geometry))
                .flags(
                    vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE
                        | vk::BuildAccelerationStructureFlagsKHR::ALLOW_DATA_ACCESS
                        | vk::BuildAccelerationStructureFlagsKHR::ALLOW_UPDATE,
                )
                .mode(mode);
            let mut size = vk::AccelerationStructureBuildSizesInfoKHR::default();
            unsafe {
                self.device
                    .acceleration_structure
                    .get_acceleration_structure_build_sizes(
                        vk::AccelerationStructureBuildTypeKHR::DEVICE,
                        &build_info,
                        slice::from_ref(&instance_count),
                        &mut size,
                    );
            };
            size
        });
        let buffer = self.create_buffer(
            lifetime,
            &BufferRequest {
                size: build_size.acceleration_structure_size,
                ty: BufferType::AccelerationStructure,
                memory_location: MemoryLocation::Device,
            },
        )?;
        let scratch = self.create_buffer(
            lifetime,
            &BufferRequest {
                ty: BufferType::Storage,
                size: build_size
                    .build_scratch_size
                    .max(update_size.update_scratch_size),
                memory_location: MemoryLocation::Device,
            },
        )?;
        let instances = self.create_buffer(
            lifetime,
            &BufferRequest {
                size: mem::size_of::<TlasInstanceData>() as vk::DeviceSize
                    * instance_count as vk::DeviceSize,
                ty: BufferType::Storage,
                memory_location: MemoryLocation::Device,
            },
        )?;
        let as_info = vk::AccelerationStructureCreateInfoKHR::default()
            .ty(vk::AccelerationStructureTypeKHR::TOP_LEVEL)
            .buffer(self.buffer(&buffer).buffer)
            .size(build_size.acceleration_structure_size)
            .offset(0);
        let acceleration_structure = unsafe {
            self.device
                .acceleration_structure
                .create_acceleration_structure(&as_info, None)
                .map_err(Error::from)?
        };
        let tlas = Tlas {
            access: Access::default(),
            timestamp: 0,
            acceleration_structure,
            instances,
            scratch,
        };
        let pool = self.pools.entry(lifetime).or_default();
        Ok(Handle::new(lifetime, pool.epoch, &mut pool.tlases, tlas))
    }
}

impl Tlas {
    pub(crate) fn device_address(&self, device: &Device) -> vk::DeviceAddress {
        acceleration_structure_device_address(device, self.acceleration_structure)
    }

    pub(crate) fn destroy(&self, device: &Device) {
        let loader = &device.acceleration_structure;
        unsafe {
            loader.destroy_acceleration_structure(self.acceleration_structure, None);
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum TlasBuildMode {
    Build = vk::BuildAccelerationStructureModeKHR::BUILD.as_raw() as isize,
    Update = vk::BuildAccelerationStructureModeKHR::UPDATE.as_raw() as isize,
}

impl From<TlasBuildMode> for vk::BuildAccelerationStructureModeKHR {
    fn from(mode: TlasBuildMode) -> Self {
        vk::BuildAccelerationStructureModeKHR::from_raw(mode as i32)
    }
}

/// The instance in a [`Tlas`].
#[derive(Debug, Clone)]
pub struct TlasInstance {
    pub blas: Handle<Blas>,
    pub transform: Mat4,
    /// A custom index that can be queried in shaders.
    pub index: u32,
}

impl Context {
    pub fn build_tlas(
        &mut self,
        tlas: &Handle<Tlas>,
        mode: TlasBuildMode,
        instances: &[TlasInstance],
    ) -> Result<&mut Self, Error> {
        let instance_data: Vec<_> = instances
            .iter()
            .map(|instance| {
                let transform = instance.transform.transpose().to_cols_array();
                let index_bytes = instance.index.to_le_bytes();
                TlasInstanceData {
                    blas_address: self.blas(&instance.blas).device_address(&self.device),
                    transform: array::from_fn(|index| transform[index]),
                    index: array::from_fn(|index| index_bytes[index]),
                    flags: 0,
                    mask: 0xff,
                    ..Default::default()
                }
            })
            .collect();
        let geometry = &tlas_geometry_info(&self.device, self.buffer(&self.tlas(tlas).instances));
        let build_info = vk::AccelerationStructureBuildGeometryInfoKHR::default()
            .dst_acceleration_structure(self.tlas(tlas).acceleration_structure)
            .src_acceleration_structure(self.tlas(tlas).acceleration_structure)
            .ty(vk::AccelerationStructureTypeKHR::TOP_LEVEL)
            .mode(mode.into())
            .flags(
                vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE
                    | vk::BuildAccelerationStructureFlagsKHR::ALLOW_DATA_ACCESS
                    | vk::BuildAccelerationStructureFlagsKHR::ALLOW_UPDATE,
            )
            .geometries(slice::from_ref(geometry))
            .scratch_data(vk::DeviceOrHostAddressKHR {
                device_address: self
                    .buffer(&self.tlas(tlas).scratch)
                    .device_address(&self.device),
            });
        let build_ranges = vk::AccelerationStructureBuildRangeInfoKHR::default()
            .primitive_count(instance_data.len() as u32);
        self.write_buffers(&[BufferWrite {
            buffer: self.tlas(tlas).instances.clone(),
            data: bytemuck::cast_slice(&instance_data).into(),
        }])?;
        let scratch_access = Access {
            stage: vk::PipelineStageFlags2::ACCELERATION_STRUCTURE_BUILD_KHR,
            access: vk::AccessFlags2::ACCELERATION_STRUCTURE_READ_KHR
                | vk::AccessFlags2::ACCELERATION_STRUCTURE_WRITE_KHR,
        };
        let blas_accesses: Vec<_> = instances
            .iter()
            .map(|instance| (instance.blas.clone(), STRUCTURE_SRC_ACCESS))
            .collect();
        let tlas_accesses = [(tlas.clone(), STRUCTURE_DST_ACCESS)];
        let buffer_accesses = [
            (self.tlas(tlas).instances.clone(), STRUCTURE_SRC_ACCESS),
            (self.tlas(tlas).scratch.clone(), scratch_access),
        ];
        self.access_resources(&[], &buffer_accesses, &blas_accesses, &tlas_accesses)?;
        unsafe {
            self.device
                .acceleration_structure
                .cmd_build_acceleration_structures(
                    self.command_buffer().buffer,
                    slice::from_ref(&build_info),
                    slice::from_ref(&slice::from_ref(&build_ranges)),
                );
        }
        Ok(self)
    }
}

// The same layout as vk::AccelerationStructureInstanceKHR.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::NoUninit, Default)]
struct TlasInstanceData {
    transform: [f32; 12],
    index: [u8; 3],
    mask: u8,
    binding_table_offset: [u8; 3],
    flags: u8,
    blas_address: vk::DeviceAddress,
}

fn tlas_geometry_info(
    device: &Device,
    instance_buffer: &Buffer,
) -> vk::AccelerationStructureGeometryKHR<'static> {
    let geometry_data = vk::AccelerationStructureGeometryDataKHR {
        instances: vk::AccelerationStructureGeometryInstancesDataKHR::default()
            .array_of_pointers(false)
            .data(vk::DeviceOrHostAddressConstKHR {
                device_address: instance_buffer.device_address(device),
            }),
    };
    vk::AccelerationStructureGeometryKHR::default()
        .geometry_type(vk::GeometryTypeKHR::INSTANCES)
        .flags(vk::GeometryFlagsKHR::empty())
        .geometry(geometry_data)
}

const STRUCTURE_SRC_ACCESS: Access = Access {
    stage: vk::PipelineStageFlags2::ACCELERATION_STRUCTURE_BUILD_KHR,
    // Not an error, this is actually what the spec says.
    access: vk::AccessFlags2::SHADER_READ,
};

const STRUCTURE_DST_ACCESS: Access = Access {
    stage: vk::PipelineStageFlags2::ACCELERATION_STRUCTURE_BUILD_KHR,
    access: vk::AccessFlags2::ACCELERATION_STRUCTURE_WRITE_KHR,
};
