//! # The shader system
//!
//! Shaders are handled somewhat untradictionally compared to other renderes. First of all, only
//! compute shaders are supported. This simplifies the whole system quite a bit. Secondly, instead
//! of declaring shader bindings in the shaders themselves, they are declared in Rust and inserted
//! where `#include "<bindings>"` is in the GLSL source code. This is in some sense the reverse of
//! the more usual shader reflections, where the bindings are obtained by parsing the intermediate
//! representation instead. This also removes the busy work of managing binding indices. Resources
//! are bound their their names instead. Lastly, all descriptors are managed through descriptor
//! buffers instead of the classic descriptor sets. Descriptor buffers are arguably simpler than
//! descriptor sets (but perhaps harder to debug). Binding a resource only involves a memcpy into
//! a preallocated buffer, which makes it very fast.

use std::collections::HashMap;
use std::{mem, ops, slice};

use ash::vk;

use crate::Extent;

use super::command::CommandBuffer;
use super::device::Device;
use super::glsl::{render_bindings, render_shader};
use super::sync::Access;
use super::{Buffer, Context, Error, Handle, Image, ImageFormat, Lifetime, Sampler, Tlas};

#[derive(Debug, Clone)]
pub enum BindingType {
    StorageBuffer {
        ty: &'static str,
        array: bool,
        writes: bool,
    },
    UniformBuffer {
        ty: &'static str,
    },
    AccelerationStructure,
    SampledImage {
        count: Option<u32>,
    },
    StorageImage {
        count: Option<u32>,
        writes: bool,
        format: ImageFormat,
    },
}

impl BindingType {
    fn descriptor_type(&self) -> vk::DescriptorType {
        match self {
            Self::StorageBuffer { .. } => vk::DescriptorType::STORAGE_BUFFER,
            Self::UniformBuffer { .. } => vk::DescriptorType::UNIFORM_BUFFER,
            Self::AccelerationStructure => vk::DescriptorType::ACCELERATION_STRUCTURE_KHR,
            Self::SampledImage { .. } => vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            Self::StorageImage { .. } => vk::DescriptorType::STORAGE_IMAGE,
        }
    }

    fn count(&self) -> Option<u32> {
        match self {
            Self::SampledImage { count } | Self::StorageImage { count, .. } => *count,
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Binding {
    pub name: &'static str,
    pub ty: BindingType,
}

#[macro_export]
macro_rules! binding {
    (storage_buffer, $ty:ident, $name:ident, $array:expr, $writes:expr) => {
        Binding {
            name: stringify!($name),
            ty: BindingType::StorageBuffer {
                ty: stringify!($ty),
                array: $array,
                writes: $writes,
            },
        }
    };
    (uniform_buffer, $ty:ident, $name:ident) => {
        Binding {
            name: stringify!($name),
            ty: BindingType::UniformBuffer {
                ty: stringify!($ty),
            },
        }
    };
    (acceleration_structure, $name:ident) => {
        Binding {
            name: stringify!($name),
            ty: BindingType::AccelerationStructure,
        }
    };
    (sampled_image, $name:ident, $count:expr) => {
        Binding {
            name: stringify!($name),
            ty: BindingType::SampledImage { count: $count },
        }
    };
    (storage_image, $format:expr, $name:ident, $count:expr, $writes:expr) => {
        Binding {
            name: stringify!($name),
            ty: BindingType::StorageImage {
                count: $count,
                writes: $writes,
                format: $format,
            },
        }
    };
}

fn create_descriptor_layout(
    device: &Device,
    bindings: &[Binding],
) -> Result<vk::DescriptorSetLayout, Error> {
    let flags: Vec<_> = bindings
        .iter()
        .map(|binding| {
            binding
                .ty
                .count()
                .map(|_| vk::DescriptorBindingFlags::VARIABLE_DESCRIPTOR_COUNT)
                .unwrap_or_default()
        })
        .collect();
    let layout_bindings: Vec<_> = bindings
        .iter()
        .enumerate()
        .map(|(location, binding)| {
            vk::DescriptorSetLayoutBinding::default()
                .binding(location as u32)
                .descriptor_type(binding.ty.descriptor_type())
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
                .descriptor_count(binding.ty.count().unwrap_or(1))
        })
        .collect();
    let mut binding_flags =
        vk::DescriptorSetLayoutBindingFlagsCreateInfo::default().binding_flags(&flags);
    let layout_info = vk::DescriptorSetLayoutCreateInfo::default()
        .flags(vk::DescriptorSetLayoutCreateFlags::DESCRIPTOR_BUFFER_EXT)
        .bindings(&layout_bindings)
        .push_next(&mut binding_flags);
    unsafe {
        device
            .create_descriptor_set_layout(&layout_info, None)
            .map_err(Error::from)
    }
}

#[derive(Debug, Clone, Copy)]
struct ShaderBinding {
    access_flags: vk::AccessFlags2,
    descriptor_size: usize,
    descriptor_offset: usize,
    ty: vk::DescriptorType,
}

impl ShaderBinding {
    fn new(
        device: &Device,
        ty: &BindingType,
        layout: vk::DescriptorSetLayout,
        index: u32,
    ) -> ShaderBinding {
        let properties = &device.descriptor_buffer_properties;
        let (access_flags, descriptor_size) = match ty {
            BindingType::StorageBuffer { writes, .. } => {
                let access_flags = vk::AccessFlags2::SHADER_STORAGE_READ
                    | writes
                        .then_some(vk::AccessFlags2::SHADER_STORAGE_WRITE)
                        .unwrap_or_default();
                (access_flags, properties.storage_buffer_descriptor_size)
            }
            BindingType::StorageImage { writes, .. } => {
                let access_flags = vk::AccessFlags2::SHADER_STORAGE_READ
                    | writes
                        .then_some(vk::AccessFlags2::SHADER_STORAGE_WRITE)
                        .unwrap_or_default();
                (access_flags, properties.storage_image_descriptor_size)
            }
            BindingType::UniformBuffer { .. } => (
                vk::AccessFlags2::UNIFORM_READ,
                properties.uniform_buffer_descriptor_size,
            ),
            BindingType::AccelerationStructure => {
                let descriptor_size = properties.acceleration_structure_descriptor_size;
                let access_flags = vk::AccessFlags2::ACCELERATION_STRUCTURE_READ_KHR;
                (access_flags, descriptor_size)
            }
            BindingType::SampledImage { .. } => {
                let descriptor_size = properties.combined_image_sampler_descriptor_size;
                (vk::AccessFlags2::SHADER_SAMPLED_READ, descriptor_size)
            }
        };
        let descriptor_offset = unsafe {
            device
                .descriptor_buffer
                .get_descriptor_set_layout_binding_offset(layout, index) as usize
        };
        ShaderBinding {
            ty: ty.descriptor_type(),
            access_flags,
            descriptor_offset,
            descriptor_size,
        }
    }
}

#[derive(Debug, Clone)]
enum BoundResource {
    Buffer(Handle<Buffer>, vk::AccessFlags2),
    Image(Handle<Image>, vk::AccessFlags2),
    AccelerationStructure(Handle<Tlas>),
}

#[derive(Debug)]
pub(super) struct BoundShader {
    shader: Handle<Shader>,
    bound: HashMap<&'static str, BoundResource>,
    /// `true` if this shader wast just dispatched.
    has_been_dispatched: bool,
    /// `true` if push constants have been pushed.
    constants_has_been_pushed: bool,
}

impl BoundShader {
    fn new(shader: Handle<Shader>) -> Self {
        Self {
            bound: HashMap::default(),
            has_been_dispatched: false,
            constants_has_been_pushed: false,
            shader,
        }
    }

    fn bind_image(
        &mut self,
        name: &'static str,
        image: &Handle<Image>,
        flags: vk::AccessFlags2,
    ) -> bool {
        self.bound
            .insert(name, BoundResource::Image(image.clone(), flags));
        mem::take(&mut self.has_been_dispatched)
    }

    fn bind_buffer(
        &mut self,
        name: &'static str,
        buffer: &Handle<Buffer>,
        flags: vk::AccessFlags2,
    ) -> bool {
        self.bound
            .insert(name, BoundResource::Buffer(buffer.clone(), flags));
        mem::take(&mut self.has_been_dispatched)
    }

    fn bind_acceleration_structure(&mut self, name: &'static str, tlas: &Handle<Tlas>) -> bool {
        self.bound
            .insert(name, BoundResource::AccelerationStructure(tlas.clone()));
        mem::take(&mut self.has_been_dispatched)
    }
}

#[derive(Debug)]
pub struct DescriptorBuffer {
    pub buffer: Buffer,
    pub data: *mut u8,
    pub size: usize,
    pub bound_range: ops::Range<usize>,
}

impl DescriptorBuffer {
    fn write(
        &mut self,
        device: &Device,
        binding: &ShaderBinding,
        index: usize,
        get_info: &vk::DescriptorGetInfoEXT,
    ) {
        unsafe {
            let offset = self.bound_range.start
                + binding.descriptor_offset
                + binding.descriptor_size * index;
            let data = slice::from_raw_parts_mut(self.data.add(offset), binding.descriptor_size);
            device.descriptor_buffer.get_descriptor(get_info, data);
        }
    }

    fn write_images(
        &mut self,
        device: &Device,
        views: &[vk::ImageView],
        sampler: Option<vk::Sampler>,
        binding: &ShaderBinding,
    ) {
        for (index, view) in views.iter().enumerate() {
            let descriptor_image_info = vk::DescriptorImageInfo::default()
                .image_view(*view)
                .sampler(sampler.unwrap_or_default());
            let get_info =
                vk::DescriptorGetInfoEXT::default()
                    .ty(binding.ty)
                    .data(vk::DescriptorDataEXT {
                        p_combined_image_sampler: &descriptor_image_info as *const _,
                    });
            self.write(device, binding, index, &get_info);
        }
    }

    fn write_buffer(
        &mut self,
        device: &Device,
        address: vk::DeviceAddress,
        size: vk::DeviceSize,
        binding: &ShaderBinding,
    ) {
        let descriptor_buffer_info = vk::DescriptorAddressInfoEXT::default()
            .address(address)
            .range(size);
        let get_info =
            vk::DescriptorGetInfoEXT::default()
                .ty(binding.ty)
                .data(vk::DescriptorDataEXT {
                    p_uniform_buffer: &descriptor_buffer_info as *const _,
                });
        self.write(device, binding, 0, &get_info);
    }

    fn write_acceleration_structure(
        &mut self,
        device: &Device,
        address: vk::DeviceAddress,
        binding: &ShaderBinding,
    ) {
        let get_info = vk::DescriptorGetInfoEXT::default()
            .ty(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
            .data(vk::DescriptorDataEXT {
                acceleration_structure: address,
            });
        self.write(device, binding, 0, &get_info);
    }

    fn check_range_increase(&self, size: usize) {
        assert!(
            size + self.bound_range.end < self.size,
            "out of descriptor range"
        );
    }

    // Allocate new range of `size` bytes. This is where following writes will be located.
    fn allocate_range(&mut self, size: vk::DeviceSize) {
        self.check_range_increase(size as usize);
        self.bound_range.start = self.bound_range.end;
        self.bound_range.end += size as usize;
    }

    // When a bound shader is run multiple times and one or more bindings change, the
    // descriptor buffer data has to be duplicated in order to not overwrite data for the
    // previous dispatch.
    fn maybe_duplicate_range(&mut self, duplicate: bool) {
        if duplicate {
            let range_size = self.bound_range.end - self.bound_range.start;
            self.check_range_increase(range_size);
            unsafe {
                let start = self.data.add(self.bound_range.start);
                self.data.add(range_size).copy_from(start, range_size);
            }
            self.allocate_range(range_size as vk::DeviceSize);
        }
    }
}

#[derive(Debug)]
pub struct Shader {
    pipeline: vk::Pipeline,
    layout: vk::PipelineLayout,
    descriptor_layout: vk::DescriptorSetLayout,
    bindings: HashMap<&'static str, ShaderBinding>,
    push_constant_size: Option<u32>,
    block_size: Extent,
}

impl Shader {
    pub fn descriptor_size(&self, device: &Device) -> vk::DeviceSize {
        let function = &device.descriptor_buffer;
        unsafe { function.get_descriptor_set_layout_size(self.descriptor_layout) }
    }

    pub fn destroy(&self, device: &Device) {
        unsafe {
            device.destroy_pipeline(self.pipeline, None);
            device.destroy_pipeline_layout(self.layout, None);
            device.destroy_descriptor_set_layout(self.descriptor_layout, None);
        }
    }
}

fn create_pipeline_layout(
    device: &Device,
    descriptor_layout: vk::DescriptorSetLayout,
    push_constant_size: Option<u32>,
) -> Result<vk::PipelineLayout, Error> {
    let mut layout_info =
        vk::PipelineLayoutCreateInfo::default().set_layouts(slice::from_ref(&descriptor_layout));
    let mut push_constant_range = vk::PushConstantRange::default();
    if let Some(size) = push_constant_size {
        push_constant_range = push_constant_range
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .size(size);
        layout_info = layout_info.push_constant_ranges(slice::from_ref(&push_constant_range));
    }
    let pipeline_layout = unsafe { device.create_pipeline_layout(&layout_info, None)? };
    Ok(pipeline_layout)
}

pub struct ShaderRequest<'a> {
    pub source: &'a str,
    pub block_size: Extent,
    pub bindings: &'a [Binding],
    pub push_constant_size: Option<u32>,
}

impl Context {
    fn create_shader_module(&self, request: &ShaderRequest) -> Result<vk::ShaderModule, Error> {
        let compiler = shaderc::Compiler::new().unwrap();
        let mut options = shaderc::CompileOptions::new().unwrap();
        options.set_optimization_level(shaderc::OptimizationLevel::Performance);
        options.set_include_callback(|name, _, _, _| {
            let content = if name == "<bindings>" {
                render_bindings(request.bindings)
            } else {
                self.includes.get(name).cloned().unwrap_or_else(|| {
                    panic!("include {name} not found");
                })
            };
            Ok(shaderc::ResolvedInclude {
                resolved_name: name.into(),
                content,
            })
        });
        let Extent { width, height } = request.block_size;
        let source = render_shader(width, height, request.source);
        let shader_kind = shaderc::ShaderKind::Compute;
        let output = compiler
            .compile_into_spirv(&source, shader_kind, "shader", "main", Some(&options))
            .map_err(Error::from)?;
        let shader_info = vk::ShaderModuleCreateInfo::default().code(output.as_binary());
        unsafe {
            self.device
                .create_shader_module(&shader_info, None)
                .map_err(Error::from)
        }
    }

    pub fn create_shader(
        &mut self,
        lifetime: Lifetime,
        request: &ShaderRequest,
    ) -> Result<Handle<Shader>, Error> {
        let module = self.create_shader_module(request)?;
        let descriptor_layout = create_descriptor_layout(&self.device, request.bindings)?;
        let pipeline_layout =
            create_pipeline_layout(&self.device, descriptor_layout, request.push_constant_size)?;
        let stage_info = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(module)
            .name(c"main");
        let pipeline_info = vk::ComputePipelineCreateInfo::default()
            .flags(vk::PipelineCreateFlags::DESCRIPTOR_BUFFER_EXT)
            .stage(stage_info)
            .layout(pipeline_layout);
        let pipeline = unsafe {
            let pipelines = self.device.create_compute_pipelines(
                vk::PipelineCache::null(),
                slice::from_ref(&pipeline_info),
                None,
            );
            self.device.destroy_shader_module(module, None);
            *pipelines.map_err(|(_, err)| err)?.first().unwrap()
        };
        let pipeline = Shader {
            push_constant_size: request.push_constant_size,
            bindings: request
                .bindings
                .iter()
                .enumerate()
                .map(|(index, binding)| {
                    let shader_binding = ShaderBinding::new(
                        &self.device,
                        &binding.ty,
                        descriptor_layout,
                        index as u32,
                    );
                    (binding.name, shader_binding)
                })
                .collect(),
            layout: pipeline_layout,
            block_size: request.block_size,
            descriptor_layout,
            pipeline,
        };
        let pool = self.pool_mut(lifetime);
        Ok(Handle::new(
            lifetime,
            pool.epoch,
            &mut pool.shaders,
            pipeline,
        ))
    }
}

fn current_descriptor_buffer(command_buffers: &mut [CommandBuffer]) -> &mut DescriptorBuffer {
    &mut command_buffers.first_mut().unwrap().descriptor_buffer
}

impl Context {
    pub fn bind_shader(&mut self, shader: &Handle<Shader>) -> &mut Self {
        self.bound_shader = Some(BoundShader::new(shader.clone()));
        unsafe {
            self.device.cmd_bind_pipeline(
                self.command_buffer().buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.shader(shader).pipeline,
            );
        }
        let descriptor_size = self.shader(shader).descriptor_size(&self.device);
        current_descriptor_buffer(&mut self.command_buffers).allocate_range(descriptor_size);
        self
    }

    fn bound_shader(&self) -> &BoundShader {
        self.bound_shader.as_ref().expect("no shader bound")
    }

    fn bound_shader_mut(&mut self) -> &mut BoundShader {
        self.bound_shader.as_mut().expect("no shader bound")
    }

    fn binding(&self, name: &'static str) -> &ShaderBinding {
        let error = || panic!("no binding with name {name}");
        let shader = self.shader(&self.bound_shader().shader);
        shader.bindings.get(name).unwrap_or_else(error)
    }

    /// Push constants to the currently bound shader.
    pub fn push_constant(&mut self, constant: &impl bytemuck::NoUninit) -> &mut Self {
        let shader = self.shader(&self.bound_shader().shader);
        let bytes = bytemuck::bytes_of(constant);
        assert_eq!(
            shader.push_constant_size,
            Some(bytes.len() as u32),
            "push constant doesn't match shader"
        );
        unsafe {
            let stage = vk::ShaderStageFlags::COMPUTE;
            let buffer = self.command_buffer().buffer;
            self.device
                .cmd_push_constants(buffer, shader.layout, stage, 0, bytes)
        }
        self.bound_shader_mut().constants_has_been_pushed = true;
        self
    }

    fn bind_images(
        &mut self,
        name: &'static str,
        sampler: Option<&Handle<Sampler>>,
        images: &[Handle<Image>],
    ) -> &mut Self {
        let binding = *self.binding(name);
        let bound_shader = self.bound_shader_mut();
        let duplicate_descriptor = images.iter().fold(false, |duplicate, image| {
            duplicate | bound_shader.bind_image(name, image, binding.access_flags)
        });
        self.command_buffer_mut()
            .descriptor_buffer
            .maybe_duplicate_range(duplicate_descriptor);
        let image_views: Vec<_> = images.iter().map(|image| self.image(image).view).collect();
        let sampler = sampler.map(|sampler| self.sampler(sampler).sampler);
        current_descriptor_buffer(&mut self.command_buffers).write_images(
            &self.device,
            &image_views,
            sampler,
            &binding,
        );
        self
    }

    /// Bind array of storage images to the currently bound shader.
    pub fn bind_storage_images(
        &mut self,
        name: &'static str,
        images: &[Handle<Image>],
    ) -> &mut Self {
        self.bind_images(name, None, images);
        self
    }

    /// Bind array of sampled images to the currently bound shader.
    pub fn bind_sampled_images(
        &mut self,
        name: &'static str,
        sampler: &Handle<Sampler>,
        images: &[Handle<Image>],
    ) -> &mut Self {
        self.bind_images(name, Some(sampler), images)
    }

    /// Bind a single storage image to the currently bound shader.
    pub fn bind_storage_image(&mut self, name: &'static str, image: &Handle<Image>) -> &mut Self {
        self.bind_storage_images(name, &[image.clone()])
    }

    /// Bind a single sampled image to the currently bound shader.
    pub fn bind_sampled_image(
        &mut self,
        name: &'static str,
        sampler: &Handle<Sampler>,
        image: &Handle<Image>,
    ) {
        self.bind_sampled_images(name, sampler, &[image.clone()]);
    }

    /// Bind buffer to the currently bound shader.
    pub fn bind_buffer(&mut self, name: &'static str, buffer: &Handle<Buffer>) -> &mut Self {
        let binding = *self.binding(name);
        let duplicate_descriptor =
            self.bound_shader_mut()
                .bind_buffer(name, buffer, binding.access_flags);
        let buffer = self.buffer(buffer);
        let (address, size) = (buffer.device_address(&self.device), buffer.size);
        let descriptor_buffer = current_descriptor_buffer(&mut self.command_buffers);
        descriptor_buffer.maybe_duplicate_range(duplicate_descriptor);
        descriptor_buffer.write_buffer(&self.device, address, size, &binding);
        self
    }

    /// Bind (TLAS) top level acceleration structure to the currently bound shader.
    pub fn bind_acceleration_structure(
        &mut self,
        name: &'static str,
        tlas: &Handle<Tlas>,
    ) -> &mut Self {
        let binding = *self.binding(name);
        let duplicate_descriptor = self
            .bound_shader_mut()
            .bind_acceleration_structure(name, tlas);
        let tlas = self.tlas(tlas);
        let address = tlas.device_address(&self.device);
        let descriptor_buffer = current_descriptor_buffer(&mut self.command_buffers);
        descriptor_buffer.maybe_duplicate_range(duplicate_descriptor);
        descriptor_buffer.write_acceleration_structure(&self.device, address, &binding);
        self
    }

    fn set_descriptor(&self, offset: u64, layout: vk::PipelineLayout) {
        let point = vk::PipelineBindPoint::COMPUTE;
        let buffer = self.command_buffer().buffer;
        unsafe {
            let loader = &self.device.descriptor_buffer;
            loader.cmd_set_descriptor_buffer_offsets(buffer, point, layout, 0, &[0], &[offset]);
        }
    }

    fn check_all_bindings_are_bound(&self, bound_shader: &BoundShader) {
        let bindings = &self.shader(&bound_shader.shader).bindings;
        let mut missing = bindings
            .keys()
            .filter(|binding| !bound_shader.bound.contains_key(*binding));
        if let Some(missing) = missing.next() {
            panic!("binding `{missing}` is not bound");
        }
    }

    /// Dispatch bound shader with `width` * `height` threads. The actual number of threads
    /// launched is the next multiple of the block extent.
    pub fn dispatch(&mut self, width: u32, height: u32) -> Result<&mut Self, Error> {
        self.bound_shader_mut().has_been_dispatched = true;

        let (mut buffers, mut images, mut tlases) = (Vec::new(), Vec::new(), Vec::new());
        let create_access = |access: vk::AccessFlags2| Access {
            stage: vk::PipelineStageFlags2::COMPUTE_SHADER,
            access,
        };

        for resource in self.bound_shader().bound.values().cloned() {
            match resource {
                BoundResource::Buffer(buffer, access) => {
                    buffers.push((buffer, create_access(access)));
                }
                BoundResource::Image(image, access) => {
                    images.push((image, create_access(access)));
                }
                BoundResource::AccelerationStructure(tlas) => {
                    let access = vk::AccessFlags2::ACCELERATION_STRUCTURE_READ_KHR;
                    tlases.push((tlas.clone(), create_access(access)));
                }
            }
        }

        self.access_resources(&images, &buffers, &[], &tlases)?;

        let bound_shader = self.bound_shader();
        self.check_all_bindings_are_bound(bound_shader);

        let shader = self.shader(&bound_shader.shader);
        let offset = self.command_buffer().descriptor_buffer.bound_range.start as vk::DeviceSize;
        self.set_descriptor(offset, shader.layout);

        if shader.push_constant_size.is_some() && !bound_shader.constants_has_been_pushed {
            panic!("push constant is missing");
        }

        unsafe {
            self.device.cmd_dispatch(
                self.command_buffer().buffer,
                width.div_ceil(shader.block_size.width),
                height.div_ceil(shader.block_size.height),
                1,
            );
        }

        Ok(self)
    }
}
