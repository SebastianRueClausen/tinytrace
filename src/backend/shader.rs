use std::collections::HashMap;
use std::{fs, mem, ops, slice};

use ash::vk;

use super::device::Device;
use super::glsl::render_shader;
use super::sync::Access;
use super::{Buffer, Context, Handle, Image, Sampler};
use crate::{
    backend::Lifetime,
    error::{Error, Result},
};

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
        count: u32,
    },
    StorageImage {
        count: u32,
        writes: bool,
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

    fn count(&self) -> u32 {
        match self {
            Self::SampledImage { count } | Self::StorageImage { count, .. } => *count,
            _ => 1,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Binding {
    pub name: &'static str,
    pub ty: BindingType,
}

fn create_descriptor_layout(
    device: &Device,
    bindings: &[Binding],
) -> Result<vk::DescriptorSetLayout> {
    let flags: Vec<_> = bindings
        .iter()
        .map(|binding| {
            (binding.ty.count() > 1)
                .then_some(vk::DescriptorBindingFlags::VARIABLE_DESCRIPTOR_COUNT)
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
                .stage_flags(vk::ShaderStageFlags::ALL)
                .descriptor_count(binding.ty.count())
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

fn create_shader_module(
    device: &Device,
    source: &str,
    vk::Extent2D { width, height }: vk::Extent2D,
    bindings: &[Binding],
) -> Result<vk::ShaderModule> {
    let compiler = shaderc::Compiler::new().unwrap();
    let mut options = shaderc::CompileOptions::new().unwrap();
    options.set_include_callback(|name, _, _, _| {
        fs::read_to_string(name)
            .map_err(|err| format!("failed to import file {name}: {err}"))
            .map(|content| shaderc::ResolvedInclude {
                resolved_name: name.into(),
                content,
            })
    });
    let source = render_shader(width, height, source, bindings);
    let shader_kind = shaderc::ShaderKind::Compute;
    let output = compiler
        .compile_into_spirv(&source, shader_kind, "shader", "main", Some(&options))
        .map_err(Error::from)?;
    let shader_info = vk::ShaderModuleCreateInfo::default().code(output.as_binary());
    unsafe {
        device
            .create_shader_module(&shader_info, None)
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
}

#[derive(Debug)]
pub(super) struct BoundShader {
    shader: Handle<Shader>,
    bound_descriptors: HashMap<&'static str, BoundResource>,
    /// `true` if this shader wast just dispatched.
    has_been_dispatched: bool,
}

impl BoundShader {
    fn bind_image(
        &mut self,
        name: &'static str,
        image: &Handle<Image>,
        flags: vk::AccessFlags2,
    ) -> bool {
        self.bound_descriptors
            .insert(name, BoundResource::Image(image.clone(), flags));
        mem::take(&mut self.has_been_dispatched)
    }

    fn bind_buffer(
        &mut self,
        name: &'static str,
        buffer: &Handle<Buffer>,
        flags: vk::AccessFlags2,
    ) -> bool {
        self.bound_descriptors
            .insert(name, BoundResource::Buffer(buffer.clone(), flags));
        mem::take(&mut self.has_been_dispatched)
    }
}

pub(super) struct DescriptorBuffer {
    pub buffer: Handle<Buffer>,
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

    fn check_range_increase(&self, size: usize) {
        assert!(
            size + self.bound_range.end < self.size,
            "out of descriptor range"
        );
    }

    fn bind_range(&mut self, size: vk::DeviceSize) {
        self.check_range_increase(size as usize);
        self.bound_range.start = self.bound_range.end;
        self.bound_range.end += size as usize;
    }

    fn maybe_duplicate_range(&mut self, duplicate: bool) {
        if duplicate {
            let range_size = self.bound_range.end - self.bound_range.start;
            self.check_range_increase(range_size);
            unsafe {
                let start = self.data.add(self.bound_range.start);
                self.data.add(range_size).copy_from(start, range_size);
            }
            self.bind_range(range_size as vk::DeviceSize);
        }
    }
}

#[derive(Debug)]
pub struct Shader {
    pub pipeline: vk::Pipeline,
    pub layout: vk::PipelineLayout,
    pub descriptor_layout: vk::DescriptorSetLayout,
    bindings: HashMap<&'static str, ShaderBinding>,
    pub block_size: vk::Extent2D,
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
) -> Result<vk::PipelineLayout> {
    let layout_info =
        vk::PipelineLayoutCreateInfo::default().set_layouts(slice::from_ref(&descriptor_layout));
    let pipeline_layout = unsafe { device.create_pipeline_layout(&layout_info, None)? };
    Ok(pipeline_layout)
}

impl Context {
    pub fn create_shader(
        &mut self,
        source: &str,
        block_size: vk::Extent2D,
        bindings: &[Binding],
    ) -> Result<Handle<Shader>> {
        let module = create_shader_module(&self.device, source, block_size, bindings)?;
        let descriptor_layout = create_descriptor_layout(&self.device, bindings)?;
        let pipeline_layout = create_pipeline_layout(&self.device, descriptor_layout)?;
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
            bindings: bindings
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
            block_size,
            descriptor_layout,
            pipeline,
        };
        let pool = self.pool_mut(Lifetime::Static);
        Ok(Handle::new(
            Lifetime::Static,
            pool.epoch,
            &mut pool.shaders,
            pipeline,
        ))
    }

    pub fn bind_shader(&mut self, shader: &Handle<Shader>) {
        self.bound_shader = Some(BoundShader {
            bound_descriptors: HashMap::default(),
            shader: shader.clone(),
            has_been_dispatched: false,
        });
        let bind_point = vk::PipelineBindPoint::COMPUTE;
        let pipeline = self.shader(shader);
        unsafe {
            self.device
                .cmd_bind_pipeline(*self.command_buffer, bind_point, pipeline.pipeline);
        }
        self.descriptor_buffer
            .bind_range(pipeline.descriptor_size(&self.device));
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

    pub fn bind_images(
        &mut self,
        name: &'static str,
        sampler: Option<&Handle<Sampler>>,
        images: &[Handle<Image>],
    ) {
        let binding = *self.binding(name);
        let bound_shader = self.bound_shader_mut();
        let duplicate_descriptor = images.iter().fold(false, |duplicate, image| {
            duplicate | bound_shader.bind_image(name, image, binding.access_flags)
        });
        self.descriptor_buffer
            .maybe_duplicate_range(duplicate_descriptor);
        let image_views: Vec<_> = images.iter().map(|image| self.image(image).view).collect();
        let sampler = sampler.map(|sampler| self.sampler(sampler).sampler);
        self.descriptor_buffer
            .write_images(&self.device, &image_views, sampler, &binding);
    }

    pub fn bind_storage_images(&mut self, name: &'static str, images: &[Handle<Image>]) {
        self.bind_images(name, None, images);
    }

    pub fn bind_sampled_images(
        &mut self,
        name: &'static str,
        sampler: &Handle<Sampler>,
        images: &[Handle<Image>],
    ) {
        self.bind_images(name, Some(sampler), images);
    }

    pub fn bind_storage_image(&mut self, name: &'static str, image: &Handle<Image>) {
        self.bind_storage_images(name, &[image.clone()]);
    }

    pub fn bind_sampled_image(
        &mut self,
        name: &'static str,
        sampler: &Handle<Sampler>,
        image: &Handle<Image>,
    ) {
        self.bind_sampled_images(name, sampler, &[image.clone()]);
    }

    pub fn bind_buffer(&mut self, name: &'static str, buffer: &Handle<Buffer>) {
        let binding = *self.binding(name);
        let bound_shader = self.bound_shader_mut();
        let duplicate_descriptor = bound_shader.bind_buffer(name, buffer, binding.access_flags);
        let buffer = self.buffer(buffer);
        let (address, size) = (buffer.device_address(&self.device), buffer.size);
        self.descriptor_buffer
            .maybe_duplicate_range(duplicate_descriptor);
        self.descriptor_buffer
            .write_buffer(&self.device, address, size, &binding);
    }

    pub fn dispatch(&mut self, width: u32, height: u32) {
        let bound_shader = self.bound_shader_mut();
        bound_shader.has_been_dispatched = true;

        let create_access = |access: vk::AccessFlags2| Access {
            stage: vk::PipelineStageFlags2::COMPUTE_SHADER,
            access,
        };

        let (mut buffers, mut images) = (Vec::new(), Vec::new());
        bound_shader
            .bound_descriptors
            .values()
            .for_each(|resource| match resource.clone() {
                BoundResource::Buffer(buffer, access) => {
                    buffers.push((buffer, create_access(access)));
                }
                BoundResource::Image(image, access) => {
                    images.push((image, create_access(access)));
                }
            });
        self.access_resources(&images, &buffers);

        let shader = self.shader(&self.bound_shader().shader);
        let offset = self.descriptor_buffer.bound_range.start as vk::DeviceSize;
        unsafe {
            self.device
                .descriptor_buffer
                .cmd_set_descriptor_buffer_offsets(
                    *self.command_buffer,
                    vk::PipelineBindPoint::COMPUTE,
                    shader.layout,
                    0,
                    &[0],
                    &[offset],
                );
        }
        let width = width.div_ceil(shader.block_size.width);
        let height = height.div_ceil(shader.block_size.height);
        unsafe {
            self.device
                .cmd_dispatch(*self.command_buffer, width, height, 1);
        }
    }
}
