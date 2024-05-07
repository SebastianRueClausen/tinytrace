use std::{fs, slice};

use ash::vk;

use super::device::Device;
use super::{Context, Handle};
use crate::{backend::Lifetime, error::Error};

const SHADER_PRELUDE: &str = r#"
#version 460
#extension GL_GOOGLE_include_directive: require
#extension GL_EXT_nonuniform_qualifier: require
"#;

#[derive(Debug, Clone)]
pub enum BindingType {
    StorageBuffer {
        ty: &'static str,
        count: u32,
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
    fn to_glsl(&self, name: &str, set: u32, index: u32) -> String {
        let classifier = |writes| if writes { "" } else { "readonly " };
        let brackets = |count| if count > 1 { "[]" } else { "" };
        match self {
            Self::StorageBuffer { ty, count, writes } => {
                let (brackets, classifier) = (brackets(*count), classifier(*writes));
                format!("{classifier}buffer Set{set}Binding{index} {{ {ty} {name}{brackets}; }};")
            }
            Self::UniformBuffer { ty } => {
                format!("uniform Set{set}Index{index} {{ {ty} {name}; }};")
            }
            Self::AccelerationStructure => {
                format!("uniform accelerationStructureEXT {name};")
            }
            Self::SampledImage { count } => {
                format!("uniform sampled2D {name}{};", brackets(*count))
            }
            Self::StorageImage { count, writes } => {
                let (brackets, classifier) = (brackets(*count), classifier(*writes));
                format!("{classifier} uniform image2D {name}{brackets};")
            }
        }
    }

    fn descriptor_type(&self) -> vk::DescriptorType {
        match self {
            Self::StorageBuffer { .. } => vk::DescriptorType::STORAGE_BUFFER,
            Self::UniformBuffer { .. } => vk::DescriptorType::UNIFORM_BUFFER,
            Self::AccelerationStructure => vk::DescriptorType::ACCELERATION_STRUCTURE_KHR,
            Self::SampledImage { .. } => vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            Self::StorageImage { .. } => vk::DescriptorType::STORAGE_IMAGE,
        }
    }

    #[allow(dead_code)]
    fn descriptor_size(&self, device: &Device) -> usize {
        let props = &device.descriptor_buffer_properties;
        match self {
            Self::StorageBuffer { .. } => props.storage_buffer_descriptor_size,
            Self::UniformBuffer { .. } => props.uniform_buffer_descriptor_size,
            Self::AccelerationStructure => props.acceleration_structure_descriptor_size,
            Self::SampledImage { .. } => props.sampled_image_descriptor_size,
            Self::StorageImage { .. } => props.storage_image_descriptor_size,
        }
    }

    fn count(&self) -> u32 {
        match self {
            Self::StorageBuffer { count, .. }
            | Self::SampledImage { count }
            | Self::StorageImage { count, .. } => *count,
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
) -> Result<vk::DescriptorSetLayout, Error> {
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

fn bindings_to_glsl(set: u32, bindings: &[Binding]) -> String {
    let to_glsl = |(index, binding): (usize, &Binding)| {
        let ty = binding.ty.to_glsl(binding.name, set, index as u32);
        format!("layout (set = {set}, binding = {index}) {ty}\n")
    };
    bindings.iter().enumerate().map(to_glsl).collect()
}

fn create_shader_module(
    device: &Device,
    source: &str,
    vk::Extent2D { width, height }: vk::Extent2D,
    bindings: &[Binding],
) -> Result<vk::ShaderModule, Error> {
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

    // Compose shader source.
    let grid_size = format!("layout (local_size_x = {width}, local_size_y = {height}) in;\n");
    let bindings = bindings_to_glsl(0, bindings);
    let source = format!("{SHADER_PRELUDE}{grid_size}{bindings}{source}");

    // Compile shader.
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

#[derive(Debug)]
pub struct Shader {
    pub pipeline: vk::Pipeline,
    pub layout: vk::PipelineLayout,
    pub descriptor_layout: vk::DescriptorSetLayout,
    pub bindings: Vec<Binding>,
    pub grid_size: vk::Extent2D,
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

impl Context {
    fn create_pipeline_layout(
        &self,
        descriptor_layout: vk::DescriptorSetLayout,
    ) -> Result<vk::PipelineLayout, Error> {
        let layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(slice::from_ref(&descriptor_layout));
        let pipeline_layout = unsafe { self.device.create_pipeline_layout(&layout_info, None)? };
        Ok(pipeline_layout)
    }

    pub fn create_shader(
        &mut self,
        source: &str,
        grid_size: vk::Extent2D,
        bindings: &[Binding],
    ) -> Result<Handle<Shader>, Error> {
        let module = create_shader_module(&self.device, source, grid_size, bindings)?;
        let descriptor_layout = create_descriptor_layout(&self.device, bindings)?;
        let pipeline_layout = self.create_pipeline_layout(descriptor_layout)?;
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
            bindings: bindings.to_vec(),
            layout: pipeline_layout,
            grid_size,
            descriptor_layout,
            pipeline,
        };
        let pool = self.pool_mut(Lifetime::Static);
        Ok(Handle::new(
            Lifetime::Static,
            pool.epoch,
            &mut pool.pipelines,
            pipeline,
        ))
    }
}
