use std::{fs, slice};

use ash::vk;
use std::ffi::CString;

use super::device::Device;
use super::{descriptor::DescriptorLayout, Context, Handle};
use crate::{backend::Lifetime, error::Error};

#[derive(Debug)]
pub struct Shader<'a> {
    pub source: &'a str,
    pub stage: vk::ShaderStageFlags,
}

const SHADER_PRELUDE: &str = r#"
#version 460
#extension GL_GOOGLE_include_directive: require
#extension GL_EXT_nonuniform_qualifier: require
"#;

fn compose_shader(source: &str, layouts: impl Iterator<Item = String>) -> String {
    let descriptors: String = layouts.collect();
    format!("{SHADER_PRELUDE}{descriptors}{source}")
}

fn create_shader_info<'a>(
    shader: &Shader,
    module: vk::ShaderModule,
    entry_point: &'a CString,
) -> vk::PipelineShaderStageCreateInfo<'a> {
    vk::PipelineShaderStageCreateInfo::default()
        .stage(shader.stage)
        .module(module)
        .name(entry_point)
}

#[derive(Debug)]
pub struct PipelineLayout<'a> {
    pub descriptor_layouts: &'a [Handle<DescriptorLayout>],
    pub push_constant: Option<vk::PushConstantRange>,
}

#[derive(Debug)]
pub struct Pipeline {
    pub pipeline: vk::Pipeline,
    pub layout: vk::PipelineLayout,
    pub descriptor_layouts: Vec<Handle<DescriptorLayout>>,
}

impl Pipeline {
    pub fn destroy(&self, device: &Device) {
        unsafe {
            device.destroy_pipeline(self.pipeline, None);
            device.destroy_pipeline_layout(self.layout, None);
        }
    }
}

impl Context {
    fn create_shader_module(
        &self,
        shader: &Shader,
        descriptor_layouts: &[Handle<DescriptorLayout>],
    ) -> Result<vk::ShaderModule, Error> {
        let compiler = shaderc::Compiler::new().unwrap();
        let shader_kind = match shader.stage {
            vk::ShaderStageFlags::COMPUTE => shaderc::ShaderKind::Compute,
            vk::ShaderStageFlags::RAYGEN_KHR => shaderc::ShaderKind::RayGeneration,
            vk::ShaderStageFlags::ANY_HIT_KHR => shaderc::ShaderKind::AnyHit,
            vk::ShaderStageFlags::CLOSEST_HIT_KHR => shaderc::ShaderKind::ClosestHit,
            _ => panic!("invalid shader stage: {:?}", shader.stage),
        };
        let mut options = shaderc::CompileOptions::new().unwrap();
        options.set_include_callback(|name, _, _, _| {
            fs::read_to_string(name)
                .map(|content| shaderc::ResolvedInclude {
                    resolved_name: name.into(),
                    content,
                })
                .map_err(|err| format!("failed to import file {name}: {err}"))
        });
        let descriptors = descriptor_layouts
            .iter()
            .enumerate()
            .map(|(set, layout)| self.descriptor_layout(layout).to_glsl(set as u32));
        let source = compose_shader(shader.source, descriptors);
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

    fn create_pipeline_layout(
        &self,
        layout: &PipelineLayout,
    ) -> Result<(vk::PipelineLayout, vk::ShaderStageFlags), Error> {
        let set_layouts: Vec<vk::DescriptorSetLayout> = layout
            .descriptor_layouts
            .iter()
            .map(|layout| **self.descriptor_layout(layout))
            .collect();
        let mut layout_info = vk::PipelineLayoutCreateInfo::default().set_layouts(&set_layouts);
        let push_constant_stages = layout
            .push_constant
            .as_ref()
            .map(|push_constant| {
                layout_info = layout_info.push_constant_ranges(slice::from_ref(push_constant));
                push_constant.stage_flags
            })
            .unwrap_or_default();
        let pipeline_layout = unsafe { self.device.create_pipeline_layout(&layout_info, None)? };
        Ok((pipeline_layout, push_constant_stages))
    }

    pub fn create_compute_pipeline(
        &mut self,
        shader: &Shader,
        layout: &PipelineLayout,
    ) -> Result<Handle<Pipeline>, Error> {
        let module = self.create_shader_module(shader, layout.descriptor_layouts)?;
        let (pipeline_layout, _) = self.create_pipeline_layout(layout)?;
        let entry_point = CString::new("main").unwrap();
        let pipeline_info = vk::ComputePipelineCreateInfo::default()
            .flags(vk::PipelineCreateFlags::DESCRIPTOR_BUFFER_EXT)
            .stage(create_shader_info(shader, module, &entry_point))
            .layout(pipeline_layout);
        let pipeline = unsafe {
            *self
                .device
                .create_compute_pipelines(
                    vk::PipelineCache::null(),
                    slice::from_ref(&pipeline_info),
                    None,
                )
                .map_err(|(_, err)| err)?
                .first()
                .unwrap()
        };
        unsafe {
            self.device.destroy_shader_module(module, None);
        }
        let pipeline = Pipeline {
            descriptor_layouts: layout.descriptor_layouts.to_vec(),
            layout: pipeline_layout,
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
