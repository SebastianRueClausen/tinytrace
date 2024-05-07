use std::{fs, slice};

use ash::vk;

use super::device::Device;
use super::{descriptor::DescriptorLayout, Context, Handle};
use crate::{backend::Lifetime, error::Error};

const SHADER_PRELUDE: &str = r#"
#version 460
#extension GL_GOOGLE_include_directive: require
#extension GL_EXT_nonuniform_qualifier: require
"#;

#[derive(Debug)]
pub struct Shader {
    pub pipeline: vk::Pipeline,
    pub layout: vk::PipelineLayout,
    pub descriptor_layouts: Vec<Handle<DescriptorLayout>>,
    pub grid_size: vk::Extent2D,
}

impl Shader {
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
        source: &str,
        vk::Extent2D { width, height }: vk::Extent2D,
        descriptor_layouts: &[Handle<DescriptorLayout>],
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
        let descriptors: String = descriptor_layouts
            .iter()
            .enumerate()
            .map(|(set, layout)| self.descriptor_layout(layout).to_glsl(set as u32))
            .collect();
        let grid_size = format!("layout (local_size_x = {width}, local_size_y = {height}) in;\n");
        let source = format!("{SHADER_PRELUDE}{grid_size}{descriptors}{}", source);
        // Compile shader.
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

    fn create_pipeline_layout(
        &self,
        layouts: &[Handle<DescriptorLayout>],
    ) -> Result<vk::PipelineLayout, Error> {
        let set_layouts: Vec<vk::DescriptorSetLayout> = layouts
            .iter()
            .map(|layout| **self.descriptor_layout(layout))
            .collect();
        let layout_info = vk::PipelineLayoutCreateInfo::default().set_layouts(&set_layouts);
        let pipeline_layout = unsafe { self.device.create_pipeline_layout(&layout_info, None)? };
        Ok(pipeline_layout)
    }

    pub fn create_shader(
        &mut self,
        source: &str,
        grid_size: vk::Extent2D,
        layouts: &[Handle<DescriptorLayout>],
    ) -> Result<Handle<Shader>, Error> {
        let module = self.create_shader_module(source, grid_size, layouts)?;
        let pipeline_layout = self.create_pipeline_layout(layouts)?;
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
            descriptor_layouts: layouts.to_vec(),
            layout: pipeline_layout,
            grid_size,
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
