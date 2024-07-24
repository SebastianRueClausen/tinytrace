use ash::vk;

use crate::backend::{
    Binding, BindingType, Buffer, Context, Handle, Image, Lifetime, Shader, ShaderRequest,
};
use crate::binding;
use crate::error::Result;

pub struct PostProcess {
    display: Handle<Shader>,
}

impl PostProcess {
    pub fn new(context: &mut Context) -> Result<Self> {
        context.add_include("tonemap", include_str!("tonemap.glsl").to_string());
        let bindings = &[
            binding!(uniform_buffer, Constants, constants),
            binding!(
                storage_image,
                super::RENDER_TARGET_FORMAT,
                render_target,
                None,
                false
            ),
            binding!(
                storage_image,
                context.surface_format(),
                swapchain,
                None,
                true
            ),
        ];
        let display = context.create_shader(
            Lifetime::Static,
            &ShaderRequest {
                block_size: vk::Extent2D::default().width(32).height(32),
                source: include_str!("display.glsl"),
                includes: &["tonemap", "scene"],
                bindings,
            },
        )?;
        Ok(Self { display })
    }

    pub fn run(
        &self,
        context: &mut Context,
        constants: &Handle<Buffer>,
        render_target: &Handle<Image>,
        swapchain: &Handle<Image>,
    ) -> Result<()> {
        context
            .bind_shader(&self.display)
            .bind_buffer("constants", constants)
            .bind_storage_image("render_target", render_target)
            .bind_storage_image("swapchain", swapchain);
        let vk::Extent3D { width, height, .. } = context.image(swapchain).extent;
        context.dispatch(width, height)?;
        Ok(())
    }
}
