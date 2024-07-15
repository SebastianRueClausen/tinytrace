use ash::vk;

use crate::backend::{
    Binding, BindingType, Buffer, Handle, Image, Lifetime, Shader, ShaderRequest,
};
use crate::binding;
use crate::error::Result;
use crate::scene::Scene;
use crate::{asset, Context};

pub struct Integrator {
    pub integrate: Handle<Shader>,
}

impl Integrator {
    pub fn new(context: &mut Context, scene: &asset::Scene) -> Result<Self> {
        let bindings = &[
            binding!(uniform_buffer, Constants, constants),
            binding!(storage_buffer, Vertex, vertices, true, false),
            binding!(storage_buffer, float, colors, true, false),
            binding!(storage_buffer, uint, indices, true, false),
            binding!(storage_buffer, Material, materials, true, false),
            binding!(storage_buffer, Instance, instances, true, false),
            binding!(acceleration_structure, acceleration_structure),
            binding!(
                storage_image,
                super::RENDER_TARGET_FORMAT,
                target,
                None,
                true
            ),
            binding!(sampled_image, textures, Some(scene.textures.len() as u32)),
        ];

        let integrate = context.create_shader(
            Lifetime::Scene,
            &ShaderRequest {
                block_size: vk::Extent2D::default().width(32).height(32),
                source: include_str!("integrate.glsl"),
                includes: &["scene", "brdf", "math", "sample"],
                bindings,
            },
        )?;

        Ok(Self { integrate })
    }

    pub fn integrate(
        &self,
        context: &mut Context,
        constants: &Handle<Buffer>,
        scene: &Scene,
        target: &Handle<Image>,
    ) {
        context
            .bind_shader(&self.integrate)
            .bind_buffer("constants", constants)
            .bind_buffer("vertices", &scene.vertices)
            .bind_buffer("colors", &scene.colors)
            .bind_buffer("indices", &scene.indices)
            .bind_buffer("materials", &scene.materials)
            .bind_buffer("instances", &scene.instances)
            .bind_sampled_images("textures", &scene.texture_sampler, &scene.textures)
            .bind_acceleration_structure("acceleration_structure", &scene.tlas)
            .bind_storage_image("target", target);
        let vk::Extent3D { width, height, .. } = context.image(target).extent;
        context.dispatch(width, height).unwrap();
    }
}
