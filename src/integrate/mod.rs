use crate::scene::Scene;
use crate::{Context, Error};
use tinytrace_backend::{
    binding, Binding, BindingType, Buffer, Extent, Handle, Image, Lifetime, Shader, ShaderRequest,
};

pub struct Integrator {
    pub integrate: Handle<Shader>,
}

impl Integrator {
    pub fn new(context: &mut Context) -> Result<Self, Error> {
        let bindings = &[
            binding!(uniform_buffer, Constants, constants),
            binding!(uniform_buffer, Scene, scene),
            binding!(acceleration_structure, acceleration_structure),
            binding!(
                storage_image,
                super::RENDER_TARGET_FORMAT,
                target,
                None,
                true
            ),
            binding!(sampled_image, textures, Some(1024)),
        ];
        Ok(Self {
            integrate: context.create_shader(
                Lifetime::Renderer,
                &ShaderRequest {
                    block_size: Extent::new(32, 32),
                    source: include_str!("integrate.glsl"),
                    push_constant_size: None,
                    bindings,
                },
            )?,
        })
    }

    pub fn integrate(
        &self,
        context: &mut Context,
        constants: &Handle<Buffer>,
        scene: &Scene,
        target: &Handle<Image>,
    ) -> Result<(), Error> {
        context
            .bind_shader(&self.integrate)
            .bind_buffer("constants", constants)
            .bind_buffer("scene", &scene.scene_data)
            .register_indirect_buffer("vertices", &scene.vertices, false)
            .register_indirect_buffer("indices", &scene.indices, false)
            .register_indirect_buffer("materials", &scene.materials, false)
            .register_indirect_buffer("instances", &scene.instances, false)
            .register_indirect_buffer("emissive_triangles", &scene.emissive_triangles, false)
            .bind_sampled_images("textures", &scene.texture_sampler, &scene.textures)
            .bind_acceleration_structure("acceleration_structure", &scene.tlas)
            .bind_storage_image("target", target);
        let Extent { width, height } = context.image(target).extent();
        context.dispatch(width, height)?;
        Ok(())
    }
}
