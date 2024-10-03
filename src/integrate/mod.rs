mod restir;

use crate::scene::Scene;
use crate::{Config, Context, Error, RestirConfig};
use restir::RestirState;
use tinytrace_backend::{
    binding, Binding, BindingType, Buffer, Extent, Handle, Image, Lifetime, Shader, ShaderRequest,
};

pub struct Integrator {
    pub integrate: Handle<Shader>,
    pub(super) restir_state: RestirState,
}

impl Integrator {
    pub fn new(context: &mut Context, restir_config: &RestirConfig) -> Result<Self, Error> {
        let bindings = &[
            binding!(uniform_buffer, Constants, constants),
            binding!(uniform_buffer, Scene, scene),
            binding!(uniform_buffer, HashGrid, reservoir_hash_grid),
            binding!(uniform_buffer, HashGrid, reservoir_update_hash_grid),
            binding!(uniform_buffer, RestirData, restir_data),
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
            restir_state: RestirState::new(context, restir_config)?,
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
        config: &Config,
        constants: &Handle<Buffer>,
        scene: &Scene,
        target: &Handle<Image>,
    ) -> Result<(), Error> {
        self.restir_state.prepare(context, &config.restir)?;
        context
            .bind_shader(&self.integrate)
            .bind_buffer("constants", constants)
            .bind_buffer("scene", &scene.scene_data)
            .register_indirect_buffer("vertices", &scene.vertices, false)
            .register_indirect_buffer("indices", &scene.indices, false)
            .register_indirect_buffer("materials", &scene.materials, false)
            .register_indirect_buffer("instances", &scene.instances, false)
            .register_indirect_buffer("emissive_triangles", &scene.emissive_triangles, false)
            .bind_buffer(
                "reservoir_hash_grid",
                &self.restir_state.reservoir_hash_grid.data,
            )
            .register_indirect_buffer(
                "reservoir_hash_grid",
                &self.restir_state.reservoir_hash_grid.keys,
                false,
            )
            .bind_buffer(
                "reservoir_update_hash_grid",
                &self.restir_state.update_hash_grid.data,
            )
            .register_indirect_buffer(
                "reservoir_update_hash_grid",
                &self.restir_state.update_hash_grid.keys,
                true,
            )
            .bind_buffer("restir_data", &self.restir_state.data)
            .register_indirect_buffer("reservoir_updates", &self.restir_state.updates, true)
            .register_indirect_buffer(
                "reservoir_update_counts",
                &self.restir_state.update_counts,
                true,
            )
            .register_indirect_buffer(
                "reservoir_sample_counts",
                &self.restir_state.sample_counts,
                true,
            )
            .register_indirect_buffer("reservoirs", &self.restir_state.reservoirs, false)
            .bind_sampled_images("textures", &scene.texture_sampler, &scene.textures)
            .bind_acceleration_structure("acceleration_structure", &scene.tlas)
            .bind_storage_image("target", target);
        let Extent { width, height } = context.image(target).extent();
        context.dispatch(width, height)?;
        self.restir_state.update_reservoirs(context, constants)
    }
}
