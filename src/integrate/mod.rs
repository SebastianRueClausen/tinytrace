mod restir;

use crate::backend::{
    Binding, BindingType, Buffer, Handle, Image, Lifetime, Shader, ShaderRequest,
};
use crate::error::Result;
use crate::scene::Scene;
use crate::Context;
use crate::{binding, RestirConfig};
use ash::vk;
use restir::RestirState;

pub struct Integrator {
    pub integrate: Handle<Shader>,
    pub(super) restir_state: RestirState,
}

impl Integrator {
    pub fn new(context: &mut Context, restir_config: &RestirConfig) -> Result<Self> {
        let bindings = &[
            binding!(uniform_buffer, Constants, constants),
            binding!(storage_buffer, Vertex, vertices, true, false),
            binding!(storage_buffer, uint, indices, true, false),
            binding!(storage_buffer, Material, materials, true, false),
            binding!(storage_buffer, Instance, instances, true, false),
            binding!(storage_buffer, uint64_t, reservoir_keys, true, true),
            binding!(storage_buffer, uint64_t, reservoir_update_keys, true, true),
            binding!(storage_buffer, Reservoir, reservoirs, true, true),
            binding!(storage_buffer, Reservoir, reservoir_updates, true, true),
            binding!(storage_buffer, uint, reservoir_update_counts, true, true),
            binding!(storage_buffer, uint, reservoir_sample_counts, true, true),
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
                    block_size: vk::Extent2D::default().width(32).height(32),
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
    ) -> Result<()> {
        self.restir_state.clear_updates(context)?;
        context
            .bind_shader(&self.integrate)
            .bind_buffer("constants", constants)
            .bind_buffer("vertices", &scene.vertices)
            .bind_buffer("indices", &scene.indices)
            .bind_buffer("materials", &scene.materials)
            .bind_buffer("instances", &scene.instances)
            .bind_buffer(
                "reservoir_keys",
                &self.restir_state.reservoir_hash_grid.keys,
            )
            .bind_buffer(
                "reservoir_update_keys",
                &self.restir_state.update_hash_grid.keys,
            )
            .bind_buffer("reservoirs", &self.restir_state.reservoirs)
            .bind_buffer("reservoir_updates", &self.restir_state.updates)
            .bind_buffer("reservoir_update_counts", &self.restir_state.update_counts)
            .bind_buffer(
                "reservoir_sample_counts",
                &self.restir_state.reservoir_sample_counts,
            )
            .bind_sampled_images("textures", &scene.texture_sampler, &scene.textures)
            .bind_acceleration_structure("acceleration_structure", &scene.tlas)
            .bind_storage_image("target", target);
        let vk::Extent3D { width, height, .. } = context.image(target).extent;
        context.dispatch(width, height)?;
        self.restir_state.update_reservoirs(context, constants)
    }
}
