mod restir;

use std::mem;

use ash::vk;

use crate::backend::{
    Binding, BindingType, Buffer, BufferRequest, BufferType, Handle, Image, Lifetime,
    MemoryLocation, Shader, ShaderRequest,
};
use crate::binding;
use crate::error::Result;
use crate::scene::Scene;
use crate::{asset, Context};

pub(super) struct HashGrid {
    pub hashes: Handle<Buffer>,
    pub values: Handle<Buffer>,
    pub layout: HashGridLayout,
}

impl HashGrid {
    pub(super) fn new(
        context: &mut Context,
        value_size: usize,
        layout: HashGridLayout,
    ) -> Result<Self> {
        let hashes = context.create_buffer(
            Lifetime::Scene,
            &BufferRequest {
                size: (mem::size_of::<u64>() * layout.capacity as usize) as vk::DeviceSize,
                ty: BufferType::Storage,
                memory_location: MemoryLocation::Device,
            },
        )?;
        let values = context.create_buffer(
            Lifetime::Scene,
            &BufferRequest {
                size: (value_size * layout.capacity as usize) as vk::DeviceSize,
                ty: BufferType::Storage,
                memory_location: MemoryLocation::Device,
            },
        )?;
        let hash_grid = Self {
            hashes,
            values,
            layout,
        };
        hash_grid.clear(context)?;
        Ok(hash_grid)
    }

    fn clear(&self, context: &mut Context) -> Result<()> {
        context.fill_buffer(&self.hashes, u32::MAX)?;
        context.fill_buffer(&self.values, 0)?;
        Ok(())
    }
}

#[repr(C)]
#[derive(Default, Debug, Clone, Copy, bytemuck::NoUninit, bytemuck::AnyBitPattern)]
pub struct HashGridLayout {
    scene_scale: f32,
    capacity: u32,
    bucket_size: u32,
    padding: u32,
}

pub struct Integrator {
    pub integrate: Handle<Shader>,
    pub update_reservoirs: Handle<Shader>,
    pub(super) reservoir_pools: HashGrid,
    pub(super) reservoir_updates: HashGrid,
}

impl Integrator {
    pub fn new(context: &mut Context, scene: &asset::Scene) -> Result<Self> {
        let bindings = &[
            binding!(uniform_buffer, Constants, constants),
            binding!(storage_buffer, Vertex, vertices, true, false),
            binding!(storage_buffer, uint, indices, true, false),
            binding!(storage_buffer, Material, materials, true, false),
            binding!(storage_buffer, Instance, instances, true, false),
            binding!(storage_buffer, uint64_t, reservoir_pool_hashes, true, true),
            binding!(
                storage_buffer,
                uint64_t,
                reservoir_update_hashes,
                true,
                true
            ),
            binding!(storage_buffer, ReservoirPool, reservoir_pools, true, true),
            binding!(
                storage_buffer,
                ReservoirUpdate,
                reservoir_updates,
                true,
                true
            ),
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
                push_constant_size: None,
                includes: &[
                    "scene",
                    "brdf",
                    "math",
                    "sample",
                    "debug",
                    "restir",
                    "constants",
                ],
                bindings,
            },
        )?;
        let bindings = &[
            binding!(uniform_buffer, Constants, constants),
            binding!(storage_buffer, uint64_t, reservoir_pool_hashes, true, true),
            binding!(
                storage_buffer,
                uint64_t,
                reservoir_update_hashes,
                true,
                true
            ),
            binding!(storage_buffer, ReservoirPool, reservoir_pools, true, true),
            binding!(
                storage_buffer,
                ReservoirUpdate,
                reservoir_updates,
                true,
                true
            ),
        ];
        let update_reservoirs = context.create_shader(
            Lifetime::Scene,
            &ShaderRequest {
                block_size: vk::Extent2D::default().width(256).height(1),
                source: include_str!("update_reservoirs.glsl"),
                includes: &["random", "restir", "math", "constants"],
                push_constant_size: None,
                bindings,
            },
        )?;
        let hash_grid_layout = |bucket_size, capacity| HashGridLayout {
            padding: 0,
            scene_scale: 20.0,
            bucket_size,
            capacity,
        };
        Ok(Self {
            reservoir_pools: HashGrid::new(
                context,
                mem::size_of::<restir::ReservoirPool>(),
                hash_grid_layout(64, 0xfffff),
            )?,
            reservoir_updates: HashGrid::new(
                context,
                mem::size_of::<restir::ReservoirUpdate>(),
                hash_grid_layout(1, 0xffff),
            )?,
            update_reservoirs,
            integrate,
        })
    }

    pub fn integrate(
        &self,
        context: &mut Context,
        constants: &Handle<Buffer>,
        scene: &Scene,
        target: &Handle<Image>,
    ) -> Result<()> {
        self.reservoir_updates.clear(context)?;
        context
            .bind_shader(&self.integrate)
            .bind_buffer("constants", constants)
            .bind_buffer("vertices", &scene.vertices)
            .bind_buffer("indices", &scene.indices)
            .bind_buffer("materials", &scene.materials)
            .bind_buffer("instances", &scene.instances)
            .bind_buffer("reservoir_pool_hashes", &self.reservoir_pools.hashes)
            .bind_buffer("reservoir_update_hashes", &self.reservoir_updates.hashes)
            .bind_buffer("reservoir_pools", &self.reservoir_pools.values)
            .bind_buffer("reservoir_updates", &self.reservoir_updates.values)
            .bind_sampled_images("textures", &scene.texture_sampler, &scene.textures)
            .bind_acceleration_structure("acceleration_structure", &scene.tlas)
            .bind_storage_image("target", target);
        let vk::Extent3D { width, height, .. } = context.image(target).extent;
        context.dispatch(width, height)?;
        context
            .bind_shader(&self.update_reservoirs)
            .bind_buffer("constants", constants)
            .bind_buffer("reservoir_pool_hashes", &self.reservoir_pools.hashes)
            .bind_buffer("reservoir_update_hashes", &self.reservoir_updates.hashes)
            .bind_buffer("reservoir_pools", &self.reservoir_pools.values)
            .bind_buffer("reservoir_updates", &self.reservoir_updates.values)
            .dispatch(self.reservoir_updates.layout.capacity, 1)?;
        Ok(())
    }
}
