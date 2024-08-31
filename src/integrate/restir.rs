use std::mem;

use ash::vk;
use glam::Vec3;
use half::f16;

use crate::{
    backend::{
        self, Binding, BindingType, Buffer, BufferRequest, BufferType, Context, Handle, Lifetime,
        MemoryLocation, Shader, ShaderRequest,
    },
    binding, error,
    hash_grid::{HashGrid, HashGridLayout},
    RestirConfig,
};

#[repr(C)]
#[derive(bytemuck::Pod, bytemuck::Zeroable, Clone, Copy, Debug)]
pub(super) struct BounceSurface {
    position: Vec3,
    normal: [f16; 2],
}

#[repr(C)]
#[derive(bytemuck::Pod, bytemuck::Zeroable, Clone, Copy, Debug)]
pub(super) struct Path {
    origin: BounceSurface,
    destination: BounceSurface,
    radiance: Vec3,
    generator: u32,
}

#[repr(C)]
#[derive(bytemuck::Zeroable, bytemuck::Pod, Clone, Copy, Debug)]
pub(super) struct Reservoir {
    path: Path,
    weight_sum: f32,
    weight: f32,
    sample_count: u32,
}

pub struct RestirState {
    pub reservoir_hash_grid: HashGrid,
    pub update_hash_grid: HashGrid,
    pub reservoirs: Handle<Buffer>,
    pub updates: Handle<Buffer>,
    pub update_counts: Handle<Buffer>,
    pub reservoir_sample_counts: Handle<Buffer>,
    pub update_reservoirs: Handle<Shader>,
}

fn create_buffer(
    context: &mut Context,
    count: u32,
    value_size: usize,
) -> Result<Handle<Buffer>, backend::Error> {
    context.create_buffer(
        Lifetime::Renderer,
        &BufferRequest {
            size: count as vk::DeviceSize * value_size as vk::DeviceSize,
            ty: BufferType::Storage,
            memory_location: MemoryLocation::Device,
        },
    )
}

impl RestirState {
    pub fn new(context: &mut Context, config: &RestirConfig) -> error::Result<Self> {
        let bindings = &[
            binding!(uniform_buffer, Constants, constants),
            binding!(storage_buffer, uint64_t, reservoir_keys, true, true),
            binding!(storage_buffer, uint64_t, reservoir_update_keys, true, true),
            binding!(storage_buffer, uint, reservoir_update_counts, true, true),
            binding!(storage_buffer, Reservoir, reservoirs, true, true),
            binding!(storage_buffer, Reservoir, reservoir_updates, true, true),
        ];
        let update_reservoirs = context.create_shader(
            Lifetime::Renderer,
            &ShaderRequest {
                block_size: vk::Extent2D::default().width(256).height(1),
                source: include_str!("update_reservoirs.glsl"),
                push_constant_size: None,
                bindings,
            },
        )?;
        Ok(Self {
            update_reservoirs,
            reservoir_hash_grid: HashGrid::new(
                context,
                HashGridLayout::new(config.reservoir_hash_grid_capacity, 32, config.scene_scale),
            )?,
            update_hash_grid: HashGrid::new(
                context,
                HashGridLayout::new(config.update_hash_grid_capacity, 1, config.scene_scale),
            )?,
            reservoirs: create_buffer(
                context,
                config.reservoir_hash_grid_capacity * config.reservoirs_per_cell,
                mem::size_of::<Reservoir>(),
            )?,
            updates: create_buffer(
                context,
                config.update_hash_grid_capacity * config.updates_per_cell,
                mem::size_of::<Reservoir>(),
            )?,
            update_counts: create_buffer(
                context,
                config.update_hash_grid_capacity,
                mem::size_of::<u32>(),
            )?,
            reservoir_sample_counts: create_buffer(
                context,
                config.reservoir_hash_grid_capacity,
                mem::size_of::<u32>(),
            )?,
        })
    }

    pub fn update_reservoirs(
        &self,
        context: &mut Context,
        constants: &Handle<Buffer>,
    ) -> error::Result<()> {
        context
            .bind_shader(&self.update_reservoirs)
            .bind_buffer("constants", constants)
            .bind_buffer("reservoir_keys", &self.reservoir_hash_grid.keys)
            .bind_buffer("reservoir_update_keys", &self.update_hash_grid.keys)
            .bind_buffer("reservoir_update_counts", &self.update_counts)
            .bind_buffer("reservoirs", &self.reservoirs)
            .bind_buffer("reservoir_updates", &self.updates)
            .dispatch(self.update_hash_grid.layout.capacity, 1)?;
        Ok(())
    }

    pub fn clear_updates(&self, context: &mut Context) -> error::Result<()> {
        context.fill_buffer(&self.update_counts, 0)?;
        self.update_hash_grid.clear(context)
    }
}
