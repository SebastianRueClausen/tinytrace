use std::mem;

use glam::Vec3;
use half::f16;

use crate::hash_grid::HashGrid;
use crate::{Error, RestirConfig, RestirReplay};
use tinytrace_backend::{
    binding, Binding, BindingType, Buffer, BufferRequest, BufferType, BufferWrite, Context, Extent,
    Handle, Lifetime, MemoryLocation, Shader, ShaderRequest,
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
    radiance: [f16; 3],
    bounce_count: u16,
    generator: u32,
}

#[repr(C)]
#[derive(bytemuck::Zeroable, bytemuck::Pod, Clone, Copy, Debug)]
pub(super) struct Reservoir {
    path: Path,
    weight_sum: f32,
    weight: f32,
    sample_count: u16,
    padding: u16,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::NoUninit)]
struct RestirConstants {
    scene_scale: f32,
    updates_per_cell: u32,
    reservoirs_per_cell: u32,
    replay: RestirReplay,
}

impl RestirConstants {
    fn new(config: &RestirConfig) -> Self {
        Self {
            scene_scale: config.scene_scale,
            updates_per_cell: config.updates_per_cell,
            reservoirs_per_cell: config.reservoirs_per_cell,
            replay: config.replay,
        }
    }
}

pub struct RestirState {
    pub reservoir_hash_grid: HashGrid,
    pub update_hash_grid: HashGrid,
    pub reservoirs: Handle<Buffer>,
    pub updates: Handle<Buffer>,
    pub update_counts: Handle<Buffer>,
    pub reservoir_sample_counts: Handle<Buffer>,
    pub constants: Handle<Buffer>,
    pub update_reservoirs: Handle<Shader>,
}

fn create_buffer(
    context: &mut Context,
    count: u32,
    value_size: usize,
) -> Result<Handle<Buffer>, tinytrace_backend::Error> {
    context.create_buffer(
        Lifetime::Renderer,
        &BufferRequest {
            size: count as u64 * value_size as u64,
            ty: BufferType::Storage,
            memory_location: MemoryLocation::Device,
        },
    )
}

impl RestirState {
    pub fn new(context: &mut Context, config: &RestirConfig) -> Result<Self, Error> {
        let bindings = &[
            binding!(uniform_buffer, Constants, constants),
            binding!(uniform_buffer, RestirConstants, restir_constants),
            binding!(uniform_buffer, HashGrid, reservoir_hash_grid),
            binding!(uniform_buffer, HashGrid, reservoir_update_hash_grid),
            binding!(storage_buffer, uint, reservoir_update_counts, true, true),
            binding!(storage_buffer, Reservoir, reservoirs, true, true),
            binding!(storage_buffer, Reservoir, reservoir_updates, true, true),
        ];
        let update_reservoirs = context.create_shader(
            Lifetime::Renderer,
            &ShaderRequest {
                block_size: Extent::new(256, 1),
                source: include_str!("update_reservoirs.glsl"),
                push_constant_size: None,
                bindings,
            },
        )?;
        Ok(Self {
            update_reservoirs,
            reservoir_hash_grid: HashGrid::new(
                context,
                config.scene_scale,
                config.reservoir_hash_grid_capacity,
                32,
            )?,
            update_hash_grid: HashGrid::new(
                context,
                config.scene_scale,
                config.reservoir_hash_grid_capacity,
                32,
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
            constants: create_buffer(context, 1, mem::size_of::<RestirConstants>())?,
        })
    }

    pub fn update_reservoirs(
        &self,
        context: &mut Context,
        constants: &Handle<Buffer>,
    ) -> Result<(), Error> {
        context
            .bind_shader(&self.update_reservoirs)
            .bind_buffer("constants", constants)
            .bind_buffer("restir_constants", &self.constants)
            .bind_buffer("reservoir_hash_grid", &self.reservoir_hash_grid.data)
            .register_indirect_buffer("reservoir_hash_grid", &self.reservoir_hash_grid.keys, false)
            .bind_buffer("reservoir_update_hash_grid", &self.update_hash_grid.data)
            .register_indirect_buffer(
                "reservoir_update_hash_grid",
                &self.update_hash_grid.keys,
                false,
            )
            .bind_buffer("reservoir_update_counts", &self.update_counts)
            .bind_buffer("reservoirs", &self.reservoirs)
            .bind_buffer("reservoir_updates", &self.updates)
            .dispatch(self.update_hash_grid.capacity, 1)?;
        Ok(())
    }

    pub fn prepare(&self, context: &mut Context, config: &RestirConfig) -> Result<(), Error> {
        context
            .fill_buffer(&self.update_counts, 0)?
            .write_buffers(&[BufferWrite {
                buffer: self.constants.clone(),
                data: bytemuck::bytes_of(&RestirConstants::new(config)).into(),
            }])?;
        self.update_hash_grid.clear(context)?;
        Ok(())
    }
}
