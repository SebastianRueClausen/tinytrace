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
struct RestirData {
    reservoirs: u64,
    reservoir_updates: u64,
    update_counts: u64,
    samples_counts: u64,
    scene_scale: f32,
    updates_per_cell: u32,
    reservoirs_per_cell: u32,
    replay: RestirReplay,
}

pub struct RestirState {
    pub reservoir_hash_grid: HashGrid,
    pub update_hash_grid: HashGrid,
    pub reservoirs: Handle<Buffer>,
    pub updates: Handle<Buffer>,
    pub update_counts: Handle<Buffer>,
    pub sample_counts: Handle<Buffer>,
    pub data: Handle<Buffer>,
    pub update_reservoirs: Handle<Shader>,
}

fn create_buffer(
    context: &mut Context,
    count: u32,
    value_size: usize,
    ty: BufferType,
) -> Result<Handle<Buffer>, tinytrace_backend::Error> {
    context.create_buffer(
        Lifetime::Renderer,
        &BufferRequest {
            size: count as u64 * value_size as u64,
            memory_location: MemoryLocation::Device,
            ty,
        },
    )
}

impl RestirState {
    pub fn new(context: &mut Context, config: &RestirConfig) -> Result<Self, Error> {
        let bindings = &[
            binding!(uniform_buffer, Constants, constants),
            binding!(uniform_buffer, RestirData, restir_data),
            binding!(uniform_buffer, HashGrid, reservoir_hash_grid),
            binding!(uniform_buffer, HashGrid, reservoir_update_hash_grid),
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
                BufferType::Storage,
            )?,
            updates: create_buffer(
                context,
                config.update_hash_grid_capacity * config.updates_per_cell,
                mem::size_of::<Reservoir>(),
                BufferType::Storage,
            )?,
            update_counts: create_buffer(
                context,
                config.update_hash_grid_capacity,
                mem::size_of::<u32>(),
                BufferType::Storage,
            )?,
            sample_counts: create_buffer(
                context,
                config.reservoir_hash_grid_capacity,
                mem::size_of::<u32>(),
                BufferType::Storage,
            )?,
            data: create_buffer(
                context,
                1,
                mem::size_of::<RestirData>(),
                BufferType::Uniform,
            )?,
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
            .bind_buffer("restir_data", &self.data)
            .register_indirect_buffer("reservoirs", &self.reservoirs, false)
            .register_indirect_buffer("updates", &self.updates, false)
            .register_indirect_buffer("update_counts", &self.update_counts, false)
            .bind_buffer("reservoir_hash_grid", &self.reservoir_hash_grid.data)
            .register_indirect_buffer("reservoir_hash_grid", &self.reservoir_hash_grid.keys, false)
            .bind_buffer("reservoir_update_hash_grid", &self.update_hash_grid.data)
            .register_indirect_buffer(
                "reservoir_update_hash_grid",
                &self.update_hash_grid.keys,
                false,
            )
            .dispatch(self.update_hash_grid.capacity, 1)?;
        Ok(())
    }

    pub fn prepare(&self, context: &mut Context, config: &RestirConfig) -> Result<(), Error> {
        let RestirConfig {
            scene_scale,
            updates_per_cell,
            reservoirs_per_cell,
            replay,
            ..
        } = *config;
        let data = RestirData {
            reservoirs: context.buffer_device_address(&self.reservoirs),
            reservoir_updates: context.buffer_device_address(&self.updates),
            update_counts: context.buffer_device_address(&self.update_counts),
            samples_counts: context.buffer_device_address(&self.sample_counts),
            scene_scale,
            updates_per_cell,
            reservoirs_per_cell,
            replay,
        };
        context
            .fill_buffer(&self.update_counts, 0)?
            .write_buffers(&[BufferWrite {
                buffer: self.data.clone(),
                data: bytemuck::bytes_of(&data).into(),
            }])?;
        self.update_hash_grid.clear(context)?;
        Ok(())
    }
}
