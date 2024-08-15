use glam::Vec3;
use half::f16;

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
#[derive(bytemuck::AnyBitPattern, Clone, Copy, Debug)]
pub(super) struct ReservoirUpdate {
    paths: [Path; 64],
    weights: [f32; 64],
    update_count: u32,
}

#[repr(C)]
#[derive(bytemuck::Zeroable, bytemuck::Pod, Clone, Copy, Debug)]
pub(super) struct Reservoir {
    path: Path,
    weight_sum: f32,
    weight: f32,
    sample_count: u32,
}

#[repr(C)]
#[derive(bytemuck::AnyBitPattern, Clone, Copy, Debug)]
pub(super) struct ReservoirPool {
    reservoirs: [Reservoir; 4],
}
