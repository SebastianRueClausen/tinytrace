mod import;
mod normal;

#[cfg(test)]
mod test;

use std::path::Path;

use glam::{Mat4, Vec3};
use half::f16;

use normal::TangentFrame;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct DirectionalLight {
    pub direction: [f16; 3],
    pub irradiance: [f16; 3],
}

impl Default for DirectionalLight {
    fn default() -> Self {
        Self {
            direction: [f16::ZERO; 3],
            irradiance: [f16::ONE; 3],
        }
    }
}

unsafe impl bytemuck::NoUninit for DirectionalLight {}

#[derive(Debug, Clone, Copy)]
pub struct BoundingSphere {
    pub center: Vec3,
    pub radius: f32,
}

unsafe impl bytemuck::NoUninit for BoundingSphere {}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Mesh {
    pub bounding_sphere: BoundingSphere,
    pub vertex_offset: u32,
    pub vertex_count: u32,
    pub index_offset: u32,
    pub index_count: u32,
    pub material: u32,
    pub color_offset: u32,
}

unsafe impl bytemuck::NoUninit for Mesh {}

#[derive(Clone, Debug)]
pub struct Model {
    pub mesh_indices: Vec<u32>,
}

#[derive(Clone, Debug)]
pub struct Instance {
    pub transform: Mat4,
    pub model_index: Option<u32>,
    pub children: Vec<Instance>,
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub enum TextureKind {
    Albedo,
    Normal,
    Specular,
    Emissive,
}

#[derive(Clone, Debug)]
pub struct Texture {
    pub kind: TextureKind,
    pub width: u32,
    pub height: u32,
    pub mips: Vec<Box<[u8]>>,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct Material {
    pub albedo_texture: u16,
    pub normal_texture: u16,
    pub specular_texture: u16,
    pub emissive_texture: u16,
    pub base_color: [f16; 4],
    pub emissive: [f16; 3],
    pub metallic: f16,
    pub roughness: f16,
    pub ior: f16,
}

unsafe impl bytemuck::NoUninit for Material {}

#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
pub struct Vertex {
    pub texcoord: [f16; 2],
    pub tangent_frame: TangentFrame,
}

unsafe impl bytemuck::NoUninit for Vertex {}

#[derive(Debug, Default)]
pub struct Scene {
    pub directional_light: DirectionalLight,
    pub vertices: Vec<Vertex>,
    pub positions: Vec<i16>,
    pub colors: Vec<f16>,
    pub indices: Vec<u32>,
    pub textures: Vec<Texture>,
    pub materials: Vec<Material>,
    pub meshes: Vec<Mesh>,
    pub models: Vec<Model>,
    pub instances: Vec<Instance>,
}

fn add_item<T>(vector: &mut Vec<T>, item: T) -> usize {
    vector.push(item);
    vector.len() - 1
}

impl Scene {
    fn add_texture(&mut self, texture: Texture) -> u16 {
        add_item(&mut self.textures, texture) as u16
    }

    fn add_material(&mut self, material: Material) -> u32 {
        add_item(&mut self.materials, material) as u32
    }

    fn add_mesh(&mut self, mesh: Mesh) -> u32 {
        add_item(&mut self.meshes, mesh) as u32
    }

    pub fn from_gltf<P: AsRef<Path>>(path: P) -> Result<Self, gltf::Error> {
        import::Data::new(path.as_ref()).and_then(|data| import::load_scene(&data))
    }
}

pub const INVALID_INDEX: u32 = u32::MAX;
