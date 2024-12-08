pub mod gltf;
pub mod material;
mod normal;
mod texture;

use std::{borrow::Cow, collections::HashSet, error::Error};

use glam::{Mat4, Vec2, Vec3, Vec4};
use half::f16;

use material::ProcessedMaterial;
use normal::TangentFrame;

pub use material::{Base, Coat, Emission, Fuzz, Geometry, Material, Param, Specular, Transmission};
pub(crate) use texture::ProcessedTexture;
pub use texture::Texture;

use crate::math;

#[derive(Clone, Debug)]
pub struct Mesh<'a> {
    pub positions: &'a [Vec3],
    pub texture_coordinates: &'a [Vec2],
    pub indices: Option<&'a [u32]>,
    pub normals: Option<&'a [Vec3]>,
    pub tangents: Option<&'a [Vec4]>,
}

#[derive(Clone, Debug)]
pub struct Model {
    pub mesh_index: u32,
    pub material_index: u32,
}

#[derive(Clone, Debug, Default)]
pub struct Instance {
    pub transform: Mat4,
    pub models: Vec<Model>,
    pub children: Vec<Instance>,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
pub(crate) struct ProcessedVertex {
    pub texture_coordinates: [f16; 2],
    pub tangent_frame: TangentFrame,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
pub(crate) struct BoundingSphere {
    pub center: Vec3,
    pub radius: f32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
pub(crate) struct ProcessedMesh {
    pub bounding_sphere: BoundingSphere,
    pub vertex_offset: u32,
    pub vertex_count: u32,
    pub index_offset: u32,
    pub index_count: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
pub(crate) struct EmissiveTriangle {
    pub positions: [[i16; 3]; 3],
    pub hash: u16,
    pub texture_coordinates: [[f16; 2]; 3],
    pub instance_index: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Default)]
pub struct Scene {
    pub(crate) positions: Vec<[i16; 3]>,
    pub(crate) indices: Vec<u32>,
    pub(crate) vertices: Vec<ProcessedVertex>,
    pub(crate) textures: Vec<ProcessedTexture>,
    pub(crate) materials: Vec<ProcessedMaterial>,
    pub(crate) meshes: Vec<ProcessedMesh>,
    pub(crate) emissive_materials: HashSet<u32>,
    pub root: Instance,
}

impl Scene {
    pub fn insert_mesh(&mut self, mesh: &Mesh) -> u32 {
        let mesh_index = self.meshes.len() as u32;
        let vertex_offset = self.positions.len() as u32;
        let vertex_count = mesh.positions.len() as u32;

        // Process positions.
        let bounding_sphere = bounding_sphere(&mesh.positions);
        self.positions.extend(mesh.positions.iter().map(|position| {
            let position = (*position - bounding_sphere.center) / bounding_sphere.radius;
            assert!(position.abs().max_element() <= 1.0);
            position
                .to_array()
                .map(|value| math::quantize_snorm(value, 16) as i16)
        }));

        // Process indices.
        let index_offset = self.indices.len();
        let index_count = match mesh.indices {
            Some(mesh_indices) => {
                self.indices.extend_from_slice(mesh_indices);
                mesh_indices.len()
            }
            None => {
                self.indices.extend(0u32..vertex_count);
                vertex_count as usize
            }
        };

        // Process normals.
        let normals = match mesh.normals {
            Some(normals) => Cow::Borrowed(normals),
            None => {
                let indices = &self.indices[index_offset..index_offset + index_count];
                Cow::Owned(generate_normals(mesh.positions, indices))
            }
        };

        // Process tangents.
        let tangents = match mesh.tangents {
            Some(tangents) => Cow::Borrowed(tangents),
            None => {
                let indices = &self.indices[index_offset..index_offset + index_count];
                Cow::Owned(generate_tangents(
                    mesh.positions,
                    mesh.texture_coordinates,
                    &normals,
                    indices,
                ))
            }
        };

        // Process vertices.
        self.vertices.extend(
            mesh.texture_coordinates
                .iter()
                .zip(normals.iter())
                .zip(tangents.iter())
                .map(|((texture_coordinate, normal), tangent)| ProcessedVertex {
                    texture_coordinates: texture_coordinate.to_array().map(f16::from_f32),
                    tangent_frame: TangentFrame::new(*normal, *tangent),
                }),
        );

        self.meshes.push(ProcessedMesh {
            index_offset: index_offset as u32,
            index_count: index_count as u32,
            bounding_sphere,
            vertex_offset,
            vertex_count,
        });

        mesh_index
    }

    fn emissive_triangles_in_model(
        &self,
        model: &Model,
        instance_index: u32,
        emissive_triangles: &mut Vec<EmissiveTriangle>,
    ) {
        if !self.emissive_materials.contains(&model.material_index) {
            return;
        }
        let mesh = &self.meshes[model.mesh_index as usize];
        let index_range =
            mesh.index_offset as usize..(mesh.index_offset + mesh.index_count) as usize;
        emissive_triangles.extend(self.indices[index_range].chunks(3).enumerate().map(
            |(offset, triangle)| {
                let indices: [u32; 3] = triangle.try_into().unwrap();
                let positions =
                    indices.map(|index| self.positions[(mesh.vertex_offset + index) as usize]);
                let texture_coordinates = indices.map(|index| {
                    self.vertices[(mesh.vertex_offset + index) as usize].texture_coordinates
                });
                EmissiveTriangle {
                    hash: (mesh.index_offset / 3 + offset as u32 % u16::MAX as u32) as u16,
                    instance_index,
                    texture_coordinates,
                    positions,
                }
            },
        ));
    }

    pub(crate) fn emissive_triangles(&self) -> Vec<EmissiveTriangle> {
        let mut find_emissive_triangles =
            |index, instance: &Instance, mut state: Vec<EmissiveTriangle>| {
                for model in &instance.models {
                    self.emissive_triangles_in_model(model, index, &mut state);
                }
                state
            };
        traverse_instance_tree(&self.root, Vec::new(), &mut find_emissive_triangles)
    }
}

pub trait SceneImporter {
    type Error: Error;

    fn insert_in_scene(&self, scene: &mut Scene) -> Result<(), Self::Error>;
    fn new_scene(&self) -> Result<Scene, Self::Error> {
        let mut scene = Scene::default();
        self.insert_in_scene(&mut scene)?;
        Ok(scene)
    }
}

pub(crate) fn traverse_instance_tree<T, F>(instance: &Instance, default: T, visitor: &mut F) -> T
where
    F: FnMut(u32, &Instance, T) -> T,
{
    fn traverse_instance_tree_inner<T, F>(
        instance: &Instance,
        index: u32,
        value: T,
        visitor: &mut F,
    ) -> (u32, T)
    where
        F: FnMut(u32, &Instance, T) -> T,
    {
        let value = visitor(index, instance, value);
        instance
            .children
            .iter()
            .fold((index + 1, value), |(index, value), instance| {
                traverse_instance_tree_inner(instance, index, value, visitor)
            })
    }
    traverse_instance_tree_inner(instance, 0, default, visitor).1
}

pub fn generate_normals(positions: &[Vec3], indices: &[u32]) -> Vec<Vec3> {
    let mut normals = vec![Vec3::ZERO; positions.len()];
    for triangle in indices.chunks(3) {
        let [i0, i1, i2] = triangle else {
            panic!("indices isn't multiple of 3");
        };
        let [i0, i1, i2] = [i0, i1, i2].map(|i| *i as usize);
        let normal = (positions[i1] - positions[i0]).cross(positions[i2] - positions[i0]);
        for index in [i0, i1, i2] {
            normals[index] += normal;
        }
    }
    for normal in &mut normals {
        *normal = normal.normalize();
    }
    normals
}

pub fn generate_tangents(
    positions: &[Vec3],
    texture_coordinates: &[Vec2],
    normals: &[Vec3],
    indices: &[u32],
) -> Vec<Vec4> {
    let tangents = vec![Vec4::ZERO; positions.len()];
    let mut generator = TangentGenerator {
        positions,
        tex_coords: texture_coordinates,
        normals,
        indices,
        tangents,
    };
    if !mikktspace::generate_tangents(&mut generator) {
        panic!("failed to generate tangents");
    }
    for tangent in &mut generator.tangents {
        tangent[3] *= -1.0;
    }
    generator.tangents
}

struct TangentGenerator<'a> {
    positions: &'a [Vec3],
    tex_coords: &'a [Vec2],
    normals: &'a [Vec3],
    indices: &'a [u32],
    tangents: Vec<Vec4>,
}

impl<'a> TangentGenerator<'a> {
    fn index(&self, face: usize, vertex: usize) -> usize {
        self.indices[face * 3 + vertex] as usize
    }
}

impl<'a> mikktspace::Geometry for TangentGenerator<'a> {
    fn num_faces(&self) -> usize {
        self.indices.len() / 3
    }

    fn num_vertices_of_face(&self, _: usize) -> usize {
        3
    }

    fn position(&self, face: usize, vert: usize) -> [f32; 3] {
        self.positions[self.index(face, vert)].into()
    }

    fn normal(&self, face: usize, vert: usize) -> [f32; 3] {
        self.normals[self.index(face, vert)].into()
    }

    fn tex_coord(&self, face: usize, vert: usize) -> [f32; 2] {
        self.tex_coords[self.index(face, vert)].into()
    }

    fn set_tangent_encoded(&mut self, tangent: [f32; 4], face: usize, vert: usize) {
        let index = self.index(face, vert);
        self.tangents[index] = tangent.into();
    }
}

fn bounding_sphere(positions: &[Vec3]) -> BoundingSphere {
    let center = positions
        .iter()
        .enumerate()
        .fold(Vec3::ZERO, |mean, (index, position)| {
            mean + (*position - mean) / (index + 1) as f32
        });
    let radius = positions
        .iter()
        .map(|position| (*position - center).length())
        .max_by(|a, b| a.total_cmp(&b))
        .unwrap_or_default();
    BoundingSphere { center, radius }
}
