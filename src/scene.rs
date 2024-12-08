use std::borrow::Cow;
use std::mem;

use glam::{Mat4, Quat, Vec3};
use half::f16;

use crate::{asset, Error};
use tinytrace_backend::{
    Blas, BlasBuild, BlasRequest, Buffer, BufferRange, BufferRequest, BufferType, BufferWrite,
    Context, Filter, Handle, Image, ImageRequest, ImageWrite, Lifetime, MemoryLocation, Offset,
    Sampler, SamplerRequest, Tlas, TlasBuildMode, TlasInstance, VertexFormat,
};

pub struct Scene {
    pub vertices: Handle<Buffer>,
    pub indices: Handle<Buffer>,
    pub materials: Handle<Buffer>,
    pub instances: Handle<Buffer>,
    pub emissive_triangles: Handle<Buffer>,
    pub scene_data: Handle<Buffer>,
    pub textures: Vec<Handle<Image>>,
    pub texture_sampler: Handle<Sampler>,
    pub blases: Vec<Handle<Blas>>,
    pub tlas: Handle<Tlas>,
}

impl Scene {
    pub fn new(context: &mut Context, scene: &asset::Scene) -> Result<Self, Error> {
        let (scene_instances, _) = asset::traverse_instance_tree(
            &scene.root,
            (Vec::<Instance>::new(), Mat4::IDENTITY),
            &mut |_, instance, (mut instances, parent_transform)| {
                let transform = parent_transform * instance.transform;
                for model in &instance.models {
                    let mesh = &scene.meshes[model.mesh_index as usize];
                    let position_transform = transform
                        * Mat4::from_scale_rotation_translation(
                            Vec3::splat(mesh.bounding_sphere.radius),
                            Quat::IDENTITY,
                            mesh.bounding_sphere.center,
                        );
                    instances.push(Instance {
                        normal_transform: transform.inverse().transpose(),
                        inverse_transform: position_transform.inverse(),
                        transform: position_transform,
                        mesh: model.mesh_index,
                        material: model.material_index,
                        index_offset: mesh.index_offset,
                        vertex_offset: mesh.vertex_offset,
                    });
                }
                (instances, transform)
            },
        );

        let emissive_triangle_data = scene.emissive_triangles();

        let positions = create_buffer(context, &scene.positions)?;
        let vertices = create_buffer(context, &scene.vertices)?;
        let indices = create_buffer(context, &scene.indices)?;
        let materials = create_buffer(context, &scene.materials)?;
        let instances = create_buffer(context, &scene_instances)?;
        let emissive_triangles = create_buffer(context, &emissive_triangle_data)?;

        let scene_data = context.create_buffer(
            Lifetime::Scene,
            &BufferRequest {
                size: mem::size_of::<SceneData>() as u64,
                ty: BufferType::Uniform,
                memory_location: MemoryLocation::Device,
            },
        )?;

        context.write_buffers(&[
            scene_buffer_write(&positions, &scene.positions),
            scene_buffer_write(&vertices, &scene.vertices),
            scene_buffer_write(&indices, &scene.indices),
            scene_buffer_write(&materials, &scene.materials),
            scene_buffer_write(&instances, &scene_instances),
            scene_buffer_write(&emissive_triangles, &emissive_triangle_data),
            scene_buffer_write(
                &scene_data,
                &[SceneData {
                    vertices: context.buffer_device_address(&vertices),
                    indices: context.buffer_device_address(&indices),
                    instances: context.buffer_device_address(&instances),
                    emissive_triangles: context.buffer_device_address(&emissive_triangles),
                    materials: context.buffer_device_address(&materials),
                    emissive_triangle_count: emissive_triangle_data.len() as u32,
                    padding: 0,
                }],
            ),
        ])?;

        let textures: Vec<_> = scene
            .textures
            .iter()
            .map(|texture| create_texture(context, texture))
            .collect::<Result<_, _>>()?;
        let image_writes: Vec<_> = scene
            .textures
            .iter()
            .zip(textures.iter())
            .map(|(texture, image)| ImageWrite {
                offset: Offset::default(),
                extent: context.image(image).extent(),
                image: image.clone(),
                mips: Cow::Borrowed(&texture.data),
            })
            .collect();
        context.write_images(&image_writes).unwrap();

        let texture_sampler = context.create_sampler(
            Lifetime::Scene,
            &SamplerRequest {
                filter: Filter::Linear,
                max_anisotropy: Some(16.0),
                clamp_to_edge: false,
            },
        )?;

        let blases: Vec<_> = scene
            .meshes
            .iter()
            .map(|mesh| {
                context.create_blas(
                    Lifetime::Scene,
                    &BlasRequest {
                        vertex_format: VertexFormat::Snorm16,
                        vertex_stride: mem::size_of::<[i16; 3]>() as u64,
                        triangle_count: mesh.index_count / 3,
                        vertex_count: mesh.vertex_count,
                        first_vertex: mesh.vertex_offset,
                    },
                )
            })
            .collect::<Result<_, tinytrace_backend::Error>>()?;

        let blas_builds: Vec<_> = blases
            .iter()
            .zip(scene.meshes.iter())
            .map(|(blas, mesh)| BlasBuild {
                blas: blas.clone(),
                vertices: BufferRange {
                    buffer: positions.clone(),
                    offset: 0,
                },
                indices: BufferRange {
                    buffer: indices.clone(),
                    offset: mesh.index_offset as u64 * mem::size_of::<u32>() as u64,
                },
            })
            .collect();
        context.build_blases(&blas_builds).unwrap();

        let tlas_instances: Vec<_> = scene_instances
            .iter()
            .enumerate()
            .map(|(index, instance)| instance.tlas_instance(&blases, index as u32))
            .collect();
        let tlas = context.create_tlas(Lifetime::Scene, tlas_instances.len() as u32)?;

        context.build_tlas(&tlas, TlasBuildMode::Build, &tlas_instances)?;

        Ok(Self {
            vertices,
            indices,
            materials,
            instances,
            emissive_triangles,
            scene_data,
            textures,
            texture_sampler,
            blases,
            tlas,
        })
    }
}

fn create_buffer<T>(
    context: &mut Context,
    data: &[T],
) -> Result<Handle<Buffer>, tinytrace_backend::Error> {
    context.create_buffer(
        Lifetime::Scene,
        &BufferRequest {
            memory_location: MemoryLocation::Device,
            size: mem::size_of_val(data) as u64,
            ty: BufferType::Storage,
        },
    )
}

fn scene_buffer_write<'a, T: bytemuck::NoUninit>(
    buffer: &Handle<Buffer>,
    data: &'a [T],
) -> BufferWrite<'a> {
    BufferWrite {
        data: bytemuck::cast_slice(data).into(),
        buffer: buffer.clone(),
    }
}

fn create_texture(
    context: &mut Context,
    texture: &asset::ProcessedTexture,
) -> Result<Handle<Image>, tinytrace_backend::Error> {
    context.create_image(
        Lifetime::Scene,
        &ImageRequest {
            extent: texture.extent,
            format: texture.format,
            mip_level_count: texture.data.len() as u32,
            memory_location: MemoryLocation::Device,
        },
    )
}

#[repr(C)]
#[derive(bytemuck::NoUninit, Debug, Default, Clone, Copy)]
pub struct EmissiveTriangle {
    pub positions: [[i16; 3]; 3],
    pub hash: u16,
    pub tex_coords: [[f16; 2]; 3],
    pub instance: u32,
}

#[repr(C)]
#[derive(bytemuck::NoUninit, Debug, Clone, Copy)]
struct Instance {
    transform: Mat4,
    inverse_transform: Mat4,
    normal_transform: Mat4,
    vertex_offset: u32,
    index_offset: u32,
    material: u32,
    mesh: u32,
}

impl Instance {
    fn tlas_instance(&self, blases: &[Handle<Blas>], index: u32) -> TlasInstance {
        TlasInstance {
            transform: self.transform,
            blas: blases[self.mesh as usize].clone(),
            index,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::NoUninit)]
struct SceneData {
    vertices: u64,
    indices: u64,
    instances: u64,
    emissive_triangles: u64,
    materials: u64,
    emissive_triangle_count: u32,
    padding: u32,
}
