use std::mem;

use ash::vk;
use glam::{Mat4, Quat, Vec3};
use half::f16;

use crate::asset;
use crate::backend::resource::{BlasBuild, BlasRequest, BufferRange, TlasInstance};
use crate::backend::{
    Blas, Buffer, BufferRequest, BufferType, BufferWrite, Context, Handle, Image, ImageRequest,
    ImageWrite, Lifetime, MemoryLocation, Sampler, SamplerRequest, Tlas,
};

use super::error::Result;

pub struct Scene {
    pub positions: Handle<Buffer>,
    pub vertices: Handle<Buffer>,
    pub indices: Handle<Buffer>,
    pub materials: Handle<Buffer>,
    pub meshes: Handle<Buffer>,
    pub instances: Handle<Buffer>,
    pub textures: Vec<Handle<Image>>,
    pub texture_sampler: Handle<Sampler>,
    pub blases: Vec<Handle<Blas>>,
    pub tlas: Handle<Tlas>,
}

impl Scene {
    pub fn new(context: &mut Context, scene: &asset::Scene) -> Result<Self> {
        let mut objects = Vec::new();

        for instance in &scene.instances {
            flatten_instance_tree(scene, instance, Mat4::IDENTITY, &mut objects);
        }

        let instance_data = create_instances(&objects);

        let positions = create_buffer(context, &scene.positions)?;
        let vertices = create_buffer(context, &scene.vertices)?;
        let indices = create_buffer(context, &scene.indices)?;
        let materials = create_buffer(context, &scene.materials)?;
        let meshes = create_buffer(context, &scene.meshes)?;
        let instances = create_buffer(context, &instance_data)?;

        context.write_buffers(&[
            scene_buffer_write(&positions, &scene.positions),
            scene_buffer_write(&vertices, &scene.vertices),
            scene_buffer_write(&indices, &scene.indices),
            scene_buffer_write(&materials, &scene.materials),
            scene_buffer_write(&meshes, &scene.meshes),
            scene_buffer_write(&instances, &instance_data),
        ])?;

        let textures: Vec<_> = scene
            .textures
            .iter()
            .map(|texture| create_texture(context, texture))
            .collect::<Result<_>>()?;
        let image_writes: Vec<_> = scene
            .textures
            .iter()
            .zip(textures.iter())
            .map(|(texture, image)| ImageWrite {
                offset: vk::Offset3D::default(),
                extent: context.image(image).extent,
                image: image.clone(),
                mips: &texture.mips,
            })
            .collect();
        context.write_images(&image_writes).unwrap();

        let texture_sampler = context.create_sampler(
            Lifetime::Scene,
            &SamplerRequest {
                address_mode: vk::SamplerAddressMode::REPEAT,
                filter: vk::Filter::LINEAR,
                max_anisotropy: Some(16.0),
            },
        )?;

        let blases: Vec<_> = scene
            .meshes
            .iter()
            .map(|mesh| {
                context.create_blas(
                    Lifetime::Scene,
                    &BlasRequest {
                        vertex_format: vk::Format::R16G16B16_SNORM,
                        vertex_stride: mem::size_of::<[i16; 3]>() as vk::DeviceSize,
                        triangle_count: mesh.index_count / 3,
                        vertex_count: mesh.vertex_count,
                        first_vertex: mesh.vertex_offset,
                    },
                )
            })
            .collect::<Result<_>>()?;

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
                    offset: mesh.index_offset as vk::DeviceSize
                        * mem::size_of::<u32>() as vk::DeviceSize,
                },
            })
            .collect();
        context.build_blases(&blas_builds).unwrap();

        let tlas_instances = create_tlas_instances(scene, &objects, &blases);
        let tlas = context.create_tlas(Lifetime::Scene, tlas_instances.len() as u32)?;

        let build_mode = vk::BuildAccelerationStructureModeKHR::BUILD;
        context.build_tlas(&tlas, build_mode, &tlas_instances)?;

        Ok(Self {
            positions,
            vertices,
            indices,
            materials,
            meshes,
            instances,
            textures,
            texture_sampler,
            blases,
            tlas,
        })
    }
}

struct Object {
    mesh_index: u32,
    material: u32,
    transform: Mat4,
}

fn flatten_instance_tree(
    scene: &asset::Scene,
    instance: &asset::Instance,
    parent_transform: Mat4,
    objects: &mut Vec<Object>,
) -> Mat4 {
    let transform = parent_transform * instance.transform;

    if let Some(model_index) = instance.model_index {
        let model = &scene.models[model_index as usize];
        objects.extend(model.mesh_indices.iter().copied().map(|mesh_index| Object {
            material: scene.meshes[mesh_index as usize].material,
            transform,
            mesh_index,
        }));
    }

    for child in &instance.children {
        flatten_instance_tree(scene, child, transform, objects);
    }

    transform
}

fn create_tlas_instances(
    scene: &asset::Scene,
    objects: &[Object],
    blases: &[Handle<Blas>],
) -> Vec<TlasInstance> {
    objects
        .iter()
        .map(|object| {
            let asset::BoundingSphere { radius, center } =
                scene.meshes[object.mesh_index as usize].bounding_sphere;
            let transform =
                Mat4::from_scale_rotation_translation(Vec3::splat(radius), Quat::IDENTITY, center);
            TlasInstance {
                transform: object.transform * transform,
                blas: blases[object.mesh_index as usize].clone(),
            }
        })
        .collect()
}

fn create_instances(objects: &[Object]) -> Vec<Instance> {
    objects
        .iter()
        .map(|object| Instance {
            normal_transform: object
                .transform
                .inverse()
                .transpose()
                .to_cols_array()
                .map(f16::from_f32),
            transform: object.transform,
            material: object.material,
            padding: Default::default(),
        })
        .collect()
}

fn texture_kind_format(kind: asset::TextureKind) -> vk::Format {
    match kind {
        asset::TextureKind::Albedo => vk::Format::BC1_RGBA_SRGB_BLOCK,
        asset::TextureKind::Normal => vk::Format::BC5_UNORM_BLOCK,
        asset::TextureKind::Specular => vk::Format::BC5_UNORM_BLOCK,
        asset::TextureKind::Emissive => vk::Format::BC1_RGB_SRGB_BLOCK,
    }
}

fn create_buffer<T>(context: &mut Context, data: &[T]) -> Result<Handle<Buffer>> {
    context.create_buffer(
        Lifetime::Scene,
        &BufferRequest {
            memory_location: MemoryLocation::Device,
            size: mem::size_of_val(data) as vk::DeviceSize,
            ty: BufferType::Storage,
        },
    )
}

fn scene_buffer_write<'a, T: bytemuck::NoUninit>(
    buffer: &Handle<Buffer>,
    data: &'a [T],
) -> BufferWrite<'a> {
    BufferWrite {
        data: bytemuck::cast_slice(data),
        buffer: buffer.clone(),
    }
}

fn create_texture(context: &mut Context, texture: &asset::Texture) -> Result<Handle<Image>> {
    context.create_image(
        Lifetime::Scene,
        &ImageRequest {
            extent: vk::Extent3D::default()
                .width(texture.width)
                .height(texture.height)
                .depth(1),
            format: texture_kind_format(texture.kind),
            mip_level_count: texture.mips.len() as u32,
            memory_location: MemoryLocation::Device,
        },
    )
}

#[repr(C)]
#[derive(bytemuck::NoUninit, Clone, Copy)]
struct Instance {
    transform: Mat4,
    normal_transform: [f16; 16],
    material: u32,
    padding: [u32; 3],
}
