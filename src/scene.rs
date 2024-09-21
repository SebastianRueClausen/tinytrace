use std::borrow::Cow;
use std::{array, mem};

use ash::vk;
use glam::{Mat4, Quat, Vec3};
use half::f16;

use crate::Error;
use tinytrace_backend::{
    Blas, BlasBuild, BlasRequest, Buffer, BufferRange, BufferRequest, BufferType, BufferWrite,
    Context, Handle, Image, ImageFormat, ImageRequest, ImageWrite, Lifetime, MemoryLocation,
    Sampler, SamplerRequest, Tlas, TlasInstance,
};

pub struct Scene {
    pub vertices: Handle<Buffer>,
    pub indices: Handle<Buffer>,
    pub materials: Handle<Buffer>,
    pub instances: Handle<Buffer>,
    pub emissive_triangles: Handle<Buffer>,
    pub textures: Vec<Handle<Image>>,
    pub texture_sampler: Handle<Sampler>,
    pub blases: Vec<Handle<Blas>>,
    pub tlas: Handle<Tlas>,
}

impl Scene {
    pub fn new(context: &mut Context, scene: &tinytrace_asset::Scene) -> Result<Self, Error> {
        let mut scene_instances = Vec::new();

        for instance in &scene.instances {
            flatten_instance_tree(
                scene,
                instance,
                Mat4::IDENTITY,
                &scene.meshes,
                &mut scene_instances,
            );
        }

        let emissive_triangle_data: Vec<EmissiveTriangle> = scene_instances
            .iter()
            .enumerate()
            .flat_map(|(index, instance)| instance.emissive_triangles(scene, index as u32))
            .collect();

        let positions = create_buffer(context, &scene.positions)?;
        let vertices = create_buffer(context, &scene.vertices)?;
        let indices = create_buffer(context, &scene.indices)?;
        let materials = create_buffer(context, &scene.materials)?;
        let instances = create_buffer(context, &scene_instances)?;
        let emissive_triangles = create_buffer(context, &emissive_triangle_data)?;

        context.write_buffers(&[
            scene_buffer_write(&positions, &scene.positions),
            scene_buffer_write(&vertices, &scene.vertices),
            scene_buffer_write(&indices, &scene.indices),
            scene_buffer_write(&materials, &scene.materials),
            scene_buffer_write(&instances, &scene_instances),
            scene_buffer_write(&emissive_triangles, &emissive_triangle_data),
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
                offset: vk::Offset3D::default(),
                extent: context.image(image).extent,
                image: image.clone(),
                mips: Cow::Borrowed(&texture.mips),
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
                    offset: mesh.index_offset as vk::DeviceSize
                        * mem::size_of::<u32>() as vk::DeviceSize,
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

        let build_mode = vk::BuildAccelerationStructureModeKHR::BUILD;
        context.build_tlas(&tlas, build_mode, &tlas_instances)?;

        Ok(Self {
            vertices,
            indices,
            materials,
            instances,
            emissive_triangles,
            textures,
            texture_sampler,
            blases,
            tlas,
        })
    }
}

fn flatten_instance_tree(
    scene: &tinytrace_asset::Scene,
    instance: &tinytrace_asset::Instance,
    parent_transform: Mat4,
    meshes: &[tinytrace_asset::Mesh],
    instances: &mut Vec<Instance>,
) -> Mat4 {
    let transform = parent_transform * instance.transform;
    if let Some(model_index) = instance.model_index {
        let model = &scene.models[model_index as usize];
        instances.extend(model.mesh_indices.iter().copied().map(|mesh| {
            let tinytrace_asset::Mesh {
                bounding_sphere: tinytrace_asset::BoundingSphere { radius, center },
                vertex_offset,
                index_offset,
                material,
                ..
            } = meshes[mesh as usize];
            let position_transform = transform
                * Mat4::from_scale_rotation_translation(
                    Vec3::splat(radius),
                    Quat::IDENTITY,
                    center,
                );
            Instance {
                normal_transform: transform.inverse().transpose(),
                inverse_transform: position_transform.inverse(),
                transform: position_transform,
                mesh,
                material,
                index_offset,
                vertex_offset,
            }
        }));
    }
    for child in &instance.children {
        flatten_instance_tree(scene, child, transform, meshes, instances);
    }
    transform
}

fn texture_kind_format(kind: tinytrace_asset::TextureKind) -> ImageFormat {
    match kind {
        tinytrace_asset::TextureKind::Albedo => ImageFormat::RgbaBc1Srgb,
        tinytrace_asset::TextureKind::Normal | tinytrace_asset::TextureKind::Specular => {
            ImageFormat::RgBc5Unorm
        }
        tinytrace_asset::TextureKind::Emissive => ImageFormat::RgbBc1Srgb,
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
        data: bytemuck::cast_slice(data).into(),
        buffer: buffer.clone(),
    }
}

fn create_texture(
    context: &mut Context,
    texture: &tinytrace_asset::Texture,
) -> Result<Handle<Image>, tinytrace_backend::Error> {
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

    fn emissive_triangles<'a>(
        &'a self,
        scene: &'a tinytrace_asset::Scene,
        instance_index: u32,
    ) -> impl Iterator<Item = EmissiveTriangle> + 'a {
        let mesh = &scene.meshes[self.mesh as usize];
        mesh.emissive_triangles.iter().map(move |triangle_index| {
            let base_index = *triangle_index as usize * 3;
            let indices: [usize; 3] = array::from_fn(|vertex| {
                (scene.indices[base_index + vertex] + mesh.vertex_offset) as usize
            });
            EmissiveTriangle {
                tex_coords: indices.map(|index| scene.vertices[index].tex_coord),
                positions: indices.map(|index| {
                    array::from_fn(|coordinate| scene.positions[index * 3 + coordinate])
                }),
                instance: instance_index,
                hash: (triangle_index % u16::MAX as u32) as u16,
            }
        })
    }
}
