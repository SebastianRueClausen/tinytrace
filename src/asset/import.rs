use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::{array, fs, io, mem};

use super::normal::TangentFrame;
use super::{
    normal, BoundingSphere, Error, Instance, Material, Mesh, Model, Scene, Texture, TextureKind,
    Vertex,
};
use bytemuck::AnyBitPattern;
use glam::{Mat4, Vec2, Vec3, Vec4};
use gltf::Gltf;
use half::f16;

#[derive(Default)]
struct Fallback {
    indices: HashMap<TextureKind, u16>,
}

impl Fallback {
    fn insert(&mut self, scene: &mut Scene, specs: &TextureSpecs) -> u16 {
        *self.indices.entry(specs.kind).or_insert_with(|| {
            let mip = vec![specs.fallback; 16];
            let mip = compress_bytes(specs.kind, 4, 4, bytemuck::cast_slice(&mip));
            scene.add_texture(Texture {
                mips: vec![mip.into_boxed_slice()],
                kind: specs.kind,
                width: 4,
                height: 4,
            })
        })
    }
}

pub struct Data {
    gltf: Gltf,
    buffer_data: Vec<Vec<u8>>,
    parent_path: PathBuf,
}

impl Data {
    pub fn new(path: &Path) -> Result<Self, Error> {
        let file = fs::File::open(path)?;
        let gltf = Gltf::from_reader(io::BufReader::new(file))?;
        let parent_path = path.parent().unwrap_or(path).to_owned();
        let buffer_data: Vec<_> = gltf
            .buffers()
            .map(|buffer| match buffer.source() {
                gltf::buffer::Source::Bin => gltf.blob.clone().ok_or(gltf::Error::MissingBlob),
                gltf::buffer::Source::Uri(uri) => {
                    let binary_path: PathBuf =
                        [parent_path.as_path(), Path::new(uri)].iter().collect();
                    fs::read(binary_path).map_err(gltf::Error::from)
                }
            })
            .collect::<Result<_, _>>()?;
        Ok(Self {
            buffer_data,
            parent_path,
            gltf,
        })
    }

    fn image(&self, source: gltf::image::Source) -> Result<image::DynamicImage, image::ImageError> {
        match source {
            gltf::image::Source::View { view, mime_type } => {
                let format =
                    image::ImageFormat::from_mime_type(mime_type).expect("invalid mime type");
                let input = self.buffer_data(&view, None, 0);
                image::load(io::Cursor::new(&input), format)
            }
            gltf::image::Source::Uri { uri, .. } => {
                let path: PathBuf = [self.parent_path.as_path(), Path::new(uri)]
                    .iter()
                    .collect();
                image::open(path)
            }
        }
    }

    fn buffer_data(
        &self,
        view: &gltf::buffer::View,
        byte_count: Option<usize>,
        offset: usize,
    ) -> &[u8] {
        let start = view.offset() + offset;
        let end = start + byte_count.unwrap_or(view.length() - offset);
        &self.buffer_data[view.buffer().index()][start..end]
    }

    fn accessor_data(&self, accessor: &gltf::Accessor) -> &[u8] {
        let view = accessor.view().unwrap();
        let bytes = accessor.count() * accessor.size();
        self.buffer_data(&view, Some(bytes), accessor.offset())
    }
}

fn load_material(
    data: &Data,
    scene: &mut Scene,
    fallback: &mut Fallback,
    material: gltf::Material,
) -> Result<Material, Error> {
    let mut load = |specs, accessor: Option<gltf::Texture>, transform: fn(&mut [u8])| {
        Ok::<u16, gltf::Error>(if let Some(accessor) = accessor {
            let image = data.image(accessor.source().source())?;
            scene.add_texture(create_texture(image, specs, true, transform))
        } else {
            fallback.insert(scene, specs)
        })
    };
    let accessor = material.pbr_metallic_roughness().base_color_texture();
    let albedo = load(&ALBEDO_SPECS, accessor.map(|i| i.texture()), |_| ())?;
    let accessor = material.emissive_texture();
    let emissive = load(&EMISSIVE_SPECS, accessor.map(|i| i.texture()), |_| ())?;
    let accessor = material.normal_texture();
    let normal = load(&NORMAL_SPECS, accessor.map(|i| i.texture()), |pixel| {
        let normal = Vec3::from_array(array::from_fn(|index| pixel[index] as f32 / 255.0));
        let uv = normal::encode_octahedron(normal * 2.0 - 1.0);
        pixel[0] = normal::quantize_unorm(uv.x, 8) as u8;
        pixel[1] = normal::quantize_unorm(uv.y, 8) as u8;
    })?;
    let accessor = material
        .pbr_metallic_roughness()
        .metallic_roughness_texture();
    let specular = load(&SPECULAR_SPECS, accessor.map(|i| i.texture()), |pixel| {
        pixel[0] = pixel[2]
    })?;
    let base_emissive =
        Vec3::from_array(material.emissive_factor()) * material.emissive_strength().unwrap_or(1.0);
    let base_color = material
        .pbr_metallic_roughness()
        .base_color_factor()
        .map(f16::from_f32);
    Ok(Material {
        metallic: f16::from_f32(material.pbr_metallic_roughness().metallic_factor()),
        roughness: f16::from_f32(material.pbr_metallic_roughness().roughness_factor()),
        ior: material.ior().map(f16::from_f32).unwrap_or(DEFAULT_IOR),
        emissive: base_emissive.to_array().map(f16::from_f32),
        albedo_texture: albedo,
        normal_texture: normal,
        specular_texture: specular,
        emissive_texture: emissive,
        base_color,
    })
}

fn load_indices(
    data: &Data,
    primitive: &gltf::Primitive,
    vertex_count: u32,
) -> Result<Vec<u32>, Error> {
    use gltf::accessor::{DataType, Dimensions};
    let Some(accessor) = primitive.indices() else {
        return Ok((0..vertex_count).collect());
    };
    let format_error = || Error::VertexFormat {
        data_type: accessor.data_type(),
        dimensions: accessor.dimensions(),
        expected_data_type: DataType::U32,
        expected_dimensions: Dimensions::Scalar,
        attribute: "index".to_owned(),
    };
    if accessor.dimensions() != Dimensions::Scalar {
        return Err(format_error());
    }
    let index_data = data.accessor_data(&accessor);
    let indices = match accessor.data_type() {
        DataType::U32 => index_data
            .chunks(4)
            .map(bytemuck::pod_read_unaligned)
            .collect(),
        DataType::U16 => index_data
            .chunks(2)
            .map(|bytes| bytemuck::pod_read_unaligned::<u16>(bytes) as u32)
            .collect(),
        _ => {
            return Err(format_error());
        }
    };
    Ok(indices)
}

fn fallback_material(scene: &mut Scene, fallback: &mut Fallback) -> u32 {
    let albedo = fallback.insert(scene, &ALBEDO_SPECS);
    let normal = fallback.insert(scene, &NORMAL_SPECS);
    let specular = fallback.insert(scene, &SPECULAR_SPECS);
    let emissive = fallback.insert(scene, &EMISSIVE_SPECS);
    scene.add_material(Material {
        albedo_texture: albedo,
        normal_texture: normal,
        specular_texture: specular,
        emissive_texture: emissive,
        base_color: DEFAULT_COLOR,
        emissive: DEFAULT_EMISSIVE,
        metallic: DEFAULT_METALLIC,
        roughness: DEFAULT_ROUGHNESS,
        ior: DEFAULT_IOR,
    })
}

fn read_from_accessor<T: AnyBitPattern>(data: &Data, accessor: &gltf::Accessor) -> Vec<T> {
    data.accessor_data(accessor)
        .chunks(mem::size_of::<T>())
        .map(bytemuck::pod_read_unaligned)
        .collect()
}

fn load_mesh(
    data: &Data,
    scene: &mut Scene,
    fallback: &mut Fallback,
    primitive: gltf::Primitive,
) -> Result<Mesh, Error> {
    use gltf::accessor::{DataType, Dimensions};
    let material = primitive
        .material()
        .index()
        .map(|material| material as u32)
        .unwrap_or_else(|| fallback_material(scene, fallback));
    let accessor = primitive.get(&gltf::Semantic::Positions).unwrap();
    verify_accessor("positions", &accessor, DataType::F32, Dimensions::Vec3)?;
    let positions = read_from_accessor::<Vec3>(data, &accessor);
    let indices = load_indices(data, &primitive, positions.len() as u32).unwrap();
    let normals = match primitive.get(&gltf::Semantic::Normals) {
        None => generate_normals(&positions, &indices),
        Some(accessor) => {
            verify_accessor("normals", &accessor, DataType::F32, Dimensions::Vec3)?;
            read_from_accessor(data, &accessor)
        }
    };
    let tex_coords = match primitive.get(&gltf::Semantic::TexCoords(0)) {
        None => vec![Vec2::ZERO; normals.len()],
        Some(accessor) => {
            verify_accessor(
                "texture coordinates",
                &accessor,
                DataType::F32,
                Dimensions::Vec2,
            )?;
            read_from_accessor(data, &accessor)
        }
    };
    let tangents = match primitive.get(&gltf::Semantic::Tangents) {
        None => generate_tangents(&positions, &tex_coords, &normals, &indices),
        Some(accessor) => {
            verify_accessor("tangents", &accessor, DataType::F32, Dimensions::Vec4)?;
            read_from_accessor(data, &accessor)
        }
    };
    let bounding_sphere = bounding_sphere(&primitive);
    let vertices = tex_coords
        .iter()
        .cloned()
        .zip(normals.iter())
        .zip(tangents.iter())
        .map(|((tex_coord, normal), tangent)| Vertex {
            tangent_frame: TangentFrame::new(*normal, *tangent),
            tex_coord: tex_coord.to_array().map(f16::from_f32),
        });
    extend_vec(
        &mut scene.positions,
        positions.iter().flat_map(|position| {
            let position = (*position - bounding_sphere.center) / bounding_sphere.radius;
            debug_assert!(position.max_element() <= 1.0);
            position
                .to_array()
                .map(|value| normal::quantize_snorm(value, 16) as i16)
        }),
    );
    Ok(Mesh {
        vertex_count: vertices.len() as u32,
        index_count: indices.len() as u32,
        vertex_offset: extend_vec(&mut scene.vertices, vertices) as u32,
        index_offset: extend_vec(&mut scene.indices, indices) as u32,
        bounding_sphere,
        material,
    })
}

fn load_model(
    data: &Data,
    scene: &mut Scene,
    fallback: &mut Fallback,
    mesh: gltf::Mesh,
) -> Result<Model, Error> {
    let meshes: Vec<_> = mesh.primitives().collect();
    let mesh_indices: Vec<u32> = meshes
        .into_iter()
        .map(|primitive| {
            let mesh = load_mesh(data, scene, fallback, primitive)?;
            Ok(scene.add_mesh(mesh))
        })
        .collect::<Result<_, Error>>()?;
    Ok(Model { mesh_indices })
}

pub fn load_scene(data: &Data) -> Result<Scene, Error> {
    let mut scene = Scene::default();
    let mut fallback = Fallback::default();
    scene.instances = load_instances(data.gltf.scenes().flat_map(|scene| scene.nodes()));
    scene.materials = data
        .gltf
        .materials()
        .map(|material| load_material(data, &mut scene, &mut fallback, material))
        .collect::<Result<_, _>>()?;
    scene.models = data
        .gltf
        .meshes()
        .map(|mesh| load_model(data, &mut scene, &mut fallback, mesh))
        .collect::<Result<_, _>>()?;
    Ok(scene)
}

fn bounding_sphere(primitive: &gltf::Primitive) -> BoundingSphere {
    let bounding_box = primitive.bounding_box();
    let [min, max] = [bounding_box.min, bounding_box.max].map(Vec3::from);
    let center = min + (max - min) * 0.5;
    BoundingSphere {
        radius: (center - max).length(),
        center,
    }
}

fn load_instances<'a>(nodes: impl Iterator<Item = gltf::Node<'a>>) -> Vec<Instance> {
    nodes
        .map(|node| Instance {
            model_index: node.mesh().map(|mesh| mesh.index() as u32),
            transform: Mat4::from_cols_array_2d(&node.transform().matrix()),
            children: load_instances(node.children()),
        })
        .collect()
}

struct TextureSpecs {
    kind: TextureKind,
    fallback: [u8; 4],
}

fn create_texture(
    mut image: image::DynamicImage,
    specs: &TextureSpecs,
    create_mips: bool,
    mut encode: impl FnMut(&mut [u8]) + Copy,
) -> Texture {
    let width = image.width().next_multiple_of(4);
    let height = image.height().next_multiple_of(4);
    let mip_level_count = if create_mips {
        let extent = u32::min(width, height) as f32;
        let count = extent.log2().floor() as u32;
        count.saturating_sub(2) + 1
    } else {
        1
    };
    let min_width = width >> (mip_level_count - 1);
    let min_height = height >> (mip_level_count - 1);
    assert!(
        min_width >= 4 && min_height >= 4,
        "smallest mip is too small: {min_width} x {min_height}",
    );
    let filter = image::imageops::FilterType::Lanczos3;
    let mips = (0..mip_level_count)
        .map(|level| {
            image = if level == 0 {
                image.resize_exact(width, height, filter)
            } else {
                let width = (image.width() / 2).next_multiple_of(4);
                let height = (image.height() / 2).next_multiple_of(4);
                image.resize_exact(width, height, filter)
            };
            let mut mip = image.clone().into_rgba8();
            mip.pixels_mut().for_each(|pixel| encode(&mut pixel.0));
            compress_bytes(specs.kind, mip.width(), mip.height(), &mip.into_raw())
                .into_boxed_slice()
        })
        .collect();
    Texture {
        kind: specs.kind,
        mips,
        width,
        height,
    }
}

fn compress_bytes(kind: TextureKind, width: u32, height: u32, bytes: &[u8]) -> Vec<u8> {
    let format = match kind {
        TextureKind::Albedo | TextureKind::Emissive => texpresso::Format::Bc1,
        TextureKind::Normal | TextureKind::Specular => texpresso::Format::Bc5,
    };
    let params = texpresso::Params {
        algorithm: texpresso::Algorithm::RangeFit,
        ..Default::default()
    };
    let size = format.compressed_size(width as usize, height as usize);
    let mut output = vec![0x0; size];
    format.compress(bytes, width as usize, height as usize, params, &mut output);
    output
}

fn verify_accessor(
    name: &str,
    accessor: &gltf::Accessor,
    data_type: gltf::accessor::DataType,
    dimensions: gltf::accessor::Dimensions,
) -> Result<(), Error> {
    let error = || Error::VertexFormat {
        expected_data_type: data_type,
        expected_dimensions: dimensions,
        data_type: accessor.data_type(),
        dimensions: accessor.dimensions(),
        attribute: name.to_owned(),
    };
    if accessor.data_type() != data_type || accessor.dimensions() != dimensions {
        Err(error())
    } else {
        Ok(())
    }
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
    tex_coords: &[Vec2],
    normals: &[Vec3],
    indices: &[u32],
) -> Vec<Vec4> {
    let tangents = vec![Vec4::ZERO; positions.len()];
    let mut generator = TangentGenerator {
        positions,
        tex_coords,
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

fn extend_vec<T>(vec: &mut Vec<T>, items: impl IntoIterator<Item = T>) -> usize {
    let index = vec.len();
    vec.extend(items);
    index
}

const ALBEDO_SPECS: TextureSpecs = TextureSpecs {
    kind: TextureKind::Albedo,
    fallback: [u8::MAX; 4],
};

const NORMAL_SPECS: TextureSpecs = TextureSpecs {
    kind: TextureKind::Normal,
    // Octahedron encoded normal pointing straight out.
    fallback: [128; 4],
};

const SPECULAR_SPECS: TextureSpecs = TextureSpecs {
    kind: TextureKind::Specular,
    fallback: [u8::MAX; 4],
};

const EMISSIVE_SPECS: TextureSpecs = TextureSpecs {
    kind: TextureKind::Emissive,
    fallback: [u8::MAX; 4],
};

const DEFAULT_IOR: f16 = f16::from_f32_const(1.4);
const DEFAULT_METALLIC: f16 = f16::ZERO;
const DEFAULT_ROUGHNESS: f16 = f16::ONE;

const DEFAULT_COLOR: [f16; 4] = [f16::ONE; 4];
const DEFAULT_EMISSIVE: [f16; 3] = [f16::ZERO; 3];
