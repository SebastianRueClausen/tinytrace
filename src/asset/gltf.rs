use std::{
    array,
    ffi::OsStr,
    fs, io, mem,
    path::{Path, PathBuf},
};

use bytemuck::AnyBitPattern;
use glam::{Mat4, Vec2, Vec3, Vec4};
use gltf::accessor::{DataType, Dimensions};
use gltf::Gltf;
use tinytrace_backend::{Extent, ImageFormat};

use crate::{
    asset::{Mesh, Param},
    math,
};

use super::{Instance, Material, Model, ProcessedTexture, Scene, SceneImporter, Texture};

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("failed parsing gltf file: {0}")]
    Gltf(#[from] gltf::Error),
    #[error("failed loading file: {0}")]
    Load(#[from] io::Error),
    #[error("failed decoding image: {0}")]
    LoadImage(#[from] png::DecodingError),
    #[error(
        "unsupported format {data_type:?} {dimensions:?} expected \
        {expected_data_type:?} {expected_dimensions:?} for {attribute}"
    )]
    VertexFormat {
        data_type: gltf::accessor::DataType,
        dimensions: gltf::accessor::Dimensions,
        expected_data_type: gltf::accessor::DataType,
        expected_dimensions: gltf::accessor::Dimensions,
        attribute: String,
    },
}

pub struct GltfImporter {
    gltf: Gltf,
    buffer_data: Vec<Vec<u8>>,
    parent_path: PathBuf,
}

impl GltfImporter {
    pub fn new<P: AsRef<Path>>(path: P) -> Result<Self, Error> {
        let file = fs::File::open(path.as_ref())?;
        let gltf = Gltf::from_reader(io::BufReader::new(file))?;
        let parent_path = path.as_ref().parent().unwrap_or(path.as_ref()).to_owned();
        let buffer_data: Vec<_> = gltf
            .buffers()
            .map(|buffer| match buffer.source() {
                gltf::buffer::Source::Bin => todo!("binary GLTF not supported yet"),
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

    fn buffer_data(
        &self,
        view: &gltf::buffer::View,
        size: usize,
        count: usize,
        offset: usize,
    ) -> impl Iterator<Item = u8> + '_ {
        self.buffer_data[view.buffer().index()][view.offset() + offset..]
            .chunks(view.stride().unwrap_or(size))
            .take(count)
            .flat_map(move |chunk| &chunk[..size])
            .copied()
    }

    fn image_data(&self, source: gltf::image::Source) -> Result<Texture<[u8; 4]>, Error> {
        let bytes: Vec<u8> = match source {
            gltf::image::Source::View { view, mime_type } => {
                assert_eq!(
                    mime_type, "image/png",
                    "only png images are supported for now"
                );
                self.buffer_data(&view, 1, view.length(), 0).collect()
            }
            gltf::image::Source::Uri { uri, .. } => {
                let path: PathBuf = [self.parent_path.as_path(), Path::new(uri)]
                    .iter()
                    .collect();
                if let Some(extension) = path.extension().and_then(OsStr::to_str) {
                    assert_eq!(extension, "png", "only png images are supported for now");
                }
                fs::read(&path)?
            }
        };
        load_png(&bytes)
    }

    fn accessor_data(&self, accessor: &gltf::Accessor) -> impl Iterator<Item = u8> + '_ {
        let view = accessor.view().unwrap();
        self.buffer_data(&view, accessor.size(), accessor.count(), accessor.offset())
    }

    fn read_from_accessor<T: AnyBitPattern>(&self, accessor: &gltf::Accessor) -> Vec<T> {
        let data: Vec<u8> = self.accessor_data(accessor).collect();
        // TODO: It may be possible to simply cast the vector here. It depends on the alignment guarantees.
        data.chunks(mem::size_of::<T>())
            .map(bytemuck::pod_read_unaligned)
            .collect()
    }
}

fn load_png(data: &[u8]) -> Result<Texture<[u8; 4]>, Error> {
    let mut decoder = png::Decoder::new(data);
    decoder.set_transformations(
        png::Transformations::ALPHA | png::Transformations::normalize_to_color8(),
    );

    let mut reader = decoder.read_info()?;
    let mut buffer = vec![[0u8; 4]; reader.output_buffer_size() / 4];

    let info = reader.next_frame(bytemuck::cast_slice_mut(&mut buffer))?;
    buffer.truncate(info.buffer_size());

    Ok(Texture {
        extent: Extent::new(info.width, info.height),
        data: buffer.into_boxed_slice(),
    })
}

fn load_indices(
    data: &GltfImporter,
    primitive: &gltf::Primitive,
) -> Option<Result<Vec<u32>, Error>> {
    let accessor = primitive.indices()?;
    let index_data: Vec<u8> = data.accessor_data(&accessor).collect();
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
            return Some(Err(Error::VertexFormat {
                data_type: accessor.data_type(),
                dimensions: accessor.dimensions(),
                expected_data_type: DataType::U32,
                expected_dimensions: Dimensions::Scalar,
                attribute: "index".to_owned(),
            }));
        }
    };
    Some(Ok(indices))
}

fn load_positions(data: &GltfImporter, primitive: &gltf::Primitive) -> Result<Vec<Vec3>, Error> {
    let accessor = primitive.get(&gltf::Semantic::Positions).unwrap();
    verify_accessor("positions", &accessor, DataType::F32, Dimensions::Vec3)?;
    Ok(data.read_from_accessor(&accessor))
}

fn load_texture_coordinates(
    data: &GltfImporter,
    primitive: &gltf::Primitive,
    vertex_count: usize,
) -> Result<Vec<Vec2>, Error> {
    match primitive.get(&gltf::Semantic::TexCoords(0)) {
        // It's not clear what to do here. You need texture coordinates to generate tangents by
        // the GLTF specs.
        None => Ok(vec![Vec2::ZERO; vertex_count]),
        Some(accessor) => {
            verify_accessor(
                "texture coordinates",
                &accessor,
                DataType::F32,
                Dimensions::Vec2,
            )?;
            Ok(data.read_from_accessor(&accessor))
        }
    }
}

fn load_mesh(
    data: &GltfImporter,
    scene: &mut Scene,
    primitive: &gltf::Primitive,
) -> Result<u32, Error> {
    let positions = load_positions(data, primitive)?;
    let texture_coordinates = load_texture_coordinates(data, primitive, positions.len())?;
    let indices = load_indices(data, &primitive).transpose()?;
    let normals = primitive
        .get(&gltf::Semantic::Normals)
        .map(|accessor| {
            verify_accessor("normals", &accessor, DataType::F32, Dimensions::Vec3)
                .map(|_| data.read_from_accessor(&accessor))
        })
        .transpose()?;
    let tangents = primitive
        .get(&gltf::Semantic::Tangents)
        .map(|accessor| {
            verify_accessor("tangents", &accessor, DataType::F32, Dimensions::Vec4)
                .map(|_| data.read_from_accessor(&accessor))
        })
        .transpose()?;
    Ok(scene.insert_mesh(&Mesh {
        positions: &positions,
        texture_coordinates: &texture_coordinates,
        indices: indices.as_deref(),
        normals: normals.as_deref(),
        tangents: tangents.as_deref(),
    }))
}

fn load_material(
    data: &GltfImporter,
    scene: &mut Scene,
    gltf: gltf::Material,
) -> Result<u32, Error> {
    let mut material = Material::default();

    // Base color.
    let base_color = Vec4::from_array(gltf.pbr_metallic_roughness().base_color_factor());

    if let Some(texture) = gltf
        .pbr_metallic_roughness()
        .base_color_texture()
        .map(|info| info.texture())
    {
        material.base.color = Param::Texture(
            data.image_data(texture.source().source())?
                .transform(|rgba| {
                    (base_color
                        * Vec4::from_array(rgba.map(|v| math::dequantize_unorm(v.into(), 8))))
                    .truncate()
                }),
        );
    } else {
        material.base.color = Param::Constant(base_color.truncate());
    }

    // Metallic roughness.
    let metalness = gltf.pbr_metallic_roughness().metallic_factor();
    let roughness = gltf.pbr_metallic_roughness().roughness_factor();

    if let Some(texture) = gltf
        .pbr_metallic_roughness()
        .metallic_roughness_texture()
        .map(|info| info.texture())
    {
        let image = data.image_data(texture.source().source())?;
        material.base.metalness = Param::Texture(
            image.transform(|rgba| math::dequantize_unorm(rgba[2].into(), 8) * metalness),
        );
        material.specular.roughness = Param::Texture(
            image.transform(|rgba| math::dequantize_unorm(rgba[1].into(), 8) * roughness),
        );
    } else {
        material.base.metalness = metalness.into();
        material.specular.roughness = roughness.into();
    }

    // Normal.
    if let Some(texture) = gltf.normal_texture().map(|info| info.texture()) {
        let image = data.image_data(texture.source().source())?;
        material.geometry.normal = Param::Texture(image.transform(|rgba| {
            let normal = Vec4::from_array(array::from_fn(|index| {
                math::dequantize_unorm(rgba[index].into(), 8)
            }))
            .truncate();
            normal * 2.0 - 1.0
        }));
    }

    // Emissive.
    let emissive_factor = Vec3::from_array(gltf.emissive_factor());
    if let Some(texture) = gltf.emissive_texture().map(|info| info.texture()) {
        let image = data.image_data(texture.source().source())?;
        material.emission.color = Param::Texture(image.transform(|rgba| {
            emissive_factor
                * Vec4::from_array(rgba.map(|v| math::dequantize_unorm(v.into(), 8))).truncate()
        }));
    } else {
        material.emission.color = emissive_factor.into();
    }

    if let Some(strength) = gltf.emissive_strength() {
        material.emission.luminance = strength.into();
    }

    // Specular.
    if let Some(specular) = gltf.specular() {
        let specular_factor = specular.specular_factor();
        let specular_color = Vec3::from_array(specular.specular_color_factor());

        if let Some(texture) = specular.specular_texture().map(|info| info.texture()) {
            let image = data.image_data(texture.source().source())?;
            material.specular.weight = Param::Texture(
                image.transform(|rgba| math::dequantize_unorm(rgba[3].into(), 8) * specular_factor),
            );
        } else {
            material.specular.weight = specular_factor.into();
        }

        if let Some(texture) = specular.specular_color_texture().map(|info| info.texture()) {
            let image = data.image_data(texture.source().source())?;
            material.specular.color = Param::Texture(image.transform(|rgba| {
                specular_color
                    * Vec4::from_array(rgba.map(|v| math::dequantize_unorm(v.into(), 8))).truncate()
            }));
        } else {
            material.specular.color = specular_color.into();
        }
    }

    // IOR.
    if let Some(ior) = gltf.ior() {
        material.specular.ior = ior.into();
    }

    // Transmission.
    if let Some(transmission) = gltf.transmission() {
        let transmission_factor = transmission.transmission_factor();
        if let Some(texture) = transmission
            .transmission_texture()
            .map(|info| info.texture())
        {
            let image = data.image_data(texture.source().source())?;
            material.specular.weight =
                Param::Texture(image.transform(|rgba| {
                    transmission_factor * math::dequantize_unorm(rgba[0].into(), 8)
                }));
        } else {
            material.transmission.weight = transmission_factor.into();
        }

        material.transmission.color = match &material.base.color {
            Param::Constant(value) => Param::Constant(1.0 - *value),
            Param::Texture(texture) => Param::Texture(texture.transform(|rgb| 1.0 - *rgb)),
        };
    }

    Ok(scene.insert_material(&material))
}

fn load_models(
    data: &GltfImporter,
    scene: &mut Scene,
    mesh: gltf::Mesh,
) -> Result<Vec<Model>, Error> {
    mesh.primitives()
        .map(|primitive| {
            Ok(Model {
                mesh_index: load_mesh(data, scene, &primitive)?,
                material_index: load_material(data, scene, primitive.material())?,
            })
        })
        .collect()
}

fn load_instances<'a>(
    nodes: impl Iterator<Item = gltf::Node<'a>>,
    data: &GltfImporter,
    scene: &mut Scene,
) -> Result<Vec<Instance>, Error> {
    nodes
        .map(|node| {
            Ok(Instance {
                models: node
                    .mesh()
                    .map(|mesh| load_models(data, scene, mesh))
                    .transpose()?
                    .unwrap_or_default(),
                transform: Mat4::from_cols_array_2d(&node.transform().matrix()),
                children: load_instances(node.children(), data, scene)?,
            })
        })
        .collect()
}

impl SceneImporter for GltfImporter {
    type Error = Error;

    fn insert_in_scene(&self, scene: &mut Scene) -> Result<(), Self::Error> {
        let mut instances = load_instances(self.gltf.nodes(), &self, scene)?;
        scene.textures.push(ProcessedTexture {
            data: Box::new([vec![0x0; 32 * 32].into_boxed_slice()]),
            extent: Extent::new(32, 32),
            format: ImageFormat::R8Unorm,
        });
        scene.root.children.append(&mut instances);
        Ok(())
    }
}

fn verify_accessor(
    name: &str,
    accessor: &gltf::Accessor,
    data_type: gltf::accessor::DataType,
    dimensions: gltf::accessor::Dimensions,
) -> Result<(), Error> {
    if accessor.data_type() != data_type || accessor.dimensions() != dimensions {
        Err(Error::VertexFormat {
            expected_data_type: data_type,
            expected_dimensions: dimensions,
            data_type: accessor.data_type(),
            dimensions: accessor.dimensions(),
            attribute: name.to_owned(),
        })
    } else {
        Ok(())
    }
}
