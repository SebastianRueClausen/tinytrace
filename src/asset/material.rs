use super::{
    texture::{ProcessedTexture, Texture},
    Scene,
};
use glam::{Vec3, Vec4};

#[derive(Debug, Clone)]
pub enum Param<T> {
    Constant(T),
    Texture(Texture<T>),
}

impl<T> From<T> for Param<T> {
    fn from(value: T) -> Self {
        Self::Constant(value)
    }
}

impl<T: Default + Copy> Param<T> {
    fn constant_or_default(&self) -> T {
        match self {
            Param::Constant(constant) => *constant,
            Param::Texture(_) => T::default(),
        }
    }

    pub fn texture(&self) -> Option<&Texture<T>> {
        match self {
            Param::Texture(texture) => Some(texture),
            Param::Constant(_) => None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Base {
    pub weight: Param<f32>,
    pub color: Param<Vec3>,
    pub metalness: Param<f32>,
    pub diffuse_roughness: Param<f32>,
}

impl Default for Base {
    fn default() -> Self {
        Self {
            weight: 1.0.into(),
            color: Vec3::splat(0.8).into(),
            metalness: 0.0.into(),
            diffuse_roughness: 0.0.into(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct Specular {
    pub weight: Param<f32>,
    pub color: Param<Vec3>,
    pub roughness: Param<f32>,
    pub roughness_anisotropy: Param<f32>,
    pub ior: Param<f32>,
    pub rotation: Param<f32>,
}

impl Default for Specular {
    fn default() -> Self {
        Self {
            weight: 1.0.into(),
            color: Vec3::ONE.into(),
            roughness: 0.3.into(),
            roughness_anisotropy: 0.0.into(),
            ior: 1.5.into(),
            rotation: 0.0.into(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct Transmission {
    pub weight: Param<f32>,
    pub color: Param<Vec3>,
    pub depth: Param<f32>,
}

impl Default for Transmission {
    fn default() -> Self {
        Self {
            weight: 0.0.into(),
            color: Vec3::ONE.into(),
            depth: 0.0.into(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct Coat {
    pub weight: Param<f32>,
    pub color: Param<Vec3>,
    pub roughness: Param<f32>,
    pub roughness_anisotropy: Param<f32>,
    pub ior: Param<f32>,
    pub darkening: Param<f32>,
    pub rotation: Param<f32>,
}

impl Default for Coat {
    fn default() -> Self {
        Self {
            weight: 0.0.into(),
            color: Vec3::ONE.into(),
            roughness: 0.0.into(),
            roughness_anisotropy: 0.0.into(),
            ior: 1.6.into(),
            darkening: 1.0.into(),
            rotation: 0.0.into(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct Fuzz {
    pub weight: Param<f32>,
    pub color: Param<Vec3>,
    pub roughness: Param<f32>,
}

impl Default for Fuzz {
    fn default() -> Self {
        Self {
            weight: 0.0.into(),
            color: Vec3::ONE.into(),
            roughness: 0.5.into(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct Emission {
    pub luminance: Param<f32>,
    pub color: Param<Vec3>,
}

impl Default for Emission {
    fn default() -> Self {
        Self {
            luminance: 0.0.into(),
            color: Vec3::ONE.into(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct Geometry {
    pub opacity: Param<f32>,
    pub normal: Param<Vec3>,
}

impl Default for Geometry {
    fn default() -> Self {
        Self {
            opacity: 1.0.into(),
            normal: Vec3::Z.into(),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct Material {
    pub base: Base,
    pub specular: Specular,
    pub transmission: Transmission,
    pub coat: Coat,
    pub fuzz: Fuzz,
    pub emission: Emission,
    pub geometry: Geometry,
}

impl Material {
    pub(crate) fn is_emissive(&self) -> bool {
        match &self.emission.luminance {
            Param::Constant(value) => *value > 0.0,
            Param::Texture(texture) => texture.data.iter().any(|value| *value > 0.0),
        }
    }
}

macro_rules! define_material_structs {
    ( $( $field:ident: $type:ty, )*) => {
        #[repr(C)]
        #[repr(align(16))]
        #[derive(Debug, Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
        pub struct MaterialConstants {
            $(pub $field: $type, )*
        }
        #[repr(C)]
        #[repr(align(4))]
        #[derive(Debug, Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
        pub struct MaterialTextures {
            $(pub $field: u32, )*
        }
    };
}

define_material_structs! {
    base_color: Vec4,
    specular_color: Vec4,
    transmission_color: Vec4,
    coat_color: Vec4,
    fuzz_color: Vec4,
    emission_color: Vec4,
    geometry_normal: Vec4,
    base_weight: f32,
    base_metalness: f32,
    base_diffuse_roughness: f32,
    specular_weight: f32,
    specular_roughness: f32,
    specular_roughness_anisotropy: f32,
    specular_ior: f32,
    specular_rotation: f32,
    transmission_weight: f32,
    transmission_depth: f32,
    coat_weight: f32,
    coat_roughness: f32,
    coat_roughness_anisotropy: f32,
    coat_ior: f32,
    coat_darkening: f32,
    coat_rotation: f32,
    fuzz_weight: f32,
    fuzz_roughness: f32,
    emission_luminance: f32,
    geometry_opacity: f32,
}

impl Material {
    pub fn constants(&self) -> MaterialConstants {
        let vec3 = |vector: &Param<Vec3>| vector.constant_or_default().extend(0.0);
        MaterialConstants {
            base_color: vec3(&self.base.color),
            specular_color: vec3(&self.specular.color),
            transmission_color: vec3(&self.transmission.color),
            coat_color: vec3(&self.coat.color),
            fuzz_color: vec3(&self.fuzz.color),
            emission_color: vec3(&self.emission.color),
            geometry_normal: vec3(&self.geometry.normal),
            base_weight: self.base.weight.constant_or_default(),
            base_metalness: self.base.metalness.constant_or_default(),
            base_diffuse_roughness: self.base.diffuse_roughness.constant_or_default(),
            specular_weight: self.specular.weight.constant_or_default(),
            specular_roughness: self.specular.roughness.constant_or_default(),
            specular_roughness_anisotropy: self.specular.roughness_anisotropy.constant_or_default(),
            specular_ior: self.specular.ior.constant_or_default(),
            transmission_weight: self.transmission.weight.constant_or_default(),
            transmission_depth: self.transmission.depth.constant_or_default(),
            coat_weight: self.coat.weight.constant_or_default(),
            coat_roughness: self.coat.roughness.constant_or_default(),
            coat_roughness_anisotropy: self.coat.roughness_anisotropy.constant_or_default(),
            coat_ior: self.coat.ior.constant_or_default(),
            coat_darkening: self.coat.darkening.constant_or_default(),
            fuzz_weight: self.fuzz.weight.constant_or_default(),
            fuzz_roughness: self.fuzz.roughness.constant_or_default(),
            emission_luminance: self.emission.luminance.constant_or_default(),
            geometry_opacity: self.geometry.opacity.constant_or_default(),
            specular_rotation: self.specular.rotation.constant_or_default().into(),
            coat_rotation: self.coat.rotation.constant_or_default().into(),
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
pub(crate) struct ProcessedMaterial {
    pub constants: MaterialConstants,
    pub textures: MaterialTextures,
    pub padding: u32,
}

impl Scene {
    fn process_and_insert_texture<T, P>(&mut self, texture: Option<&Texture<T>>, process: P) -> u32
    where
        P: FnOnce(&Texture<T>) -> ProcessedTexture,
    {
        texture
            .map(|texture| {
                self.textures.push(process(texture));
                self.textures.len() as u32 - 1
            })
            .unwrap_or(NULL_TEXTURE_INDEX)
    }

    fn insert_normal_texture(&mut self, texture: Option<&Texture<Vec3>>) -> u32 {
        self.process_and_insert_texture(texture, ProcessedTexture::from_normal_map)
    }

    fn insert_color_texture(&mut self, texture: Option<&Texture<Vec3>>) -> u32 {
        self.process_and_insert_texture(texture, ProcessedTexture::from_color_texture)
    }

    fn insert_scalar_unorm_texture(&mut self, texture: Option<&Texture<f32>>) -> u32 {
        self.process_and_insert_texture(texture, ProcessedTexture::from_scalar_unorm_texture)
    }

    fn insert_scalar_float_texture(&mut self, texture: Option<&Texture<f32>>) -> u32 {
        self.process_and_insert_texture(texture, ProcessedTexture::from_scalar_float_texture)
    }

    pub fn insert_material(&mut self, material: &Material) -> u32 {
        let index = self.materials.len() as u32;
        if material.is_emissive() {
            self.emissive_materials.insert(index);
        }

        let textures = MaterialTextures {
            base_color: self.insert_color_texture(material.base.color.texture()),
            specular_color: self.insert_color_texture(material.specular.color.texture()),
            transmission_color: self.insert_color_texture(material.transmission.color.texture()),
            coat_color: self.insert_color_texture(material.coat.color.texture()),
            fuzz_color: self.insert_color_texture(material.fuzz.color.texture()),
            emission_color: self.insert_color_texture(material.emission.color.texture()),
            geometry_normal: self.insert_normal_texture(material.geometry.normal.texture()),
            base_weight: self.insert_scalar_unorm_texture(material.base.weight.texture()),
            base_metalness: self.insert_scalar_unorm_texture(material.base.metalness.texture()),
            base_diffuse_roughness: self
                .insert_scalar_unorm_texture(material.base.diffuse_roughness.texture()),
            specular_weight: self.insert_scalar_unorm_texture(material.specular.weight.texture()),
            specular_roughness: self
                .insert_scalar_unorm_texture(material.specular.roughness.texture()),
            specular_roughness_anisotropy: self
                .insert_scalar_unorm_texture(material.specular.roughness_anisotropy.texture()),
            transmission_weight: self
                .insert_scalar_unorm_texture(material.transmission.weight.texture()),
            coat_weight: self.insert_scalar_unorm_texture(material.coat.weight.texture()),
            coat_roughness: self.insert_scalar_unorm_texture(material.coat.roughness.texture()),
            coat_roughness_anisotropy: self
                .insert_scalar_unorm_texture(material.coat.roughness_anisotropy.texture()),
            coat_darkening: self.insert_scalar_unorm_texture(material.coat.darkening.texture()),
            fuzz_weight: self.insert_scalar_unorm_texture(material.fuzz.weight.texture()),
            fuzz_roughness: self.insert_scalar_unorm_texture(material.fuzz.roughness.texture()),
            geometry_opacity: self.insert_scalar_unorm_texture(material.geometry.opacity.texture()),
            emission_luminance: self
                .insert_scalar_float_texture(material.emission.luminance.texture()),
            coat_ior: self.insert_scalar_float_texture(material.coat.ior.texture()),
            specular_ior: self.insert_scalar_float_texture(material.specular.ior.texture()),
            transmission_depth: self
                .insert_scalar_float_texture(material.transmission.depth.texture()),
            // FIXME: This should not be float textures.
            specular_rotation: self
                .insert_scalar_float_texture(material.specular.rotation.texture()),
            coat_rotation: self.insert_scalar_float_texture(material.coat.rotation.texture()),
        };

        self.materials.push(ProcessedMaterial {
            constants: material.constants(),
            textures,
            padding: 0,
        });

        index
    }
}

pub(super) const NULL_TEXTURE_INDEX: u32 = u32::MAX;
