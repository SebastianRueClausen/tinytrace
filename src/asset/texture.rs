use std::{iter::Sum, ops::Div};

use glam::Vec3;
use half::f16;
use tinytrace_backend::{Extent, ImageFormat};

use crate::math;

#[derive(Clone, Debug)]
pub struct Texture<T> {
    pub extent: Extent,
    pub data: Box<[T]>,
}

impl<T> Texture<T> {
    pub fn texel(&self, x: u32, y: u32) -> &T {
        &self.data[(y * self.extent.width + x) as usize]
    }

    pub fn set_texel(&mut self, x: u32, y: u32, value: T) {
        self.data[(y * self.extent.width + x) as usize] = value;
    }

    pub fn transform<R, F: FnMut(&T) -> R>(&self, transform: F) -> Texture<R> {
        Texture {
            extent: self.extent,
            data: self.data.iter().map(transform).collect(),
        }
    }
}

impl<T: Default + Clone> Texture<T> {
    pub fn empty(extent: Extent) -> Self {
        Self {
            data: vec![T::default(); (extent.width * extent.height) as usize].into_boxed_slice(),
            extent,
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct ProcessedTexture {
    pub data: Box<[Box<[u8]>]>,
    pub extent: Extent,
    pub format: ImageFormat,
}

impl ProcessedTexture {
    fn new(data: Box<[Box<[u8]>]>, format: ImageFormat, extent: Extent) -> Self {
        Self {
            format,
            extent,
            data,
        }
    }

    fn process_texture<T, P>(texture: &Texture<T>, process: P) -> Box<[Box<[u8]>]>
    where
        T: Copy + Default + Sum<T> + Div<f32, Output = T>,
        P: Fn(Texture<T>) -> Box<[u8]>,
    {
        generate_mip_chain(texture.clone())
            .into_iter()
            .map(process)
            .collect()
    }

    pub fn from_scalar_float_texture(texture: &Texture<f32>) -> Self {
        let data = Self::process_texture(texture, |Texture { data, .. }| {
            data.iter()
                .flat_map(|value| f16::from_f32(*value).to_le_bytes())
                .collect()
        });
        Self::new(data, ImageFormat::R16Float, texture.extent)
    }

    pub fn from_scalar_unorm_texture(texture: &Texture<f32>) -> Self {
        let data = Self::process_texture(texture, |Texture { data, .. }| {
            data.iter()
                .map(|value| math::quantize_unorm(*value, 8) as u8)
                .collect()
        });
        Self::new(data, ImageFormat::R8Unorm, texture.extent)
    }

    pub fn from_color_texture(texture: &Texture<Vec3>) -> Self {
        let data = Self::process_texture(texture, |Texture { data, extent }| {
            let rgba: Vec<u8> = data
                .iter()
                .flat_map(|color| {
                    let color = color
                        .extend(1.0)
                        .to_array()
                        .map(|float| math::quantize_unorm(float, 8) as u8);
                    color
                })
                .collect();
            compress_bytes(texpresso::Format::Bc1, extent, &rgba)
        });
        Self::new(data, ImageFormat::RgbaBc1Srgb, texture.extent)
    }

    pub fn from_normal_map(texture: &Texture<Vec3>) -> Self {
        let data = Self::process_texture(texture, |Texture { data, extent }| {
            let rgba: Vec<u8> = data
                .iter()
                .flat_map(|normal| {
                    let encoded = math::encode_octahedron(*normal);
                    let r = math::quantize_unorm(encoded.x, 8) as u8;
                    let g = math::quantize_unorm(encoded.y, 8) as u8;
                    [r, g, 0, 0]
                })
                .collect();
            compress_bytes(texpresso::Format::Bc5, extent, &rgba)
        });
        Self::new(data, ImageFormat::RgBc5Unorm, texture.extent)
    }
}

fn generate_mip_chain<T>(texture: Texture<T>) -> Vec<Texture<T>>
where
    T: Default + Sum<T> + Div<f32, Output = T> + Copy,
{
    let mut chain = vec![texture];
    loop {
        let last = chain.last().unwrap();
        if last.extent.width < 2 || last.extent.height < 2 {
            break chain;
        }
        let mut output: Texture<T> = Texture::empty(last.extent.mip_level(1));
        for y in 0..output.extent.height {
            for x in 0..output.extent.width {
                let sum = [(0, 0), (0, 1), (1, 0), (1, 1)]
                    .map(|(x_offset, y_offset)| *last.texel(x * 2 + x_offset, y * 2 + y_offset))
                    .into_iter()
                    .sum::<T>();
                output.set_texel(x, y, sum / 4.0);
            }
        }
        chain.push(output);
    }
}

fn compress_bytes(format: texpresso::Format, extent: Extent, rgba: &[u8]) -> Box<[u8]> {
    let (width, height) = (extent.width as usize, extent.height as usize);
    let mut output = vec![0x0; format.compressed_size(width, height)];
    let params = texpresso::Params {
        algorithm: texpresso::Algorithm::ClusterFit,
        ..Default::default()
    };
    format.compress(rgba, width, height, params, &mut output);
    output.into_boxed_slice()
}
