#![warn(clippy::all)]

use std::{borrow::Cow, mem};

use glam::{UVec2, Vec2};
use tinytrace_backend::{
    binding, Binding, BindingType, Buffer, BufferRequest, BufferType, BufferWrite, Context, Error,
    Extent, Filter, Handle, Image, ImageFormat, ImageRequest, ImageWrite, Lifetime, MemoryLocation,
    Offset, Sampler, SamplerRequest, Shader, ShaderRequest,
};

#[repr(C)]
#[derive(bytemuck::NoUninit, Clone, Copy)]
struct DrawParameters {
    screen_size_in_points: Vec2,
    area_offset: UVec2,
    area_size: UVec2,
    texture_index: u32,
    vertex_offset: i32,
    index_start: u32,
    index_count: u32,
}

pub struct RenderRequest {
    pub primitives: Vec<egui::ClippedPrimitive>,
    pub textures_delta: egui::TexturesDelta,
    pub pixels_per_point: f32,
}

pub struct Renderer {
    sampler: Handle<Sampler>,
    draw: Handle<Shader>,
    textures: Vec<(epaint::TextureId, Handle<Image>)>,
}

impl Renderer {
    pub fn new(context: &mut Context, target_format: ImageFormat) -> Result<Self, Error> {
        let bindings = &[
            binding!(storage_buffer, Vertex, vertices, true, false),
            binding!(storage_buffer, uint, indices, true, false),
            binding!(storage_image, target_format, render_target, None, true),
            binding!(sampled_image, textures, Some(1024)),
        ];
        let draw = context.create_shader(
            Lifetime::Static,
            &ShaderRequest {
                source: include_str!("draw.glsl"),
                block_size: Extent::new(32, 32),
                push_constant_size: Some(mem::size_of::<DrawParameters>() as u32),
                bindings,
            },
        )?;
        let sampler = context.create_sampler(
            Lifetime::Static,
            &SamplerRequest {
                filter: Filter::Linear,
                max_anisotropy: None,
                clamp_to_edge: true,
            },
        )?;
        Ok(Self {
            textures: Vec::new(),
            draw,
            sampler,
        })
    }

    fn texture_index(&self, id: epaint::TextureId) -> usize {
        self.textures
            .iter()
            .position(|(texture_id, _)| *texture_id == id)
            .expect("invalid texture id")
    }

    pub fn render(
        &mut self,
        context: &mut Context,
        target: &Handle<Image>,
        request: &RenderRequest,
    ) -> Result<(), Error> {
        let target_extent = context.image(target).extent();
        let screen_size_in_points =
            Vec2::new(target_extent.width as f32, target_extent.height as f32)
                / request.pixels_per_point;
        for id in &request.textures_delta.free {
            // FIXME: Memory leak.
            self.textures.swap_remove(self.texture_index(*id));
        }
        let image_writes: Vec<_> = request
            .textures_delta
            .set
            .iter()
            .map(|(id, delta)| {
                let (bytes, width, height) = match &delta.image {
                    egui::ImageData::Color(image) => {
                        let bytes = bytemuck::cast_slice::<_, u8>(&image.pixels).to_vec();
                        (bytes, image.width() as u32, image.height() as u32)
                    }
                    egui::ImageData::Font(image) => {
                        let bytes = image
                            .srgba_pixels(None)
                            .flat_map(|pixel| pixel.to_array())
                            .collect();
                        (bytes, image.width() as u32, image.height() as u32)
                    }
                };
                let extent = Extent::new(width, height);
                let (offset, image) = if let Some([x, y]) = delta.pos {
                    let (_, image) = &self.textures[self.texture_index(*id)];
                    (Offset::new(x as i32, y as i32), image.clone())
                } else {
                    let image = create_texture(context, extent)?;
                    self.textures.push((*id, image.clone()));
                    (Offset::default(), image)
                };
                Ok(ImageWrite {
                    mips: Cow::Owned(vec![bytes.into_boxed_slice()]),
                    image,
                    offset,
                    extent,
                })
            })
            .collect::<Result<_, Error>>()?;
        context.write_images(&image_writes)?;
        let (mut vertices, mut indices, mut draws) =
            (Vec::<u8>::new(), Vec::<u8>::new(), Vec::new());
        for primitive in &request.primitives {
            let (area_offset, area_size) = area_offset_size(
                &primitive.clip_rect,
                request.pixels_per_point,
                target_extent,
            );
            match &primitive.primitive {
                epaint::Primitive::Callback(_) => todo!(),
                epaint::Primitive::Mesh(mesh) => {
                    let vertex_offset = (vertices.len() / mem::size_of::<epaint::Vertex>()) as i32;
                    let index_start = (indices.len() / mem::size_of::<u32>()) as u32;
                    vertices.extend_from_slice(bytemuck::cast_slice(&mesh.vertices));
                    indices.extend_from_slice(bytemuck::cast_slice(&mesh.indices));
                    draws.push(DrawParameters {
                        texture_index: self.texture_index(mesh.texture_id) as u32,
                        index_count: mesh.indices.len() as u32,
                        screen_size_in_points,
                        vertex_offset,
                        index_start,
                        area_offset,
                        area_size,
                    });
                }
            }
        }
        let vertex_buffer = create_buffer(context, vertices.len() as u64)?;
        let index_buffer = create_buffer(context, indices.len() as u64)?;
        context.write_buffers(&[
            BufferWrite {
                buffer: vertex_buffer.clone(),
                data: vertices.into(),
            },
            BufferWrite {
                buffer: index_buffer.clone(),
                data: indices.into(),
            },
        ])?;
        let images: Vec<_> = self
            .textures
            .iter()
            .map(|(_, image)| image.clone())
            .collect();
        context
            .bind_shader(&self.draw)
            .bind_buffer("vertices", &vertex_buffer)
            .bind_buffer("indices", &index_buffer)
            .bind_storage_image("render_target", target)
            .bind_sampled_images("textures", &self.sampler, &images);
        for draw in &draws {
            context
                .push_constant(draw)
                .dispatch(draw.area_size.x, draw.area_size.y)?;
        }
        Ok(())
    }
}

fn create_buffer(context: &mut Context, size: u64) -> Result<Handle<Buffer>, Error> {
    context.create_buffer(
        Lifetime::Frame,
        &BufferRequest {
            memory_location: MemoryLocation::Device,
            ty: BufferType::Storage,
            size,
        },
    )
}

fn create_texture(context: &mut Context, extent: Extent) -> Result<Handle<Image>, Error> {
    // TODO: Fix static lifetime.
    context.create_image(
        Lifetime::Static,
        &ImageRequest {
            format: ImageFormat::Rgba8Srgb,
            memory_location: MemoryLocation::Device,
            mip_level_count: 1,
            extent,
        },
    )
}

fn area_offset_size(
    clip_rect: &epaint::Rect,
    pixels_per_point: f32,
    surface_extent: Extent,
) -> (UVec2, UVec2) {
    let transform =
        |value: f32, min, max| u32::clamp((pixels_per_point * value).round() as u32, min, max);
    let clip_min_x = transform(clip_rect.min.x, 0, surface_extent.width);
    let clip_min_y = transform(clip_rect.min.y, 0, surface_extent.height);
    let clip_max_x = transform(clip_rect.max.x, clip_min_x, surface_extent.width);
    let clip_max_y = transform(clip_rect.max.y, clip_min_y, surface_extent.height);
    let offset = UVec2::new(clip_min_x, clip_min_y);
    let extent = UVec2::new(clip_max_x - clip_min_x, clip_max_y - clip_min_y);
    (offset, extent)
}
