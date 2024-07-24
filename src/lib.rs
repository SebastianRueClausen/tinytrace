#![warn(clippy::all)]

#[cfg(test)]
mod test;

pub mod asset;
pub mod backend;
pub mod camera;
pub mod error;
mod integrate;
mod post_process;
pub mod scene;

use std::mem;

use ash::vk;
use backend::{
    Buffer, BufferRequest, BufferType, BufferWrite, Context, Handle, Image, ImageRequest, Lifetime,
    MemoryLocation,
};
use camera::Camera;
use error::Error;
use glam::{Mat4, UVec2, Vec2, Vec4};
use integrate::Integrator;
use post_process::PostProcess;
use raw_window_handle::{RawDisplayHandle, RawWindowHandle};
use scene::Scene;

pub struct Renderer {
    pub context: Context,
    pub scene: Scene,
    pub integrator: Integrator,
    pub post_process: Option<PostProcess>,
    pub render_target: Handle<Image>,
    pub constants: Handle<Buffer>,
    pub camera: Camera,
    pub extent: vk::Extent2D,
    pub accumulated_frame_count: u32,
    pub config: Config,
}

impl Renderer {
    pub fn new(
        window: Option<(RawWindowHandle, RawDisplayHandle)>,
        extent: vk::Extent2D,
    ) -> Result<Self, Error> {
        let scene = asset::Scene::default();
        let mut context = Context::new(window, extent)?;
        context.add_include("constants", include_str!("includes/constants.glsl").into());
        context.add_include("brdf", include_str!("includes/brdf.glsl").into());
        context.add_include("scene", include_str!("includes/scene.glsl").into());
        context.add_include("math", include_str!("includes/math.glsl").into());
        context.add_include("random", include_str!("includes/random.glsl").into());
        context.add_include("sample", include_str!("includes/sample.glsl").into());

        let render_target = create_render_target(&mut context, extent)?;
        let constants = context.create_buffer(
            Lifetime::Static,
            &BufferRequest {
                size: mem::size_of::<Constants>() as vk::DeviceSize,
                memory_location: MemoryLocation::Device,
                ty: BufferType::Uniform,
            },
        )?;
        let integrator = Integrator::new(&mut context, &scene)?;
        let post_process = if window.is_some() {
            Some(PostProcess::new(&mut context)?)
        } else {
            None
        };
        let scene = Scene::new(&mut context, &scene)?;
        Ok(Self {
            camera: Camera::new(Vec2 {
                x: extent.width as f32,
                y: extent.height as f32,
            }),
            config: Config::default(),
            accumulated_frame_count: 0,
            context,
            scene,
            render_target,
            integrator,
            post_process,
            extent,
            constants,
        })
    }

    pub fn set_scene(&mut self, scene: &asset::Scene) -> Result<(), Error> {
        self.context.advance_lifetime(Lifetime::Scene)?;
        self.scene = Scene::new(&mut self.context, scene)?;
        self.integrator = Integrator::new(&mut self.context, scene)?;
        Ok(())
    }

    pub fn resize(&mut self, extent: vk::Extent2D) -> Result<(), Error> {
        self.context.resize_surface(extent)?;
        self.render_target = create_render_target(&mut self.context, extent)?;
        self.camera.aspect = extent.width as f32 / extent.height as f32;
        self.extent = extent;
        self.reset_accumulation();
        Ok(())
    }

    fn render_to_target(&mut self) -> Result<(), Error> {
        let (view, proj) = (self.camera.view(), self.camera.proj());
        let proj_view = proj * view;
        let constants = Constants {
            screen_size: UVec2::new(self.extent.width, self.extent.height),
            frame_index: self.context.frame_index() as u32,
            inverse_view: view.inverse(),
            inverse_proj: proj.inverse(),
            camera_position: self.camera.position.extend(0.0),
            accumulated_frame_count: self.accumulated_frame_count,
            sample_count: self.config.sample_count,
            bounce_count: self.config.bounce_count,
            proj_view,
            view,
            proj,
            ..Default::default()
        };

        self.context.write_buffers(&[BufferWrite {
            buffer: self.constants.clone(),
            data: bytemuck::bytes_of(&constants),
        }])?;

        self.integrator.integrate(
            &mut self.context,
            &self.constants,
            &self.scene,
            &self.render_target,
        );

        Ok(())
    }

    pub fn reset_accumulation(&mut self) {
        self.accumulated_frame_count = 0;
    }

    pub fn render_to_surface(&mut self) -> Result<(), Error> {
        self.render_to_target()?;
        let post_process = self
            .post_process
            .as_mut()
            .expect("renderer doesn't have a surface");
        let swapchain_image = self.context.swapchain_image()?;
        post_process.run(
            &mut self.context,
            &self.constants,
            &self.render_target,
            &swapchain_image,
        )?;
        self.accumulated_frame_count += 1;
        self.context.present(&swapchain_image).map_err(Error::from)
    }

    pub fn render_to_texture(&mut self) -> Result<Box<[Vec4]>, Error> {
        self.render_to_target()?;
        let mut download = self.context.download(&[], &[self.render_target.clone()])?;
        let data = download.images.remove(&self.render_target).unwrap();
        Ok(bytemuck::allocation::cast_slice_box(data))
    }
}

fn create_render_target(
    context: &mut Context,
    extent: vk::Extent2D,
) -> Result<Handle<Image>, backend::Error> {
    context.create_image(
        Lifetime::Surface,
        &ImageRequest {
            memory_location: MemoryLocation::Device,
            format: RENDER_TARGET_FORMAT,
            mip_level_count: 1,
            extent: vk::Extent3D {
                width: extent.width,
                height: extent.height,
                depth: 1,
            },
        },
    )
}

#[repr(C)]
#[derive(bytemuck::NoUninit, Clone, Copy, Default)]
struct Constants {
    view: Mat4,
    proj: Mat4,
    proj_view: Mat4,
    inverse_view: Mat4,
    inverse_proj: Mat4,
    camera_position: Vec4,
    screen_size: UVec2,
    frame_index: u32,
    accumulated_frame_count: u32,
    sample_count: u32,
    bounce_count: u32,
    padding: [u32; 2],
}

/// Configuration of the renderer.
pub struct Config {
    /// The number of ray bounces.
    pub bounce_count: u32,
    /// The number of samples per pixel.
    pub sample_count: u32,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            bounce_count: 4,
            sample_count: 4,
        }
    }
}

const RENDER_TARGET_FORMAT: vk::Format = vk::Format::R32G32B32A32_SFLOAT;
