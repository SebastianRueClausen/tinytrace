#![warn(clippy::all)]

#[cfg(test)]
mod test;

pub mod asset;
pub mod backend;
pub mod camera;
pub mod config;
pub mod error;
mod hash_grid;
mod integrate;
mod post_process;
pub mod scene;

use std::{mem, time::Duration};

use ash::vk;
use backend::{
    Buffer, BufferRequest, BufferType, BufferWrite, Context, Handle, Image, ImageFormat,
    ImageRequest, Lifetime, MemoryLocation,
};
use camera::Camera;
pub use config::{Config, RestirConfig, RestirReplay, SampleStrategy};
pub use error::Error;
use glam::{Mat4, UVec2, Vec4};
use hash_grid::HashGridLayout;
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
        let mut context = Context::new(window)?;
        context.add_include("constants", include_str!("includes/constants.glsl").into());
        context.add_include(
            "octahedron",
            include_str!("includes/octahedron.glsl").into(),
        );
        context.add_include("brdf", include_str!("includes/brdf.glsl").into());
        context.add_include("scene", include_str!("includes/scene.glsl").into());
        context.add_include("math", include_str!("includes/math.glsl").into());
        context.add_include("random", include_str!("includes/random.glsl").into());
        context.add_include("sample", include_str!("includes/sample.glsl").into());
        context.add_include("debug", include_str!("includes/debug.glsl").into());
        context.add_include("hash_grid", include_str!("includes/hash_grid.glsl").into());
        context.add_include("restir", include_str!("includes/restir.glsl").into());

        let render_target = create_render_target(&mut context, extent)?;
        let constants = context.create_buffer(
            Lifetime::Static,
            &BufferRequest {
                size: mem::size_of::<Constants>() as vk::DeviceSize,
                memory_location: MemoryLocation::Device,
                ty: BufferType::Uniform,
            },
        )?;
        let config = Config::default();
        let integrator = Integrator::new(&mut context, &config.restir)?;
        let post_process = if window.is_some() {
            Some(PostProcess::new(&mut context)?)
        } else {
            None
        };
        Ok(Self {
            camera: Camera::new(extent.width as f32 / extent.height as f32),
            scene: Scene::new(&mut context, &scene)?,
            accumulated_frame_count: 0,
            config,
            context,
            render_target,
            integrator,
            post_process,
            extent,
            constants,
        })
    }

    pub fn set_scene(&mut self, scene: &asset::Scene) -> Result<(), Error> {
        self.context
            .clear_resources_with_lifetime(Lifetime::Scene)?;
        self.scene = Scene::new(&mut self.context, scene)?;
        Ok(())
    }

    pub fn config(&self) -> &Config {
        &self.config
    }

    pub fn set_config(&mut self, config: Config) -> Result<(), Error> {
        if self.config != config {
            self.reset_accumulation();
            if self.config.restir != config.restir {
                self.context
                    .clear_resources_with_lifetime(Lifetime::Renderer)?;
                self.integrator = Integrator::new(&mut self.context, &config.restir)?;
            }
        }
        self.config = config;
        Ok(())
    }

    pub fn resize(&mut self, extent: vk::Extent2D) -> Result<(), Error> {
        self.context.resize_surface()?;
        self.render_target = create_render_target(&mut self.context, extent)?;
        self.camera.aspect = extent.width as f32 / extent.height as f32;
        self.extent = extent;
        self.reset_accumulation();
        Ok(())
    }

    pub fn render_to_target(&mut self) -> Result<(), Error> {
        let (view, proj) = (self.camera.view(), self.camera.proj());
        let proj_view = proj * view;
        let constants = Constants {
            screen_size: UVec2::new(self.extent.width, self.extent.height),
            frame_index: self.context.frame_index() as u32,
            camera_position: self.camera.position.extend(0.0),
            accumulated_frame_count: self.accumulated_frame_count,
            sample_count: self.config.sample_count,
            bounce_count: self.config.bounce_count,
            reservoir_hash_grid: self.integrator.restir_state.reservoir_hash_grid.layout,
            reservoir_update_hash_grid: self.integrator.restir_state.update_hash_grid.layout,
            use_world_space_restir: self.config.restir.enabled.into(),
            inverse_view: view.inverse(),
            inverse_proj: proj.inverse(),
            tonemap: self.config.tonemap.into(),
            sample_strategy: self.config.sample_strategy,
            reservoirs_per_cell: self.config.restir.reservoirs_per_cell,
            reservoir_updates_per_cell: self.config.restir.updates_per_cell,
            restir_replay: self.config.restir.replay,
            proj_view,
            view,
            proj,
        };

        self.context.write_buffers(&[BufferWrite {
            buffer: self.constants.clone(),
            data: bytemuck::bytes_of(&constants).into(),
        }])?;

        self.context.insert_timestamp("before integrate");
        self.integrator.integrate(
            &mut self.context,
            &self.constants,
            &self.scene,
            &self.render_target,
        )?;

        self.context.insert_timestamp("after integrate");
        self.accumulated_frame_count += 1;

        Ok(())
    }

    pub fn reset_accumulation(&mut self) {
        self.accumulated_frame_count = 0;
    }

    pub fn timings(&mut self) -> Option<Timings> {
        Some(Timings {
            integrate: self.context.timestamp("after integrate")?
                - self.context.timestamp("before integrate")?,
            post_process: self.context.timestamp("after post process")?
                - self.context.timestamp("before post process")?,
        })
    }

    pub fn prepare_to_present(&mut self) -> Result<Handle<Image>, Error> {
        let swapchain_image = self.context.swapchain_image()?;
        let post_process = self
            .post_process
            .as_mut()
            .expect("renderer doesn't have a surface");
        self.context.insert_timestamp("before post process");
        post_process.run(
            &mut self.context,
            &self.constants,
            &self.render_target,
            &swapchain_image,
        )?;
        self.context.insert_timestamp("after post process");
        Ok(swapchain_image)
    }

    pub fn present(&mut self) -> Result<(), Error> {
        self.context.present().map_err(Error::from)
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

#[derive(Debug)]
pub struct Timings {
    pub integrate: Duration,
    pub post_process: Duration,
}

#[repr(C)]
#[derive(bytemuck::NoUninit, Debug, Clone, Copy, Default)]
struct Constants {
    view: Mat4,
    proj: Mat4,
    proj_view: Mat4,
    inverse_view: Mat4,
    inverse_proj: Mat4,
    camera_position: Vec4,
    frame_index: u32,
    accumulated_frame_count: u32,
    sample_count: u32,
    bounce_count: u32,
    reservoir_hash_grid: HashGridLayout,
    reservoir_update_hash_grid: HashGridLayout,
    screen_size: UVec2,
    use_world_space_restir: u32,
    tonemap: u32,
    sample_strategy: SampleStrategy,
    reservoir_updates_per_cell: u32,
    reservoirs_per_cell: u32,
    restir_replay: RestirReplay,
}

const RENDER_TARGET_FORMAT: ImageFormat = ImageFormat::Rgba32Float;
