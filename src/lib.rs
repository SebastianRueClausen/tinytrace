#![warn(clippy::all)]

pub mod backend;
pub mod error;
pub mod scene;

use ash::vk;
use backend::Context;
use error::Error;
use glam::Vec3;
use raw_window_handle::{RawDisplayHandle, RawWindowHandle};

pub struct Renderer {
    pub context: Context,
}

impl Renderer {
    pub fn new(
        window: Option<(RawWindowHandle, RawDisplayHandle, vk::Extent2D)>,
    ) -> Result<Self, Error> {
        Ok(Self {
            context: Context::new(window)?,
        })
    }

    pub fn render(&self, width: usize, height: usize) -> Vec<Vec3> {
        vec![Vec3::ZERO; width * height]
    }
}
