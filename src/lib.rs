#![warn(clippy::all)]

pub mod backend;
mod error;

use backend::Context;
use error::Error;
use glam::Vec3;

pub struct Renderer {
    pub context: Context,
}

impl Renderer {
    pub fn new() -> Result<Self, Error> {
        Ok(Self {
            context: Context::new()?,
        })
    }

    pub fn render(&self, width: usize, height: usize) -> Vec<Vec3> {
        vec![Vec3::ZERO; width * height]
    }
}
