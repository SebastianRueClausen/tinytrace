use glam::Vec3;

#[derive(Default)]
pub struct Renderer {}

impl Renderer {
    pub fn new() -> Self {
        Self {}
    }

    pub fn render(&self, width: usize, height: usize) -> Vec<Vec3> {
        vec![Vec3::ZERO; width * height]
    }
}
