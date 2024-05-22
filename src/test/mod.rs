use crate::Renderer;
use ash::vk;

#[test]
fn create_renderer() {
    let extent = vk::Extent2D::default().width(1024).height(1024);
    let _ = Renderer::new(None, extent).unwrap();
}
