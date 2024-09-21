use tinytrace_backend::Extent;

use crate::Renderer;

#[test]
fn create_renderer() {
    let extent = Extent::new(1024, 1024);
    let _ = Renderer::new(None, extent).unwrap();
}
