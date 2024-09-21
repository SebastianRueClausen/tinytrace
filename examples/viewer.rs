use tinytrace_viewer::Viewer;
use winit::event_loop::EventLoop;

fn main() {
    let scene = tinytrace_asset::Scene::from_gltf("scenes/cornell_box.gltf").unwrap();
    let mut app = Viewer::new(scene);
    let event_loop = EventLoop::new().unwrap();
    event_loop.run_app(&mut app).unwrap();
}
