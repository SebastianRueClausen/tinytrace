use tinytrace::asset::{self, SceneImporter};
use tinytrace_viewer::Viewer;
use winit::event_loop::EventLoop;

fn main() {
    let importer = asset::gltf::GltfImporter::new("scenes/cornell_box.gltf").unwrap();
    let scene = importer.new_scene().unwrap();
    let mut app = Viewer::new(scene);
    let event_loop = EventLoop::new().unwrap();
    event_loop.run_app(&mut app).unwrap();
}
