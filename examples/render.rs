use std::time::{Duration, Instant};

use ash::vk;
use bit_set::BitSet;
use glam::{Vec2, Vec3};
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use tinytrace::camera::{Camera, CameraMove};
use tinytrace::Renderer;
use winit::application::ApplicationHandler;
use winit::event::{ElementState, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, ModifiersState, PhysicalKey};
use winit::window::{Window, WindowId};

struct App {
    state: Option<(Window, Renderer)>,
    scene: tinytrace::asset::Scene,
    inputs: Inputs,
    last_update: Instant,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        let window = event_loop
            .create_window(Window::default_attributes())
            .unwrap();

        let extent = vk::Extent2D {
            width: window.inner_size().width,
            height: window.inner_size().height,
        };

        let handles = (
            window.window_handle().unwrap().as_raw(),
            window.display_handle().unwrap().as_raw(),
        );

        let mut renderer = Renderer::new(Some(handles), extent).unwrap();
        renderer.set_scene(&self.scene).unwrap();
        renderer.context.execute_commands().unwrap();

        self.state = Some((window, renderer));
        event_loop.set_control_flow(ControlFlow::Poll);
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: WindowId,
        event: WindowEvent,
    ) {
        let Some((window, _)) = &mut self.state else {
            return;
        };
        if window.id() != window_id {
            return;
        }
        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if let PhysicalKey::Code(key) = event.physical_key {
                    match event.state {
                        ElementState::Pressed => self.inputs.key_pressed(key),
                        ElementState::Released => self.inputs.key_released(key),
                    }
                };
            }
            WindowEvent::ModifiersChanged(modifiers) => {
                self.inputs.modifier_change(modifiers.state());
            }
            WindowEvent::CursorMoved { position, .. } => {
                self.inputs.mouse_moved(Vec2 {
                    x: position.x as f32,
                    y: position.y as f32,
                });
            }
            _ => (),
        }
    }

    fn about_to_wait(&mut self, _: &ActiveEventLoop) {
        let Some((window, renderer)) = &mut self.state else {
            return;
        };

        let dt = self.last_update.elapsed();
        self.last_update = Instant::now();

        let camera_move = self.inputs.camera_move(&renderer.camera, dt);
        renderer.camera.move_by(camera_move);

        renderer.render_to_surface().unwrap();
        window.request_redraw();
    }
}

#[derive(Default)]
struct Inputs {
    keys_pressed: BitSet,
    modifier_state: ModifiersState,
    mouse_position: Option<Vec2>,
    mouse_delta: Option<Vec2>,
    camera_velocity: Vec3,
}

impl Inputs {
    fn mouse_moved(&mut self, to: Vec2) {
        let position = self.mouse_position.unwrap_or(to);
        let delta = self.mouse_delta.unwrap_or_default();
        self.mouse_delta = Some(delta + (position - to));
        self.mouse_position = Some(to);
    }

    fn key_pressed(&mut self, key: KeyCode) {
        self.keys_pressed.insert(key as usize);
    }

    fn key_released(&mut self, key: KeyCode) {
        self.keys_pressed.remove(key as usize);
    }

    fn is_key_pressed(&self, key: KeyCode) -> bool {
        self.keys_pressed.contains(key as usize)
    }

    fn mouse_delta(&mut self) -> Vec2 {
        self.mouse_delta.take().unwrap_or(Vec2::ZERO)
    }

    fn modifier_change(&mut self, modifier_state: ModifiersState) {
        self.modifier_state = modifier_state;
    }

    fn camera_move(&mut self, camera: &Camera, dt: Duration) -> CameraMove {
        let mut camera_move = CameraMove::default();
        let dt = dt.as_secs_f32();

        let acceleration = 1.0;
        let drag = 8.0;
        let sensitivity = 0.05;

        self.camera_velocity -= self.camera_velocity * drag * dt.min(1.0);

        if self.is_key_pressed(KeyCode::KeyW) {
            self.camera_velocity -= camera.forward * acceleration * dt;
        }
        if self.is_key_pressed(KeyCode::KeyS) {
            self.camera_velocity += camera.forward * acceleration * dt;
        }

        let right = camera.right();

        if self.is_key_pressed(KeyCode::KeyA) {
            self.camera_velocity -= right * acceleration * dt;
        }
        if self.is_key_pressed(KeyCode::KeyD) {
            self.camera_velocity += right * acceleration * dt;
        }

        camera_move.translation = self.camera_velocity;

        let mouse_delta = self.mouse_delta();
        if self.modifier_state.shift_key() {
            camera_move.yaw += sensitivity * mouse_delta.x;
            camera_move.pitch += sensitivity * mouse_delta.y;
        }

        camera_move
    }
}

fn main() {
    let scene = tinytrace::asset::Scene::from_gltf("scenes/cornell_box.gltf").unwrap();
    // let scene = tinytrace::asset::Scene::from_gltf("../glTF-Sample-Assets/Models/Duck/glTF/Duck.gltf").unwrap();
    // let mut red_box = tinytrace::asset::Scene::from_gltf("../glTF-Sample-Assets/Models/Box/glTF/Box.gltf").unwrap();

    let mut app = App {
        last_update: Instant::now(),
        inputs: Inputs::default(),
        state: None,
        scene,
    };

    let event_loop = EventLoop::new().unwrap();
    event_loop.run_app(&mut app).unwrap();
}
