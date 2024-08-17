use std::time::{Duration, Instant};

use ash::vk;
use bit_set::BitSet;
use glam::{Vec2, Vec3};
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use tinytrace::camera::{Camera, CameraMove};
use tinytrace::error::{ErrorKind, Result};
use tinytrace::{asset, backend, Renderer};
use tinytrace_egui::{RenderRequest as GuiRenderRequest, Renderer as GuiRenderer};
use winit::application::ApplicationHandler;
use winit::event::{ElementState, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, ModifiersState, PhysicalKey};
use winit::window::{Window, WindowId};

struct App {
    render_state: Option<RenderState>,
    scene: asset::Scene,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        event_loop.set_control_flow(ControlFlow::Poll);
        let window = event_loop
            .create_window(Window::default_attributes())
            .unwrap();
        let extent = vk::Extent2D::default()
            .width(window.inner_size().width)
            .height(window.inner_size().height);
        let window_handles = Some((
            window.window_handle().unwrap().as_raw(),
            window.display_handle().unwrap().as_raw(),
        ));
        let mut renderer = Renderer::new(window_handles, extent).unwrap();
        renderer.set_scene(&self.scene).unwrap();
        self.render_state = Some(RenderState {
            gui: Gui::new(&mut renderer.context, event_loop).unwrap().into(),
            camera_controller: CameraController::default(),
            last_update: Instant::now(),
            window,
            renderer,
        });
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: WindowId,
        event: WindowEvent,
    ) {
        let Some(render_state) = &mut self.render_state else {
            return;
        };
        if render_state.window.id() != window_id {
            return;
        }
        if let WindowEvent::CloseRequested = event {
            event_loop.exit();
        } else {
            render_state.handle_window_event(&event);
        }
    }

    fn about_to_wait(&mut self, _: &ActiveEventLoop) {
        if let Some(render_state) = &mut self.render_state {
            render_state.render();
        }
    }
}

struct RenderState {
    window: Window,
    renderer: Renderer,
    last_update: Instant,
    gui: Gui,
    camera_controller: CameraController,
}

impl RenderState {
    fn handle_window_event(&mut self, event: &WindowEvent) {
        let response = self
            .gui
            .egui_winit_state
            .on_window_event(&self.window, &event);
        if response.consumed {
            return;
        }
        match event {
            WindowEvent::KeyboardInput { event, .. } => {
                if let PhysicalKey::Code(key) = event.physical_key {
                    match event.state {
                        ElementState::Pressed => self.camera_controller.key_pressed(key),
                        ElementState::Released => self.camera_controller.key_released(key),
                    }
                };
            }
            WindowEvent::ModifiersChanged(modifiers) => {
                self.camera_controller.modifier_change(modifiers.state());
            }
            WindowEvent::CursorMoved { position, .. } => {
                self.camera_controller
                    .mouse_moved(Vec2::new(position.x as f32, position.y as f32));
            }
            _ => (),
        }
    }

    fn render(&mut self) {
        self.window.request_redraw();
        let dt = self.last_update.elapsed();
        self.last_update = Instant::now();
        let previous_camera = self.renderer.camera;
        let egui_output = self.run_gui();
        let camera_move = self
            .camera_controller
            .camera_move(&self.renderer.camera, dt);
        self.renderer.camera.move_by(camera_move);
        if previous_camera.different_from(&self.renderer.camera) {
            self.renderer.reset_accumulation();
        }
        self.renderer.render_to_target().unwrap();
        let swapchain_image = match self.renderer.prepare_to_present() {
            Ok(swapchain_image) => swapchain_image,
            Err(error) => {
                self.handle_error(&error);
                return;
            }
        };
        self.render_gui(&swapchain_image, egui_output).unwrap();
        if let Err(error) = self.renderer.present() {
            self.handle_error(&error);
        }
    }

    fn handle_error(&mut self, error: &tinytrace::Error) {
        if let ErrorKind::Backend(backend::Error::SurfaceOutdated) = error.kind {
            self.renderer
                .resize(vk::Extent2D {
                    width: self.window.inner_size().width,
                    height: self.window.inner_size().height,
                })
                .unwrap();
        } else {
            panic!("unexpected error: {error}")
        }
    }

    fn run_gui(&mut self) -> egui::FullOutput {
        let raw_input = self.gui.egui_winit_state.take_egui_input(&self.window);
        self.gui.egui_context.run(raw_input, |egui_context| {
            egui::Window::new("Render Control")
                .resizable(true)
                .show(egui_context, |ui| {
                    ui.collapsing("Camera", |ui| {
                        camera_gui(&mut self.camera_controller, &mut self.renderer.camera, ui);
                    });
                });
        })
    }

    fn render_gui(
        &mut self,
        render_target: &backend::Handle<backend::Image>,
        egui_output: egui::FullOutput,
    ) -> Result<()> {
        let pixels_per_point = self.gui.egui_context.pixels_per_point();
        self.gui.renderer.render(
            &mut self.renderer.context,
            render_target,
            &GuiRenderRequest {
                textures_delta: egui_output.textures_delta,
                primitives: self
                    .gui
                    .egui_context
                    .tessellate(egui_output.shapes, pixels_per_point),
                pixels_per_point,
            },
        )?;
        Ok(())
    }
}

struct Gui {
    egui_context: egui::Context,
    egui_winit_state: egui_winit::State,
    renderer: GuiRenderer,
}

impl Gui {
    fn new(context: &mut backend::Context, event_loop: &ActiveEventLoop) -> Result<Self> {
        let egui_context = egui::Context::default();
        let viewport_id = egui_context.viewport_id();
        Ok(Self {
            renderer: GuiRenderer::new(context, context.surface_format())?,
            egui_winit_state: egui_winit::State::new(
                egui_context.clone(),
                viewport_id,
                event_loop,
                None,
                None,
                None,
            ),
            egui_context,
        })
    }
}

fn camera_gui(camera_controller: &mut CameraController, camera: &mut Camera, ui: &mut egui::Ui) {
    egui::Grid::new("transform").show(ui, |ui| {
        ui.label("Acceleration");
        ui.add(egui::DragValue::new(&mut camera_controller.acceleration).speed(0.1));
        ui.end_row();

        ui.label("Sensitivity");
        ui.add(egui::DragValue::new(&mut camera_controller.sensitivity).speed(0.1));
        ui.end_row();

        ui.label("Position");
        ui.add(egui::DragValue::new(&mut camera.position.x));
        ui.add(egui::DragValue::new(&mut camera.position.y));
        ui.add(egui::DragValue::new(&mut camera.position.z));
        ui.end_row();

        ui.label("Field of view");
        ui.drag_angle(&mut camera.fov);
        ui.end_row();
    });
}

struct CameraController {
    velocity: Vec3,
    acceleration: f32,
    sensitivity: f32,
    drag: f32,
    keys_pressed: BitSet,
    modifier_state: ModifiersState,
    mouse_position: Option<Vec2>,
    mouse_delta: Option<Vec2>,
}

impl CameraController {
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

        self.velocity -= self.velocity * self.drag * dt.min(1.0);

        if self.is_key_pressed(KeyCode::KeyW) {
            self.velocity -= camera.forward * self.acceleration * dt;
        }
        if self.is_key_pressed(KeyCode::KeyS) {
            self.velocity += camera.forward * self.acceleration * dt;
        }

        let right = camera.right();

        if self.is_key_pressed(KeyCode::KeyA) {
            self.velocity -= right * self.acceleration * dt;
        }
        if self.is_key_pressed(KeyCode::KeyD) {
            self.velocity += right * self.acceleration * dt;
        }

        camera_move.translation = self.velocity;

        let mouse_delta = self.mouse_delta();
        if self.modifier_state.shift_key() {
            camera_move.yaw += self.sensitivity * mouse_delta.x;
            camera_move.pitch += self.sensitivity * mouse_delta.y;
        }

        camera_move
    }
}

impl Default for CameraController {
    fn default() -> Self {
        Self {
            keys_pressed: BitSet::default(),
            modifier_state: ModifiersState::default(),
            mouse_position: None,
            mouse_delta: None,
            velocity: Vec3::default(),
            acceleration: 1.0,
            sensitivity: 0.05,
            drag: 8.0,
        }
    }
}

fn main() {
    let scene = tinytrace::asset::Scene::from_gltf("scenes/cornell_box.gltf").unwrap();
    let mut app = App {
        render_state: None,
        scene,
    };
    let event_loop = EventLoop::new().unwrap();
    event_loop.run_app(&mut app).unwrap();
}
