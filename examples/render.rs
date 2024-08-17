use std::time::{Duration, Instant};

use ash::vk;
use bit_set::BitSet;
use glam::{Vec2, Vec3};
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use tinytrace::camera::{Camera, CameraMove};
use tinytrace::error::{ErrorKind, Result};
use tinytrace::{backend, Renderer};
use tinytrace_egui::{RenderRequest as GuiRenderRequest, Renderer as GuiRenderer};
use winit::application::ApplicationHandler;
use winit::event::{ElementState, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, ModifiersState, PhysicalKey};
use winit::window::{Window, WindowId};

struct App {
    state: Option<(Window, Renderer)>,
    scene: tinytrace::asset::Scene,
    gui: Option<Gui>,
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

        let mut renderer =
            Renderer::new(Some(handles), extent).unwrap_or_else(|error| panic!("{error:#?}"));
        renderer.set_scene(&self.scene).unwrap();

        self.gui = Gui::new(&mut renderer.context, event_loop).unwrap().into();
        self.state = (window, renderer).into();
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
        if let Some(gui) = &mut self.gui {
            if gui
                .egui_winit_state
                .on_window_event(&window, &event)
                .consumed
            {
                return;
            }
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
        if camera_move.moves() {
            renderer.reset_accumulation();
        }

        renderer.camera.move_by(camera_move);
        renderer.render_to_target().unwrap();
        let swapchain_image = match renderer.prepare_to_present() {
            Ok(swapchain_image) => swapchain_image,
            Err(error) => {
                self.handle_error(&error);
                return;
            }
        };

        if let Some(gui) = &mut self.gui {
            gui.render(
                &mut renderer.context,
                &swapchain_image,
                &window,
                &mut self.inputs,
            )
            .unwrap();
        }

        if let Err(error) = renderer.present() {
            self.handle_error(&error);
        }

        let Some((window, _)) = &mut self.state else {
            return;
        };
        window.request_redraw();
    }
}

impl App {
    fn handle_error(&mut self, error: &tinytrace::Error) {
        let Some((window, renderer)) = &mut self.state else {
            return;
        };
        if let ErrorKind::Backend(backend::Error::SurfaceOutdated) = error.kind {
            let window_size = window.inner_size();
            renderer
                .resize(vk::Extent2D {
                    width: window_size.width,
                    height: window_size.height,
                })
                .unwrap();
        } else {
            panic!("unexpected error: {error}")
        }
        window.request_redraw();
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

    fn render(
        &mut self,
        context: &mut backend::Context,
        render_target: &backend::Handle<backend::Image>,
        window: &winit::window::Window,
        _input: &mut Inputs,
    ) -> Result<()> {
        let raw_input = self.egui_winit_state.take_egui_input(window);
        let output = self.egui_context.run(raw_input, |egui_context| {
            gui_window(egui_context);
        });
        let pixels_per_point = self.egui_context.pixels_per_point();
        self.renderer.render(
            context,
            render_target,
            &GuiRenderRequest {
                textures_delta: output.textures_delta,
                primitives: self
                    .egui_context
                    .tessellate(output.shapes, pixels_per_point),
                pixels_per_point,
            },
        )?;
        Ok(())
    }
}

fn gui_window(egui_context: &egui::Context) {
    egui::Window::new("Render Control")
        .resizable(true)
        .show(egui_context, |ui| {
            ui.label("hello world");
        });
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
    let mut app = App {
        last_update: Instant::now(),
        inputs: Inputs::default(),
        gui: None,
        state: None,
        scene,
    };
    let event_loop = EventLoop::new().unwrap();
    event_loop.run_app(&mut app).unwrap();
}
