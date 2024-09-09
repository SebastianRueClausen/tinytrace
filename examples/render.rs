use std::path::PathBuf;
use std::time::{Duration, Instant};
use std::{mem, thread};

use ash::vk;
use bit_set::BitSet;
use egui::RichText;
use glam::{Vec2, Vec3};
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use tinytrace::camera::{Camera, CameraMove};
use tinytrace::error::ErrorKind;
use tinytrace::{asset, backend, Renderer, RestirReplay, SampleStrategy, Timings};
use tinytrace_egui::{RenderRequest as GuiRenderRequest, Renderer as GuiRenderer};
use winit::application::ApplicationHandler;
use winit::dpi::PhysicalSize;
use winit::event::{ElementState, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, ModifiersState, PhysicalKey};
use winit::window::{Window, WindowId};

struct App {
    render_state: Option<RenderState>,
    scene_controller: SceneController,
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
        renderer.set_scene(&self.scene_controller.scene).unwrap();
        self.render_state = Some(RenderState {
            gui: Gui::new(&mut renderer.context, event_loop).unwrap().into(),
            camera_controller: CameraController::default(),
            renderer_controller: RendererController::default(),
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
            render_state.render(&mut self.scene_controller);
        }
    }
}

struct RenderState {
    window: Window,
    renderer: Renderer,
    last_update: Instant,
    camera_controller: CameraController,
    renderer_controller: RendererController,
    gui: Gui,
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
            WindowEvent::Resized(size) => self.handle_resize(*size),
            WindowEvent::CursorMoved { position, .. } => {
                self.camera_controller
                    .mouse_moved(Vec2::new(position.x as f32, position.y as f32));
            }
            _ => (),
        }
    }

    fn render(&mut self, scene_controller: &mut SceneController) {
        self.window.request_redraw();
        if let Some(scene) = scene_controller.load.update() {
            self.renderer.set_scene(&scene).unwrap();
            scene_controller.scene = scene;
        }
        let dt = self.last_update.elapsed();
        self.last_update = Instant::now();
        let previous_camera = self.renderer.camera;
        let egui_output = self.run_gui(scene_controller);
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

    fn handle_resize(&mut self, size: PhysicalSize<u32>) {
        self.renderer
            .resize(vk::Extent2D {
                width: size.width,
                height: size.height,
            })
            .unwrap();
    }

    fn handle_error(&mut self, error: &tinytrace::Error) {
        if let ErrorKind::Backend(backend::Error::SurfaceOutdated) = error.kind {
            self.handle_resize(self.window.inner_size());
        } else {
            panic!("unexpected error: {error}")
        }
    }

    fn run_gui(&mut self, scene_controller: &mut SceneController) -> egui::FullOutput {
        let raw_input = self.gui.egui_winit_state.take_egui_input(&self.window);
        self.gui.egui_context.run(raw_input, |egui_context| {
            egui::Window::new("Render Control")
                .resizable(true)
                .show(egui_context, |ui| {
                    ui.collapsing("Scene", |ui| {
                        scene_controller.gui(ui);
                    });
                    ui.collapsing("Camera", |ui| {
                        self.camera_controller.gui(&mut self.renderer.camera, ui);
                    });
                    ui.collapsing("Renderer", |ui| {
                        self.renderer_controller.gui(ui);
                        if &self.renderer_controller.config != self.renderer.config() {
                            self.renderer
                                .set_config(self.renderer_controller.config)
                                .unwrap();
                        }
                    });
                    if let Some(timings) = self.renderer.timings() {
                        timings_gui(&timings, ui);
                    }
                });
        })
    }

    fn render_gui(
        &mut self,
        render_target: &backend::Handle<backend::Image>,
        egui_output: egui::FullOutput,
    ) -> Result<(), tinytrace::Error> {
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
    fn new(
        context: &mut backend::Context,
        event_loop: &ActiveEventLoop,
    ) -> Result<Self, tinytrace::Error> {
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

fn timings_gui(timings: &Timings, ui: &mut egui::Ui) {
    egui::Grid::new("timings").show(ui, |ui| {
        ui.label(RichText::new("Stage").strong());
        ui.label(RichText::new("Duration").strong());
        ui.end_row();
        let mut timing = |label, duration: Duration| {
            ui.label(label);
            ui.label(format!("{:?}", duration));
            ui.end_row();
        };
        timing("Integrate", timings.integrate);
        timing("Post Process", timings.post_process);
    });
}

#[derive(Default)]
enum SceneLoad {
    #[default]
    Empty,
    Loaded(PathBuf),
    Error(String),
    Loading {
        thread: thread::JoinHandle<Result<asset::Scene, asset::Error>>,
        path: PathBuf,
    },
}

impl SceneLoad {
    fn start_loading(&mut self, path: PathBuf) {
        *self = Self::Loading {
            path: path.clone(),
            thread: thread::spawn(move || asset::Scene::from_gltf(&path)),
        };
    }

    fn gui(&self, ui: &mut egui::Ui) {
        match self {
            Self::Loaded(path) => {
                ui.label(format!("'{path:?}' is loaded"));
            }
            Self::Error(error) => {
                ui.label(format!("Failed to load scene: {error}"));
            }
            Self::Loading { path, .. } => {
                ui.horizontal(|ui| {
                    ui.add(egui::Spinner::new());
                    ui.label(format!("Loading scene {path:?}"))
                });
            }
            Self::Empty => (),
        };
    }

    fn update(&mut self) -> Option<asset::Scene> {
        match mem::take(self) {
            Self::Loading { thread, path } if thread.is_finished() => {
                match thread.join().unwrap() {
                    Ok(scene) => {
                        *self = Self::Loaded(path.clone());
                        Some(scene)
                    }
                    Err(error) => {
                        *self = Self::Error(error.to_string());
                        None
                    }
                }
            }
            stage => {
                *self = stage;
                None
            }
        }
    }
}

#[derive(Default)]
struct SceneController {
    path: String,
    load: SceneLoad,
    scene: asset::Scene,
}

impl SceneController {
    fn gui(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            let label = ui.label("Scene path");
            ui.text_edit_singleline(&mut self.path)
                .labelled_by(label.id);
            if ui.button("Load").clicked() {
                self.load.start_loading(PathBuf::from(&self.path));
            }
        });
        self.load.gui(ui);
    }
}

#[derive(Default)]
struct RendererController {
    config: tinytrace::Config,
}

impl RendererController {
    fn gui(&mut self, ui: &mut egui::Ui) {
        egui::Grid::new("renderer").show(ui, |ui| {
            let mut drag_value = |value: &mut u32, label: &str| {
                ui.label(label);
                ui.add(egui::DragValue::new(value));
                ui.end_row();
            };
            drag_value(&mut self.config.sample_count, "Sample count");
            drag_value(&mut self.config.bounce_count, "Bounce count");
            egui::ComboBox::from_label("Sample strategy")
                .selected_text(format!("{}", self.config.sample_strategy))
                .show_ui(ui, |ui| {
                    let mut option = |value| {
                        ui.selectable_value(
                            &mut self.config.sample_strategy,
                            value,
                            format!("{value}"),
                        );
                    };
                    option(SampleStrategy::UniformHemisphere);
                    option(SampleStrategy::CosineHemisphere);
                    option(SampleStrategy::Brdf);
                });
            ui.end_row();
            let mut checkbox = |value: &mut bool, label: &str| {
                ui.add(egui::Checkbox::new(value, label));
                ui.end_row();
            };
            checkbox(&mut self.config.tonemap, "Enable tonemapping");
            checkbox(&mut self.config.restir.enabled, "Enable world space ReSTIR");
            ui.add_enabled_ui(self.config.restir.enabled, |ui| {
                egui::Grid::new("restir").show(ui, |ui| {
                    let mut drag_value = |value: &mut u32, label: &str, range| {
                        ui.label(label);
                        let drag_value = egui::DragValue::new(value)
                            .range(range)
                            .clamp_to_range(true)
                            .update_while_editing(false);
                        ui.add(drag_value);
                        ui.end_row();
                    };
                    drag_value(
                        &mut self.config.restir.reservoir_hash_grid_capacity,
                        "Reservoir hash grid capacity",
                        0xfff..=0xffffff,
                    );
                    drag_value(
                        &mut self.config.restir.update_hash_grid_capacity,
                        "Update hash grid capacity",
                        0xfff..=0xffffff,
                    );
                    drag_value(
                        &mut self.config.restir.reservoirs_per_cell,
                        "Reservoirs per cell",
                        1..=256,
                    );
                    drag_value(
                        &mut self.config.restir.updates_per_cell,
                        "Reservoir updates per cell",
                        1..=256,
                    );
                    ui.label("Scene scale");
                    let drag_value = egui::DragValue::new(&mut self.config.restir.scene_scale)
                        .range(0.1..=100.0)
                        .clamp_to_range(true)
                        .update_while_editing(false);
                    ui.add(drag_value);
                    ui.end_row();
                    egui::ComboBox::from_label("Replay")
                        .selected_text(format!("{}", self.config.restir.replay))
                        .show_ui(ui, |ui| {
                            let mut option = |value| {
                                ui.selectable_value(
                                    &mut self.config.restir.replay,
                                    value,
                                    format!("{value}"),
                                );
                            };
                            option(RestirReplay::None);
                            option(RestirReplay::First);
                            option(RestirReplay::Full);
                        });
                    ui.end_row();
                });
            });
        });
    }
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
    fn gui(&mut self, camera: &mut Camera, ui: &mut egui::Ui) {
        egui::Grid::new("transform").show(ui, |ui| {
            ui.label("Acceleration");
            ui.add(egui::DragValue::new(&mut self.acceleration).speed(0.1));
            ui.end_row();

            ui.label("Sensitivity");
            ui.add(egui::DragValue::new(&mut self.sensitivity).speed(0.1));
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
        let right = camera.right();

        if self.is_key_pressed(KeyCode::KeyW) {
            self.velocity -= camera.forward * self.acceleration * dt;
        }
        if self.is_key_pressed(KeyCode::KeyS) {
            self.velocity += camera.forward * self.acceleration * dt;
        }
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
        scene_controller: SceneController {
            scene,
            ..Default::default()
        },
    };
    let event_loop = EventLoop::new().unwrap();
    event_loop.run_app(&mut app).unwrap();
}
