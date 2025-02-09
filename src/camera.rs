use std::f32::consts::{FRAC_PI_2, TAU};

use glam::{Mat4, Vec3};

#[derive(Debug, Clone, Copy)]
pub struct Camera {
    pub position: Vec3,
    pub yaw: f32,
    pub pitch: f32,
    pub z_near: f32,
    pub z_far: f32,
    pub fov: f32,
}

impl Default for Camera {
    fn default() -> Self {
        Self {
            position: Vec3::ZERO,
            fov: FRAC_PI_2,
            z_near: 0.1,
            z_far: 400.0,
            yaw: 0.0,
            pitch: 0.0,
        }
    }
}

impl Camera {
    pub const UP: Vec3 = Vec3::Y;

    pub fn forward(&self) -> Vec3 {
        let x = f32::cos(self.yaw) * f32::cos(self.pitch);
        let y = f32::sin(self.pitch);
        let z = f32::sin(self.yaw) * f32::cos(self.pitch);
        Vec3::new(x, y, z).normalize()
    }

    pub fn right(&self) -> Vec3 {
        -self.forward().cross(Self::UP).normalize()
    }

    pub fn proj(&self, aspect: f32) -> Mat4 {
        let mut proj = Mat4::perspective_infinite_reverse_rh(self.fov, aspect, self.z_near);
        proj.z_axis[3] = 1.0;
        proj
    }

    pub fn view(&self) -> Mat4 {
        Mat4::look_at_rh(self.position, self.position + self.forward(), Self::UP)
    }

    pub fn move_by(&mut self, delta: CameraMove) {
        self.position += delta.translation;
        self.yaw = (self.yaw - delta.yaw) % TAU;
        self.pitch = (self.pitch - delta.pitch).clamp(-1.553, 1.553);
    }

    pub fn different_from(&self, other: &Camera) -> bool {
        const DIFFERENCE_THRESHOLD: f32 = 1e-4;
        Vec3::max_element((self.position - other.position).abs()) > DIFFERENCE_THRESHOLD
            || (self.yaw - other.yaw).abs() > DIFFERENCE_THRESHOLD
            || (self.pitch - other.pitch).abs() > DIFFERENCE_THRESHOLD
            || (self.fov - other.fov).abs() > DIFFERENCE_THRESHOLD
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct CameraMove {
    pub translation: Vec3,
    pub yaw: f32,
    pub pitch: f32,
}
