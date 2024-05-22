use glam::{Mat4, Vec2, Vec3};

#[derive(Debug)]
pub struct Camera {
    pub position: Vec3,
    pub forward: Vec3,
    pub yaw: f32,
    pub pitch: f32,
    pub z_near: f32,
    pub z_far: f32,
    pub fov: f32,
    pub aspect: f32,
}

impl Camera {
    pub const UP: Vec3 = Vec3::Y;

    pub fn new(surface_size: Vec2) -> Self {
        let position = Vec3::new(-0.284, 1.1, 1.0);
        let forward = Vec3::X;
        let yaw = 0.0;
        let pitch = 0.0;
        let z_near = 0.1;
        let z_far = 400.0;
        let fov = std::f32::consts::FRAC_PI_2;
        let aspect = surface_size.x / surface_size.y;
        Self {
            position,
            forward,
            yaw,
            pitch,
            z_near,
            z_far,
            fov,
            aspect,
        }
    }

    pub fn right(&self) -> Vec3 {
        self.forward.cross(Self::UP).normalize()
    }

    pub fn proj(&self) -> Mat4 {
        let mut proj = Mat4::perspective_infinite_reverse_rh(self.fov, self.aspect, self.z_near);
        proj.z_axis[3] = 1.0;
        proj
    }

    pub fn view(&self) -> Mat4 {
        Mat4::look_at_rh(self.position, self.position + self.forward, Self::UP)
    }

    pub fn move_by(&mut self, delta: CameraMove) {
        self.position += delta.translation;
        self.yaw = (self.yaw + delta.yaw) % std::f32::consts::TAU;
        self.pitch = (self.pitch - delta.pitch).clamp(-1.553, 1.553);
        self.forward = Vec3::new(
            f32::cos(self.yaw) * f32::cos(self.pitch),
            f32::sin(self.pitch),
            f32::sin(self.yaw) * f32::cos(self.pitch),
        )
        .normalize();
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct CameraMove {
    pub translation: Vec3,
    pub yaw: f32,
    pub pitch: f32,
}
