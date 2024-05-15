use std::f32::consts::{PI, TAU};

use glam::{Vec2, Vec2Swizzles, Vec3, Vec3Swizzles, Vec4};

#[repr(C)]
#[derive(Default, Clone, Copy, Debug)]
pub struct TangentFrame {
    /// `0..10` - octahedron mapped normal u-coordinate unorm.
    /// `10..20` - octahedron mapped normal v-coordinate unorm.
    /// `20..31` - tangent rotation unorm.
    /// `31` - bitangent sign (negative if 0).
    pub encoded: u32,
}

unsafe impl bytemuck::NoUninit for TangentFrame {}

impl TangentFrame {
    pub fn new(normal: Vec3, tangent: Vec4) -> Self {
        let octahedron = encode_octahedron(normal);
        let u = quantize_unorm(octahedron.x, 10) as u32;
        let v = quantize_unorm(octahedron.y, 10) as u32;
        // This is a bit weird, but because of the precision loss from the octahedron encoding
        // (and perhaps quantization), we can't be certain that the normal and decoded normal
        // produces the same orthonormal vector. The `normal.x.abs() > normal.z.abs()` predicate
        // from `orthonormal` may evaluate differently. Therefore the decoded normal should always
        // be used to choose the orthonormal vector
        let orthonormal = orthonormal(decode_octahedron(Vec2 {
            x: dequantize_unorm(u, 10),
            y: dequantize_unorm(v, 10),
        }));
        let angle = angle_around_normal(normal, orthonormal, tangent.truncate()) / TAU;
        let angle = quantize_unorm(angle, 11) as u32;
        Self {
            encoded: (if tangent.w >= 0.0 { 1 } else { 0 }) << 31 | angle << 20 | v << 10 | u,
        }
    }
}

fn decode_octahedron(octahedron: Vec2) -> Vec3 {
    let normal = octahedron * 2.0 - 1.0;
    let mut normal = normal.xy().extend(1.0 - normal.x.abs() - normal.y.abs());
    let t = (-normal.z).clamp(0.0, 1.0);
    normal.x += if normal.x >= 0.0 { -t } else { t };
    normal.y += if normal.y >= 0.0 { -t } else { t };
    normal.normalize()
}

/// Return the angle between `a` and `b` around the `normal` vector.
/// `a` and `b` should both be orthogonal with `normal`.
/// The angle will be counter clockwise and between `0.0` and `std::f32::consts::PI`.
fn angle_around_normal(normal: Vec3, a: Vec3, b: Vec3) -> f32 {
    let dot = a.dot(b);
    let det = normal.dot(a.cross(b));

    let angle = f32::atan2(det, dot);

    if angle < 0.0 {
        angle + 2.0 * PI
    } else {
        angle
    }
}

/// Deterministic orthonormal vector to `normal`.
pub fn orthonormal(normal: Vec3) -> Vec3 {
    if normal.x.abs() > normal.z.abs() {
        Vec3::new(-normal.y, normal.x, 0.0).normalize()
    } else {
        Vec3::new(0.0, -normal.z, normal.y).normalize()
    }
}

pub fn encode_octahedron(mut normal: Vec3) -> Vec2 {
    fn wrap(value: Vec2) -> Vec2 {
        (Vec2::ONE - value.yx().abs()) * value.signum()
    }
    normal /= normal.x.abs() + normal.y.abs() + normal.z.abs();
    if normal.z < 0.0 {
        wrap(normal.xy()) * 0.5 + 0.5
    } else {
        normal.xy() * 0.5 + 0.5
    }
}

pub fn quantize_unorm(v: f32, n: i32) -> i32 {
    let scale = ((1i32 << n) - 1i32) as f32;
    let v = if v >= 0f32 { v } else { 0f32 };
    let v = if v <= 1f32 { v } else { 1f32 };
    (v * scale + 0.5f32) as i32
}

pub fn dequantize_unorm(value: u32, n: u32) -> f32 {
    value as f32 / ((1_i32 << n) - 1) as f32
}
