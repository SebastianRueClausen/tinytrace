use glam::{Vec2, Vec2Swizzles, Vec3, Vec3Swizzles};

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

pub fn decode_octahedron(octahedron: Vec2) -> Vec3 {
    let normal = octahedron * 2.0 - 1.0;
    let mut normal = normal.xy().extend(1.0 - normal.x.abs() - normal.y.abs());
    let t = (-normal.z).clamp(0.0, 1.0);
    normal.x += if normal.x >= 0.0 { -t } else { t };
    normal.y += if normal.y >= 0.0 { -t } else { t };
    normal.normalize()
}

pub fn quantize_unorm(v: f32, n: i32) -> i32 {
    let scale = ((1i32 << n) - 1i32) as f32;
    let v = if v >= 0f32 { v } else { 0f32 };
    let v = if v <= 1f32 { v } else { 1f32 };
    (v * scale + 0.5f32) as i32
}

pub fn quantize_snorm(mut v: f32, n: i32) -> i32 {
    let scale = ((1 << (n - 1)) - 1) as f32;
    let round = if v >= 0.0 { 0.5 } else { -0.5 };
    v = if v >= -1.0 { v } else { -1.0 };
    v = if v <= 1.0 { v } else { 1.0 };
    (v * scale + round) as i32
}

pub fn dequantize_unorm(value: u32, n: u32) -> f32 {
    value as f32 / ((1_i32 << n) - 1) as f32
}
