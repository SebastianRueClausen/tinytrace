use core::f32;

use glam::{Vec3, Vec4};
use tinytrace_backend::Extent;

#[allow(unused)]
fn mean_squared_difference(a: &[Vec4], b: &[Vec4]) -> f32 {
    assert_eq!(a.len(), b.len());
    let sum: f32 = a
        .iter()
        .zip(b.iter())
        .map(|(a, b)| a.truncate().distance_squared(b.truncate()))
        .sum();
    sum / a.len() as f32
}

fn mean_absolute_percentage_error(correct: &[Vec4], prediction: &[Vec4]) -> f32 {
    assert_eq!(correct.len(), prediction.len());
    let sum: f32 = correct
        .iter()
        .zip(prediction.iter())
        .map(|(correct, prediction)| {
            let percentage_error =
                (*correct - *prediction).abs() / correct.abs().max(Vec4::splat(f32::EPSILON));
            percentage_error.truncate().element_sum() / 3.0
        })
        .sum();
    sum / correct.len() as f32
}

fn benchmark() {
    let bounce_count = 4;
    let ground_truth_config = tinytrace::Config {
        // Uniform hemisphere is gives very different results, could perhaps be a bug.
        sample_strategy: tinytrace::SampleStrategy::CosineHemisphere,
        light_sampling: tinytrace::LightSampling::OnHit,
        sample_count: 1 << 16,
        bounce_count,
        tonemap: false,
    };
    let test_config = tinytrace::Config {
        sample_strategy: tinytrace::SampleStrategy::Brdf,
        light_sampling: tinytrace::LightSampling::NextEventEstimation,
        sample_count: 32,
        bounce_count,
        tonemap: false,
    };
    let extent = Extent::new(64, 64);
    let mut renderer = tinytrace::Renderer::new(None, extent).unwrap();
    renderer.camera.position = Vec3::new(0.0, 1.0, 2.0);
    renderer.camera.yaw = 1.57;
    renderer.camera.pitch = 0.0;
    renderer.camera.fov = f32::to_radians(75.0);
    let scene = tinytrace_asset::Scene::from_gltf("scenes/cornell_box.gltf").unwrap();
    renderer.set_scene(&scene).unwrap();
    renderer.set_config(ground_truth_config).unwrap();
    let unbiased_output = renderer.render_to_texture().unwrap();
    renderer.set_config(test_config).unwrap();
    let test_output = renderer.render_to_texture().unwrap();
    let error = mean_absolute_percentage_error(&unbiased_output, &test_output);
    println!("mean absolute percentage error (MAPE): {error}");
}

fn main() {
    benchmark();
}
