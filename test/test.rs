use glam::Vec3;

fn mean_squared_error(a: &[Vec3], b: &[Vec3]) -> f32 {
    assert_eq!(a.len(), b.len());
    let sum: f32 = a
        .iter()
        .zip(b.iter())
        .map(|(a, b)| a.distance_squared(*b))
        .sum();
    sum / a.len() as f32
}

#[test]
fn cornell_box() {
    let num_samples = 128;
    let width = 128;
    let height = 128;

    let mut correct = vec![Vec3::new(0.0, 0.0, 0.0); width * height];

    let mut scene = smallpt::Scene::init();

    // Bottom
    scene.add(Box::new(smallpt::Plane::new(
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
        smallpt::Material::new(
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.75, 0.75, 0.75),
            smallpt::BSDF::Diffuse,
        ),
    )));

    // Left
    scene.add(Box::new(smallpt::Plane::new(
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(1.0, 0.0, 0.0),
        smallpt::Material::new(
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.75, 0.25, 0.25),
            smallpt::BSDF::Diffuse,
        ),
    )));

    // Right
    scene.add(Box::new(smallpt::Plane::new(
        Vec3::new(99.0, 0.0, 0.0),
        Vec3::new(-1.0, 0.0, 0.0),
        smallpt::Material::new(
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.25, 0.25, 0.75),
            smallpt::BSDF::Diffuse,
        ),
    )));

    // Front
    scene.add(Box::new(smallpt::Plane::new(
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(0.0, 0.0, 1.0),
        smallpt::Material::new(
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.75, 0.75, 0.75),
            smallpt::BSDF::Diffuse,
        ),
    )));

    // Back
    scene.add(Box::new(smallpt::Plane::new(
        Vec3::new(0.0, 0.0, 170.0),
        Vec3::new(0.0, 0.0, -1.0),
        smallpt::Material::new(
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 0.0),
            smallpt::BSDF::Diffuse,
        ),
    )));

    // Top
    scene.add(Box::new(smallpt::Plane::new(
        Vec3::new(0.0, 81.6, 0.0),
        Vec3::new(0.0, -1.0, 0.0),
        smallpt::Material::new(
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.75, 0.75, 0.75),
            smallpt::BSDF::Diffuse,
        ),
    )));

    // Light (emissive rectangle)
    scene.add(Box::new(smallpt::Rectangle::new(
        Vec3::new(50.0, 81.5, 50.0),
        Vec3::new(0.0, -1.0, 0.0),
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(0.0, 0.0, 1.0),
        33.0,
        33.0,
        smallpt::Material::new(
            Vec3::new(12.0, 12.0, 12.0),
            Vec3::new(0.0, 0.0, 0.0),
            smallpt::BSDF::Diffuse,
        ),
    )));

    let camera = smallpt::Camera {
        origin: Vec3::new(50.0, 50.0, 200.0),
        forward: Vec3::new(0.0, -0.05, -1.0).normalize(),
        right: Vec3::new(1.0, 0.0, 0.0).normalize(),
        up: Vec3::new(0.0, 1.0, 0.0).normalize(),
    };

    let mut num_rays = 0;
    smallpt::trace(
        &scene,
        &camera,
        width,
        height,
        num_samples,
        &mut correct,
        &mut num_rays,
    );

    let renderer = tinytrace::Renderer::new().unwrap();
    let test = renderer.render(width, height);

    let error = mean_squared_error(&correct, &test);
    println!("{error}");
}
