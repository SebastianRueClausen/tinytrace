use super::{Binding, BindingType, ImageFormat};
use std::fmt::Write;

const PRELUDE: &str = r#"
#version 460
#extension GL_GOOGLE_include_directive: require
#extension GL_EXT_nonuniform_qualifier: require
#extension GL_EXT_shader_explicit_arithmetic_types_float16: require
#extension GL_EXT_shader_16bit_storage: require
#extension GL_EXT_shader_explicit_arithmetic_types_int16: require
#extension GL_EXT_ray_query: require
#extension GL_EXT_ray_tracing_position_fetch: require
"#;

fn render_binding(name: &str, ty: &BindingType, index: u32) -> String {
    let classifier = |writes| if writes { "" } else { "readonly" };
    match ty {
        BindingType::StorageBuffer { ty, array, writes } => {
            let classifier = classifier(*writes);
            let brackets = if *array { "[]" } else { "" };
            format!("{classifier} buffer Binding{index} {{ {ty} {name}{brackets}; }};")
        }
        BindingType::UniformBuffer { ty } => {
            format!("uniform Binding{index} {{ {ty} {name}; }};")
        }
        BindingType::AccelerationStructure => {
            format!("uniform accelerationStructureEXT {name};")
        }
        BindingType::SampledImage { count } => {
            let brackets = if count.is_some() { "[]" } else { "" };
            format!("uniform sampler2D {name}{brackets};")
        }
        BindingType::StorageImage { count, writes, .. } => {
            let brackets = if count.is_some() { "[]" } else { "" };
            format!("{} uniform image2D {name}{brackets};", classifier(*writes))
        }
    }
}

fn binding_format_specifier(ty: &BindingType) -> &'static str {
    if let BindingType::StorageImage { format, .. } = ty {
        match *format {
            ImageFormat::Rgba8Srgb | ImageFormat::Rgba8Unorm | ImageFormat::Bgra8Unorm => ", rgba8",
            ImageFormat::Rgba32Float => ", rgba32f",
            ImageFormat::RgBc5Unorm | ImageFormat::RgbBc1Srgb | ImageFormat::RgbaBc1Srgb => {
                panic!("block compressed formats not usable as storage images")
            }
        }
    } else {
        ""
    }
}

pub fn render_shader(
    width: u32,
    height: u32,
    source: &str,
    bindings: &[Binding],
    includes: &[&str],
) -> String {
    let mut glsl = PRELUDE.to_string();
    writeln!(
        &mut glsl,
        "layout (local_size_x = {width}, local_size_y = {height}) in;"
    )
    .unwrap();
    let glsl = includes.iter().fold(glsl, |mut glsl, include| {
        writeln!(&mut glsl, "#include \"{include}\"").unwrap();
        glsl
    });
    let mut glsl = bindings
        .iter()
        .enumerate()
        .fold(glsl, |mut glsl, (index, binding)| {
            let format_specifier = binding_format_specifier(&binding.ty);
            let binding = render_binding(binding.name, &binding.ty, index as u32);
            writeln!(
                &mut glsl,
                "layout (set = 0, binding = {index}{format_specifier}) {binding}"
            )
            .unwrap();
            glsl
        });
    write!(&mut glsl, "{source}").unwrap();
    glsl
}
