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
#extension GL_ARB_gpu_shader_int64: require
#extension GL_NV_shader_atomic_int64: require
#extension GL_EXT_buffer_reference2: require
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
            ImageFormat::RgBc5Unorm
            | ImageFormat::RgbBc1Srgb
            | ImageFormat::RgbaBc1Srgb
            | ImageFormat::RgbBc1Unorm => {
                panic!("block compressed formats not usable as storage images")
            }
        }
    } else {
        ""
    }
}

pub fn render_bindings(bindings: &[Binding]) -> String {
    bindings
        .iter()
        .enumerate()
        .fold(String::new(), |mut glsl, (index, binding)| {
            let format_specifier = binding_format_specifier(&binding.ty);
            let binding = render_binding(binding.name, &binding.ty, index as u32);
            writeln!(
                &mut glsl,
                "layout (set = 0, binding = {index}{format_specifier}) {binding}"
            )
            .unwrap();
            glsl
        })
}

pub fn render_shader(width: u32, height: u32, source: &str) -> String {
    format!("{PRELUDE}\nlayout (local_size_x = {width}, local_size_y = {height}) in;\n{source}")
}
