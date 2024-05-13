use super::{Binding, BindingType};
use std::fmt::Write;

const PRELUDE: &str = r#"
#version 460
#extension GL_GOOGLE_include_directive: require
#extension GL_EXT_nonuniform_qualifier: require
"#;

fn render_binding(name: &str, ty: &BindingType, index: u32) -> String {
    let classifier = |writes| if writes { "" } else { "readonly " };
    match ty {
        BindingType::StorageBuffer { ty, array, writes } => {
            let classifier = classifier(*writes);
            let brackets = if *array { "[]" } else { "" };
            format!("{classifier}buffer Binding{index} {{ {ty} {name}{brackets}; }};")
        }
        BindingType::UniformBuffer { ty } => {
            format!("uniform Binding{index} {{ {ty} {name}; }};")
        }
        BindingType::AccelerationStructure => {
            format!("uniform accelerationStructureEXT {name};")
        }
        BindingType::SampledImage { count } => {
            let brackets = if *count > 1 { "[]" } else { "" };
            format!("uniform sampled2D {name}{};", brackets)
        }
        BindingType::StorageImage { count, writes } => {
            let brackets = if *count > 1 { "[]" } else { "" };
            format!("{} uniform image2D {name}{brackets};", classifier(*writes))
        }
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
        writeln!(&mut glsl, "#include\"{include}\"").unwrap();
        glsl
    });
    let mut glsl = bindings
        .iter()
        .enumerate()
        .fold(glsl, |mut glsl, (index, binding)| {
            let binding = render_binding(binding.name, &binding.ty, index as u32);
            writeln!(&mut glsl, "layout (set = 0, binding = {index}) {binding}").unwrap();
            glsl
        });
    write!(&mut glsl, "{source}").unwrap();
    glsl
}
