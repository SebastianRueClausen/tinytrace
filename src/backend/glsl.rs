use super::{Binding, BindingType};

const PRELUDE: &str = r#"
#version 460
#extension GL_GOOGLE_include_directive: require
#extension GL_EXT_nonuniform_qualifier: require
"#;

fn render_binding(name: &str, ty: &BindingType, set: u32, index: u32) -> String {
    let classifier = |writes| if writes { "" } else { "readonly " };
    match ty {
        BindingType::StorageBuffer { ty, array, writes } => {
            let classifier = classifier(*writes);
            let brackets = if *array { "[]" } else { "" };
            format!("{classifier}buffer Set{set}Binding{index} {{ {ty} {name}{brackets}; }};")
        }
        BindingType::UniformBuffer { ty } => {
            format!("uniform Set{set}Index{index} {{ {ty} {name}; }};")
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

fn render_bindings(set: u32, bindings: &[Binding]) -> String {
    let to_glsl = |(index, binding): (usize, &Binding)| {
        let binding = render_binding(binding.name, &binding.ty, set, index as u32);
        format!("layout (set = {set}, binding = {index}) {binding}\n")
    };
    bindings.iter().enumerate().map(to_glsl).collect()
}

pub fn render_shader(width: u32, height: u32, source: &str, bindings: &[Binding]) -> String {
    let block_size = format!("layout (local_size_x = {width}, local_size_y = {height}) in;\n");
    let bindings = render_bindings(0, bindings);
    format!("{PRELUDE}{block_size}{bindings}{source}")
}