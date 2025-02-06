#[cfg(test)]
mod test;

use tinytrace_backend::Context;

pub fn add_includes(context: &mut Context) {
    context.add_include("material", include_str!("material.glsl").to_string());
    context.add_include("microfacet", include_str!("microfacet.glsl").to_string());
    context.add_include("fresnel", include_str!("fresnel.glsl").to_string());
    context.add_include("coat_brdf", include_str!("coat_brdf.glsl").to_string());
    context.add_include(
        "diffuse_brdf",
        include_str!("diffuse_brdf.glsl").to_string(),
    );
    context.add_include("fuzz_brdf", include_str!("fuzz_brdf.glsl").to_string());
    context.add_include("metal_brdf", include_str!("metal_brdf.glsl").to_string());
    context.add_include("bsdf", include_str!("bsdf.glsl").to_string());
    context.add_include(
        "specular_brdf",
        include_str!("specular_brdf.glsl").to_string(),
    );
    context.add_include(
        "specular_btdf",
        include_str!("specular_btdf.glsl").to_string(),
    );
}
