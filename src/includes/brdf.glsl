#ifndef BRDF
#define BRDF

#include "random"
#include "math"
#include "brdf"

// The cutoff for roughness where a material is treated as a perfect mirror.
const float ROUGHNESS_DELTA_CUTOFF = 0.01;

struct SurfaceProperties {
    float metallic;
    float roughness;
    vec3 albedo;
    vec3 fresnel_min;
    float fresnel_max;
    vec3 view_direction;
    float normal_dot_view;
    float anisotropy_strength;
    vec2 anisotropy_direction;
    bool is_anisotropic;
};

struct ScatterProperties {
    vec3 direction;
    vec3 half_vector;
    float normal_dot_half;
    float normal_dot_scatter;
    float view_dot_half;
};

ScatterProperties create_scatter_properties(SurfaceProperties surface, vec3 direction, vec3 normal) {
    ScatterProperties scatter;
    scatter.direction = direction;
    scatter.half_vector = normalize(surface.view_direction + scatter.direction);
    scatter.normal_dot_half = saturate(dot(normal, scatter.half_vector));
    scatter.normal_dot_scatter = saturate(dot(normal, scatter.direction));
    scatter.view_dot_half = saturate(dot(surface.view_direction, scatter.half_vector));
    return scatter;
}

// The Schlick approximation of the Fresnel effect. The Fresnel effect describes
// how light reflects off a surface at different angles. Steep angles result in weak
// reflections, while shallow angles result in strong reflections.
//
// The Schlick approximation was first presented by
// Christophe Schlick in 1994 (Schlick, 1994) - https://www.researchgate.net/publication/354065225_The_Schlick_Fresnel_Approximation.
vec3 fresnel_schlick(vec3 fresnel_min, float fresnel_max, float view_dot_half) {
    float flipped = 1.0 - view_dot_half;
    float flipped_2 = pow2(flipped);
    return fresnel_min + (fresnel_max - fresnel_min) * (pow2(flipped_2) * flipped);
}

// Models the shadowing and masking effect caused by the microfacets of a surface.
// This function uses an optimization based on the Filament PBR document:
// - https://google.github.io/filament/Filament.html.
//
// The calculation is an approximation of:
// G1(normal_dot_view) * G1(normal_dot_scatter) / (4 * normal_dot_view * normal_dot_scatter),
// where G1 represents the unidirectional shadowing.
float ggx_visibility(float roughness, float normal_dot_view, float normal_dot_scatter) {
    float lambda_view = normal_dot_scatter * (normal_dot_view * (1.0 - roughness) + roughness);
    float lambda_light = normal_dot_view * (normal_dot_scatter * (1.0 - roughness) + roughness);
    return 0.5 / (normal_dot_view + normal_dot_scatter);
}

// Calculates the geometric shadowing of rays hitting a surface point based on roughness
// and incident angle. This is typically referred to as G1.
//
// Originally described by Bruce D. Smith in 1967 (Smith, 1967) - https://ieeexplore.ieee.org/document/1138991.
float ggx_visibility_unidirectional(float roughness, float cos_theta) {
    float roughness_squared = roughness * roughness;
    float denom = cos_theta + sqrt(roughness_squared + (1.0 - roughness_squared) * (cos_theta * cos_theta));
    return (2.0 * cos_theta) / denom;
}

// Models the distribution of microfacet orientations, known as the GGX distribution.
//
// Originally described by Bruce Walter in 2007 (Walter, 2007) - https://www.graphics.cornell.edu/~bjw/microfacetbsdf.pdf.
float ggx_normal_dist(float roughness, float normal_dot_half) {
    float a = normal_dot_half * roughness;
    float k = roughness / (1.0 - normal_dot_half * normal_dot_half + a * a);
    return k * k * INVERSE_PI;
}

vec3 ggx_specular(SurfaceProperties surface, ScatterProperties scatter) {
    vec3 fresnel = fresnel_schlick(surface.fresnel_min, surface.fresnel_max, scatter.view_dot_half);
    if (surface.roughness <= ROUGHNESS_DELTA_CUTOFF) return fresnel;
    return ggx_normal_dist(surface.roughness, scatter.normal_dot_half)
        * ggx_visibility(surface.roughness, surface.normal_dot_view, scatter.normal_dot_scatter)
        * fresnel;
}

// The Burley diffuse model adds highlights based on surface roughness to the Lambertian diffuse model.
// It is not energy conserving like the Lambertian model, but perceptually looks quite a bit better.
//
// Source: Brent Burley, 2012 (Burley, 2012) - https://disneyanimation.com/publications/physically-based-shading-at-disney/
vec3 burley_diffuse(SurfaceProperties surface, ScatterProperties scatter) {
    vec3 light_scatter = fresnel_schlick(vec3(1.0), surface.fresnel_max, scatter.normal_dot_scatter);
    vec3 view_scatter  = fresnel_schlick(vec3(1.0), surface.fresnel_max, surface.normal_dot_view);
    vec3 diffuse_color = surface.albedo * (1.0 - surface.metallic);
    return diffuse_color * light_scatter * view_scatter * INVERSE_PI;
}

vec3 fresnel_min(float ior, vec3 albedo, float metallic) {
    return mix(vec3(pow2((ior - 1.0) / (ior + 1.0))), albedo, metallic);
}

vec3 brdf(SurfaceProperties surface, ScatterProperties scatter) {
    return ggx_specular(surface, scatter) + burley_diffuse(surface, scatter);
}

float ggx_normal_dist_anisotropic(
    float normal_dot_half, float tangent_dot_half, float bitangent_dot_half,
    float tangent_roughness, float bitangent_roughness
) {
    float roughness = tangent_roughness * bitangent_roughness;
    vec3 f = vec3(bitangent_roughness * tangent_dot_half, tangent_roughness * bitangent_dot_half, roughness * normal_dot_half);
    return roughness * pow2(roughness / length_squared(f)) * INVERSE_PI;
}

float ggx_visibility_anisotropic(
    float normal_dot_scatter, float normal_dot_view, float tangent_dot_view, float bitangent_dot_view,
    float tangent_dot_scatter, float bitangent_dot_scatter, float tangent_roughness, float bitangent_roughness
) {
    float view = normal_dot_scatter * length(
        vec3(tangent_roughness * tangent_dot_view, bitangent_roughness * bitangent_dot_view, normal_dot_view)
    );
    float scatter = normal_dot_view * length(
        vec3(tangent_roughness * tangent_dot_scatter, bitangent_roughness * bitangent_dot_scatter, normal_dot_scatter)
    );
    return saturate(0.5 / (view + scatter));
}

vec3 ggx_specular_anisotropic(Basis surface_basis, SurfaceProperties surface, ScatterProperties scatter) {
    vec3 fresnel = fresnel_schlick(surface.fresnel_min, surface.fresnel_max, scatter.view_dot_half);
    // if (surface.roughness <= ROUGHNESS_DELTA_CUTOFF) return fresnel;
    vec3 tangent = transform_to_basis(surface_basis, normalize(vec3(surface.anisotropy_direction, 0.0)));
    vec3 bitangent = normalize(cross(surface_basis.normal, tangent));

    float tangent_roughness = mix(surface.roughness, 1.0, pow2(surface.anisotropy_strength));
    float bitangent_roughness = clamp(surface.roughness, 0.001, 1.0);

    float normal_dist = ggx_normal_dist_anisotropic(
        scatter.normal_dot_half, dot(tangent, scatter.half_vector), dot(bitangent, scatter.half_vector),
        tangent_roughness, bitangent_roughness
    );
    float visibility = ggx_visibility_anisotropic(
        scatter.normal_dot_scatter, surface.normal_dot_view,
        dot(tangent, surface.view_direction), dot(bitangent, surface.view_direction),
        dot(tangent, scatter.direction), dot(bitangent, scatter.direction),
        tangent_roughness, bitangent_roughness
    );
    return vec3(normal_dist * visibility * fresnel);
}

vec3 brdf(Basis surface_basis, SurfaceProperties surface, ScatterProperties scatter) {
    vec3 specular;
    if (surface.is_anisotropic) {
        specular = ggx_specular_anisotropic(surface_basis, surface, scatter);
    } else {
        specular = ggx_specular(surface, scatter);
    }
    return specular + burley_diffuse(surface, scatter);
}

#define LobeType uint
const LobeType SPECULAR_LOBE = 1;
const LobeType DIFFUSE_LOBE = 2;

#endif
