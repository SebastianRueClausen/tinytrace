#ifndef BRDF
#define BRDF

#include "brdf"
#include "math"
#include "microfacet"
#include "random"

struct SurfaceProperties {
    float metallic;
    float roughness;
    vec3 albedo;
    vec3 fresnel_min;
    float fresnel_max;
    vec3 view_direction;
    float anisotropic_rotation;
    float sheen, sheen_tint;
    vec2 alpha;
};

struct ScatterProperties {
    vec3 direction;
    float view_dot_half;
    float scatter_dot_half;
};

ScatterProperties
create_scatter_properties(SurfaceProperties surface, vec3 direction, vec3 normal) {
    ScatterProperties scatter;
    scatter.direction = direction;
    vec3 half_vector = normalize(surface.view_direction + scatter.direction);
    scatter.view_dot_half = dot(surface.view_direction, half_vector);
    scatter.scatter_dot_half = dot(direction, half_vector);
    return scatter;
}

float fresnel_weight(float cos_theta) { return pow5(1.0 - cos_theta); }
vec3 fresnel_schlick(vec3 fresnel_min, float fresnel_max, float cos_theta) {
    return fresnel_min + (fresnel_max - fresnel_min) * fresnel_weight(cos_theta);
}

vec3 specular_lobe(
    vec3 local_view, vec3 local_scatter, SurfaceProperties surface, ScatterProperties scatter
) {
    vec3 fresnel = fresnel_schlick(surface.fresnel_min, surface.fresnel_max, scatter.view_dot_half);
    float visibility = microfacet_visibility(surface.alpha, local_scatter, local_view);
    float distribution =
        microfacet_distribution(surface.alpha, normalize(local_view + local_scatter));
    return (visibility * distribution * fresnel) /
        (4.0 * cos_theta(local_view) * cos_theta(local_scatter));
}

vec3 diffuse_lobe(
    vec3 local_view, vec3 local_scatter, SurfaceProperties surface, ScatterProperties scatter
) {
    float view_fresnel = fresnel_weight(abs(cos_theta(local_view))),
          scatter_fresnel = fresnel_weight(abs(cos_theta(local_scatter)));
    float retro_reflection = 2.0 * surface.roughness * pow2(scatter.scatter_dot_half);
    float lambert_strength = (1.0 - 0.5 * scatter_fresnel) * (1.0 - 0.5 * view_fresnel);
    float retro_strength = retro_reflection *
        (scatter_fresnel + view_fresnel + scatter_fresnel * view_fresnel * (retro_reflection - 1.0)
        );
    float sheen_strength = pow5(1.0 - abs(scatter.scatter_dot_half)) * surface.sheen;
    vec3 sheen_color =
        mix(vec3(1.0), surface.albedo / luminance(surface.albedo), surface.sheen_tint);
    vec3 diffuse = (lambert_strength + 0.5 * retro_strength) * surface.albedo * INVERSE_PI +
        sheen_strength * sheen_color;
    return diffuse * (1.0 - surface.metallic);
}

vec3 fresnel_min(float ior, vec3 albedo, float metallic) {
    return mix(vec3(pow2((ior - 1.0) / (ior + 1.0))), albedo, metallic);
}

vec3 brdf(Basis basis, SurfaceProperties surface, ScatterProperties scatter) {
    vec3 local_view = transform_from_basis(basis, surface.view_direction);
    vec3 local_scatter = transform_from_basis(basis, scatter.direction);
    return specular_lobe(local_view, local_scatter, surface, scatter) +
        diffuse_lobe(local_view, local_scatter, surface, scatter);
}

#define LobeType uint
const LobeType SPECULAR_LOBE = 1;
const LobeType DIFFUSE_LOBE = 2;

#endif
