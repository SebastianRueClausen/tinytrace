#ifndef BRDF
#define BRDF

#include "constants"
#include "random"

struct SurfaceProperties {
    float metallic;
    float roughness;
    vec3 albedo;
    vec3 fresnel_min;
    float fresnel_max;
    vec3 view_direction;
    float normal_dot_view;
};

struct ScatterProperties {
    vec3 direction;
    vec3 half_vector;
    float normal_dot_half;
    float normal_dot_scatter;
    float view_dot_half;
};

vec3 fresnel_schlick(vec3 fresnel_min, float fresnel_max, float view_dot_half) {
    float flipped = 1.0 - view_dot_half;
    float flipped_2 = flipped * flipped;
    float flipped_5 = flipped * flipped_2 * flipped_2;
    return fresnel_min + (fresnel_max - fresnel_min) * flipped_5;
}

float ggx_visibility(SurfaceProperties surface, ScatterProperties scatter) {
    float alpha_squared = surface.roughness * surface.roughness;
    float lambda_view = scatter.normal_dot_scatter * sqrt(
        (surface.normal_dot_view - alpha_squared * surface.normal_dot_view)
            * surface.normal_dot_view
            + alpha_squared
    );
    float lambda_light = surface.normal_dot_view * sqrt(
        (scatter.normal_dot_scatter - alpha_squared * scatter.normal_dot_scatter)
            * scatter.normal_dot_scatter
            + alpha_squared
    );
    return 0.5 / (lambda_view + lambda_light);
}

float ggx_normal_dist(SurfaceProperties surface, ScatterProperties scatter) {
    float alpha = scatter.normal_dot_half * surface.roughness;
    float k = surface.roughness / ((1.0 - scatter.normal_dot_half * scatter.normal_dot_half) + alpha * alpha);
    return k * k * (1.0 / PI);
}

vec3 ggx_specular(SurfaceProperties surface, ScatterProperties scatter) {
    float d = ggx_normal_dist(surface, scatter);
    float v = ggx_visibility(surface, scatter);
    vec3 f = fresnel_schlick(surface.fresnel_min, surface.fresnel_max, scatter.view_dot_half);
    return d * v * f;
}

vec3 burley_diffuse(SurfaceProperties surface, ScatterProperties scatter) {
    vec3 light_scatter = fresnel_schlick(vec3(1.0), surface.fresnel_max, scatter.normal_dot_scatter);
    vec3 view_scatter  = fresnel_schlick(vec3(1.0), surface.fresnel_max, surface.normal_dot_view);
    vec3 diffuse_color = surface.albedo * (1.0 - surface.metallic);
    return diffuse_color * light_scatter * view_scatter * (1.0 / PI);
}

#endif