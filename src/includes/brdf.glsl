#ifndef BRDF
#define BRDF

#include "constants"

struct SurfaceProperties {
    float16_t metallic;
    float16_t roughness;
    f16vec3 albedo;
    f16vec3 fresnel_min;
    float16_t fresnel_max;
    f16vec3 view_direction;
    float16_t normal_dot_view;
};

struct ScatterProperties {
    f16vec3 direction;
    f16vec3 half_vector;
    float16_t normal_dot_half;
    float16_t normal_dot_scatter;
    float16_t view_dot_half;
};

f16vec3 fresnel_schlick(f16vec3 fresnel_min, float16_t fresnel_max, float16_t view_dot_half) {
    float16_t flipped = 1.0hf - view_dot_half;
    float16_t flipped_2 = flipped * flipped;
    float16_t flipped_5 = flipped * flipped_2 * flipped_2;
    return fresnel_min + (fresnel_max - fresnel_min) * flipped_5;
}

float16_t ggx_visibility(SurfaceProperties surface, ScatterProperties scatter) {
    float16_t alpha_squared = surface.roughness * surface.roughness;
    float16_t lambda_view = scatter.normal_dot_scatter * sqrt(
        (surface.normal_dot_view - alpha_squared * surface.normal_dot_view)
            * surface.normal_dot_view
            + alpha_squared
    );

    float16_t lambda_light = surface.normal_dot_view * sqrt(
        (scatter.normal_dot_scatter - alpha_squared * scatter.normal_dot_scatter)
            * scatter.normal_dot_scatter
            + alpha_squared
    );

    return 0.5hf / (lambda_view + lambda_light);
}

float16_t ggx_normal_dist(SurfaceProperties surface, ScatterProperties scatter) {
    float16_t alpha = scatter.normal_dot_half * surface.roughness;
    float16_t k = surface.roughness / ((1.0hf - scatter.normal_dot_half * scatter.normal_dot_half) + alpha * alpha);
    return k * k * (1.0hf / PI);
}

f16vec3 ggx_specular(SurfaceProperties surface, ScatterProperties scatter) {
    float16_t d = ggx_normal_dist(surface, scatter);
    float16_t v = ggx_visibility(surface, scatter);
    f16vec3 f = fresnel_schlick(surface.fresnel_min, surface.fresnel_max, scatter.view_dot_half);
    return d * v * f;
}

f16vec3 lambert_diffuse(SurfaceProperties surface) {
    f16vec3 diffuse_color = surface.albedo * (1.0hf - surface.metallic);
    return diffuse_color * (1.0hf / PI);
}

f16vec3 burley_diffuse(SurfaceProperties surface, ScatterProperties scatter) {
    f16vec3 light_scatter = fresnel_schlick(f16vec3(1.0hf), surface.fresnel_max, scatter.normal_dot_scatter);
    f16vec3 view_scatter  = fresnel_schlick(f16vec3(1.0hf), surface.fresnel_max, surface.normal_dot_view);
    return light_scatter * view_scatter * (1.0hf / PI);
}

#endif