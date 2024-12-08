#ifndef MICROFACET
#define MICROFACET

#include "math"

// D
float microfacet_distribution(vec2 alpha, vec3 microfacet_normal) {
    float tan_squared = tan_theta_squared(microfacet_normal);
    if (isinf(tan_squared))
        return 0.0;
    float e = tan_squared *
        (pow2(cos_phi(microfacet_normal) / alpha.x) + pow2(sin_phi(microfacet_normal) / alpha.y));
    return 1.0 /
        (PI * alpha.x * alpha.y * pow2(cos_theta_squared(microfacet_normal)) * pow2(1.0 + e));
}

float microfacet_lambda(vec2 alpha, vec3 view) {
    float tan_theta_squared = tan_theta_squared(view);
    if (isinf(tan_theta_squared))
        return 0.0;
    float alpha_squared = pow2(cos_phi(view) * alpha.x) + pow2(sin_phi(view) * alpha.y);
    return (sqrt(1.0 + alpha_squared * tan_theta_squared) - 1) / 2.0;
}

// G: Shadowing and masking.
float microfacet_visibility(vec2 alpha, vec3 incident, vec3 outgoing) {
    return 1.0 / (1.0 + microfacet_lambda(alpha, outgoing) + microfacet_lambda(alpha, incident));
}

// G1: Shadowing or masking.
float microfacet_masking(vec2 alpha, vec3 view) {
    return 1.0 / (1.0 + microfacet_lambda(alpha, view));
}

float microfacet_density(vec2 alpha, vec3 view, vec3 microfacet_normal) {
    return microfacet_masking(alpha, view) * saturate(dot(view, microfacet_normal)) *
        microfacet_distribution(alpha, microfacet_normal) / abs(cos_theta(view));
}

vec3 microfacet_sample(vec2 alpha, vec3 view, vec2 random) {
    view = normalize(vec3(view.xy * alpha, view.z));
    float phi = 2.0 * PI * random.x;
    float z = (1.0 - random.y) * (1.0 + view.z) - view.z;
    float sin_theta = sqrt(saturate(1.0 - pow2(z)));
    vec3 half_vector = vec3(sin_theta * cos(phi), sin_theta * sin(phi), z) + view;
    return normalize(vec3(half_vector.xy * alpha, half_vector.z));
}

vec2 microfacet_roughness_to_alpha(float roughness, float anisotropy) {
    float roughness_squared = pow2(roughness);
    float alpha_x = roughness_squared * sqrt(2.0 / (1.0 + pow2(1.0 - anisotropy)));
    float alpha_y = (1.0 - anisotropy) * alpha_x;
    return vec2(max(1e-4, alpha_x), max(1e-4, alpha_y));
}

#endif
