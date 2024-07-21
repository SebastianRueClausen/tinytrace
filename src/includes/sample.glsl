#ifndef SAMPLE
#define SAMPLE

#include "math"
#include "random"
#include "brdf"

vec2 sample_disk(inout Generator generator) {
    vec2 u = 2.0 * vec2(random_float(generator), random_float(generator)) - 1.0;
    if (u.x == 0.0 && u.y == 0.0) return vec2(0.0);
    float phi, r;
    if (abs(u.x) > abs(u.y)) {
        r = u.x;
        phi = (u.y / u.x) * (PI / 4.0);
    } else {
        r = u.y;
        phi = (PI / 2.0) - (u.x / u.y) * (PI / 4.0);
    }
    return r * vec2(cos(phi), sin(phi));
}

vec3 cosine_hemisphere_sample(inout Generator generator) {
    vec2 d = sample_disk(generator);
    return vec3(d, sqrt(max(0.0, 1.0 - d.x * d.x - d.y * d.y)));
}

float cosine_hemisphere_pdf(float normal_dot_scatter) {
    return normal_dot_scatter / PI;
}

// Importance sample the GGX microfacet BRDF. This first finds a random
// microfacet normal based on the GGX distribution, which is treated as a
// perfect mirror to generate a scatter ray.
//
// A popular solution is to use the VNDF (Visible Normal Distribution Function)
// algorithm presented by (Heitz, 2018)[https://jcgt.org/published/0007/04/01/].
// This is a variation based on (Dupuy, 2023)
// [https://cdrdv2-public.intel.com/782052/sampling-visible-ggx-normals.pdf].
//
// It samples a hemisphere above a certain point, or specifically,
// a spherical cap with height `-view.z`.
//
// `view` is the local view direction. Returns the scatter direction in local
// space. `half_vector` is the sampled normal (or half vector as the
// microfacet is treated as a perfect mirror).
vec3 ggx_sample(vec3 view, float roughness, out vec3 half_vector, inout Generator generator) {
    // Very low roughness values cause issues, so instead, treat it as a delta reflection.
    if (roughness <= ROUGHNESS_DELTA_CUTOFF) return vec3(-view.xy, view.z);
    // Stretch hemisphere based on roughness values. This hemisphere is used to
    // sample normals within a visible area.
    vec3 hemisphere = normalize(vec3(roughness * view.xy, view.z));
    // Sample the hemisphere with sperical cap.
    float phi = 2.0 * PI * random_float(generator);
	float z = ((1.0 - random_float(generator)) * (1.0f + hemisphere.z)) - hemisphere.z;
	float sin_theta = sqrt(saturate(1.0f - z * z));
    // Compute half vector.
    half_vector = vec3(sin_theta * cos(phi), sin_theta * sin(phi), z) + hemisphere;
    // Reproject onto hemisphere.
    half_vector = normalize(vec3(roughness * half_vector.xy, max(0.0, half_vector.z)));
    // Calculate scatter using the half vector as the normal.
    return reflect(-view, half_vector);
}

// The PDF of `ggx_sample`. All vectors are expected to be in local space.
float ggx_pdf(vec3 view, vec3 half_vector, float roughness) {
    // Handle the situation with delta reflections for materials with very low
    // roughness.
    if (roughness <= ROUGHNESS_DELTA_CUTOFF) return 1.0;
	float pdf = ggx_visibility_unidirectional(roughness, view.z)
        * ggx_normal_dist(roughness, half_vector.z);
	// Account for reflection.
    float reflection_jacobian = 4.0 * dot(view, half_vector);
    return pdf / reflection_jacobian;
}

#endif
