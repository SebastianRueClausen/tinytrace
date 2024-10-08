#ifndef SAMPLE
#define SAMPLE

#include "brdf"
#include "math"
#include "random"

#define SampleStrategy uint
const SampleStrategy UNIFORM_HEMISPHERE_SAMPLING = 1;
const SampleStrategy COSINE_HEMISPHERE_SAMPLING= 2;
const SampleStrategy BRDF_SAMPLING = 2;

vec3 uniform_hemisphere_sample(inout Generator generator) {
    float z = random_float(generator);
    float r = sqrt(max(0.0, 1.0 - z * z));
    float phi = 2.0 * PI * random_float(generator);
    return vec3(r * cos(phi), r * sin(phi), z);
}

vec2 sample_disk(inout Generator generator) {
    vec2 u = 2.0 * random_vec2(generator) - 1.0;
    if (u.x == 0.0 && u.y == 0.0) return vec2(0.0);
    float phi, r;
    if (abs(u.x) > abs(u.y)) {
        r = u.x;
        phi = (u.y / u.x) * PI_OVER_4;
    } else {
        r = u.y;
        phi = PI_OVER_2 - (u.x / u.y) * PI_OVER_4;
    }
    return r * vec2(cos(phi), sin(phi));
}

vec3 sample_triangle(inout Generator generator) {
    float b0 = random_float(generator) / 2.0, b1 = random_float(generator) / 2.0;
    float offset = b1 - b0;
    if (offset > 0.0) {
        b1 += offset;
    } else {
        b0 -= offset;
    }
    return vec3(b0, b1, 1.0 - b0 - b1);
}

vec3 cosine_hemisphere_sample(inout Generator generator) {
    vec2 d = sample_disk(generator);
    return vec3(d, sqrt(max(0.0, 1.0 - d.x * d.x - d.y * d.y)));
}

float cosine_hemisphere_density(float normal_dot_scatter) {
    return normal_dot_scatter / PI;
}

// Importance sample the GGX microfacet BRDF. This first finds a random
// microfacet normal based on the GGX distribution, which is treated as a
// perfect mirror to generate a scatter ray.
//
// A popular solution is to use the VNDF (Visible Normal Distribution Function)
// algorithm presented by (Heitz, 2018) - https://jcgt.org/published/0007/04/01/.
// This is a variation based on (Dupuy, 2023)
// - https://cdrdv2-public.intel.com/782052/sampling-visible-ggx-normals.pdf.
//
// It samples a hemisphere above a certain point, or specifically,
// a spherical cap with height `-view.z`.
//
// `view` is the local view direction. Returns the scatter direction in local
// space. `half_vector` is the sampled normal (or half vector as the
// microfacet is treated as a perfect mirror).
vec3 ggx_sample(vec3 view, vec2 roughness, out vec3 half_vector, inout Generator generator) {
    // Very low roughness values cause issues, so instead, treat it as a delta reflection.
    if (all(lessThanEqual(roughness, vec2(ROUGHNESS_DELTA_CUTOFF)))) return vec3(-view.xy, view.z);
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
float ggx_density(vec3 view, vec3 half_vector, float roughness) {
    // Handle the situation with delta reflections for materials with very low
    // roughness.
    if (roughness <= ROUGHNESS_DELTA_CUTOFF) return 1.0;
	float density = ggx_visibility_unidirectional(roughness, view.z)
        * ggx_normal_dist(roughness, half_vector.z);
	// Account for reflection.
    float reflection_jacobian = 4.0 * dot(view, half_vector);
    return density / reflection_jacobian;
}

#endif
