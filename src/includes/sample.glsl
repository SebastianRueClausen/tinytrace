#ifndef SAMPLE
#define SAMPLE

#include "brdf"
#include "math"
#include "random"

#define SampleStrategy uint
const SampleStrategy UNIFORM_HEMISPHERE_SAMPLING = 1;
const SampleStrategy COSINE_HEMISPHERE_SAMPLING = 2;
const SampleStrategy BRDF_SAMPLING = 2;

vec3 uniform_hemisphere_sample(inout Generator generator) {
    float z = random_float(generator);
    float r = sqrt(max(0.0, 1.0 - z * z));
    float phi = 2.0 * PI * random_float(generator);
    return vec3(r * cos(phi), r * sin(phi), z);
}

vec2 sample_disk(vec2 random) {
    vec2 u = 2.0 * random - 1.0;
    if (u.x == 0.0 && u.y == 0.0)
        return vec2(0.0);
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

vec2 sample_disk(inout Generator generator) { return sample_disk(random_vec2(generator)); }

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

vec3 cosine_hemisphere_sample(vec2 random) {
    float r = sqrt(random.x);
    float theta = 2.0 * PI * random.y;
    float x = r * cos(theta);
    float y = r * sin(theta);
    float z = sqrt(max(0.0, 1.0 - x * x - y * y));
    return vec3(x, y, z);
}

float cosine_hemisphere_density(float normal_dot_scatter) {
    if (normal_dot_scatter <= 1e-6)
        return 1e-6 / PI;
    return normal_dot_scatter / PI;
}

#endif
