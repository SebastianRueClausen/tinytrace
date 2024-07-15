#ifndef SAMPLE
#define SAMPLE

#include "math"
#include "random"

struct Sample {
    float pdf;
    vec3 scatter;
};

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

#endif