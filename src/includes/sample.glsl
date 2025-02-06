#ifndef SAMPLE
#define SAMPLE

#include "math"
#include "random"

vec3 sample_triangle(vec2 random) {
    random /= 2.0;
    float offset = random.y - random.x;
    if (offset > 0.0) {
        random.y += offset;
    } else {
        random.x -= offset;
    }
    return vec3(random.x, random.y, 1.0 - random.x - random.y);
}

vec3 cosine_hemisphere_sample(vec2 random) {
    float r = sqrt(random.x);
    float theta = 2.0 * PI * random.y;
    float x = r * cos(theta), y = r * sin(theta);
    float z = sqrt(max(0.0, 1.0 - x * x - y * y));
    return vec3(x, y, z);
}

float cosine_hemisphere_density(float normal_dot_scatter) {
    if (normal_dot_scatter <= 1e-6)
        return 1e-6 / PI;
    return normal_dot_scatter / PI;
}

#endif
