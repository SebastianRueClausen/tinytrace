#ifndef MATH
#define MATH

const float TAU = 6.283185307179586;
const float PI = 3.1415926535897932;
const float INVERSE_PI = 1.0 / PI;
const float INVERSE_2_PI = 1.0 / (2.0 * PI);
const float PI_OVER_2 = PI / 2.0;
const float PI_OVER_3 = PI / 3.0;
const float PI_OVER_4 = PI / 4.0;

// An orthonormal basis.
struct Basis {
    vec3 normal, tangent, bitangent;
};

vec3 transform_to_basis(Basis basis, vec3 direction) {
    return normalize(direction.x * basis.tangent + direction.y * basis.bitangent + direction.z * basis.normal);
}

vec3 transform_from_basis(Basis basis, vec3 direction) {
    return normalize(vec3(dot(direction, basis.tangent), dot(direction, basis.bitangent), dot(direction, basis.normal)));
}

// Use Gram Schmidt to find the orthonormal vector to `normal` closests to `tangent`.
vec3 gram_schmidt(vec3 normal, vec3 tangent) {
    return normalize(tangent - normal * dot(normal, tangent));
}

float saturate(float value) { return clamp(value, 0.0, 1.0); }
float pow2(float value) { return value * value; }
float length_squared(vec2 value) { return dot(value, value); }
float length_squared(vec3 value) { return dot(value, value); }

#endif
