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

vec2 interpolate(vec3 barycentric, f16vec2 a, f16vec2 b, f16vec2 c) {
    return barycentric.x * a + barycentric.y * b + barycentric.z * c;
}

vec3 interpolate(vec3 barycentric, vec3 a, vec3 b, vec3 c) {
    return barycentric.x * a + barycentric.y * b + barycentric.z * c;
}

struct Ray {
    vec3 direction;
    vec3 origin;
};

// Returns the barycentric coordinates of ray triangle intersection.
vec3 triangle_intersection(vec3 triangle[3], Ray ray) {
    vec3 edge_to_origin = ray.origin - triangle[0];
    vec3 edge_2 = triangle[2] - triangle[0];
    vec3 edge_1 = triangle[1] - triangle[0];
    vec3 r = cross(ray.direction, edge_2);
    vec3 s = cross(edge_to_origin, edge_1);
    float inverse_det = 1.0 / dot(r, edge_1);
    float v1 = dot(r, edge_to_origin);
    float v2 = dot(s, ray.direction);
    float b = v1 * inverse_det;
    float c = v2 * inverse_det;
    return vec3(1.0 - b - c, b, c);
}

#endif
