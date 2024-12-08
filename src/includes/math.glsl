#ifndef MATH
#define MATH

// Mathematical constants
const float TAU = 6.283185307179586;
const float PI = 3.1415926535897932;
const float INVERSE_PI = 1.0 / PI;
const float INVERSE_2_PI = 1.0 / (2.0 * PI);
const float PI_OVER_2 = PI / 2.0;
const float PI_OVER_3 = PI / 3.0;
const float PI_OVER_4 = PI / 4.0;

// Numerical constants

// The distance to the next float after 1.0
const float FLT_EPSILON = 1.1920929e-7;

// An orthonormal basis.
struct Basis {
    vec3 normal, tangent, bitangent;
};

vec3 transform_to_basis(Basis basis, vec3 direction) {
    return normalize(
        direction.x * basis.tangent + direction.y * basis.bitangent + direction.z * basis.normal
    );
}

vec3 transform_from_basis(Basis basis, vec3 direction) {
    return normalize(vec3(
        dot(direction, basis.tangent), dot(direction, basis.bitangent), dot(direction, basis.normal)
    ));
}

struct LocalRotation {
    mat2 transform;
    mat2 inverse_transform;
};

LocalRotation local_frame_rotation(in float angle) {
    LocalRotation rotation;
    if (angle == 0.0 || angle == 2.0 * PI) {
        mat2 identity = mat2(1.0, 0.0, 0.0, 1.0);
        rotation.transform = identity;
        rotation.inverse_transform = identity;
    } else {
        float cos_angle = cos(angle), sin_angle = sin(angle);
        rotation.transform = mat2(cos_angle, sin_angle, -sin_angle, cos_angle);
        rotation.inverse_transform = mat2(cos_angle, -sin_angle, sin_angle, cos_angle);
    }
    return rotation;
}

vec3 rotate_local(vec3 local, LocalRotation rotation) {
    return vec3(rotation.transform * local.xy, local.z);
}

vec3 inverse_rotate_local(vec3 rotated, LocalRotation rotation) {
    return vec3(rotation.inverse_transform * rotated.xy, rotated.z);
}

vec3 gram_schmidt(vec3 normal, vec3 tangent) {
    return normalize(tangent - normal * dot(normal, tangent));
}

float luminance(vec3 color) { return dot(color, vec3(0.2126, 0.7152, 0.0722)); }
vec3 safe_normalize(vec3 vector) { return vector / max(length(vector), 1e-8); }
float saturate(float value) { return clamp(value, 0.0, 1.0); }
float pow2(float value) { return value * value; }
vec3 pow2(vec3 value) { return value * value; }
float pow5(float value) { return value * pow2(value) * pow2(value); }
float pow6(float value) { return value * pow5(value); }
float min_component(vec3 v) { return min(v.x, min(v.y, v.z)); }
float max_component(vec3 v) { return max(v.x, max(v.y, v.z)); }
float length_squared(vec2 value) { return dot(value, value); }
float length_squared(vec3 value) { return dot(value, value); }
float cos_theta(vec3 direction) { return direction.z; }
float cos_theta_squared(vec3 direction) { return pow2(direction.z); }
float sin_theta_squared(vec3 direction) { return max(0.0, 1.0 - cos_theta_squared(direction)); }
float sin_theta(vec3 direction) { return sqrt(sin_theta_squared(direction)); }
float tan_theta_squared(vec3 direction) {
    return sin_theta_squared(direction) / cos_theta_squared(direction);
}
float sin_phi(vec3 direction) {
    float sin_theta = sin_theta(direction);
    return abs(sin_theta) < 0.0001 ? 0.0 : clamp(direction.y / sin_theta, -1.0, 1.0);
}
float cos_phi(vec3 direction) {
    float sin_theta = sin_theta(direction);
    return abs(sin_theta) < 0.0001 ? 1.0 : clamp(direction.x / sin_theta, -1.0, 1.0);
}
bool is_same_hemisphere(vec3 a, vec3 b) { return cos_theta(a) * cos_theta(b) >= 0.0; }

vec2 interpolate(vec3 barycentric, f16vec2 a, f16vec2 b, f16vec2 c) {
    return barycentric.x * a + barycentric.y * b + barycentric.z * c;
}

vec3 interpolate(vec3 barycentric, vec3 a, vec3 b, vec3 c) {
    return barycentric.x * a + barycentric.y * b + barycentric.z * c;
}

vec3 rotate_around_z_axis(vec3 vector, float angle) {
    float sin = sin(angle), cos = cos(angle);
    return vec3(vector.x * cos - vector.y * sin, vector.x * sin - vector.y * cos, vector.z);
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
