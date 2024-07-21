#ifndef MATH
#define MATH

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

float saturate(float value) {
    return clamp(value, 0.0, 1.0);
}

#endif
