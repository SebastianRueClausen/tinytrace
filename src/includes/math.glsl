#ifndef MATH
#define MATH

struct Basis {
    vec3 normal, tangent, bitangent;
};

vec3 transform_to_basis(Basis basis, vec3 direction) {
    return normalize(direction.x * basis.tangent + direction.y * basis.bitangent + direction.z * basis.normal);
}

vec3 gram_schmidt(vec3 normal, vec3 tangent) {
    return normalize(tangent - normal * dot(normal, tangent));
}

#endif