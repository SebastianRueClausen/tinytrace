#include "constants"
#include "octahedron"

struct EmissiveTriangle {
    int16_t positions[3][3];
    f16vec2 texcoords[3];
    uint16_t padding;
    uint instance;
};

struct Vertex {
    f16vec2 texcoord;
    uint tangent_frame;
};

struct TangentFrame {
    vec3 normal;
    vec3 tangent;
    float bitangent_sign;
};

vec3 deterministic_orthonormal_vector(vec3 normal) {
    if (abs(normal.x) > abs(normal.z)) {
        return vec3(-normal.y, normal.x, 0.0);
    } else {
        return vec3(0.0, -normal.z, normal.y);
    }
}

float dequantize_unorm(uint bits, uint value) {
    return float(value) / float((1 << bits) - 1);
}

float dequantize_snorm(uint bits, int value) {
    return float(value) / float((1 << (bits - 1)) - 1);
}

float dequantize_snorm(int16_t value) {
    return dequantize_snorm(16, int(value));
}

vec3 dequantize_snorm(int16_t values[3]) {
    return vec3(dequantize_snorm(values[0]), dequantize_snorm(values[1]), dequantize_snorm(values[2]));
}

TangentFrame decode_tangent_frame(uint encoded) {
    const uint UV_MASK = 0x3ff;
    const uint ANGLE_MASK = 0x7ff;
    TangentFrame tangent_frame;

    vec2 uv = vec2(
        dequantize_unorm(10, encoded & UV_MASK),
        dequantize_unorm(10, (encoded >> 10) & UV_MASK)
    );

    tangent_frame.normal = octahedron_decode(uv);
    vec3 orthonormal = deterministic_orthonormal_vector(tangent_frame.normal);
    float angle = dequantize_unorm(11, (encoded >> 20) & ANGLE_MASK) * TAU;

    tangent_frame.tangent = orthonormal
        * cos(angle) + cross(tangent_frame.normal, orthonormal)
        * sin(angle);
    tangent_frame.bitangent_sign = encoded >> 31 == 1 ? 1.0 : -1.0;

    return tangent_frame;
}

struct Material {
    uint16_t albedo_texture, normal_texture, specular_texture, emissive_texture;
    float16_t base_color[4], emissive[3];
    float16_t metallic, roughness, ior;
};

struct BoundingSphere {
    float x, y, z;
    float radius;
};

struct Instance {
    mat4 transform, inverse_transform, normal_transform;
    uint vertex_offset, index_offset, material, padding;
};

const uint INVALID_INDEX = 4294967295;
