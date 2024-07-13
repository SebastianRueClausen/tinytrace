#include "constants"

struct Vertex {
    f16vec2 texcoord;
    uint tangent_frame;
};

struct TangentFrame {
    f16vec3 normal;
    f16vec3 tangent;
    float16_t bitangent_sign;
};

f16vec3 deterministic_orthonormal_vector(f16vec3 normal) {
    if (abs(normal.x) > abs(normal.z)) {
        return f16vec3(-normal.y, normal.x, 0.0hf);
    } else {
        return f16vec3(0.0hf, -normal.z, normal.y);
    }
}

float16_t dequantize_unorm(uint bits, uint value) {
    return float16_t(value) / float16_t((1 << bits) - 1);
}

f16vec2 octahedron_encode(f16vec3 vector) {
    f16vec3 normal = vector / (abs(vector.x) + abs(vector.y) + abs(vector.z));
    if (normal.z < 0.0hf) {
        float16_t x = normal.x >= 0.0hf ? 1.0hf : 0.0hf;
        float16_t y = normal.y >= 0.0hf ? 1.0hf : 0.0hf;
        f16vec2 wrapped = (f16vec2(1.0) - abs(normal.yx)) * f16vec2(x, y);
        return wrapped * 0.5hf + 0.5hf;
    } else {
        return normal.xy * 0.5hf + 0.5hf;
    }
}

f16vec3 octahedron_decode(f16vec2 octahedron) {
    f16vec2 scaled = octahedron * 2.0hf - 1.0hf;
    f16vec3 normal = f16vec3(scaled.xy, 1.0hf - abs(scaled.x) - abs(scaled.y));
    float16_t t = clamp(-normal.z, 0.0hf, 1.0hf);
    normal.x += normal.x >= 0.0hf ? -t : t;
    normal.y += normal.y >= 0.0hf ? -t : t;
    return normalize(normal);
}

TangentFrame decode_tangent_frame(uint encoded) {
    const uint UV_MASK = 0x3ff;
    const uint ANGLE_MASK = 0x7ff;
    TangentFrame tangent_frame;

    f16vec2 uv = f16vec2(
        dequantize_unorm(10, encoded & UV_MASK),
        dequantize_unorm(10, (encoded >> 10) & UV_MASK)
    );

    tangent_frame.normal = octahedron_decode(uv);
    f16vec3 orthonormal = deterministic_orthonormal_vector(tangent_frame.normal);
    float16_t angle = dequantize_unorm(11, (encoded >> 20) & ANGLE_MASK) * TAU;

    tangent_frame.tangent = orthonormal
        * cos(angle) + cross(tangent_frame.normal, orthonormal)
        * sin(angle);
    tangent_frame.bitangent_sign = encoded >> 31 == 1 ? 1.0hf : -1.0hf;

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
    mat4 transform, inverse_transform;
    f16mat4 normal_transform;
    uint vertex_offset, index_offset, color_offset, material;
};

const uint INVALID_INDEX = 4294967295;