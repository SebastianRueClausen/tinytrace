#ifndef FUZZ_BRDF
#define FUZZ_BRDF

#include "fresnel"
#include "random"
#include "sample"

// Practical Multiple-Scattering Sheen Using Linearly Transformed Cosines (Zeltner, et. al. 2022):
// https://tizianzeltner.com/projects/Zeltner2022Practical/

float fuzz_albedo(float x, float y) {
    float s = y * (0.0206607 + 1.58491 * y) / (0.0379424 + y * (1.32227 + y));
    float m = y * (-0.193854 + y * (-1.14885 + y * (1.7932 - 0.95943 * y * y))) / (0.046391 + y);
    float o = y * (0.000654023 + (-0.0207818 + 0.119681 * y) * y) / (1.26264 + y * (-1.92021 + y));
    return exp(-0.5 * pow2((x - m) / s)) / (s * sqrt(2.0 * PI)) + o;
}

vec2 ltc_inverse_coeffs(float x, float y) {
    float a = (2.58126 * x + 0.813703 * y) * y / (1.0 + 0.310327 * x * x + 2.60994 * x * y);
    float b = sqrt(1.0 - x) * (y - 1.0) * y * y * y /
        (0.0000254053 + 1.71228 * x - 1.71506 * x * y + 1.34174 * y * y);
    return vec2(a, b);
}

mat3 orthonormal_basis_ltc(vec3 normal) {
    float length_squared = length_squared(normal.xy);
    vec3 tangent = length_squared > 0.0
        ? vec3(normal.x, normal.y, 0.0) * inversesqrt(length_squared)
        : vec3(1, 0, 0);
    vec3 bitangent = vec3(-tangent.y, tangent.x, 0.0);
    return mat3(tangent, bitangent, vec3(0, 0, 1));
}

vec3 fuzz_density_and_brdf(
    MaterialConstants material, vec3 direction, vec3 wi, float a_inverse, float roughness,
    out float density
) {
    float jacobian = pow2(a_inverse / length_squared(direction));
    density = max(cos_theta(direction), 0.0) * INVERSE_PI * jacobian;
    float albedo = fuzz_albedo(cos_theta(wi), roughness);
    return material.fuzz_color.rgb * albedo * INVERSE_PI * jacobian;
}

vec3 fuzz_brdf_evaluate(MaterialConstants material, vec3 wi, vec3 wo, inout float density) {
    if (cos_theta(wi) < DENOM_TOLERANCE || cos_theta(wo) < DENOM_TOLERANCE)
        return vec3(0.0);
    float roughness = clamp(material.fuzz_roughness, 0.01, 1.0);
    vec3 w = transpose(orthonormal_basis_ltc(wi)) * wo;
    vec2 inverse_coeffs = ltc_inverse_coeffs(cos_theta(wi), roughness);
    vec3 direction =
        vec3(inverse_coeffs.x * w.x + inverse_coeffs.y * w.z, inverse_coeffs.x * w.y, w.z);
    return fuzz_density_and_brdf(material, direction, wi, inverse_coeffs.x, roughness, density);
}

vec3 fuzz_brdf_sample(
    MaterialConstants material, vec3 wi, inout Generator generator, out vec3 wo, out float density
) {
    if (cos_theta(wi) < DENOM_TOLERANCE)
        return vec3(0.0);
    float roughness = clamp(material.fuzz_roughness, 0.01, 1.0);
    vec3 direction = cosine_hemisphere_sample(random_vec2(generator));
    vec2 inverse_coeffs = ltc_inverse_coeffs(cos_theta(wi), roughness);
    vec3 w = vec3(
        direction.x / inverse_coeffs.x - direction.z * inverse_coeffs.y / inverse_coeffs.x,
        direction.y / inverse_coeffs.x, direction.z
    );
    wo = orthonormal_basis_ltc(wi) * normalize(w);
    return fuzz_density_and_brdf(material, direction, wi, inverse_coeffs.x, roughness, density);
}

vec3 fuzz_brdf_albedo(MaterialConstants material, vec3 wi) {
    return vec3(fuzz_albedo(cos_theta(wi), material.fuzz_roughness));
}

#endif
