#ifndef DIFFUSE_BRDF
#define DIFFUSE_BRDF

#include "fresnel"
#include "math"
#include "random"
#include "sample"

float oren_nayar(float cos_theta, float sigma_squared) {
    float a = 1.0 - 0.5 * (sigma_squared / (sigma_squared + 0.33));
    float b = 0.45 * sigma_squared / (sigma_squared + 0.09);
    float s = sqrt(1.0 - pow2(cos_theta));
    float g = s * (acos(cos_theta) - s * cos_theta) +
        (s / max(1e-7, cos_theta)) * (1.0 - pow(s, 3.0)) * 2.0 / 3.0;
    return a + (b * INVERSE_PI) * g;
}

vec3 energy_compensated_oren_nayar(vec3 rho, float sigma, vec3 wi, vec3 wo) {
    float sigma_squared = pow2(sigma);
    float s = dot(wi, wo) - cos_theta(wi) * cos_theta(wo);
    float s_over_t = s > 0.0 ? s / max(cos_theta(wi), cos_theta(wo)) : 0.0;
    float a = 1.0 - 0.5 * (sigma_squared / (sigma_squared + 0.33));
    float b = 0.45 * sigma_squared / (sigma_squared + 0.09);
    float on_o = oren_nayar(cos_theta(wo), sigma_squared);
    float on_i = oren_nayar(cos_theta(wi), sigma_squared);
    float average_albedo = a + (2.0 / 3.0 - 64.0 / (45.0 * PI)) * b;
    vec3 rho_ms = pow2(rho) * average_albedo /
        (vec3(1.0) - rho * max(0.0, 1.0 - average_albedo)); // OpenPBR Eq. (39)
    return (rho * INVERSE_PI) * (a + b * s_over_t) +
        (rho_ms * INVERSE_PI) * max(1e-7, 1.0 - on_o) * max(1e-7, 1.0 - on_i) /
        max(1e-7, 1.0 - average_albedo); // OpenPBR Eq. (38)
}

vec3 diffuse_brdf_evaluate(MaterialConstants material, vec3 wi, vec3 wo, inout float density) {
    if (cos_theta(wi) < DENOM_TOLERANCE || cos_theta(wo) < DENOM_TOLERANCE)
        return vec3(0.0);
    density = cosine_hemisphere_density(cos_theta(wo));
    return energy_compensated_oren_nayar(
        material.base_weight * material.base_color.rgb, PI / 2.0 * material.base_diffuse_roughness,
        wi, wo
    );
}

vec3 diffuse_brdf_sample(
    MaterialConstants material, vec3 wi, inout Generator generator, out vec3 wo, out float density
) {
    if (cos_theta(wi) < DENOM_TOLERANCE)
        return vec3(0.0);
    wo = cosine_hemisphere_sample(random_vec2(generator));
    density = cosine_hemisphere_density(cos_theta(wo));
    return energy_compensated_oren_nayar(
        material.base_weight * material.base_color.rgb, PI / 2.0 * material.base_diffuse_roughness,
        wi, wo
    );
}

vec3 diffuse_brdf_albedo(MaterialConstants material, vec3 wi) {
    if (cos_theta(wi) < DENOM_TOLERANCE)
        return vec3(0.0);
    return material.base_weight * material.base_color.rgb;
}

#endif
