#ifndef METAL_BRDF
#define METAL_BRDF

#include "fresnel"
#include "random"
#include "specular_brdf"

// The F82-tint Schlick model from [kutz2021]. OpenPBR Eq. (30)
vec3 fresnel_metal(float cos_theta, vec3 fresnel_at_normal, vec3 tint) {
    const float mu_bar = 1.0 / 7.0;
    const float denom = mu_bar * pow6(1.0 - mu_bar);
    return fresnel_schlick(fresnel_at_normal, cos_theta) -
        cos_theta * pow6(1.0 - cos_theta) * (1.0 - tint) *
        fresnel_schlick(fresnel_at_normal, mu_bar) / denom;
}

vec3 metal_brdf(
    MaterialConstants material, vec2 alpha, vec3 microfacet_normal, vec3 wi, vec3 wo,
    vec3 wi_rotated, vec3 wo_rotated, out float density
) {
    float distribution = microfacet_distribution(alpha, microfacet_normal);
    float visible_normals = distribution * microfacet_masking(alpha, wi_rotated) *
        max(0.0, dot(wi_rotated, microfacet_normal)) / max(DENOM_TOLERANCE, cos_theta(wi_rotated));
    float jacobian = 1.0 / max(abs(4.0 * dot(wi_rotated, microfacet_normal)), DENOM_TOLERANCE);
    density = max(PDF_EPSILON, visible_normals * jacobian);
    vec3 fresnel = fresnel_metal(
        abs(dot(wi_rotated, microfacet_normal)), material.base_weight * material.base_color.rgb,
        material.specular_weight * material.specular_color.rgb
    );
    float visibility = microfacet_visibility(alpha, wi_rotated, wo_rotated);
    return fresnel * distribution * visibility /
        max(4.0 * abs(cos_theta(wo)) * abs(cos_theta(wi)), DENOM_TOLERANCE);
}

vec3 metal_brdf_evaluate(MaterialConstants material, vec3 wi, vec3 wo, inout float density) {
    if (cos_theta(wi) < DENOM_TOLERANCE || cos_theta(wo) < DENOM_TOLERANCE) {
        density = PDF_EPSILON;
        return vec3(0.0);
    }
    LocalRotation rotation = local_frame_rotation(2.0 * PI * material.specular_rotation);
    vec3 wi_rotated = rotate_local(wi, rotation), wo_rotated = rotate_local(wo, rotation);
    vec2 alpha = microfacet_roughness_to_alpha(
        material.specular_roughness, material.specular_roughness_anisotropy
    );
    vec3 microfacet_normal = normalize(wo_rotated + wi_rotated);
    return metal_brdf(material, alpha, microfacet_normal, wi, wo, wi_rotated, wo_rotated, density);
}

vec3 metal_brdf_sample(
    MaterialConstants material, vec3 wi, inout Generator generator, out vec3 wo, out float density
) {
    if (cos_theta(wi) < DENOM_TOLERANCE) {
        density = PDF_EPSILON;
        return vec3(0.0);
    }
    vec2 alpha = microfacet_roughness_to_alpha(
        material.specular_roughness, material.specular_roughness_anisotropy
    );
    LocalRotation rotation = local_frame_rotation(2.0 * PI * material.specular_rotation);
    vec3 wi_rotated = rotate_local(wi, rotation);
    vec3 microfacet_normal = microfacet_sample(alpha, wi_rotated, random_vec2(generator));
    vec3 wo_rotated = -reflect(wi_rotated, microfacet_normal);
    if (!is_same_hemisphere(wi_rotated, wo_rotated))
        return vec3(0.0);
    wo = inverse_rotate_local(wo_rotated, rotation);
    return metal_brdf(material, alpha, microfacet_normal, wi, wo, wi_rotated, wo_rotated, density);
}

vec3 metal_brdf_albedo(MaterialConstants material, vec3 wi, inout Generator generator) {
    if (cos_theta(wi) < DENOM_TOLERANCE)
        return vec3(0.0);
    const int num_samples = 1;
    vec3 albedo = vec3(0.0);
    for (int n = 0; n < num_samples; ++n) {
        vec3 wo;
        float density;
        vec3 brdf = metal_brdf_sample(material, wi, generator, wo, density);
        if (length(brdf) > RADIANCE_EPSILON)
            albedo += brdf * abs(cos_theta(wo)) / max(PDF_EPSILON, density);
    }
    return albedo / float(num_samples);
}

#endif
