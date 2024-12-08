#ifndef SPECULAR_BRDF
#define SPECULAR_BRDF

#include "fresnel"
#include "microfacet"
#include "random"

float specular_ior(MaterialConstants material) {
    return material.specular_ior /
        mix(1.0, material.coat_ior, material.coat_weight); // OpenPBR Eq. (60)
}

float specular_ior_ratio(MaterialConstants material) {
    float ior = specular_ior(material);
    float fresnel_at_normal = fresnel_at_normal_incident(ior);
    float clamped_specular_weight =
        clamp(material.specular_weight, 0.0, 1.0 / max(fresnel_at_normal, DENOM_TOLERANCE));
    float epsilon =
        sign(ior - 1.0) * sqrt(clamped_specular_weight * fresnel_at_normal); // OpenPBR Eq. (26)
    return (1.0 + epsilon) / max(1.0 - epsilon, DENOM_TOLERANCE);            // OpenPBR Eq. (26)
}

vec3 specular_fresnel_with_coat(MaterialConstants material, float cos_theta, float ior) {
    float coat_refract_angle =
        sqrt(1.0 - (1.0 - pow2(cos_theta)) / pow2(material.coat_ior)); // OpenPBR Eq. (75)
    float ior_mix = mix(material.specular_ior, ior, material.coat_weight);
    return vec3(fresnel_dielectric(coat_refract_angle, ior_mix));
}

vec3 specular_brdf(
    MaterialConstants material, vec3 wi, vec3 wo, vec3 wi_rotated, vec3 wo_rotated,
    vec3 microfacet_normal, vec2 alpha, float ior, inout float density
) {
    float distribution = microfacet_distribution(alpha, microfacet_normal);
    float visible_normals = distribution * microfacet_masking(alpha, wi_rotated) *
        max(0.0, dot(wi_rotated, microfacet_normal)) / max(DENOM_TOLERANCE, cos_theta(wi_rotated));
    float jacobian = 1.0 / max(abs(4.0 * dot(wi_rotated, microfacet_normal)), DENOM_TOLERANCE);
    density = visible_normals * jacobian;
    vec3 fresnel;
    if (cos_theta(wi_rotated) > 0.0) {
        fresnel =
            specular_fresnel_with_coat(material, abs(dot(wi_rotated, microfacet_normal)), ior);
    } else {
        fresnel = vec3(fresnel_dielectric(abs(dot(wi_rotated, microfacet_normal)), ior));
    }
    vec3 brdf = fresnel * distribution * microfacet_visibility(alpha, wi_rotated, wo_rotated) /
        max(4.0 * abs(cos_theta(wo)) * abs(cos_theta(wi)), DENOM_TOLERANCE);
    return brdf * material.specular_color.rgb;
}

vec3 specular_brdf_evaluate(MaterialConstants material, vec3 wi, vec3 wo, inout float density) {
    if (!is_same_hemisphere(wo, wi))
        return vec3(0.0);
    float ior_ratio = specular_ior_ratio(material);
    float ior = cos_theta(wi) > 0.0 ? ior_ratio : 1.0 / ior_ratio;
    if (abs(ior - 1.0) < IOR_EPSILON)
        return vec3(0.0);
    vec2 alpha = microfacet_roughness_to_alpha(
        material.specular_roughness, material.specular_roughness_anisotropy
    );
    LocalRotation rotation = local_frame_rotation(2.0 * PI * material.specular_rotation);
    vec3 wi_rotated = rotate_local(wi, rotation), wo_rotated = rotate_local(wo, rotation);
    vec3 microfacet_normal = normalize(wo_rotated + wi_rotated);
    if (dot(microfacet_normal, wi_rotated) * cos_theta(wi_rotated) < 0.0 ||
        dot(microfacet_normal, wo_rotated) * cos_theta(wo_rotated) < 0.0)
        return vec3(0.0);
    return specular_brdf(
        material, wi, wo, wi_rotated, wo_rotated, microfacet_normal, alpha, ior, density
    );
}

vec3 specular_brdf_sample(
    MaterialConstants material, vec3 wi, inout Generator generator, out vec3 wo, out float density
) {
    float ior_ratio = specular_ior_ratio(material);
    float ior = cos_theta(wi) > 0.0 ? ior_ratio : 1.0 / ior_ratio;
    if (abs(ior - 1.0) < IOR_EPSILON)
        return vec3(0.0);
    vec2 alpha = microfacet_roughness_to_alpha(
        material.specular_roughness, material.specular_roughness_anisotropy
    );
    LocalRotation rotation = local_frame_rotation(2.0 * PI * material.specular_rotation);
    vec3 wi_rotated = rotate_local(wi, rotation);
    vec3 microfacet_normal;
    if (cos_theta(wi_rotated) > 0.0) {
        microfacet_normal = microfacet_sample(alpha, wi_rotated, random_vec2(generator));
    } else {
        microfacet_normal =
            microfacet_sample(alpha, vec3(wi_rotated.xy, -wi_rotated.z), random_vec2(generator));
        microfacet_normal.z *= -1.0;
    }
    vec3 wo_rotated = -reflect(wi_rotated, microfacet_normal);
    if (!is_same_hemisphere(wi_rotated, wo_rotated)) {
        density = 1.0;
        return vec3(0.0);
    }
    wo = inverse_rotate_local(wo_rotated, rotation);
    return specular_brdf(
        material, wi, wo, wi_rotated, wo_rotated, microfacet_normal, alpha, ior, density
    );
}

vec3 specular_brdf_albedo(MaterialConstants material, vec3 wi, inout Generator generator) {
    if (abs(specular_ior_ratio(material) - 1.0) < IOR_EPSILON)
        return vec3(0.0);
    const int num_samples = 1;
    vec3 albedo = vec3(0.0);
    for (int n = 0; n < num_samples; ++n) {
        vec3 wo;
        float density;
        vec3 brdf = specular_brdf_sample(material, wi, generator, wo, density);
        if (length(brdf) > RADIANCE_EPSILON)
            albedo += brdf * abs(cos_theta(wo)) / max(DENOM_TOLERANCE, density);
    }
    return albedo / float(num_samples);
}

#endif
