#ifndef COAT_BRDF
#define COAT_BRDF

#include "fresnel"
#include "microfacet"
#include "random"

vec3 coat_brdf(
    MaterialConstants material, vec3 wi, vec3 wo, vec3 wi_rotated, vec3 wo_rotated, float ior,
    vec2 alpha, inout float density
) {
    vec3 microfacet_normal = normalize(wo_rotated + wi_rotated);
    float wi_rotated_dot_normal = dot(wi_rotated, microfacet_normal);
    float distribution = microfacet_distribution(alpha, microfacet_normal);
    float visible_normals = distribution * microfacet_masking(alpha, wi_rotated) *
        max(0.0, wi_rotated_dot_normal) / cos_theta(wi_rotated);
    float jacobian = 1.0 / max(abs(4.0 * wi_rotated_dot_normal), DENOM_TOLERANCE);
    density = visible_normals * jacobian;
    float fresnel = fresnel_dielectric(abs(wi_rotated_dot_normal), ior);
    return vec3(fresnel) * distribution * microfacet_visibility(alpha, wi_rotated, wo_rotated) /
        max(4.0 * abs(cos_theta(wo)) * abs(cos_theta(wi)), DENOM_TOLERANCE);
}

vec3 coat_brdf_evaluate(MaterialConstants material, vec3 wi, vec3 wo, inout float density) {
    if (!is_same_hemisphere(wo, wi))
        return vec3(0.0);
    float ior = cos_theta(wi) > 0.0 ? material.coat_ior : 1.0 / material.coat_ior;
    if (abs(ior - 1.0) < IOR_EPSILON)
        return vec3(0.0);
    vec2 alpha =
        microfacet_roughness_to_alpha(material.coat_roughness, material.coat_roughness_anisotropy);
    LocalRotation rotation = local_frame_rotation(2.0 * PI * material.coat_rotation);
    return coat_brdf(
        material, wi, wo, rotate_local(wi, rotation), rotate_local(wo, rotation), ior, alpha,
        density
    );
}

vec3 coat_brdf_sample(
    MaterialConstants material, vec3 wi, inout Generator generator, out vec3 wo, out float density
) {
    float ior = cos_theta(wi) > 0.0 ? material.coat_ior / 1.0 : 1.0 / material.coat_ior;
    if (abs(ior - 1.0) < IOR_EPSILON)
        return vec3(0.0);
    vec2 alpha =
        microfacet_roughness_to_alpha(material.coat_roughness, material.coat_roughness_anisotropy);
    LocalRotation rotation = local_frame_rotation(2.0 * PI * material.coat_rotation);
    vec3 wi_rotated = rotate_local(wi, rotation);
    vec3 microfacet_normal = microfacet_sample(alpha, wi_rotated, random_vec2(generator));
    if (cos_theta(wi_rotated) <= 0.0)
        return vec3(0.0);
    vec3 wo_rotated = -reflect(wi_rotated, microfacet_normal);
    if (!is_same_hemisphere(wi_rotated, wo_rotated))
        return vec3(0.0);
    wo = inverse_rotate_local(wo_rotated, rotation);
    return coat_brdf(material, wi, wo, wi_rotated, wo_rotated, ior, alpha, density);
}

vec3 coat_brdf_albedo(MaterialConstants material, vec3 wi, inout Generator generator) {
    float ior = material.coat_ior / 1.0;
    if (abs(ior - 1.0) < IOR_EPSILON)
        return vec3(0.0);
    const int num_samples = 1;
    vec3 albedo = vec3(0.0);
    for (int n = 0; n < num_samples; ++n) {
        vec3 wo;
        float density;
        vec3 brdf = coat_brdf_sample(material, wi, generator, wo, density);
        if (length(brdf) > RADIANCE_EPSILON)
            albedo += brdf * abs(cos_theta(wo)) / max(PDF_EPSILON, density);
    }
    return albedo / float(num_samples);
}

#endif
