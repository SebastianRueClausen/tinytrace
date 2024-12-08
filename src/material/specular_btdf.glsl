#ifndef SPECULAR_BTDF
#define SPECULAR_BTDF

#include "fresnel"
#include "microfacet"
#include "random"

vec3 specular_btdf(
    MaterialConstants material, vec3 microfacet_normal, vec2 alpha, vec3 wi, vec3 wo,
    vec3 wi_rotated, vec3 wo_rotated, float ior, inout float density
) {
    float jacobian = pow2(ior) * abs(cos_theta(wi)) /
        max(pow2(cos_theta(wo) + ior * cos_theta(wi)), DENOM_TOLERANCE);
    float distribution = microfacet_distribution(alpha, microfacet_normal);
    float visible_normals = distribution * microfacet_masking(alpha, wi_rotated) *
        max(0.0, dot(wi_rotated, microfacet_normal)) /
        max(DENOM_TOLERANCE, abs(cos_theta(wi_rotated)));
    density = visible_normals * jacobian;
    float visibility = microfacet_visibility(alpha, wi_rotated, wo_rotated);
    float transmission =
        max(0.0, 1.0 - fresnel_dielectric(abs(dot(wi_rotated, microfacet_normal)), 1.0 / ior));
    float btdf = transmission * abs(dot(wi_rotated, microfacet_normal)) * jacobian * visibility *
        distribution / max(abs(cos_theta(wo)) * abs(cos_theta(wi)), DENOM_TOLERANCE);
    return btdf *
        (material.transmission_depth == 0.0 ? material.transmission_color.rgb : vec3(1.0));
}

vec3 specular_btdf_evaluate(MaterialConstants material, vec3 wi, vec3 wo, inout float density) {
    if (is_same_hemisphere(wo, wi)) {
        density = 1.0;
        return vec3(0.0);
    }
    float ior = cos_theta(wi) > 0.0 ? 1.0 / material.specular_ior : material.specular_ior;
    if (abs(ior - 1.0) < IOR_EPSILON) {
        vec3 tint =
            material.transmission_depth == 0.0 ? material.transmission_color.rgb : vec3(1.0);
        density = 1.0 / PDF_EPSILON;
        return tint * density / max(DENOM_TOLERANCE, abs(cos_theta(wo)));
    }
    LocalRotation rotation = local_frame_rotation(2.0 * PI * material.specular_rotation);
    vec3 wi_rotated = rotate_local(wi, rotation), wo_rotated = rotate_local(wo, rotation);
    vec3 microfacet_normal = -wo_rotated - ior * wi_rotated;
    if (dot(microfacet_normal, microfacet_normal) == 0.0)
        return vec3(0.0);
    microfacet_normal = safe_normalize(microfacet_normal);
    if (cos_theta(microfacet_normal) <= 0.0)
        microfacet_normal *= -1.0;
    vec2 alpha = microfacet_roughness_to_alpha(
        material.specular_roughness, material.specular_roughness_anisotropy
    );
    return specular_btdf(
        material, microfacet_normal, alpha, wi, wo, wi_rotated, wo_rotated, ior, density
    );
}

// Computes refraction direction using Snell's law. Returns false if the ray isn't refracted.
// Source: https://pbr-book.org/3ed-2018/Reflection_Models/Specular_Reflection_and_Transmission
bool refraction_direction(vec3 normal, float ior, vec3 wi, inout vec3 direction) {
    float cos_theta_in = dot(wi, normal);
    float sin_theta_in_squared = max(0.0, 1.0 - pow2(cos_theta_in));
    float sin_theta_tr_squared = pow2(ior) * sin_theta_in_squared;
    if (sin_theta_tr_squared >= 1.0)
        return false;
    float cos_theta_transmission = sqrt(1.0 - sin_theta_tr_squared);
    direction = ior * -wi + (ior * cos_theta_in - cos_theta_transmission) * normal;
    return true;
}

vec3 specular_btdf_sample(
    MaterialConstants material, vec3 wi, inout Generator generator, out vec3 wo, out float density
) {
    float ior = cos_theta(wi) > 0.0 ? 1.0 / material.specular_ior : material.specular_ior;
    if (abs(ior - 1.0) < IOR_EPSILON) {
        wo = -wi;
        density = 1.0 / PDF_EPSILON;
        vec3 tint =
            material.transmission_depth == 0.0 ? material.transmission_color.rgb : vec3(1.0);
        return tint * density / max(DENOM_TOLERANCE, abs(cos_theta(wo)));
    }
    LocalRotation rotation = local_frame_rotation(2.0 * PI * material.specular_rotation);
    vec3 wi_rotated = rotate_local(wi, rotation);
    vec2 alpha = microfacet_roughness_to_alpha(
        material.specular_roughness, material.specular_roughness_anisotropy
    );
    vec3 microfacet_normal;
    if (cos_theta(wi_rotated) > 0.0) {
        microfacet_normal = microfacet_sample(alpha, wi_rotated, random_vec2(generator));
    } else {
        vec3 wi_rotated_reflected = vec3(wi_rotated.xy, -cos_theta(wi_rotated));
        microfacet_normal = microfacet_sample(alpha, wi_rotated_reflected, random_vec2(generator));
        microfacet_normal.z *= -1.0;
    }
    vec3 refract_direction;
    if (!refraction_direction(microfacet_normal, ior, wi_rotated, refract_direction)) {
        density = PDF_EPSILON;
        return vec3(0.0);
    }
    vec3 wo_rotated = -safe_normalize(refract_direction);
    wo = inverse_rotate_local(wo_rotated, rotation);
    return specular_btdf(
        material, microfacet_normal, alpha, wi, wo, wi_rotated, wo_rotated, ior, density
    );
}

vec3 specular_btdf_albedo(MaterialConstants material, vec3 wi, inout Generator generator) {
    if (abs(material.specular_ior - 1.0) < IOR_EPSILON) {
        return material.transmission_depth == 0.0 ? material.transmission_color.rgb : vec3(1.0);
    }
    const int num_samples = 1;
    vec3 albedo = vec3(0.0);
    for (int n = 0; n < num_samples; ++n) {
        vec3 wo;
        float density;
        vec3 btdf = specular_btdf_sample(material, wi, generator, wo, density);
        if (length(btdf) > RADIANCE_EPSILON)
            albedo += btdf * abs(cos_theta(wo)) / max(DENOM_TOLERANCE, density);
    }
    return albedo / float(num_samples);
}

#endif
