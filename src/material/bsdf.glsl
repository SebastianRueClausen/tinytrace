#ifndef BSDF
#define BSDF

#include "material"
#include "random"

#include "coat_brdf"
#include "diffuse_brdf"
#include "fresnel"
#include "fuzz_brdf"
#include "metal_brdf"
#include "specular_brdf"
#include "specular_btdf"

const int FUZZ_BRDF_INDEX = 0;
const int COAT_BRDF_INDEX = 1;
const int METAL_BRDF_INDEX = 2;
const int SPECULAR_BRDF_INDEX = 3;
const int SPECULAR_BTDF_INDEX = 4;
const int DIFFUSE_BRDF_INDEX = 5;
const int LOBE_COUNT = 6;

struct Lobes {
    vec3 weights[LOBE_COUNT];
    vec3 albedos[LOBE_COUNT];
    float probs[LOBE_COUNT];
};

struct LobeDensities {
    float values[7];
};

void lobe_weights(
    MaterialConstants material, vec3 wi, inout Generator generator, inout Lobes lobes
) {
    bool fully_metallic = material.base_metalness == 1.0;
    bool fully_transmissive = material.transmission_weight == 1.0;

    lobes.albedos[FUZZ_BRDF_INDEX] =
        material.fuzz_weight > 0.0 ? fuzz_brdf_albedo(material, wi) : vec3(0.0);
    lobes.albedos[COAT_BRDF_INDEX] =
        material.coat_weight > 0.0 ? coat_brdf_albedo(material, wi, generator) : vec3(0.0);
    lobes.albedos[METAL_BRDF_INDEX] =
        material.base_metalness > 0.0 ? metal_brdf_albedo(material, wi, generator) : vec3(0.0);
    lobes.albedos[SPECULAR_BRDF_INDEX] =
        !fully_metallic ? specular_brdf_albedo(material, wi, generator) : vec3(0.0);
    lobes.albedos[SPECULAR_BTDF_INDEX] = !fully_metallic && material.transmission_weight > 0.0
        ? specular_btdf_albedo(material, wi, generator)
        : vec3(0.0);
    lobes.albedos[DIFFUSE_BRDF_INDEX] =
        !fully_metallic && !fully_transmissive ? diffuse_brdf_albedo(material, wi) : vec3(0.0);

    lobes.weights[FUZZ_BRDF_INDEX] = vec3(material.fuzz_weight);
    vec3 coated_base_weight =
        mix(vec3(1.0), 1.0 - lobes.albedos[FUZZ_BRDF_INDEX],
            material.fuzz_weight); // From OpenPBR Eq. (81)
    lobes.weights[COAT_BRDF_INDEX] = coated_base_weight * material.coat_weight;

    vec3 base_darkening = vec3(1.0);
    if (material.coat_weight > 0.0 && material.coat_darkening > 0.0) {
        float fresnel_weight =
            saturate(material.specular_weight * fresnel_at_normal_incident(specular_ior(material)));
        float dielectric_roughness =
            mix(1.0, material.specular_roughness, fresnel_weight); // OpenPBR Eq. (70)
        float base_roughness =
            mix(dielectric_roughness, material.specular_roughness,
                material.base_metalness); // OpenPBR Eq. (69)

        float average_fresnel = 1.0 -
            (1.0 - average_dielectric_fresnel(material.coat_ior)) /
                pow2(material.coat_ior); // OpenPBR Eq. (66)
        float fresnel = fresnel_dielectric(abs(cos_theta(wi)), material.coat_ior);
        float diffuse_reflection_coeff =
            mix(fresnel, average_fresnel, base_roughness); // OpenPBR Eq. (68)

        vec3 dielectric_base_albedo =
            mix(lobes.albedos[DIFFUSE_BRDF_INDEX], lobes.albedos[SPECULAR_BTDF_INDEX],
                material.transmission_weight);
        vec3 base_albedo =
            mix(dielectric_base_albedo, lobes.albedos[METAL_BRDF_INDEX], material.base_metalness);
        vec3 darkening_factor = (1.0 - diffuse_reflection_coeff) /
            (1.0 - base_albedo * diffuse_reflection_coeff); // OpenPBR Eq. (65)
        base_darkening =
            mix(vec3(1.0), darkening_factor,
                material.coat_weight * material.coat_darkening); // OpenPBR Eq. (71)
    }

    vec3 base_weight = coated_base_weight *
        mix(vec3(1.0),
            base_darkening * material.coat_color.rgb * (1.0 - lobes.albedos[COAT_BRDF_INDEX]),
            material.coat_weight); // From OpenPBR Eq. (92)
    lobes.weights[METAL_BRDF_INDEX] = base_weight * material.base_metalness;

    vec3 dielectric_base_weight = base_weight * vec3(1.0 - material.base_metalness);
    lobes.weights[SPECULAR_BRDF_INDEX] = dielectric_base_weight;
    lobes.weights[SPECULAR_BTDF_INDEX] = dielectric_base_weight * material.transmission_weight;

    vec3 opaque_dielectric_base_weight =
        dielectric_base_weight * (1.0 - material.transmission_weight);
    lobes.weights[DIFFUSE_BRDF_INDEX] =
        opaque_dielectric_base_weight * (1.0 - lobes.albedos[SPECULAR_BRDF_INDEX]);
}

void lobe_probabilities(MaterialConstants material, inout Lobes lobes) {
    float total_weight = 0.0;
    for (int lobe_index = 0; lobe_index < LOBE_COUNT; ++lobe_index) {
        lobes.probs[lobe_index] = length(lobes.weights[lobe_index] * lobes.albedos[lobe_index]);
        total_weight += lobes.probs[lobe_index];
    }
    total_weight = max(DENOM_TOLERANCE, total_weight);
    for (int lobe_index = 0; lobe_index < LOBE_COUNT; ++lobe_index)
        lobes.probs[lobe_index] /= total_weight;
}

Lobes bsdf_prepare(MaterialConstants material, vec3 wi, inout Generator generator) {
    Lobes lobes;
    lobe_weights(material, wi, generator, lobes);
    lobe_probabilities(material, lobes);
    return lobes;
}

vec3 bsdf_evaluate_lobes(
    MaterialConstants material, Lobes lobes, vec3 wi, vec3 wo, int skip_lobe_index,
    inout LobeDensities densities
) {
    vec3 bsdf = vec3(0.0);
    if (skip_lobe_index != FUZZ_BRDF_INDEX && lobes.probs[FUZZ_BRDF_INDEX] > 0.0)
        bsdf += lobes.weights[FUZZ_BRDF_INDEX] *
            fuzz_brdf_evaluate(material, wi, wo, densities.values[FUZZ_BRDF_INDEX]);
    if (skip_lobe_index != COAT_BRDF_INDEX && lobes.probs[COAT_BRDF_INDEX] > 0.0)
        bsdf += lobes.weights[COAT_BRDF_INDEX] *
            coat_brdf_evaluate(material, wi, wo, densities.values[COAT_BRDF_INDEX]);
    if (skip_lobe_index != METAL_BRDF_INDEX && lobes.probs[METAL_BRDF_INDEX] > 0.0)
        bsdf += lobes.weights[METAL_BRDF_INDEX] *
            metal_brdf_evaluate(material, wi, wo, densities.values[METAL_BRDF_INDEX]);
    if (skip_lobe_index != SPECULAR_BRDF_INDEX && lobes.probs[SPECULAR_BRDF_INDEX] > 0.0)
        bsdf += lobes.weights[SPECULAR_BRDF_INDEX] *
            specular_brdf_evaluate(material, wi, wo, densities.values[SPECULAR_BRDF_INDEX]);
    if (skip_lobe_index != DIFFUSE_BRDF_INDEX && lobes.probs[DIFFUSE_BRDF_INDEX] > 0.0)
        bsdf += lobes.weights[DIFFUSE_BRDF_INDEX] *
            diffuse_brdf_evaluate(material, wi, wo, densities.values[DIFFUSE_BRDF_INDEX]);
    if (skip_lobe_index != SPECULAR_BTDF_INDEX && lobes.probs[SPECULAR_BTDF_INDEX] > 0.0) {
        bsdf += lobes.weights[SPECULAR_BTDF_INDEX] *
            specular_btdf_evaluate(material, wi, wo, densities.values[SPECULAR_BTDF_INDEX]);
    }
    return bsdf;
}

float bsdf_total_pdf(Lobes lobes, LobeDensities densities) {
    float density = 0.0;
    for (int lobe_index = 0; lobe_index < LOBE_COUNT; ++lobe_index)
        density += lobes.probs[lobe_index] * densities.values[lobe_index];
    return density;
}

vec3 bsdf_evaluate(MaterialConstants material, Lobes lobes, vec3 wi, vec3 wo, inout float density) {
    LobeDensities densities;
    vec3 bsdf = bsdf_evaluate_lobes(material, lobes, wi, wo, -1, densities);
    density = bsdf_total_pdf(lobes, densities);
    return bsdf;
}

vec3 bsdf_sample(
    MaterialConstants material, Lobes lobes, vec3 wi, inout Generator generator, out vec3 wo,
    out float density, out int sampled_lobe
) {
    float random = random_float(generator);
    float accumulated_density = 0.0;
    for (int lobe_index = 0; lobe_index < LOBE_COUNT; ++lobe_index) {
        accumulated_density += lobes.probs[lobe_index];
        if (random < accumulated_density) {
            float lobe_density;
            vec3 lobe_bsdf;
            if (lobe_index == FUZZ_BRDF_INDEX) {
                lobe_bsdf = fuzz_brdf_sample(material, wi, generator, wo, lobe_density);
            } else if (lobe_index == COAT_BRDF_INDEX) {
                lobe_bsdf = coat_brdf_sample(material, wi, generator, wo, lobe_density);
            } else if (lobe_index == METAL_BRDF_INDEX) {
                lobe_bsdf = metal_brdf_sample(material, wi, generator, wo, lobe_density);
            } else if (lobe_index == SPECULAR_BRDF_INDEX) {
                lobe_bsdf = specular_brdf_sample(material, wi, generator, wo, lobe_density);
            } else if (lobe_index == SPECULAR_BTDF_INDEX) {
                lobe_bsdf = specular_btdf_sample(material, wi, generator, wo, lobe_density);
            } else if (lobe_index == DIFFUSE_BRDF_INDEX) {
                lobe_bsdf = diffuse_brdf_sample(material, wi, generator, wo, lobe_density);
            } else {
                break;
            }

            LobeDensities densities;
            vec3 bsdf = bsdf_evaluate_lobes(material, lobes, wi, wo, lobe_index, densities);
            densities.values[lobe_index] = lobe_density;
            density = bsdf_total_pdf(lobes, densities);
            sampled_lobe = lobe_index;

            return bsdf + lobes.weights[lobe_index] * lobe_bsdf;
        }
    }

    sampled_lobe = -1;
    density = 1.0;
    return vec3(0.0);
}

#endif
