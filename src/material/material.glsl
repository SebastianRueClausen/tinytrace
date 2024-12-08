#ifndef MATERIAL
#define MATERIAL

const float DENOM_TOLERANCE = 1.0e-10;
const float RADIANCE_EPSILON = 1.0e-12;
const float THROUGHPUT_EPSILON = 1.0e-6;
const float PDF_EPSILON = 1.0e-6;
const float IOR_EPSILON = 1.0e-5;

struct MaterialConstants {
    vec4 base_color;
    vec4 specular_color;
    vec4 transmission_color;
    vec4 transmission_scatter;
    vec4 coat_color;
    vec4 fuzz_color;
    vec4 emission_color;
    vec4 geometry_normal;
    vec4 geometry_coat_normal;
    float base_weight;
    float base_metalness;
    float base_diffuse_roughness;
    float specular_weight;
    float specular_roughness;
    float specular_roughness_anisotropy;
    float specular_ior;
    float specular_rotation;
    float transmission_weight;
    float transmission_depth;
    float transmission_scatter_anisotropy;
    float coat_weight;
    float coat_roughness;
    float coat_roughness_anisotropy;
    float coat_ior;
    float coat_darkening;
    float coat_rotation;
    float fuzz_weight;
    float fuzz_roughness;
    float emission_luminance;
    float geometry_opacity;
};

struct MaterialTextures {
    uint base_color;
    uint specular_color;
    uint transmission_color;
    uint transmission_scatter;
    uint coat_color;
    uint fuzz_color;
    uint emission_color;
    uint geometry_normal;
    uint geometry_coat_normal;
    uint base_weight;
    uint base_metalness;
    uint base_diffuse_roughness;
    uint specular_weight;
    uint specular_roughness;
    uint specular_roughness_anisotropy;
    uint specular_ior;
    uint specular_rotation;
    uint transmission_weight;
    uint transmission_depth;
    uint transmission_scatter_anisotropy;
    uint coat_weight;
    uint coat_roughness;
    uint coat_roughness_anisotropy;
    uint coat_ior;
    uint coat_darkening;
    uint coat_rotation;
    uint fuzz_weight;
    uint fuzz_roughness;
    uint emission_luminance;
    uint geometry_opacity;
};

struct Material {
    MaterialConstants constants;
    MaterialTextures textures;
};

#endif
