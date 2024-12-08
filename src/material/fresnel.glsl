#ifndef FRESNEL
#define FRESNEL

#include "math"

vec3 fresnel_schlick(vec3 fresnel_at_normal, float cos_theta) {
    return fresnel_at_normal + pow5(1.0 - cos_theta) * (vec3(1.0) - fresnel_at_normal);
}

float fresnel_at_normal_incident(float ior) {
    return pow2((ior - 1.0) / (ior + 1.0)); // OpenPBR Eq. (25)
}
float dieletric_fresnel_factor(float ior) {
    return log(
        (10893.0 * ior - 1438.2) / (-774.4 * pow2(ior) + 10212.0 * ior + 1.0)
    ); // OpenPBR Eq. (102)
}

// Source: https://pbr-book.org/3ed-2018/Reflection_Models/Specular_Reflection_and_Transmission
float fresnel_dielectric(float cos_theta_i, float ior) {
    float sin_theta_squared_in = 1 - pow2(cos_theta_i);
    float sin_theta_squared_tr = sin_theta_squared_in / pow2(ior);
    if (sin_theta_squared_tr >= 1.0)
        return 1.0;
    float cos_theta_t = sqrt(1.0 - sin_theta_squared_tr);
    float par = (ior * cos_theta_i - cos_theta_t) / (ior * cos_theta_i + cos_theta_t);
    float per = (cos_theta_i - ior * cos_theta_t) / (cos_theta_i + ior * cos_theta_t);
    return 0.5 * (pow2(par) + pow2(per));
}

float average_dielectric_fresnel(float ior) {
    if (ior > 1.0) {
        return dieletric_fresnel_factor(ior);
    } else if (ior < 1.0) {
        return 1.0 - pow2(ior) * (1.0 - dieletric_fresnel_factor(1.0 / ior)); // OpenPBR Eq. (103)
    } else {
        return 0.0;
    }
}

#endif
