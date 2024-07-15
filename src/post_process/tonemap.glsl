vec3 rgb_to_ycbcr(vec3 color) {
    return color * mat3(0.2126, 0.7152, 0.0722, -0.1146, -0.3854, 0.5, 0.5, -0.4542, -0.0458);
}

float rgb_to_luminance(vec3 color) {
    return dot(vec3(0.2126, 0.7152, 0.0722), color);
}

float tonemap_curve(float value) {
    return 1.0 - exp(-value);
}

vec3 tonemap_curve(vec3 value) {
    return vec3(1.0) - exp(-value);
}

vec3 neutral_tonemap(vec3 color) {
    vec3 ycbcr = rgb_to_ycbcr(color);
    float bt = tonemap_curve(length(ycbcr.yz) * 2.4);
    float desat = max((bt - 0.7) * 0.8, 0.0);
    desat *= desat;
    vec3 desat_color = mix(color.rgb, ycbcr.xxx, desat);
    float tm_lum = tonemap_curve(ycbcr.x);
    vec3 tm0 = color.rgb * max(0.0, tm_lum / max(1e-5, rgb_to_luminance(color.rgb)));
    vec3 tm1 = tonemap_curve(desat_color);
    return mix(tm0, tm1, bt * bt) * 0.97;
}