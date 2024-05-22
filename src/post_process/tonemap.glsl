f16vec3 rgb_to_ycbcr(f16vec3 col) {
    f16mat3 m = f16mat3(0.2126hf, 0.7152hf, 0.0722hf, -0.1146hf, -0.3854hf, 0.5hf, 0.5hf, -0.4542hf, -0.0458hf);
    return col * m;
}

float16_t rgb_to_luminance(f16vec3 col) {
    return dot(f16vec3(0.2126hf, 0.7152hf, 0.0722hf), col);
}

float16_t tonemap_curve(float16_t v) {
    return 1.0hf - exp(-v);
}

f16vec3 tonemap_curve(f16vec3 v) {
    return f16vec3(1.0hf) - exp(-v);
}

f16vec3 neutral_tonemap(f16vec3 col) {
    f16vec3 ycbcr = rgb_to_ycbcr(col);
    float16_t bt = tonemap_curve(length(ycbcr.yz) * 2.4hf);
    float16_t desat = max((bt - 0.7hf) * 0.8hf, 0.0hf);
    desat *= desat;
    f16vec3 desat_col = mix(col.rgb, ycbcr.xxx, desat);
    float16_t tm_lum = tonemap_curve(ycbcr.x);
    f16vec3 tm0 = col.rgb * max(0.0hf, tm_lum / max(1e-5hf, rgb_to_luminance(col.rgb)));
    f16vec3 tm1 = tonemap_curve(desat_col);
    col = mix(tm0, tm1, bt * bt);
    return col * 0.97hf;
}