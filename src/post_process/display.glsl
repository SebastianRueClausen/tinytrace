void main() {
    uvec2 pixel_index = gl_GlobalInvocationID.xy;
    if (any(greaterThanEqual(pixel_index, constants.screen_size))) {
        return;
    }
    f16vec3 color = f16vec3(imageLoad(render_target, ivec2(pixel_index)).rgb);
    color = neutral_tonemap(color);
    imageStore(swapchain, ivec2(pixel_index), vec4(vec3(color), 1.0));
}