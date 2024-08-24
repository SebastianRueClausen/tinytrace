#include "tonemap"
#include "scene"

#include "<bindings>"

void main() {
    uvec2 pixel_index = gl_GlobalInvocationID.xy;
    if (any(greaterThanEqual(pixel_index, constants.screen_size))) return;
    vec3 color = vec3(imageLoad(render_target, ivec2(pixel_index)).rgb);
    if (constants.tonemap != 0) color = neutral_tonemap(color).xyz;
    imageStore(swapchain, ivec2(pixel_index), vec4(color, 1.0));
}
