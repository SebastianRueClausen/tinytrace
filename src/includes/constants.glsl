
#ifndef CONSTANTS
#define CONSTANTS

const float16_t TAU = 6.283185307179586hf;
const float16_t PI = 3.1415926535897932hf;

struct Constants {
    mat4 view, proj, proj_view, inverse_view, inverse_proj, ray_matrix;
    vec4 camera_position;
    uvec2 screen_size;
    uint frame_index;
};

#endif