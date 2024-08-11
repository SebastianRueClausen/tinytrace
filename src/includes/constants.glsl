#ifndef CONSTANTS
#define CONSTANTS

#include "hash_grid"

const float TAU = 6.283185307179586;
const float PI = 3.1415926535897932;

struct Constants {
    mat4 view, proj, proj_view, inverse_view, inverse_proj;
    vec4 camera_position;
    uint frame_index, accumulated_frame_count, sample_count, bounce_count;
    HashGrid reservoir_hash_grid, reservoir_update_hash_grid;
    uvec2 screen_size;
};

#endif
