#ifndef CONSTANTS
#define CONSTANTS

#include "hash_grid"
#include "sample"

struct Constants {
    mat4 view, proj, proj_view, inverse_view, inverse_proj;
    vec4 camera_position;
    uint frame_index, accumulated_frame_count, sample_count, bounce_count;
    HashGrid reservoir_hash_grid, reservoir_update_hash_grid;
    uvec2 screen_size;
    uint use_world_space_restir, tonemap;
    SampleStrategy sample_strategy;
};

#endif
