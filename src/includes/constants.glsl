#ifndef CONSTANTS
#define CONSTANTS

#include "hash_grid"
#include "sample"
#include "restir"
#include "light_sampling"

struct Constants {
    mat4 view, proj, proj_view, inverse_view, inverse_proj;
    vec4 camera_position;
    uint frame_index, accumulated_frame_count, sample_count, bounce_count;
    uvec2 screen_size;
    uint use_world_space_restir, tonemap;
    SampleStrategy sample_strategy;
    LightSampling light_sampling;
};

#endif
