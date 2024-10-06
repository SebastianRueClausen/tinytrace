#ifndef CONSTANTS
#define CONSTANTS

#include "sample"
#include "light_sampling"

struct Constants {
    mat4 view, proj, proj_view, inverse_view, inverse_proj;
    vec4 camera_position;
    uint frame_index, accumulated_frame_count, sample_count, bounce_count;
    uvec2 screen_size;
    uint tonemap;
    SampleStrategy sample_strategy;
    LightSampling light_sampling;
};

#endif
