#ifndef CONSTANTS
#define CONSTANTS

#define LightSampling uint
const LightSampling LIGHT_SAMPLING_NONE = 1;
const LightSampling LIGHT_SAMPLING_NEXT_EVENT_ESTIMATION = 2;

struct Constants {
    mat4 view, proj, proj_view, inverse_view, inverse_proj;
    vec4 camera_position;
    uint frame_index, accumulated_frame_count, sample_count, bounce_count;
    uvec2 screen_size;
    uint tonemap;
    LightSampling light_sampling;
};

#endif
