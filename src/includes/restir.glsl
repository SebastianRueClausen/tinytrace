#ifndef RESTIR
#define RESTIR

#include "random"
#include "octahedron"
#include "math"

struct BounceSurface {
    float position[3];
    f16vec2 normal;
};

BounceSurface create_bounce_surface(vec3 position, vec3 normal) {
    return BounceSurface(
        float[3](position.x, position.y, position.z),
        f16vec2(octahedron_encode(normal))
    );
}

vec3 bounce_surface_position(in BounceSurface bounce_surface) {
    return vec3(bounce_surface.position[0], bounce_surface.position[1], bounce_surface.position[2]);
}

vec3 bounce_surface_normal(in BounceSurface bounce_surface) {
    return vec3(octahedron_decode(vec2(bounce_surface.normal)));
}

struct Path {
    BounceSurface origin, destination;
    float16_t radiance[3];
    uint16_t bounce_count;
    Generator generator;
};

// p hat.
float path_target_function(in Path path) {
    return float(path.radiance[0] * 0.299hf + path.radiance[1] * 0.587hf + path.radiance[2] * 0.114hf);
}

struct Reservoir {
    Path path;
    float weight_sum, weight;
    uint16_t sample_count;
    uint16_t padding;
};

const uint16_t RESERVOIR_MAX_SAMPLE_COUNT = uint16_t(512);

bool reservoir_is_valid(in Reservoir reservoir) {
    // The reservoir must have a valid sample if the weight sum isn't 0 as it only selects a sample
    // if the weight isn't 0 and keeps that until it sees another valid sample.
    return reservoir.weight_sum > 0.0;
}

Reservoir create_empty_reservoir() {
    Reservoir reservoir;
    reservoir.weight_sum = 0.0;
    reservoir.weight = 0.0;
    reservoir.sample_count = uint16_t(0);
    reservoir.path.radiance = float16_t[3](0.0hf, 0.0hf, 0.0hf);
    return reservoir;
}

void update_reservoir(inout Reservoir reservoir, inout Generator generator, Path path, float weight) {
    reservoir.weight_sum += weight;
    reservoir.sample_count++;
    if (reservoir.weight_sum > 0 && random_float(generator) < weight / reservoir.weight_sum) {
        reservoir.path = path;
    }
}

void update_reservoir_weight(inout Reservoir reservoir) {
    float target_function = path_target_function(reservoir.path);
    if (target_function > 0.0) {
        reservoir.weight = reservoir.weight_sum / (reservoir.sample_count * target_function);
    }
}

void merge_reservoirs(inout Reservoir reservoir, inout Generator generator, Reservoir other) {
    float target_function = path_target_function(other.path);
    update_reservoir(reservoir, generator, other.path, target_function * other.weight * other.sample_count);
    reservoir.sample_count += other.sample_count;
}

float path_jacobian(vec3 sample_position, Path path) {
    vec3 destination_position = bounce_surface_position(path.destination);
    vec3 destination_normal = bounce_surface_normal(path.destination);

    vec3 sample_to_destination = sample_position - destination_position;
    float sample_distance = length(sample_to_destination);
    float sample_cos_phi = saturate(abs(dot(sample_to_destination, destination_normal) / sample_distance));

    vec3 path_to_destination = bounce_surface_position(path.origin) - destination_position;
    float path_distance = length(path_to_destination);
    float path_cos_phi = saturate(abs(dot(path_to_destination, destination_normal) / path_distance));

    float div = path_cos_phi * pow2(sample_distance);
    return div > 0.0 ? sample_cos_phi * pow2(path_distance) / div : 0.0;
}

#define RestirReplay uint
const RestirReplay REPLAY_NONE = 1;
const RestirReplay REPLAY_FIRST = 2;
const RestirReplay REPLAY_FULL = 3;

bool can_reconnect_to_path(vec3 sample_position, vec3 sample_normal, uint bounce_budget, RestirReplay replay, Path path) {
    bool has_enough_bounces;
    switch (replay) {
        case REPLAY_NONE:
            has_enough_bounces = true;
            break;
        case REPLAY_FIRST:
            has_enough_bounces = bounce_budget >= 1;
            break;
        case REPLAY_FULL:
            has_enough_bounces = bounce_budget >= uint(path.bounce_count);
            break;
    };
    float reconnection_distance = length(sample_position - bounce_surface_position(path.destination));
    float cos_theta = dot(sample_normal, bounce_surface_normal(path.origin));
    return reconnection_distance > 0.2 && cos_theta > 0.906 && has_enough_bounces;
}

vec3 reservoir_hash_grid_position_jitter(inout Generator generator) {
    return random_vec3(generator) * 2.0 - 1.0;
}

float reservoir_hash_grid_level_jitter(inout Generator generator) {
    return random_float(generator) * 0.5 - 0.25;
}

#endif
