#ifndef RESTIR
#define RESTIR

#include "random"
#include "octahedron"
#include "math"

// 16 bytes.
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

// 52 bytes.
struct Path {
    BounceSurface origin, destination;
    float radiance[3];
    Generator generator;
};

// p hat.
float path_target_function(in Path path) {
    return path.radiance[0] * 0.299 + path.radiance[1] * 0.587 + path.radiance[2] * 0.114;
}

// 60 bytes.
struct Reservoir {
    Path path;
    // w, W
    float weight_sum, weight;
    // M
    uint sample_count;
};

const uint RESERVOIR_MAX_SAMPLE_COUNT = 30;

bool reservoir_is_valid(in Reservoir reservoir) {
    // The reservoir must have a valid sample if the weight sum isn't 0 as it only selects a sample
    // if the weight isn't 0 and keeps that until it sees another valid sample.
    return reservoir.weight_sum > 0.0;
}

Reservoir create_empty_reservoir() {
    Reservoir reservoir;
    reservoir.weight_sum = 0.0;
    reservoir.weight = 0.0;
    reservoir.sample_count = 0;
    reservoir.path.radiance = float[3](0.0, 0.0, 0.0);
    return reservoir;
}

void update_reservoir(inout Reservoir reservoir, inout Generator generator, Path path, float weight) {
    reservoir.weight_sum += weight;
    reservoir.sample_count += 1;
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

const float PATH_RECONNECT_DISTANCE_THRESHOLD = 0.1;
const float PATH_RECONNECT_COS_THETA_THRESHOLD = 0.906;

bool can_reconnect_to_path(vec3 sample_position, vec3 sample_normal, Path path) {
    float distance = length(sample_position - bounce_surface_position(path.origin));
    float cos_theta = dot(sample_normal, bounce_surface_normal(path.origin));
    return distance < PATH_RECONNECT_DISTANCE_THRESHOLD
        && cos_theta > PATH_RECONNECT_COS_THETA_THRESHOLD;
}

#endif
