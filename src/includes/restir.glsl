#ifndef RESTIR
#define RESTIR

#include "random"
#include "octahedron"
#include "hash_grid"
#include "math"
#include "brdf"

#define RestirReplay uint
const RestirReplay REPLAY_NONE = 1;
const RestirReplay REPLAY_FIRST = 2;
const RestirReplay REPLAY_FULL = 3;

struct RestirConstants {
    float scene_scale;
    uint updates_per_cell;
    uint reservoirs_per_cell;
    RestirReplay replay;
};

struct PathVertex {
    float position[3];
    f16vec2 normal;
};

PathVertex create_path_vertex(vec3 position, vec3 normal) {
    return PathVertex(float[3](position.x, position.y, position.z), f16vec2(octahedron_encode(normal)));
}

vec3 path_vertex_position(in PathVertex path_vertex) {
    return vec3(path_vertex.position[0], path_vertex.position[1], path_vertex.position[2]);
}

vec3 path_vertex_normal(in PathVertex path_vertex) {
    return vec3(octahedron_decode(vec2(path_vertex.normal)));
}

struct Path {
    PathVertex origin, destination;
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
    vec3 destination_position = path_vertex_position(path.destination);
    vec3 destination_normal = path_vertex_normal(path.destination);

    vec3 sample_to_destination = sample_position - destination_position;
    float sample_distance = length(sample_to_destination);
    float sample_cos_phi = saturate(abs(dot(sample_to_destination, destination_normal) / sample_distance));

    vec3 path_to_destination = path_vertex_position(path.origin) - destination_position;
    float path_distance = length(path_to_destination);
    float path_cos_phi = saturate(abs(dot(path_to_destination, destination_normal) / path_distance));

    float div = path_cos_phi * pow2(sample_distance);
    return div > 0.0 ? sample_cos_phi * pow2(path_distance) / div : 0.0;
}

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
    float reconnection_distance = length(sample_position - path_vertex_position(path.destination));
    float cos_theta = dot(sample_normal, path_vertex_normal(path.origin));
    return reconnection_distance > 0.2 && cos_theta > 0.906 && has_enough_bounces;
}

vec3 reservoir_hash_grid_position_jitter(inout Generator generator) {
    return random_vec3(generator) * 2.0 - 1.0;
}

float reservoir_hash_grid_level_jitter(inout Generator generator) {
    return random_float(generator) * 0.5 - 0.25;
}

struct PathCandidate {
    vec3 accumulated, attenuation, previous_normal;
    // This is true if the previous bounce was "diffuse", which in this context means that the
    // surface roughness is relatively high. This makes it so that when reconnecting, the different
    // solid angle won't fall outside the material BRDF.
    bool previous_bounce_was_diffuse;
    // The found path origin and destination if a path has been found.
    PathVertex origin, destination;
    // The generator from just before generating a path from `destination`.
    Generator generator;
    float scatter_density;
    uint16_t first_bounce, last_bounce;
    // True if a path has been found.
    bool is_found;
};

PathCandidate create_empty_path_candidate() {
    PathCandidate candidate;
    candidate.accumulated = vec3(0.0);
    candidate.attenuation = vec3(1.0);
    candidate.first_bounce = uint16_t(0);
    candidate.last_bounce = uint16_t(0);
    candidate.is_found = false;
    return candidate;
}

void add_light_to_path_candidate(inout PathCandidate path_candidate, vec3 light, uint bounce) {
    vec3 contribution = light * path_candidate.attenuation;
    path_candidate.last_bounce = length_squared(contribution) > 0.0 ? uint16_t(bounce) : path_candidate.last_bounce;
    path_candidate.accumulated += contribution;
}

bool can_form_path_candidate(PathCandidate path_candidate, uint bounce, LobeType lobe_type, float section_distance_squared) {
    // Only find a simple resample path per trace for now
    return !path_candidate.is_found
        // It isn't possible to reconnect the first bounce.
        && bounce != 0
        // The previous bounce must be counted as "diffuse" e.g. relatively rough so that
        // reconnecting doesn't cause the BRDF to become zero.
        && path_candidate.previous_bounce_was_diffuse
        // The current bounce must be diffuse meaning that it samples the diffuse lope. This
        // is required to create a successfull reconnection because diffuse sampling doesn't
        // depend on the incidence vector, which will be different because of the reconnection.
        && lobe_type == DIFFUSE_LOBE
        // Very short reconnection paths can cause a bunch of problems.
        && section_distance_squared > 0.025;
}

Reservoir create_reservoir_from_path_candidate(PathCandidate path_candidate, uint16_t sample_count) {
    Reservoir reservoir;
    reservoir.path.origin = path_candidate.origin;
    reservoir.path.destination = path_candidate.destination;
    reservoir.path.generator = path_candidate.generator;
    reservoir.path.bounce_count = path_candidate.last_bounce - path_candidate.first_bounce;
    for (uint i = 0; i < 3; i++) reservoir.path.radiance[i] = float16_t(path_candidate.accumulated[i]);
    reservoir.sample_count = sample_count;
    reservoir.weight_sum = path_target_function(reservoir.path) / path_candidate.scatter_density;
    update_reservoir_weight(reservoir);
    return reservoir;
}

struct ReplayPath {
    PathVertex reconnect_location;
    vec3 radiance;
    bool is_found;
};

ReplayPath create_replay_path() {
    ReplayPath replay_path;
    replay_path.is_found = false;
    return replay_path;
}

layout(std430, buffer_reference, buffer_reference_align = 8) readonly buffer Reservoirs {
    Reservoir data[];
};

layout(std430, buffer_reference, buffer_reference_align = 8) readonly buffer Counters {
    uint data[];
};

struct RestirData {
    Reservoirs reservoirs;
    Reservoirs updates;
    Counters update_counts;
    Counters sample_counts;
    float scene_scale;
    uint updates_per_cell;
    uint reservoirs_per_cell;
    RestirReplay replay;
    HashGrid reservoir_hash_grid, update_hash_grid;
};

#endif
