#ifndef RESTIR
#define RESTIR

#include "random"
#include "octahedron"

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

// 48 bytes.
struct Path {
    // 32 bytes.
    BounceSurface origin, destination;
    // 12 bytes.
    float radiance[3];
    // 4 bytes.
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

void update_reservoir_weight(inout Reservoir reservoir) {
    reservoir.weight =
        reservoir.weight_sum / (reservoir.sample_count * path_target_function(reservoir.path));
}

const uint RESERVOIR_UPDATE_COUNT = 4;

// 212 bytes.
struct ReservoirUpdate {
    // 192 bytes.
    Path paths[RESERVOIR_UPDATE_COUNT];
    // 16 bytes.
    float weights[RESERVOIR_UPDATE_COUNT];
    // 4 bytes with padding.
    uint update_count;
};

void initialize_reservoir(out Reservoir reservoir) {
    reservoir.weight_sum = 0.0;
    reservoir.weight = 0.0;
    reservoir.sample_count = 0;
}

void update_reservoir(inout Reservoir reservoir, inout Generator generator, Path path, float weight) {
    reservoir.weight_sum += weight;
    reservoir.sample_count += 1;
    if (random_float(generator) < weight / reservoir.weight_sum) {
        reservoir.path = path;
    }
}

void merge_reservoirs(inout Reservoir reservoir, inout Generator generator, Reservoir other, float p_hat) {
    update_reservoir(reservoir, generator, other.path, p_hat * other.weight * other.sample_count);
    reservoir.sample_count += other.sample_count;
}

#endif
