#ifndef HASH_GRID
#define HASH_GRID

#include "random"

const uint LEVEL_BIT_COUNT = 10;
const uint LEVEL_BIT_MASK = (1 << LEVEL_BIT_COUNT) - 1;
const uint HORIZONTAL_POSITION_BIT_COUNT = 20;
const uint HORIZONTAL_POSITION_BIT_MASK = (1 << HORIZONTAL_POSITION_BIT_COUNT) - 1;
const uint VERTICAL_POSITION_BIT_COUNT = 14;
const uint VERTICAL_POSITION_BIT_MASK = (1 << VERTICAL_POSITION_BIT_COUNT) - 1;
const uint LEVEL_BIAS = 2;
const float POSITION_BIAS = 0.0001;

const uint64_t HASH_GRID_INVALID_KEY = -1;

struct GridCell {
    ivec3 position;
    uint level;
};

layout(std430, buffer_reference, buffer_reference_align = 8) readonly buffer HashGridKeys {
    uint64_t data[];
};

struct HashGrid {
    HashGridKeys keys;
    float scene_scale;
    uint bucket_size, capacity, padding;
};

GridCell hash_grid_cell(vec3 position, vec3 camera_position, vec3 offset, float level_offset, in HashGrid hash_grid) {
    float distance_squared = dot(camera_position - position, camera_position - position);
    uint grid_level = uint(clamp(0.5 * log2(distance_squared) + LEVEL_BIAS + level_offset, 1.0, float(LEVEL_BIT_MASK)));
    float voxel_size = pow(2.0, grid_level) / (hash_grid.scene_scale * pow(2.0, LEVEL_BIAS));
    ivec3 grid_position = ivec3(floor((position + vec3(POSITION_BIAS)) / voxel_size + offset));
    return GridCell(grid_position, grid_level);
}

uint64_t hash_grid_key(GridCell grid_cell) {
    uint64_t horizontal = (uint64_t(grid_cell.position.x) & HORIZONTAL_POSITION_BIT_MASK)
        | ((uint64_t(grid_cell.position.z) & HORIZONTAL_POSITION_BIT_MASK) << HORIZONTAL_POSITION_BIT_COUNT);
    uint64_t vertical = uint64_t(grid_cell.position.y) & VERTICAL_POSITION_BIT_MASK;
    uint64_t level = uint64_t(grid_cell.level) & LEVEL_BIT_COUNT;
    return horizontal | (vertical << (HORIZONTAL_POSITION_BIT_COUNT * 2))
        | (level << (HORIZONTAL_POSITION_BIT_COUNT * 2 + VERTICAL_POSITION_BIT_COUNT));
}

uint hash_grid_hash(uint64_t key) {
    return jenkins_hash(uint(key & 0xffffffff)) ^ jenkins_hash(uint(key >> 32));
}

bool hash_grid_insert(in HashGrid hash_grid, uint64_t key, out uint index) {
    uint base_slot = min(hash_grid_hash(key) % hash_grid.capacity, hash_grid.capacity - hash_grid.bucket_size);
    uint64_t prev_key = HASH_GRID_INVALID_KEY;
    for (uint offset = 0; offset < hash_grid.bucket_size; offset++) {
        prev_key = atomicCompSwap(hash_grid.keys.data[base_slot + offset], HASH_GRID_INVALID_KEY, key);
        if (prev_key == HASH_GRID_INVALID_KEY || prev_key == key) {
            index = base_slot + offset;
            return true;
        }
    }
    return false;
}

bool hash_grid_find(in HashGrid hash_grid, uint64_t key, out uint index) {
    uint base_slot = min(hash_grid_hash(key) % hash_grid.capacity, hash_grid.capacity - hash_grid.bucket_size);
    for (uint offset = 0; offset < hash_grid.bucket_size; offset++) {
        uint64_t stored_key = hash_grid.keys.data[base_slot + offset];
        if (stored_key == key) {
            index = base_slot + offset;
            return true;
        }
    }
    return false;
}

#endif
