#define HASH_GRID_BUFFER reservoir_hashes
#define HASH_GRID_INSERT insert_reservoir
#define HASH_GRID_FIND find_reservoir
#include "hash_grid"

HashGrid create_hash_grid(uint bucket_size, uint capacity) {
    HashGrid hash_grid;
    hash_grid.camera_position = constants.camera_position.xyz;
    hash_grid.scene_scale = 10.0;
    hash_grid.bucket_size = bucket_size;
    hash_grid.capacity = capacity;
    return hash_grid;
}

void main() {
    uint update_index = gl_GlobalInvocationID.x;
    if (update_index > 1024) {
        return;
    }
    Generator generator = init_generator_from_index(update_index, constants.frame_index);
    HashGrid reservoir_hash_grid = create_hash_grid(16, 0xffff);
    uint update_count = min(reservoir_updates[update_index].update_count, 4);
    for (uint path_index = 0; path_index < update_count; path_index++) {
        Path path = reservoir_updates[update_index].paths[path_index];
        vec3 origin_position = bounce_surface_position(path.origin);
        uint64_t key = hash_grid_key(hash_grid_cell(origin_position, reservoir_hash_grid));
        uint reservoir_index;
        if (insert_reservoir(reservoir_hash_grid, key, reservoir_index)) {
            float weight = reservoir_updates[update_index].weights[path_index];
            update_reservoir(reservoirs[reservoir_index], generator, path, weight);
            update_reservoir_weight(reservoirs[reservoir_index]);
        }
    }
}
