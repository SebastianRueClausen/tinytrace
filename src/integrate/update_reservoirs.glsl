#define HASH_GRID_BUFFER reservoir_hashes
#define HASH_GRID_INSERT insert_reservoir
#define HASH_GRID_FIND find_reservoir
#include "hash_grid"

void main() {
    uint update_index = gl_GlobalInvocationID.x;
    if (update_index > constants.reservoir_update_hash_grid.capacity) {
        return;
    }
    Generator generator = init_generator_from_index(update_index, constants.frame_index);
    uint update_count = min(reservoir_updates[update_index].update_count, RESERVOIR_UPDATE_COUNT);
    for (uint path_index = 0; path_index < update_count; path_index++) {
        Path path = reservoir_updates[update_index].paths[path_index];
        vec3 origin_position = bounce_surface_position(path.origin);
        uint64_t key = hash_grid_key(hash_grid_cell(
            origin_position, constants.camera_position.xyz, vec3(0.0), constants.reservoir_hash_grid
        ));
        uint reservoir_index;
        if (insert_reservoir(constants.reservoir_hash_grid, key, reservoir_index)) {
            float weight = reservoir_updates[update_index].weights[path_index];
            Reservoir reservoir;
            initialize_reservoir(reservoir);
            update_reservoir(reservoir, generator, path, weight);
            merge_reservoirs(reservoir, generator, reservoirs[reservoir_index]);
            update_reservoir_weight(reservoir);
            reservoir.sample_count = min(reservoir.sample_count, RESERVOIR_MAX_SAMPLE_COUNT);
            reservoirs[reservoir_index] = reservoir;
        }
    }
}
