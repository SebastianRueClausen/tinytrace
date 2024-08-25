#include "random"
#include "restir"
#include "math"
#include "constants"

#include "<bindings>"

#define HASH_GRID_BUFFER reservoir_pool_hashes
#define HASH_GRID_INSERT insert_reservoir_pool
#define HASH_GRID_FIND find_reservoir_pool
#include "hash_grid"

void main() {
    uint update_index = gl_GlobalInvocationID.x;
    if (update_index > constants.reservoir_update_hash_grid.capacity) return;
    Generator generator = init_generator_from_index(update_index, constants.frame_index);
    uint update_count = min(reservoir_updates[update_index].update_count, RESERVOIR_UPDATE_COUNT);
    for (uint path_index = 0; path_index < update_count; path_index++) {
        Path path = reservoir_updates[update_index].paths[path_index];
        vec3 origin_position = bounce_surface_position(path.origin);
        uint64_t key = hash_grid_key(hash_grid_cell(
            origin_position, constants.camera_position.xyz, vec3(0.0), constants.reservoir_hash_grid
        ));
        uint reservoir_pool_index;
        if (insert_reservoir_pool(constants.reservoir_hash_grid, key, reservoir_pool_index)) {
            float weight = reservoir_updates[update_index].weights[path_index];
            // Setup initial reservoir.
            Reservoir reservoir = create_empty_reservoir();
            update_reservoir(reservoir, generator, path, weight);

            // Select random reservoir in pool to update.
            uint reservoir_index = random_uint(generator, RESERVOIR_POOL_SIZE);

            // Merge with random existing reservoir.
            merge_reservoirs(reservoir, generator, reservoir_pools[reservoir_pool_index].reservoirs[reservoir_index]);
            update_reservoir_weight(reservoir);
            reservoir.sample_count = min(reservoir.sample_count, RESERVOIR_MAX_SAMPLE_COUNT);
            reservoir_pools[reservoir_pool_index].reservoirs[reservoir_index] = reservoir;
        }
    }
}
