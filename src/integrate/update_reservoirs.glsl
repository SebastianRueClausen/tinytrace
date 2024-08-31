#include "random"
#include "restir"
#include "math"
#include "constants"

#include "<bindings>"

#define HASH_GRID_BUFFER reservoir_keys
#define HASH_GRID_INSERT insert_reservoir_cell
#define HASH_GRID_FIND find_reservoir_cell
#include "hash_grid"

void main() {
    uint cell_index = gl_GlobalInvocationID.x;
    if (cell_index > constants.reservoir_update_hash_grid.capacity) return;
    Generator generator = init_generator_from_index(cell_index, constants.frame_index);
    uint update_count = min(reservoir_update_counts[cell_index], constants.reservoir_updates_per_cell);
    uint base_index = cell_index * constants.reservoir_updates_per_cell;
    for (uint update_index = base_index; update_index < base_index + update_count; update_index++) {
        Reservoir reservoir_update = reservoir_updates[update_index];
        vec3 origin_position = bounce_surface_position(reservoir_update.path.origin);
        // vec3 offset = random_vec3(generator) * 2.0 - 1.0;
        vec3 offset = vec3(0.0);
        uint64_t key = hash_grid_key(hash_grid_cell(
            origin_position, constants.camera_position.xyz, offset, constants.reservoir_hash_grid
        ));
        uint reservoir_cell_index;
        if (insert_reservoir_cell(constants.reservoir_hash_grid, key, reservoir_cell_index)) {
            // Select random reservoir in pool to update.
            uint reservoir_index = reservoir_cell_index * constants.reservoirs_per_cell
                + random_uint(generator, constants.reservoirs_per_cell);
            // Merge with random existing reservoir.
            merge_reservoirs(reservoir_update, generator, reservoirs[reservoir_index]);
            update_reservoir_weight(reservoir_update);
            reservoir_update.sample_count = min(reservoir_update.sample_count, RESERVOIR_MAX_SAMPLE_COUNT);
            reservoirs[reservoir_index] = reservoir_update;
        }
    }
}
