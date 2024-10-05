#include "random"
#include "restir"
#include "math"
#include "constants"
#include "hash_grid"

#include "<bindings>"

void main() {
    uint cell_index = gl_GlobalInvocationID.x;
    if (cell_index > restir_data.update_hash_grid.capacity) return;
    Generator generator = init_generator_from_index(cell_index, constants.frame_index);
    uint update_count = min(restir_data.update_counts.data[cell_index], restir_data.updates_per_cell);
    uint base_index = cell_index * restir_data.updates_per_cell;
    for (uint update_index = base_index; update_index < base_index + update_count; update_index++) {
        Reservoir reservoir_update = restir_data.updates.data[update_index];
        vec3 origin_position = path_vertex_position(reservoir_update.path.origin);
        vec3 offset = reservoir_hash_grid_position_jitter(generator);
        float level_offset = reservoir_hash_grid_level_jitter(generator);
        uint64_t key = hash_grid_key(hash_grid_cell(
            origin_position, constants.camera_position.xyz, offset, level_offset, restir_data.reservoir_hash_grid
        ));
        uint reservoir_cell_index;
        if (hash_grid_insert(restir_data.reservoir_hash_grid, key, reservoir_cell_index)) {
            // Select random reservoir in pool to update.
            uint reservoir_index = reservoir_cell_index * restir_data.reservoirs_per_cell
                + random_uint(generator, restir_data.reservoirs_per_cell);
            // Merge with random existing reservoir.
            merge_reservoirs(reservoir_update, generator, restir_data.reservoirs.data[reservoir_index]);
            update_reservoir_weight(reservoir_update);
            reservoir_update.sample_count = min(reservoir_update.sample_count, RESERVOIR_MAX_SAMPLE_COUNT);
            restir_data.reservoirs.data[reservoir_index] = reservoir_update;
        }
    }
}
