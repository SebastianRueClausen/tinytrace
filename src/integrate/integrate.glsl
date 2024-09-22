#include "scene"
#include "brdf"
#include "math"
#include "sample"
#include "debug"
#include "restir"
#include "constants"

#include "<bindings>"

#define HASH_GRID_BUFFER reservoir_keys
#define HASH_GRID_INSERT insert_reservoir_cell
#define HASH_GRID_FIND find_reservoir_cell
#include "hash_grid"

#undef HASH_GRID_BUFFER
#undef HASH_GRID_INSERT
#undef HASH_GRID_FIND

#define HASH_GRID_BUFFER reservoir_update_keys
#define HASH_GRID_INSERT insert_reservoir_update
#define HASH_GRID_FIND find_reservoir_update
#include "hash_grid"

struct RayHit {
    vec3 world_position;
    Basis tangent_space;
    vec2 texcoord, texcoord_ddx, texcoord_ddy;
    int instance;
};

struct Ray {
    vec3 direction;
    vec3 origin;
};

// Returns the barycentric coordinates of ray triangle intersection.
vec3 triangle_intersection(vec3 triangle[3], Ray ray) {
    vec3 edge_to_origin = ray.origin - triangle[0];
    vec3 edge_2 = triangle[2] - triangle[0];
    vec3 edge_1 = triangle[1] - triangle[0];
    vec3 r = cross(ray.direction, edge_2);
    vec3 s = cross(edge_to_origin, edge_1);
    float inverse_det = 1.0 / dot(r, edge_1);
    float v1 = dot(r, edge_to_origin);
    float v2 = dot(s, ray.direction);
    float b = v1 * inverse_det;
    float c = v2 * inverse_det;
    return vec3(1.0 - b - c, b, c);
}

vec3 camera_ray_direction(vec2 ndc) {
    vec4 view_space_point = constants.inverse_proj * vec4(ndc.x, -ndc.y, 1.0, 1.0);
    return normalize((constants.inverse_view * vec4(view_space_point.xyz, 0.0)).xyz);
}

vec2 interpolate(vec3 barycentric, f16vec2 a, f16vec2 b, f16vec2 c) {
    return barycentric.x * a + barycentric.y * b + barycentric.z * c;
}

vec3 interpolate(vec3 barycentric, vec3 a, vec3 b, vec3 c) {
    return barycentric.x * a + barycentric.y * b + barycentric.z * c;
}

RayHit get_ray_hit(rayQueryEXT query, uint bounce, vec2 ndc) {
    RayHit hit;

    hit.instance = rayQueryGetIntersectionInstanceCustomIndexEXT(query, true);
    Instance instance = instances[hit.instance];
    mat4x3 transform = mat4x3(instance.transform);

    vec3 positions[3];
    rayQueryGetIntersectionTriangleVertexPositionsEXT(query, true, positions);

    // Transform positions from model to world space.
    for (uint i = 0; i < 3; i++) positions[i] = transform * vec4(positions[i], 1.0);

    uint base_index = 3 * rayQueryGetIntersectionPrimitiveIndexEXT(query, true) + instance.index_offset;
    uvec3 triangle_indices = uvec3(indices[base_index + 0], indices[base_index + 1], indices[base_index + 2]) + instance.vertex_offset;
    Vertex triangle_vertices[3] = Vertex[3](vertices[triangle_indices[0]], vertices[triangle_indices[1]], vertices[triangle_indices[2]]);

    vec3 barycentric = vec3(0.0, rayQueryGetIntersectionBarycentricsEXT(query, true));
    barycentric.x = 1.0 - barycentric.y - barycentric.z;

    TangentFrame tangent_frames[3] = TangentFrame[3](
        decode_tangent_frame(triangle_vertices[0].tangent_frame),
        decode_tangent_frame(triangle_vertices[1].tangent_frame),
        decode_tangent_frame(triangle_vertices[2].tangent_frame)
    );

    mat3 normal_transform = mat3(instance.normal_transform);
    hit.tangent_space.normal = normalize(normal_transform * interpolate(
        barycentric, tangent_frames[0].normal, tangent_frames[1].normal, tangent_frames[2].normal
    ));
    hit.tangent_space.tangent = normalize(normal_transform * interpolate(
        barycentric, tangent_frames[0].tangent, tangent_frames[1].tangent, tangent_frames[2].tangent
    ));
    float bitangent_sign = sign(dot(barycentric, vec3(
        tangent_frames[0].bitangent_sign, tangent_frames[1].bitangent_sign, tangent_frames[2].bitangent_sign
    )));
    hit.tangent_space.bitangent
        = normalize(bitangent_sign * cross(hit.tangent_space.normal, hit.tangent_space.tangent));

    hit.texcoord = interpolate(barycentric, triangle_vertices[0].texcoord, triangle_vertices[1].texcoord, triangle_vertices[2].texcoord);
    hit.world_position = barycentric.x * positions[0] + barycentric.y * positions[1] + barycentric.z * positions[2];

    // Calculate the differentials of the texture coordinates using ray
    // differentials if it's the first bounce.
    if (bounce == 0) {
        vec2 texel_size = 2.0 / vec2(constants.screen_size);
        Ray ray = Ray(camera_ray_direction(vec2(ndc.x + texel_size.x, ndc.y)), rayQueryGetWorldRayOriginEXT(query));
        vec3 hx = triangle_intersection(positions, ray);
        ray.direction = camera_ray_direction(vec2(ndc.x, ndc.y + texel_size.y));
        vec3 hy = triangle_intersection(positions, ray);
        hit.texcoord_ddx = interpolate(barycentric - hx, triangle_vertices[0].texcoord, triangle_vertices[1].texcoord, triangle_vertices[2].texcoord);
        hit.texcoord_ddy = interpolate(barycentric - hy, triangle_vertices[0].texcoord, triangle_vertices[1].texcoord, triangle_vertices[2].texcoord);
    }

    return hit;
}

bool trace_ray(Ray ray, uint bounce, vec2 ndc, out RayHit hit) {
    rayQueryEXT ray_query;
    rayQueryInitializeEXT(
        ray_query, acceleration_structure, gl_RayFlagsOpaqueEXT, 0xff, ray.origin, 1.0e-3, ray.direction, 1000.0
    );
    while (rayQueryProceedEXT(ray_query));
    if (rayQueryGetIntersectionTypeEXT(ray_query, true) == gl_RayQueryCommittedIntersectionNoneEXT) {
        return false;
    }
    hit = get_ray_hit(ray_query, bounce, ndc);
    return true;
}

vec2 pixel_ndc(uvec2 pixel_index, inout Generator generator) {
    vec2 offset = vec2(random_float(generator), random_float(generator)) - 0.5;
    return ((vec2(pixel_index) + offset) / vec2(constants.screen_size)) * 2.0 - 1.0;
}

float calculate_mip_level(vec2 texcoord_ddx, vec2 texcoord_ddy) {
    return 0.5 * log2(max(
        length_squared(texcoord_ddx * constants.screen_size),
        length_squared(texcoord_ddy * constants.screen_size)
    ));
}

vec3 normal_bias(vec3 position, vec3 normal) {
    return position + normal * 0.00001;
}

struct DirectLightSample {
    vec3 barycentric, position, normal;
    vec2 texcoord;
    uint instance, hash;
    float area, light_probability;
};

DirectLightSample sample_random_light(inout Generator generator) {
    uint triangle_count = emissive_triangles.length();
    EmissiveTriangle triangle = emissive_triangles[random_uint(generator, triangle_count)];
    vec3 barycentric = sample_triangle(generator);
    mat4x3 transform = mat4x3(instances[triangle.instance].transform);
    vec3 positions[3] = vec3[3](
        transform * vec4(dequantize_snorm(triangle.positions[0]), 1.0),
        transform * vec4(dequantize_snorm(triangle.positions[1]), 1.0),
        transform * vec4(dequantize_snorm(triangle.positions[2]), 1.0)
    );
    float area = max(0.00001, 0.5 * length(cross(positions[1] - positions[0], positions[2] - positions[0])));
    vec3 normal = normalize(cross(positions[1] - positions[0], positions[2] - positions[0]));
    vec3 position = normal_bias(barycentric[0] * positions[0] + barycentric[1] * positions[1] + barycentric[2] * positions[2], normal);
    vec2 texcoord = interpolate(barycentric, triangle.texcoords[0], triangle.texcoords[1], triangle.texcoords[2]);
    return DirectLightSample(
        barycentric, position, normal, texcoord, triangle.instance, uint(triangle.hash), area, 1.0 / triangle_count
    );
}

vec3 direct_light_contribution(vec3 position, vec3 normal, SurfaceProperties surface, inout Generator generator) {
    DirectLightSample light_sample = sample_random_light(generator);
    vec3 to_light = light_sample.position - position;
    float distance_squared = length_squared(to_light);
    Ray ray = Ray(to_light / sqrt(distance_squared), position);
    // Transform the area density to the solid angle density.
    float density = (1.0 / light_sample.area) * distance_squared / saturate(dot(-ray.direction, light_sample.normal)) * light_sample.light_probability;

    rayQueryEXT ray_query;
    rayQueryInitializeEXT(
        ray_query, acceleration_structure, gl_RayFlagsOpaqueEXT, 0xff, ray.origin, 1.0e-3, ray.direction, 10000.0
    );
    while (rayQueryProceedEXT(ray_query));
    if (rayQueryGetIntersectionTypeEXT(ray_query, true) == gl_RayQueryCommittedIntersectionNoneEXT) {
        return vec3(0.0);
    }

    uint instance_index = rayQueryGetIntersectionInstanceCustomIndexEXT(ray_query, true);
    if (instance_index != light_sample.instance) return vec3(0.0);
    uint triangle_index = rayQueryGetIntersectionPrimitiveIndexEXT(ray_query, true) + instances[instance_index].index_offset / 3;
    if (triangle_index % 0xffff != light_sample.hash) return vec3(0.0);

    Material material = materials[instances[light_sample.instance].material];

    // TODO: Figure out a good mip level.
    float mip_level = 4.0;
    vec3 emissive = textureLod(textures[material.emissive_texture], light_sample.texcoord, mip_level).rgb
        * vec3(material.emissive[0], material.emissive[1], material.emissive[2]);

    ScatterProperties scatter = create_scatter_properties(surface, ray.direction, normal);
    vec3 brdf = ggx_specular(surface, scatter) + burley_diffuse(surface, scatter);

    return brdf * emissive * abs(scatter.normal_dot_scatter) / density;
}

Reservoir get_reservoir_from_cell(uint cell_index, inout Generator generator) {
    uint base_index = constants.reservoirs_per_cell * cell_index;
    uint reservoir_index = random_uint(generator, constants.reservoirs_per_cell);
    return reservoirs[base_index + reservoir_index];
}

uint64_t get_reservoir_key(vec3 position, vec3 normal) {
    return hash_grid_key(hash_grid_cell(
        normal_bias(position, normal), constants.camera_position.xyz,
        vec3(0.0), 0.0, constants.reservoir_hash_grid
    ));
}

uint take_reservoir_sample_count(vec3 position, vec3 normal) {
    uint64_t key = get_reservoir_key(position, normal);
    uint cell_index;
    if (insert_reservoir_cell(constants.reservoir_hash_grid, key, cell_index)) {
        return atomicExchange(reservoir_sample_counts[cell_index], 0);
    } else {
        return 0;
    }
}

void increment_reservoir_sample_count(vec3 position, vec3 normal) {
    uint64_t key = get_reservoir_key(position, normal);
    uint cell_index;
    if (insert_reservoir_cell(constants.reservoir_hash_grid, key, cell_index)) {
        atomicAdd(reservoir_sample_counts[cell_index], 1);
    }
}

Basis create_surface_basis(Basis tangent_space, vec3 tangent_space_normal) {
    Basis surface_basis;
    surface_basis.normal = transform_to_basis(tangent_space, tangent_space_normal);
    surface_basis.tangent = gram_schmidt(surface_basis.normal, tangent_space.tangent);
    surface_basis.bitangent = normalize(cross(surface_basis.normal, surface_basis.tangent));
    return surface_basis;
}

struct TracePathConfig {
    uint bounce_count;
    bool find_resample_path, resample_path;
};

// State required when generating resample paths.
struct ResampleState {
    vec3 accumulated, attenuation, previous_normal;
    // This is true if the previous bounce was "diffuse", which in this context means that the
    // surface roughness is relatively high. This makes it so that when reconnecting, the different
    // solid angle won't fall outside the material BRDF.
    bool previous_bounce_was_diffuse;
    // The found path origin and destination if a path has been found.
    BounceSurface origin, destination;
    // The generator from just before generating a path from `destination`.
    Generator generator;
    float scatter_density;
    // True if a path has been found.
    bool is_found;
};

ResampleState create_resample_state() {
    ResampleState resample_state;
    resample_state.accumulated = vec3(0.0);
    resample_state.attenuation = vec3(1.0);
    resample_state.is_found = false;
    return resample_state;
}

struct ReplayPath {
    BounceSurface reconnect_location;
    vec3 radiance;
    bool is_found;
};

ReplayPath create_replay_path() {
    ReplayPath replay_path;
    replay_path.is_found = false;
    return replay_path;
}

struct PathState {
    vec3 accumulated, attenuation;
    // The two generators. `generator` is used to everything that doens't change the path.
    // `path_generator` generates the path. This is usefull when replaying paths, which only
    // requires replacing `path_generator`.
    Generator generator, path_generator;
    // The normalized device coordinates. This is only used to calculate the mip level for the
    // primary hit.
    vec2 ndc;
    // The current mip level. It gets increased every bounce by the surface roughness.
    float mip_level;
    // The next ray to be traced.
    Ray ray;
    ReplayPath replay_path;
};

PathState create_path_state(Generator generator, Generator path_generator, vec2 ndc, Ray ray) {
    return PathState(vec3(0.0), vec3(1.0), generator, path_generator, ndc, 1.0, ray, create_replay_path());
}

LobeType sample_scatter_direction(
    in SurfaceProperties surface, inout Generator generator, vec3 local_view,
    out vec3 local_scatter, out vec3 local_half_vector, out float density
) {
    LobeType lobe_type = 0;
    if (constants.sample_strategy == UNIFORM_HEMISPHERE_SAMPLING) {
        local_scatter = uniform_hemisphere_sample(generator);
        local_half_vector = normalize(local_scatter + local_view);
        lobe_type = DIFFUSE_LOBE;
        density = INVERSE_2_PI;
    } else if (constants.sample_strategy == COSINE_HEMISPHERE_SAMPLING) {
        local_scatter = cosine_hemisphere_sample(generator);
        local_half_vector = normalize(local_scatter + local_view);
        lobe_type = DIFFUSE_LOBE;
        density = cosine_hemisphere_density(local_scatter.z);
    } else {
        if (random_float(generator) > surface.metallic) {
            local_scatter = cosine_hemisphere_sample(generator);
            local_half_vector = normalize(local_scatter + local_view);
            lobe_type = DIFFUSE_LOBE;
        } else {
            local_scatter = ggx_sample(local_view, surface.roughness, local_half_vector, generator);
            lobe_type = SPECULAR_LOBE;
        }
        density = surface.metallic * ggx_density(local_view, local_half_vector, surface.roughness)
            + (1.0 - surface.metallic) * cosine_hemisphere_density(local_scatter.z);
    }
    return lobe_type;
}

// Trace the next path segment. Returns true false if the path has left the scene.
bool next_path_segment(inout PathState path_state, inout ResampleState resample_state, inout TracePathConfig config, uint bounce) {
    RayHit hit;
    if (!trace_ray(path_state.ray, bounce, path_state.ndc, hit)) {
        vec3 sky_color = vec3(0.0);
        path_state.accumulated += sky_color * path_state.attenuation;
        if (resample_state.is_found) {
            resample_state.accumulated += sky_color * resample_state.attenuation;
        }
        return false;
    }

    if (constants.restir_replay == REPLAY_FIRST && path_state.replay_path.is_found) {
        if (length(bounce_surface_position(path_state.replay_path.reconnect_location) - hit.world_position) < 0.01) {
            path_state.accumulated += path_state.attenuation * path_state.replay_path.radiance;
            return false;
        } else {
            path_state.replay_path.is_found = false;
            // Continuing here may not be correct.
        }
    }

    Instance instance = instances[hit.instance];
    Material material = materials[instance.material];

    // Calculate mip level for the first bounce using ray differentials.
    if (bounce == 0) {
        path_state.mip_level = calculate_mip_level(hit.texcoord_ddx, hit.texcoord_ddy);
    }

    // Create the surface basis from from the normal map.
    Basis surface_basis = create_surface_basis(hit.tangent_space, octahedron_decode(
        textureLod(textures[material.normal_texture], hit.texcoord, path_state.mip_level).xy
    ));

    vec4 albedo = textureLod(textures[material.albedo_texture], hit.texcoord, path_state.mip_level)
        * vec4(material.base_color[0], material.base_color[1], material.base_color[2], material.base_color[3]);

    vec2 metallic_roughness = textureLod(textures[material.specular_texture], hit.texcoord, path_state.mip_level).rg;
    vec3 emissive = textureLod(textures[material.emissive_texture], hit.texcoord, path_state.mip_level).rgb
        * vec3(material.emissive[0], material.emissive[1], material.emissive[2]);

    // Determine the surface properties.
    SurfaceProperties surface;
    surface.albedo = albedo.rgb;
    surface.metallic = metallic_roughness.r * material.metallic;
    surface.roughness = pow2(metallic_roughness.g * material.roughness);
    surface.view_direction = normalize(path_state.ray.origin - hit.world_position);
    surface.normal_dot_view = clamp(dot(surface_basis.normal, surface.view_direction), 0.0001, 1.0);
    surface.fresnel_min = fresnel_min(material.ior, surface.albedo, surface.metallic);
    surface.fresnel_max = 1.0;

    // Calculate the view vector local to the surface basis. This is often referred to as wo in
    // litterature.
    vec3 local_view = transform_from_basis(surface_basis, surface.view_direction);

    float scatter_density = 0.0;
    vec3 local_scatter, local_half_vector;

    bool can_reconnect = config.resample_path
        // Check that we haven't already reconnected.
        && !path_state.replay_path.is_found
        // Check that we aren't "recording" a resample path.
        && !resample_state.is_found;
    if (can_reconnect) {
        uint64_t key = hash_grid_key(hash_grid_cell(
            normal_bias(hit.world_position, surface_basis.normal),
            constants.camera_position.xyz,
            random_vec3(path_state.generator) * 2.0 - 1.0,
            random_float(path_state.generator) * 0.5 - 0.25,
            constants.reservoir_hash_grid
        ));
        uint reservoir_cell_index;
        if (find_reservoir_cell(constants.reservoir_hash_grid, key, reservoir_cell_index)) {
            Reservoir reservoir = get_reservoir_from_cell(reservoir_cell_index, path_state.generator);
            // Reconnect with the sample in the reservoir if it's valid.
            bool can_reconnect = reservoir_is_valid(reservoir)
                && can_reconnect_to_path(hit.world_position, surface_basis.normal, reservoir.path);
            if (can_reconnect) {
                path_state.path_generator = reservoir.path.generator;
                local_scatter = transform_from_basis(surface_basis, normalize(
                    bounce_surface_position(reservoir.path.destination) - hit.world_position
                ));
                local_half_vector = normalize(local_scatter + local_view);
                float jacobian = path_jacobian(hit.world_position, reservoir.path);
                scatter_density = 1.0 / reservoir.weight * clamp(jacobian, 0.00001, 1000.0);
                path_state.replay_path.radiance =
                    vec3(reservoir.path.radiance[0], reservoir.path.radiance[1], reservoir.path.radiance[2]);
                path_state.replay_path.reconnect_location = reservoir.path.destination;
                path_state.replay_path.is_found = true;
            }
        }
    }

    Generator before_sample_generator = path_state.path_generator;

    bool perform_next_event_estimation = constants.light_sampling == LIGHT_SAMPLING_NEXT_EVENT_ESTIMATION
        && surface.roughness > 0.15;

    // Next event estimation.
    if (perform_next_event_estimation) {
        vec3 direct_light =
            direct_light_contribution(hit.world_position, surface_basis.normal, surface, path_state.path_generator);
        path_state.accumulated += path_state.attenuation * direct_light;
        if (resample_state.is_found) {
            resample_state.accumulated += direct_light * resample_state.attenuation;
        }
    }

    // TODO: Figure out how to handle hitting light sources directly.
    if (bounce == 0 || !perform_next_event_estimation) {
        path_state.accumulated += emissive * path_state.attenuation;
        if (resample_state.is_found) {
            resample_state.accumulated += emissive * resample_state.attenuation;
        }
    }

    LobeType lobe_type = 0;
    if (scatter_density == 0.0) {
        lobe_type = sample_scatter_direction(surface, path_state.path_generator, local_view, local_scatter, local_half_vector, scatter_density);
    }

    bool can_form_resample_path = config.find_resample_path
        // Only find a simple resample path per trace for now
        && !resample_state.is_found
        // Don't resample if a path is being replayed.
        && !path_state.replay_path.is_found
        // It isn't possible to reconnect the first bounce.
        && bounce != 0
        // The previous bounce must be counted as "diffuse" e.g. relatively rough so that
        // reconnecting doesn't cause the BRDF to become zero.
        && resample_state.previous_bounce_was_diffuse
        // The current bounce must be diffuse meaning that it samples the diffuse lope. This
        // is required to create a successfull reconnecting because diffuse sampling doesn't
        // depend on the incidence vector, which will be different because of the reconnection.
        && lobe_type == DIFFUSE_LOBE
        // Very short reconnection paths can cause a bunch of problems.
        && length_squared(hit.world_position - path_state.ray.origin) > 0.0025;
    if (can_form_resample_path) {
        resample_state.origin = create_bounce_surface(path_state.ray.origin, resample_state.previous_normal);
        resample_state.destination = create_bounce_surface(hit.world_position, surface_basis.normal);
        resample_state.generator = before_sample_generator;
        resample_state.scatter_density = scatter_density;
        resample_state.is_found = true;
    }

    path_state.ray.direction = transform_to_basis(surface_basis, local_scatter);
    path_state.ray.origin = normal_bias(hit.world_position, surface_basis.normal);

    ScatterProperties scatter =
        create_scatter_properties(surface, path_state.ray.direction, surface_basis.normal);
    vec3 brdf = ggx_specular(surface, scatter) + burley_diffuse(surface, scatter);

    // FIXME: The minimum here fixes NaNs appearing sometimes, most likely because `scatter_density`
    // is very small (it doesn't ever appear to be 0.0, thus it is likely to be numeric issues and
    // not algebraic).
    vec3 attenuation_factor = min((brdf * abs(scatter.normal_dot_scatter)) / scatter_density, vec3(1.0));

    if (constants.restir_replay == REPLAY_NONE && path_state.replay_path.is_found) {
        path_state.accumulated += attenuation_factor * path_state.attenuation * path_state.replay_path.radiance;
        return false;
    }

    path_state.attenuation *= attenuation_factor;
    if (resample_state.is_found) resample_state.attenuation *= attenuation_factor;

    resample_state.previous_normal = surface_basis.normal;
    resample_state.previous_bounce_was_diffuse = surface.roughness > 0.25;

    // Heuristic for mip level.
    path_state.mip_level += surface.roughness;

    return true;
}

void main() {
    uvec2 pixel_index = gl_GlobalInvocationID.xy;
    if (any(greaterThanEqual(pixel_index, constants.screen_size))) return;

    Generator generator =
        init_generator_from_pixel(pixel_index, constants.screen_size, constants.frame_index);

    vec3 accumulated = vec3(0.0);
    for (uint sample_index = 0; sample_index < constants.sample_count; sample_index++) {
        TracePathConfig trace_path_config;
        if (constants.use_world_space_restir != 0) {
            bool resample_path = random_float(generator) > 0.75;
            trace_path_config = TracePathConfig(constants.bounce_count, true, resample_path);
        } else {
            trace_path_config = TracePathConfig(constants.bounce_count, false, false);
        }

        Generator path_generator = init_generator_from_index(generator.state, sample_index);

        vec2 ndc = pixel_ndc(pixel_index, generator);
        Ray ray = Ray(camera_ray_direction(ndc), constants.camera_position.xyz);

        PathState path_state = create_path_state(generator, path_generator, ndc, ray);
        ResampleState resample_state = create_resample_state();

        for (uint bounce = 0; bounce < constants.bounce_count; bounce++) {
            if (!next_path_segment(path_state, resample_state, trace_path_config, bounce)) break;
        }

        accumulated += path_state.accumulated;

        if (resample_state.is_found) {
            vec3 position = bounce_surface_position(resample_state.origin);
            vec3 normal = bounce_surface_normal(resample_state.origin);
            if (length_squared(resample_state.accumulated) > 0.0) {
                Reservoir reservoir;
                reservoir.path.origin = resample_state.origin;
                reservoir.path.destination = resample_state.destination;
                reservoir.path.generator = resample_state.generator;
                for (uint i = 0; i < 3; i++) reservoir.path.radiance[i] = resample_state.accumulated[i];
                reservoir.sample_count = take_reservoir_sample_count(position, normal) + 1;
                reservoir.weight_sum = path_target_function(reservoir.path) / resample_state.scatter_density;
                update_reservoir_weight(reservoir);

                // Insert as a reservoir update.
                uint64_t key = hash_grid_key(hash_grid_cell(
                    position, constants.camera_position.xyz, vec3(0.0), 0.0, constants.reservoir_update_hash_grid
                ));
                uint slot = hash_grid_hash(key) % constants.reservoir_update_hash_grid.capacity;
                uint update_index = atomicAdd(reservoir_update_counts[slot], 1);
                if (update_index < constants.reservoir_updates_per_cell) {
                    uint base_index = constants.reservoir_updates_per_cell * slot;
                    reservoir_updates[base_index + update_index] = reservoir;
                }
            } else {
                increment_reservoir_sample_count(position, normal);
            }
        }
    }

    accumulated /= constants.sample_count;

    if (constants.accumulated_frame_count != 0) {
        vec3 previous = imageLoad(target, ivec2(pixel_index)).xyz;
        accumulated = (previous * (constants.accumulated_frame_count - 1) + accumulated)
            / constants.accumulated_frame_count;
    }

    imageStore(target, ivec2(pixel_index), vec4(accumulated, 1.0));
}
