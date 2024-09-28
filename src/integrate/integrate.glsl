#include "restir"
#include "scene"
#include "brdf"
#include "math"
#include "sample"
#include "debug"
#include "constants"
#include "hash_grid"

#include "<bindings>"

struct RayHit {
    vec3 world_position;
    Basis tangent_space;
    vec2 texcoord, texcoord_ddx, texcoord_ddy;
    uint instance, material;
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

    hit.instance = uint(rayQueryGetIntersectionInstanceCustomIndexEXT(query, true));
    Instance instance = scene.instances.instances[hit.instance];
    hit.material = instance.material;
    mat4x3 transform = mat4x3(instance.transform);

    vec3 positions[3];
    rayQueryGetIntersectionTriangleVertexPositionsEXT(query, true, positions);

    // Transform positions from model to world space.
    for (uint i = 0; i < 3; i++) positions[i] = transform * vec4(positions[i], 1.0);

    uint base_index = 3 * rayQueryGetIntersectionPrimitiveIndexEXT(query, true) + instance.index_offset;
    uvec3 triangle_indices = uvec3(
        scene.indices.indices[base_index + 0], scene.indices.indices[base_index + 1], scene.indices.indices[base_index + 2]
    ) + instance.vertex_offset;
    Vertex triangle_vertices[3] = Vertex[3](
        scene.vertices.vertices[triangle_indices[0]], scene.vertices.vertices[triangle_indices[1]], scene.vertices.vertices[triangle_indices[2]]
    );

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
    EmissiveTriangle triangle = scene.emissive_triangles.data[random_uint(generator, scene.emissive_triangle_count)];
    vec3 barycentric = sample_triangle(generator);
    mat4x3 transform = mat4x3(scene.instances.instances[triangle.instance].transform);
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
        barycentric, position, normal, texcoord, triangle.instance, uint(triangle.hash), area, 1.0 / scene.emissive_triangle_count
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
    uint triangle_index = rayQueryGetIntersectionPrimitiveIndexEXT(ray_query, true) + scene.instances.instances[instance_index].index_offset / 3;
    if (triangle_index % 0xffff != light_sample.hash) return vec3(0.0);

    Material material = scene.materials.materials[scene.instances.instances[light_sample.instance].material];

    // TODO: Figure out a good mip level.
    float mip_level = 4.0;
    vec3 emissive = textureLod(textures[material.emissive_texture], light_sample.texcoord, mip_level).rgb
        * vec3(material.emissive[0], material.emissive[1], material.emissive[2]);

    ScatterProperties scatter = create_scatter_properties(surface, ray.direction, normal);
    vec3 brdf = ggx_specular(surface, scatter) + burley_diffuse(surface, scatter);

    return brdf * emissive * abs(scatter.normal_dot_scatter) / density;
}

Reservoir get_reservoir_from_cell(uint cell_index, inout Generator generator) {
    uint base_index = restir_constants.reservoirs_per_cell * cell_index;
    uint reservoir_index = random_uint(generator, restir_constants.reservoirs_per_cell);
    return reservoirs[base_index + reservoir_index];
}

uint64_t get_reservoir_key(vec3 position, vec3 normal) {
    return hash_grid_key(hash_grid_cell(
        normal_bias(position, normal), constants.camera_position.xyz,
        vec3(0.0), 0.0, reservoir_hash_grid
    ));
}

uint take_reservoir_sample_count(vec3 position, vec3 normal) {
    uint64_t key = get_reservoir_key(position, normal);
    uint cell_index;
    if (hash_grid_insert(reservoir_hash_grid, key, cell_index)) {
        return atomicExchange(reservoir_sample_counts[cell_index], 0);
    } else {
        return 0;
    }
}

void increment_reservoir_sample_count(vec3 position, vec3 normal) {
    uint64_t key = get_reservoir_key(position, normal);
    uint cell_index;
    if (hash_grid_insert(reservoir_hash_grid, key, cell_index)) {
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
bool next_path_segment(inout PathState path_state, inout PathCandidate path_candidate, TracePathConfig config, uint bounce) {
    RayHit hit;
    if (!trace_ray(path_state.ray, bounce, path_state.ndc, hit)) {
        vec3 sky_color = vec3(0.0);
        path_state.accumulated += sky_color * path_state.attenuation;
        if (path_candidate.is_found) {
            add_light_to_path_candidate(path_candidate, sky_color, bounce);
        }
        return false;
    }

    if (restir_constants.replay == REPLAY_FIRST && path_state.replay_path.is_found) {
        if (length(path_vertex_position(path_state.replay_path.reconnect_location) - hit.world_position) < 0.01) {
            path_state.accumulated += path_state.attenuation * path_state.replay_path.radiance;
            return false;
        } else {
            path_state.replay_path.is_found = false;
            // Continuing here may not be correct.
        }
    }

    Material material = scene.materials.materials[hit.material];

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
        && !path_candidate.is_found;
    if (can_reconnect) {
        uint64_t key = hash_grid_key(hash_grid_cell(
            normal_bias(hit.world_position, surface_basis.normal),
            constants.camera_position.xyz,
            reservoir_hash_grid_position_jitter(path_state.generator),
            reservoir_hash_grid_level_jitter(path_state.generator),
            reservoir_hash_grid
        ));
        uint reservoir_cell_index;
        if (hash_grid_find(reservoir_hash_grid, key, reservoir_cell_index)) {
            Reservoir reservoir = get_reservoir_from_cell(reservoir_cell_index, path_state.generator);
            // Reconnect with the sample in the reservoir if it's valid.
            uint bounce_budget = bounce - constants.bounce_count - 1;
            bool can_reconnect = reservoir_is_valid(reservoir)
                && can_reconnect_to_path(hit.world_position, surface_basis.normal, bounce_budget, restir_constants.replay, reservoir.path);
            if (can_reconnect) {
                path_state.path_generator = reservoir.path.generator;
                local_scatter = transform_from_basis(surface_basis, normalize(
                    path_vertex_position(reservoir.path.destination) - hit.world_position
                ));
                local_half_vector = normalize(local_scatter + local_view);
                float jacobian = path_jacobian(hit.world_position, reservoir.path);
                scatter_density = 1.0 / reservoir.weight * clamp(jacobian, 0.00001, 1000.0);
                path_state.replay_path.radiance = vec3(reservoir.path.radiance[0], reservoir.path.radiance[1], reservoir.path.radiance[2]);
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
        if (path_candidate.is_found) {
            add_light_to_path_candidate(path_candidate, direct_light, bounce);
        }
    }

    // TODO: Figure out how to handle hitting light sources directly.
    if (bounce == 0 || !perform_next_event_estimation) {
        path_state.accumulated += emissive * path_state.attenuation;
        if (path_candidate.is_found) {
            add_light_to_path_candidate(path_candidate, emissive, bounce);
        }
    }

    LobeType lobe_type = 0;
    if (scatter_density == 0.0) {
        lobe_type = sample_scatter_direction(surface, path_state.path_generator, local_view, local_scatter, local_half_vector, scatter_density);
    }

    float section_distance_squared = length_squared(hit.world_position - path_state.ray.origin);
    bool has_candidate_path = config.find_resample_path
        && !path_state.replay_path.is_found
        && can_form_path_candidate(path_candidate, bounce, lobe_type, section_distance_squared);
    if (has_candidate_path) {
        path_candidate.origin = create_path_vertex(path_state.ray.origin, path_candidate.previous_normal);
        path_candidate.destination = create_path_vertex(hit.world_position, surface_basis.normal);
        path_candidate.generator = before_sample_generator;
        path_candidate.scatter_density = scatter_density;
        path_candidate.first_bounce = uint16_t(bounce);
        path_candidate.last_bounce = path_candidate.first_bounce;
        path_candidate.is_found = true;
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

    if (restir_constants.replay == REPLAY_NONE && path_state.replay_path.is_found) {
        path_state.accumulated += attenuation_factor * path_state.attenuation * path_state.replay_path.radiance;
        return false;
    }

    path_state.attenuation *= attenuation_factor;
    if (path_candidate.is_found) path_candidate.attenuation *= attenuation_factor;

    path_candidate.previous_normal = surface_basis.normal;
    path_candidate.previous_bounce_was_diffuse = surface.roughness > 0.25;

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
        PathCandidate path_candidate = create_empty_path_candidate();

        for (uint bounce = 0; bounce < constants.bounce_count; bounce++) {
            if (!next_path_segment(path_state, path_candidate, trace_path_config, bounce)) break;
        }

        accumulated += path_state.accumulated;

        if (path_candidate.is_found) {
            vec3 position = path_vertex_position(path_candidate.origin);
            vec3 normal = path_vertex_normal(path_candidate.origin);
            if (length_squared(path_candidate.accumulated) > 0.0) {
                uint16_t sample_count = uint16_t(take_reservoir_sample_count(position, normal) + 1);
                Reservoir reservoir = create_reservoir_from_path_candidate(path_candidate, sample_count);

                // Insert as a reservoir update.
                uint64_t key = hash_grid_key(hash_grid_cell(
                    position, constants.camera_position.xyz, vec3(0.0), 0.0, reservoir_update_hash_grid
                ));
                uint slot = hash_grid_hash(key) % reservoir_update_hash_grid.capacity;
                uint update_index = atomicAdd(reservoir_update_counts[slot], 1);
                if (update_index < restir_constants.updates_per_cell) {
                    uint base_index = restir_constants.updates_per_cell * slot;
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
