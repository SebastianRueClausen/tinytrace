#include "scene"
#include "brdf"
#include "math"
#include "sample"
#include "debug"
#include "restir"
#include "constants"

#include "<bindings>"

#define HASH_GRID_BUFFER reservoir_pool_hashes
#define HASH_GRID_INSERT insert_reservoir_pool
#define HASH_GRID_FIND find_reservoir_pool
#include "hash_grid"

#undef HASH_GRID_BUFFER
#undef HASH_GRID_INSERT
#undef HASH_GRID_FIND

#define HASH_GRID_BUFFER reservoir_update_hashes
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

RayHit get_ray_hit(rayQueryEXT query, uint bounce, vec2 ndc) {
    RayHit hit;

    hit.instance = rayQueryGetIntersectionInstanceCustomIndexEXT(query, true);
    Instance instance = instances[hit.instance];

    vec3 positions[3];
    rayQueryGetIntersectionTriangleVertexPositionsEXT(query, true, positions);

    // Transform positions from model to world space.
    for (uint i = 0; i < 3; i++) {
        positions[i] = (instance.transform * vec4(positions[i], 1.0)).xyz;
    }

    uint base_index = 3 * rayQueryGetIntersectionPrimitiveIndexEXT(query, true) + instance.index_offset;
    uvec3 triangle_indices = uvec3(
        indices[base_index + 0],
        indices[base_index + 1],
        indices[base_index + 2]
    ) + instance.vertex_offset;

    Vertex triangle_vertices[3] = Vertex[3](
        vertices[triangle_indices[0]],
        vertices[triangle_indices[1]],
        vertices[triangle_indices[2]]
    );

    vec3 barycentric = vec3(0.0, rayQueryGetIntersectionBarycentricsEXT(query, true));
    barycentric.x = 1.0 - barycentric.y - barycentric.z;

    TangentFrame tangent_frames[3] = TangentFrame[3](
        decode_tangent_frame(triangle_vertices[0].tangent_frame),
        decode_tangent_frame(triangle_vertices[1].tangent_frame),
        decode_tangent_frame(triangle_vertices[2].tangent_frame)
    );

    mat3 normal_transform = mat3(instance.normal_transform);
    hit.tangent_space.normal = normalize(barycentric.x * tangent_frames[0].normal
        + barycentric.y * tangent_frames[1].normal
        + barycentric.z * tangent_frames[2].normal);
    hit.tangent_space.tangent = normalize(barycentric.x * tangent_frames[0].tangent
        + barycentric.y * tangent_frames[1].tangent
        + barycentric.z * tangent_frames[2].tangent);
    hit.tangent_space.normal = normalize(normal_transform * hit.tangent_space.normal);
    hit.tangent_space.tangent = normalize(normal_transform * hit.tangent_space.tangent);

    float bitangent_sign = sign(dot(barycentric, vec3(
        tangent_frames[0].bitangent_sign,
        tangent_frames[1].bitangent_sign,
        tangent_frames[2].bitangent_sign
    )));
    hit.tangent_space.bitangent
        = normalize(bitangent_sign * cross(hit.tangent_space.normal, hit.tangent_space.tangent));

    hit.texcoord = barycentric.x * vec2(triangle_vertices[0].texcoord)
        + barycentric.y * vec2(triangle_vertices[1].texcoord)
        + barycentric.z * vec2(triangle_vertices[2].texcoord);

    hit.world_position = barycentric.x * positions[0]
        + barycentric.y * positions[1]
        + barycentric.z * positions[2];

    // Calculate the differentials of the texture coordinates using ray
    // differentials if it's the first bounce.
    if (bounce == 0) {
        vec2 texel_size = 2.0 / vec2(constants.screen_size);

        Ray ray;
        ray.origin = rayQueryGetWorldRayOriginEXT(query);

        ray.direction = camera_ray_direction(vec2(ndc.x + texel_size.x, ndc.y));
        vec3 hx = triangle_intersection(positions, ray);

        ray.direction = camera_ray_direction(vec2(ndc.x, ndc.y + texel_size.y));
        vec3 hy = triangle_intersection(positions, ray);

        vec3 ddx = barycentric - hx;
        vec3 ddy = barycentric - hy;

        hit.texcoord_ddx = ddx.x * vec2(triangle_vertices[0].texcoord)
            + ddx.y * vec2(triangle_vertices[1].texcoord)
            + ddx.z * vec2(triangle_vertices[2].texcoord);
        hit.texcoord_ddy = ddy.x * vec2(triangle_vertices[0].texcoord)
            + ddy.y * vec2(triangle_vertices[1].texcoord)
            + ddy.z * vec2(triangle_vertices[2].texcoord);
    }

    return hit;
}

bool trace_ray(Ray ray, uint bounce, vec2 ndc, out RayHit hit) {
    rayQueryEXT ray_query;
    rayQueryInitializeEXT(
        ray_query,
        acceleration_structure,
        gl_RayFlagsOpaqueEXT,
        0xff,
        ray.origin,
        1.0e-3,
        ray.direction,
        1000.0
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

Basis create_surface_basis(Basis tangent_space, vec3 tangent_space_normal) {
    Basis surface_basis;
    surface_basis.normal = transform_to_basis(tangent_space, tangent_space_normal);
    surface_basis.tangent = gram_schmidt(surface_basis.normal, tangent_space.tangent);
    surface_basis.bitangent = normalize(cross(surface_basis.normal, surface_basis.tangent));
    return surface_basis;
}

struct TracePathConfig {
    uint bounce_count;
    bool find_resample_path;
    bool resample_path;
};

// A found (or not found) resample path.
struct ResamplePath {
    // The path found.
    Path path;
    // The probability density function of generating the path.
    float pdf;
    // True if a valid path is found.
    bool is_found;
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
    // The pdf of the found path.
    float pdf;
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
    bool has_left_scene;
};

PathState create_path_state(Generator generator, Generator path_generator, vec2 ndc, Ray ray) {
    return PathState(vec3(0.0), vec3(1.0), generator, path_generator, ndc, 1.0, ray, false);
}

LobeType importance_sample(
    in SurfaceProperties surface, inout Generator generator, vec3 local_view,
    out vec3 local_scatter, out vec3 local_half_vector, out float pdf
) {
    LobeType lobe_type = 0;
    if (constants.sample_strategy == UNIFORM_HEMISPHERE_SAMPLING) {
        local_scatter = uniform_hemisphere_sample(generator);
        local_half_vector = normalize(local_scatter + local_view);
        lobe_type = DIFFUSE_LOBE;
        pdf = 1.0 / PI;
    } else if (constants.sample_strategy == COSINE_HEMISPHERE_SAMPLING) {
        local_scatter = cosine_hemisphere_sample(generator);
        local_half_vector = normalize(local_scatter + local_view);
        lobe_type = DIFFUSE_LOBE;
        pdf = cosine_hemisphere_pdf(local_scatter.z);
    } else {
        if (random_float(generator) > surface.metallic) {
            local_scatter = cosine_hemisphere_sample(generator);
            local_half_vector = normalize(local_scatter + local_view);
            lobe_type = DIFFUSE_LOBE;
        } else {
            local_scatter = ggx_sample(local_view, surface.roughness, local_half_vector, generator);
            lobe_type = SPECULAR_LOBE;
        }
        pdf = surface.metallic * ggx_pdf(local_view, local_half_vector, surface.roughness)
            + (1.0 - surface.metallic) * cosine_hemisphere_pdf(local_scatter.z);
    }
    return lobe_type;
}

void next_path_segment(
    inout PathState path_state,
    inout ResampleState resample_state,
    inout TracePathConfig config,
    uint bounce
) {
    RayHit hit;
    if (!trace_ray(path_state.ray, bounce, path_state.ndc, hit)) {
        path_state.has_left_scene = true;
        return;
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

    float pdf = 0.0;
    vec3 local_scatter, local_half_vector;
    if (config.resample_path && bounce != 0) {
        config.resample_path = false;
        uint64_t key = hash_grid_key(hash_grid_cell(
            normal_bias(hit.world_position, surface_basis.normal),
            constants.camera_position.xyz,
            vec3(0.0),
            constants.reservoir_hash_grid
        ));
        uint reservoir_pool_index;
        if (find_reservoir_pool(constants.reservoir_hash_grid, key, reservoir_pool_index)) {
            Reservoir reservoir =
                determine_reservoir_from_pool(reservoir_pools[reservoir_pool_index], path_state.generator);
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
                pdf = 1.0 / reservoir.weight * clamp(jacobian, 0.0, 1000.0);
            }
        }
    }

    Generator before_sample_generator = path_state.path_generator;
    LobeType lobe_type = 0;
    if (pdf == 0.0) {
        lobe_type = importance_sample(
            surface, path_state.path_generator, local_view,
            local_scatter, local_half_vector, pdf
        );
    }

    bool can_form_resample_path = config.find_resample_path
        // Only find a simple resample path per trace for now
        && !resample_state.is_found
        // It isn't possible to reconnect the first bounce.
        && bounce != 0
        // The previous bounce must be counted as "diffuse" e.g. relatively rough so that
        // reconnecting doesn't cause the BRDF to become zero.
        && resample_state.previous_bounce_was_diffuse
        // The current bounce must be diffuse meaning that it samples the diffuse lope. This
        // is required to create a successfull reconnecting because diffuse sampling doesn't
        // depend on the incidence vector, which will be different because of the reconnection.
        && lobe_type == DIFFUSE_LOBE;
    if (can_form_resample_path) {
        resample_state.origin = create_bounce_surface(path_state.ray.origin, resample_state.previous_normal);
        resample_state.destination = create_bounce_surface(hit.world_position, surface_basis.normal);
        resample_state.generator = before_sample_generator;
        resample_state.pdf = pdf;
        resample_state.is_found = true;
    }

    path_state.ray.direction = transform_to_basis(surface_basis, local_scatter);
    path_state.ray.origin = normal_bias(hit.world_position, surface_basis.normal);

    ScatterProperties scatter;
    scatter.direction = path_state.ray.direction;
    scatter.half_vector = normalize(surface.view_direction + scatter.direction);
    scatter.normal_dot_half = saturate(dot(surface_basis.normal, scatter.half_vector));
    scatter.normal_dot_scatter = saturate(dot(surface_basis.normal, scatter.direction));
    scatter.view_dot_half = saturate(dot(surface.view_direction, scatter.half_vector));

    vec3 brdf = ggx_specular(surface, scatter) + burley_diffuse(surface, scatter);

    // FIXME: The minimum here fixes NaNs appearing sometimes,
    // most likely because the pdf is very small (it doesn't ever
    // appear to be 0.0, thus it is likely to be numeric issues and
    // not algebraic).
    vec3 attenuation_factor = min((brdf * abs(scatter.normal_dot_scatter)) / pdf, vec3(1.0));

    path_state.accumulated += emissive * path_state.attenuation;
    path_state.attenuation *= attenuation_factor;
    if (resample_state.is_found) {
        resample_state.accumulated += emissive * resample_state.attenuation;
        resample_state.attenuation *= attenuation_factor;
    }

    resample_state.previous_normal = surface_basis.normal;
    resample_state.previous_bounce_was_diffuse = surface.roughness > 0.25;

    // Heuristic for mip level.
    path_state.mip_level += surface.roughness;
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
            bool resample = sample_index % 2 == 1;
            trace_path_config = TracePathConfig(constants.bounce_count, resample, !resample);
        } else {
            trace_path_config = TracePathConfig(constants.bounce_count, false, false);
        }
        Generator path_generator =
            init_generator_from_pixel(pixel_index, constants.screen_size, generator.state);

        vec2 ndc = pixel_ndc(pixel_index, generator);
        Ray ray = Ray(camera_ray_direction(ndc), constants.camera_position.xyz);

        PathState path_state = create_path_state(generator, path_generator, ndc, ray);
        ResampleState resample_state = create_resample_state();

        for (uint bounce = 0; bounce < constants.bounce_count && !path_state.has_left_scene; bounce++) {
            next_path_segment(path_state, resample_state, trace_path_config, bounce);
        }

        accumulated += path_state.accumulated;

        if (resample_state.is_found /* && length_squared(resample_state.accumulated) > 0.0 */) {
            // Create path.
            Path path;
            path.origin = resample_state.origin;
            path.destination = resample_state.destination;
            path.generator = resample_state.generator;
            for (uint i = 0; i < 3; i++) path.radiance[i] = resample_state.accumulated[i];

            // Insert as a reservoir update.
            uint64_t key = hash_grid_key(hash_grid_cell(
                bounce_surface_position(path.origin),
                constants.camera_position.xyz,
                vec3(0.0),
                constants.reservoir_update_hash_grid
            ));
            uint slot = hash_grid_hash(key) % constants.reservoir_update_hash_grid.capacity;
            uint update_index = atomicAdd(reservoir_updates[slot].update_count, 1);
            if (update_index < RESERVOIR_UPDATE_COUNT) {
                reservoir_updates[slot].paths[update_index] = path;
                reservoir_updates[slot].weights[update_index] = path_target_function(path) / resample_state.pdf;
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
