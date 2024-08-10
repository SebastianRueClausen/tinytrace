
#define HASH_GRID_BUFFER reservoir_hashes
#define HASH_GRID_INSERT insert_reservoir
#define HASH_GRID_FIND find_reservoir
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

HashGrid create_hash_grid(uint bucket_size, uint capacity) {
    HashGrid hash_grid;
    hash_grid.camera_position = constants.camera_position.xyz;
    hash_grid.scene_scale = 10.0;
    hash_grid.bucket_size = bucket_size;
    hash_grid.capacity = capacity;
    return hash_grid;
}

struct ResamplePath {
    Path path;
    float pdf;
    bool is_found;
};

struct TracePathConfig {
    uint bounce_count;
    bool find_resample_path;
    bool resample_path;
};

vec3 trace_path(
    Ray ray,
    inout Generator generator,
    vec2 ndc,
    TracePathConfig config,
    out ResamplePath resample_path
) {
    vec3 accumulated = vec3(0.0), attenuation = vec3(1.0);
    vec3 previous_normal, resample_attenuation = vec3(1.0);

    resample_path.is_found = !config.find_resample_path;
    if (config.find_resample_path) {
        for (uint i = 0; i < 3; i++) {
            resample_path.path.radiance[i] = 0.0;
        }
    }

    float mip_level;
    for (uint bounce = 0; bounce < config.bounce_count; bounce++) {
        RayHit hit;

        if (!trace_ray(ray, bounce, ndc, hit)) {
            return accumulated + attenuation;
        }

        Instance instance = instances[hit.instance];
        Material material = materials[instance.material];

        // Calculate the mip level using the standard GLSL method.
        if (bounce == 0) {
            hit.texcoord_ddx *= vec2(constants.screen_size);
            hit.texcoord_ddy *= vec2(constants.screen_size);
            float max_length_squared =
                max(dot(hit.texcoord_ddx, hit.texcoord_ddx), dot(hit.texcoord_ddy, hit.texcoord_ddy));
            mip_level = 0.5 * log2(max_length_squared);
        }

        vec3 tangent_space_normal = octahedron_decode(
            textureLod(textures[material.normal_texture], hit.texcoord, mip_level).xy
        );

        Basis surface_basis;
        surface_basis.normal = transform_to_basis(hit.tangent_space, tangent_space_normal);
        surface_basis.tangent = gram_schmidt(surface_basis.normal, hit.tangent_space.tangent);
        surface_basis.bitangent = normalize(cross(surface_basis.normal, surface_basis.tangent));

        vec4 albedo = textureLod(textures[material.albedo_texture], hit.texcoord, mip_level);
        albedo *= vec4(material.base_color[0], material.base_color[1], material.base_color[2], material.base_color[3]);

        vec2 metallic_roughness = textureLod(textures[material.specular_texture], hit.texcoord, mip_level).rg;
        vec3 emissive = textureLod(textures[material.emissive_texture], hit.texcoord, mip_level).rgb;
        emissive *= vec3(material.emissive[0], material.emissive[1], material.emissive[2]);

        SurfaceProperties surface;
        surface.albedo = albedo.rgb;

        surface.metallic = metallic_roughness.r * material.metallic;
        surface.roughness = metallic_roughness.g * material.roughness;
        surface.roughness *= surface.roughness;

        // Heuristic for mip level.
        mip_level += surface.roughness;

        float dielectric_specular = (material.ior - 1.0) / (material.ior + 1.0);
        dielectric_specular *= dielectric_specular;
        surface.fresnel_min = mix(vec3(dielectric_specular), surface.albedo, surface.metallic);
        surface.fresnel_max = 1.0;

        surface.view_direction = normalize(ray.origin - hit.world_position);
        surface.normal_dot_view = clamp(dot(surface_basis.normal, surface.view_direction), 0.0001, 1.0);

        vec3 local_scatter, local_half_vector;
        vec3 local_view = transform_from_basis(surface_basis, surface.view_direction);
        if (random_float(generator) > surface.metallic) {
            local_scatter = cosine_hemisphere_sample(generator);
            local_half_vector = normalize(local_scatter + local_view);
        } else {
            local_scatter = ggx_sample(local_view, surface.roughness, local_half_vector, generator);
        }

        if (!resample_path.is_found && bounce != 0 && surface.roughness >= 0.25) {
            resample_path.path.origin = create_bounce_surface(ray.origin, previous_normal);
            resample_path.path.destination = create_bounce_surface(hit.world_position, surface_basis.normal);
            resample_path.path.generator = generator;
            resample_path.is_found = true;
        }

        ray.direction = transform_to_basis(surface_basis, local_scatter);
        ray.origin = hit.world_position + 0.0001 * surface_basis.normal;

        ScatterProperties scatter;
        scatter.direction = ray.direction;
        scatter.half_vector = normalize(surface.view_direction + scatter.direction);
        scatter.normal_dot_half = saturate(dot(surface_basis.normal, scatter.half_vector));
        scatter.normal_dot_scatter = saturate(dot(surface_basis.normal, scatter.direction));
        scatter.view_dot_half = saturate(dot(surface.view_direction, scatter.half_vector));

        vec3 brdf = ggx_specular(surface, scatter) + burley_diffuse(surface, scatter);
        float pdf = surface.metallic * ggx_pdf(local_view, local_half_vector, surface.roughness)
            + (1.0 - surface.metallic) * cosine_hemisphere_pdf(scatter.normal_dot_scatter);

        accumulated += emissive * attenuation;

        if (resample_path.is_found) {
            resample_path.pdf = pdf;
            for (uint i = 0; i < 3; i++) {
                resample_path.path.radiance[i] += emissive[i] * resample_attenuation[i];
            }
        }

        // FIXME: The minimum here fixes NaNs appearing sometimes,
        // most likely because the pdf is very small (it doesn't ever
        // appear to be 0.0, thus it is likely to be numeric issues and
        // not algebraic).
        vec3 attenuation_factor = min((brdf * abs(scatter.normal_dot_scatter)) / pdf, vec3(1.0));
        attenuation *= attenuation_factor;

        if (resample_path.is_found) {
            resample_attenuation *= attenuation_factor;
        }

        previous_normal = surface_basis.normal;
    }
    return accumulated;
}

void main() {
    uvec2 pixel_index = gl_GlobalInvocationID.xy;
    if (any(greaterThanEqual(pixel_index, constants.screen_size))) {
        return;
    }

    HashGrid reservoir_hash_grid = create_hash_grid(16, 0xffff);
    HashGrid reservoir_update_hash_grid = create_hash_grid(1, 1024);

    Generator generator = init_generator_from_pixel(pixel_index, constants.screen_size, constants.frame_index);

    vec3 accumulated = vec3(0.0);

    TracePathConfig trace_path_config = TracePathConfig(constants.bounce_count, true, false);

    for (uint sample_index = 0; sample_index < constants.sample_count; sample_index++) {
        vec2 ndc = pixel_ndc(pixel_index, generator);
        Ray ray = Ray(camera_ray_direction(ndc), constants.camera_position.xyz);

        ResamplePath resample_path;
        accumulated += trace_path(ray, generator, ndc, trace_path_config, resample_path);

        if (resample_path.is_found && sample_index == 0) {
            bool has_radiance = resample_path.path.radiance[0] != 0.0
                || resample_path.path.radiance[1] != 0.0
                || resample_path.path.radiance[2] != 0.0;
            if (!has_radiance) continue;
            uint64_t key = hash_grid_key(hash_grid_cell(bounce_surface_position(resample_path.path.origin), reservoir_update_hash_grid));
            uint slot = hash_grid_hash(key) % reservoir_update_hash_grid.capacity;
            uint update_index = atomicAdd(reservoir_updates[slot].update_count, 1);
            if (update_index < RESERVOIR_UPDATE_COUNT) {
                reservoir_updates[slot].paths[update_index] = resample_path.path;
                reservoir_updates[slot].weights[update_index] = path_target_function(resample_path.path) / resample_path.pdf;
            }
        }
    }

    accumulated /= float(constants.sample_count);

    if (constants.accumulated_frame_count != 0) {
        vec3 previous = imageLoad(target, ivec2(pixel_index)).xyz;
        accumulated = (previous * (constants.accumulated_frame_count - 1) + accumulated)
            / constants.accumulated_frame_count;
    }

    imageStore(target, ivec2(pixel_index), vec4(accumulated, 1.0));
}
