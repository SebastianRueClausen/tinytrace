#include "scene"
#include "brdf"
#include "math"
#include "sample"
#include "debug"
#include "constants"

#include "<bindings>"

struct RayHit {
    vec3 world_position;
    Basis tangent_space;
    vec2 texcoord, texcoord_ddx, texcoord_ddy;
    uint instance, material;
};

vec3 camera_ray_direction(vec2 ndc) {
    vec4 view_space_point = constants.inverse_proj * vec4(ndc.x, -ndc.y, 1.0, 1.0);
    return normalize((constants.inverse_view * vec4(view_space_point.xyz, 0.0)).xyz);
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

vec3 direct_light_contribution(vec3 position, Basis surface_basis, SurfaceProperties surface, inout Generator generator) {
    DirectLightSample light_sample = sample_random_light(scene, generator);
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
    ScatterProperties scatter = create_scatter_properties(surface, ray.direction, surface_basis.normal);
    return brdf(surface_basis, surface, scatter) * emissive * abs(scatter.normal_dot_scatter) / density;
}

Basis create_surface_basis(Basis tangent_space, vec3 tangent_space_normal) {
    Basis surface_basis;
    surface_basis.normal = transform_to_basis(tangent_space, tangent_space_normal);
    surface_basis.tangent = gram_schmidt(surface_basis.normal, tangent_space.tangent);
    surface_basis.bitangent = normalize(cross(surface_basis.normal, surface_basis.tangent));
    return surface_basis;
}

struct PathState {
    vec3 accumulated, attenuation;
    // The two generators. `generator` is used to everything that doens't change the path.
    // `path_generator` generates the path. This is usefull when replaying paths, which only
    // requires replacing `path_generator`.
    Generator generator;
    // The normalized device coordinates. This is only used to calculate the mip level for the
    // primary hit.
    vec2 ndc;
    // The current mip level. It gets increased every bounce by the surface roughness.
    float mip_level;
    // The next ray to be traced.
    Ray ray;
};

PathState create_path_state(Generator generator, vec2 ndc, Ray ray) {
    return PathState(vec3(0.0), vec3(1.0), generator, ndc, 1.0, ray);
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
            local_scatter = ggx_sample(local_view, vec2(surface.roughness), local_half_vector, generator);
            lobe_type = SPECULAR_LOBE;
        }
        density = surface.metallic * ggx_density(local_view, local_half_vector, surface.roughness)
            + (1.0 - surface.metallic) * cosine_hemisphere_density(local_scatter.z);
    }
    density = max(0.001, density);
    return lobe_type;
}

// Trace the next path segment. Returns true false if the path has left the scene.
bool next_path_segment(inout PathState path_state, uint bounce) {
    RayHit hit;
    if (!trace_ray(path_state.ray, bounce, path_state.ndc, hit)) {
        vec3 sky_color = vec3(0.0);
        if (path_state.ray.direction.y > 0.0) {
            sky_color = vec3(5.0);
        }
        path_state.accumulated += sky_color * path_state.attenuation;
        return false;
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

    SurfaceProperties surface;
    surface.anisotropy_direction = vec2(cos(material.anisotropy_rotation), sin(material.anisotropy_rotation));
    surface.anisotropy_strength = material.anisotropy_strength;
    if (material.anisotropy_texture != uint16_t(0xffff)) {
        vec3 anisotropy = textureLod(textures[material.anisotropy_texture], hit.texcoord, path_state.mip_level).rgb;
        vec2 direction = normalize(anisotropy.rg * 2.0 - 1.0);
        mat2 direction_transform = mat2(
            surface.anisotropy_direction.x, surface.anisotropy_direction.y,
            -surface.anisotropy_direction.y, surface.anisotropy_direction.x
        );
        surface.anisotropy_direction = direction_transform * normalize(anisotropy.rg * 2.0 - 1.0);
        surface.anisotropy_strength = saturate(surface.anisotropy_strength * anisotropy.b);
        surface.is_anisotropic = true;
    } else {
        surface.is_anisotropic = false;
    }

    // Determine the surface properties.
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

    bool perform_next_event_estimation = constants.light_sampling == LIGHT_SAMPLING_NEXT_EVENT_ESTIMATION
        && scene.emissive_triangle_count > 0
        && surface.roughness > 0.15;

    // Next event estimation.
    if (perform_next_event_estimation) {
        vec3 direct_light =
            direct_light_contribution(hit.world_position, surface_basis, surface, path_state.generator);
        path_state.accumulated += path_state.attenuation * direct_light;
    }

    // TODO: Figure out how to handle hitting light sources directly.
    if (bounce == 0 || !perform_next_event_estimation) {
        path_state.accumulated += emissive * path_state.attenuation;
    }

    float scatter_density = 0.0;
    vec3 local_scatter, local_half_vector;
    LobeType lobe_type = sample_scatter_direction(
        surface, path_state.generator, local_view, local_scatter, local_half_vector, scatter_density
    );

    path_state.ray.direction = transform_to_basis(surface_basis, local_scatter);
    path_state.ray.origin = hit.world_position + surface_basis.normal * 0.001;

    ScatterProperties scatter =
        create_scatter_properties(surface, path_state.ray.direction, surface_basis.normal);

    // FIXME: The minimum here fixes NaNs appearing sometimes, most likely because `scatter_density`
    // is very small (it doesn't ever appear to be 0.0, thus it is likely to be numeric issues and
    // not algebraic).
    vec3 attenuation_factor = min((brdf(surface_basis, surface, scatter) * abs(scatter.normal_dot_scatter)) / scatter_density, vec3(1.0));

    path_state.attenuation *= attenuation_factor;

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
        vec2 ndc = pixel_ndc(pixel_index, generator);
        Ray ray = Ray(camera_ray_direction(ndc), constants.camera_position.xyz);
        PathState path_state = create_path_state(generator, ndc, ray);
        for (uint bounce = 0; bounce < constants.bounce_count; bounce++) {
            if (!next_path_segment(path_state, bounce)) break;
        }
        accumulated += path_state.accumulated;
    }
    accumulated /= constants.sample_count;
    if (constants.accumulated_frame_count != 0) {
        vec3 previous = imageLoad(target, ivec2(pixel_index)).xyz;
        accumulated = (previous * (constants.accumulated_frame_count - 1) + accumulated)
            / constants.accumulated_frame_count;
    }
    imageStore(target, ivec2(pixel_index), vec4(accumulated, 1.0));
}
