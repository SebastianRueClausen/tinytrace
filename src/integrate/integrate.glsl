#include "bsdf"
#include "constants"
#include "debug"
#include "math"
#include "sample"
#include "scene"

#include "<bindings>"

struct RayHit {
    vec3 world_position;
    Basis tangent_space;
    vec2 texcoord, texcoord_ddx, texcoord_ddy;
    uint instance, material;
};

vec3 camera_ray_direction(vec2 ndc) {
    vec4 view_space_point = constants.inverse_proj * vec4(-ndc, 1.0, 1.0);
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
    for (uint i = 0; i < 3; i++)
        positions[i] = transform * vec4(positions[i], 1.0);

    uint base_index =
        3 * rayQueryGetIntersectionPrimitiveIndexEXT(query, true) + instance.index_offset;
    uvec3 triangle_indices =
        uvec3(
            scene.indices.indices[base_index + 0], scene.indices.indices[base_index + 1],
            scene.indices.indices[base_index + 2]
        ) +
        instance.vertex_offset;
    Vertex triangle_vertices[3] = Vertex[3](
        scene.vertices.vertices[triangle_indices[0]], scene.vertices.vertices[triangle_indices[1]],
        scene.vertices.vertices[triangle_indices[2]]
    );

    vec3 barycentric = vec3(0.0, rayQueryGetIntersectionBarycentricsEXT(query, true));
    barycentric.x = 1.0 - barycentric.y - barycentric.z;

    TangentFrame tangent_frames[3] = TangentFrame[3](
        decode_tangent_frame(triangle_vertices[0].tangent_frame),
        decode_tangent_frame(triangle_vertices[1].tangent_frame),
        decode_tangent_frame(triangle_vertices[2].tangent_frame)
    );

    mat3 normal_transform = mat3(instance.normal_transform);
    hit.tangent_space.normal = normalize(
        normal_transform *
        interpolate(
            barycentric, tangent_frames[0].normal, tangent_frames[1].normal,
            tangent_frames[2].normal
        )
    );
    hit.tangent_space.tangent = normalize(
        normal_transform *
        interpolate(
            barycentric, tangent_frames[0].tangent, tangent_frames[1].tangent,
            tangent_frames[2].tangent
        )
    );
    float bitangent_sign = sign(
        dot(barycentric,
            vec3(
                tangent_frames[0].bitangent_sign, tangent_frames[1].bitangent_sign,
                tangent_frames[2].bitangent_sign
            ))
    );
    hit.tangent_space.bitangent =
        normalize(bitangent_sign * cross(hit.tangent_space.normal, hit.tangent_space.tangent));

    hit.texcoord = interpolate(
        barycentric, triangle_vertices[0].texcoord, triangle_vertices[1].texcoord,
        triangle_vertices[2].texcoord
    );
    hit.world_position =
        barycentric.x * positions[0] + barycentric.y * positions[1] + barycentric.z * positions[2];

    // Calculate the differentials of the texture coordinates using ray
    // differentials if it's the first bounce.
    if (bounce == 0) {
        vec2 texel_size = 2.0 / vec2(constants.screen_size);
        Ray ray =
            Ray(camera_ray_direction(vec2(ndc.x + texel_size.x, ndc.y)),
                rayQueryGetWorldRayOriginEXT(query));
        vec3 hx = triangle_intersection(positions, ray);
        ray.direction = camera_ray_direction(vec2(ndc.x, ndc.y + texel_size.y));
        vec3 hy = triangle_intersection(positions, ray);
        hit.texcoord_ddx = interpolate(
            barycentric - hx, triangle_vertices[0].texcoord, triangle_vertices[1].texcoord,
            triangle_vertices[2].texcoord
        );
        hit.texcoord_ddy = interpolate(
            barycentric - hy, triangle_vertices[0].texcoord, triangle_vertices[1].texcoord,
            triangle_vertices[2].texcoord
        );
    }

    return hit;
}

bool trace_ray(Ray ray, uint bounce, vec2 ndc, out RayHit hit) {
    rayQueryEXT ray_query;
    rayQueryInitializeEXT(
        ray_query, acceleration_structure,
        gl_RayFlagsOpaqueEXT | gl_RayFlagsCullBackFacingTrianglesEXT, 0xff, ray.origin, 1.0e-3,
        ray.direction, 1000.0
    );
    while (rayQueryProceedEXT(ray_query))
        ;
    if (rayQueryGetIntersectionTypeEXT(ray_query, true) ==
        gl_RayQueryCommittedIntersectionNoneEXT) {
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
    return 0.5 *
        log2(
               max(length_squared(texcoord_ddx * constants.screen_size),
                   length_squared(texcoord_ddy * constants.screen_size))
        );
}

bool fetch_texture(uint index, vec2 texture_coordinates, float mip_level, inout vec4 rgba) {
    bool has_texture = index != INVALID_INDEX;
    if (has_texture)
        rgba = textureLod(textures[index], texture_coordinates, mip_level);
    return has_texture;
}

vec3 material_emissive(in Material material, vec2 texture_coordinates, float mip_level) {
    vec4 color = material.constants.emission_color;
    vec4 luminance = vec4(material.constants.emission_luminance);
    fetch_texture(material.textures.emission_color, texture_coordinates, mip_level, color);
    fetch_texture(material.textures.emission_luminance, texture_coordinates, mip_level, luminance);
    return color.xyz * luminance.x;
}

vec3 direct_light_contribution(
    vec3 position, vec3 wi, Basis surface_basis, MaterialConstants surface_material, Lobes lobes,
    inout Generator generator
) {
    DirectLightSample light_sample = sample_random_light(scene, generator);
    vec3 to_light = light_sample.position - position;
    vec3 wo = transform_from_basis(surface_basis, -to_light);
    float distance_squared = length_squared(to_light);
    Ray ray = Ray(to_light / sqrt(distance_squared), position);
    // Transform the area density to the solid angle density.
    float density = (1.0 / light_sample.area) * distance_squared /
        saturate(dot(-ray.direction, light_sample.normal)) * light_sample.light_probability;
    rayQueryEXT ray_query;
    rayQueryInitializeEXT(
        ray_query, acceleration_structure, gl_RayFlagsOpaqueEXT, 0xff, ray.origin, 1.0e-3,
        ray.direction, 10000.0
    );
    while (rayQueryProceedEXT(ray_query))
        ;
    if (rayQueryGetIntersectionTypeEXT(ray_query, true) ==
        gl_RayQueryCommittedIntersectionNoneEXT) {
        return vec3(0.0);
    }
    uint instance_index = rayQueryGetIntersectionInstanceCustomIndexEXT(ray_query, true);
    if (instance_index != light_sample.instance)
        return vec3(0.0);
    uint triangle_index = rayQueryGetIntersectionPrimitiveIndexEXT(ray_query, true) +
        scene.instances.instances[instance_index].index_offset / 3;
    if (triangle_index % 0xffff != light_sample.hash)
        return vec3(0.0);
    Material material =
        scene.materials.materials[scene.instances.instances[light_sample.instance].material];
    // TODO: Figure out a good mip level.
    vec3 emissive = material_emissive(material, light_sample.texcoord, 4.0);
    float bsdf_density;
    vec3 bsdf = bsdf_evaluate(surface_material, lobes, wi, wo, bsdf_density);
    return bsdf * emissive * abs(cos_theta(wo)) / density;
}

Basis create_surface_basis(Basis tangent_space, vec3 tangent_space_normal) {
    Basis surface_basis;
    surface_basis.normal = transform_to_basis(tangent_space, tangent_space_normal);
    surface_basis.tangent = gram_schmidt(surface_basis.normal, tangent_space.tangent);
    surface_basis.bitangent = normalize(cross(surface_basis.normal, surface_basis.tangent));
    return surface_basis;
}

struct PathState {
    vec3 accumulated, throughput;
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

MaterialConstants load_textures(in Material material, vec2 texture_coordinates, float mip_level) {
    MaterialConstants constants = material.constants;
    vec4 rgba;
    if (fetch_texture(material.textures.base_color, texture_coordinates, mip_level, rgba))
        constants.base_color = rgba;
    if (fetch_texture(material.textures.specular_color, texture_coordinates, mip_level, rgba))
        constants.specular_color = rgba;
    if (fetch_texture(material.textures.transmission_color, texture_coordinates, mip_level, rgba))
        constants.transmission_color = rgba;
    if (fetch_texture(material.textures.coat_color, texture_coordinates, mip_level, rgba))
        constants.coat_color = rgba;
    if (fetch_texture(material.textures.fuzz_color, texture_coordinates, mip_level, rgba))
        constants.fuzz_color = rgba;
    if (fetch_texture(material.textures.geometry_normal, texture_coordinates, mip_level, rgba))
        constants.geometry_normal = octahedron_decode(rgba.rg).xyzx;
    if (fetch_texture(material.textures.base_metalness, texture_coordinates, mip_level, rgba))
        constants.base_metalness = rgba.r;
    if (fetch_texture(
            material.textures.base_diffuse_roughness, texture_coordinates, mip_level, rgba
        ))
        constants.base_diffuse_roughness = rgba.r;
    if (fetch_texture(material.textures.specular_weight, texture_coordinates, mip_level, rgba))
        constants.specular_weight = rgba.r;
    if (fetch_texture(material.textures.specular_roughness, texture_coordinates, mip_level, rgba))
        constants.specular_roughness = rgba.r;
    if (fetch_texture(
            material.textures.specular_roughness_anisotropy, texture_coordinates, mip_level, rgba
        ))
        constants.specular_roughness_anisotropy = rgba.r;
    if (fetch_texture(material.textures.specular_ior, texture_coordinates, mip_level, rgba))
        constants.specular_ior = rgba.r;
    if (fetch_texture(material.textures.specular_rotation, texture_coordinates, mip_level, rgba))
        constants.specular_rotation = rgba.r;
    if (fetch_texture(material.textures.transmission_weight, texture_coordinates, mip_level, rgba))
        constants.transmission_weight = rgba.r;
    if (fetch_texture(material.textures.coat_weight, texture_coordinates, mip_level, rgba))
        constants.coat_weight = rgba.r;
    if (fetch_texture(material.textures.coat_darkening, texture_coordinates, mip_level, rgba))
        constants.coat_darkening = rgba.r;
    if (fetch_texture(material.textures.coat_rotation, texture_coordinates, mip_level, rgba))
        constants.coat_rotation = rgba.r;
    if (fetch_texture(material.textures.fuzz_weight, texture_coordinates, mip_level, rgba))
        constants.fuzz_weight = rgba.r;
    if (fetch_texture(material.textures.fuzz_roughness, texture_coordinates, mip_level, rgba))
        constants.fuzz_roughness = rgba.r;
    if (fetch_texture(material.textures.geometry_opacity, texture_coordinates, mip_level, rgba))
        constants.geometry_opacity = rgba.r;
    return constants;
}

// Trace the next path segment. Returns false if the path has left the scene.
bool next_path_segment(inout PathState path_state, uint bounce) {
    RayHit hit;
    if (!trace_ray(path_state.ray, bounce, path_state.ndc, hit)) {
        vec3 sky_color = vec3(4.0);
        path_state.accumulated += sky_color * path_state.throughput;
        return false;
    }

    // Calculate mip level for the first bounce using ray differentials.
    if (bounce == 0) {
        path_state.mip_level = calculate_mip_level(hit.texcoord_ddx, hit.texcoord_ddy);
    }

    Material material = scene.materials.materials[hit.material];
    MaterialConstants material_constants =
        load_textures(material, hit.texcoord, path_state.mip_level);
    Basis surface_basis =
        create_surface_basis(hit.tangent_space, material_constants.geometry_normal.xyz);

    vec3 wi = transform_from_basis(surface_basis, -path_state.ray.direction);
    Lobes lobes = bsdf_prepare(material_constants, wi, path_state.generator);

    bool perform_next_event_estimation =
        constants.light_sampling == LIGHT_SAMPLING_NEXT_EVENT_ESTIMATION &&
        scene.emissive_triangle_count > 0 && material_constants.specular_roughness > 0.15;

    if (perform_next_event_estimation) {
        vec3 direct_light = direct_light_contribution(
            hit.world_position, wi, surface_basis, material_constants, lobes, path_state.generator
        );
        path_state.accumulated += path_state.throughput * direct_light;
    }

    if (bounce == 0 || !perform_next_event_estimation) {
        vec3 emissive = material_emissive(material, hit.texcoord, path_state.mip_level);
        path_state.accumulated += emissive * path_state.throughput;
    }

    float scatter_density = 1.0;
    vec3 wo;
    int sampled_lobe_index;
    vec3 bsdf = bsdf_sample(
        material_constants, lobes, wi, path_state.generator, wo, scatter_density, sampled_lobe_index
    );

    path_state.ray.direction = transform_to_basis(surface_basis, wo);
    path_state.ray.origin = hit.world_position;
    path_state.throughput *= bsdf / max(scatter_density, PDF_EPSILON) * abs(cos_theta(wo));

    return true;
}

void main() {
    uvec2 pixel_index = gl_GlobalInvocationID.xy;
    if (any(greaterThanEqual(pixel_index, constants.screen_size)))
        return;
    Generator generator =
        init_generator_from_pixel(pixel_index, constants.screen_size, constants.frame_index);
    vec3 accumulated = vec3(0.0);
    for (uint sample_index = 0; sample_index < constants.sample_count; sample_index++) {
        vec2 ndc = pixel_ndc(pixel_index, generator);
        Ray ray = Ray(camera_ray_direction(ndc), constants.camera_position.xyz);
        PathState path_state = create_path_state(generator, ndc, ray);
        for (uint bounce = 0; bounce < constants.bounce_count; bounce++) {
            if (!next_path_segment(path_state, bounce))
                break;
        }
        accumulated += path_state.accumulated;
    }
    accumulated /= constants.sample_count;
    if (constants.accumulated_frame_count != 0) {
        vec3 previous = imageLoad(target, ivec2(pixel_index)).xyz;
        accumulated = (previous * (constants.accumulated_frame_count - 1) + accumulated) /
            constants.accumulated_frame_count;
    }
    imageStore(target, ivec2(pixel_index), vec4(accumulated, 1.0));
}
