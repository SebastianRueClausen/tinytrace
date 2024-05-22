

struct RayHit {
    vec3 world_position;
    f16vec3 normal;
    f16vec3 tangent;
    f16vec3 bitangent;
    f16vec2 texcoord;
    int instance;
};

f16vec3 interpolate_vec3(f16vec3 barycentric, f16vec3 a, f16vec3 b, f16vec3 c) {
    return normalize(barycentric.x * a + barycentric.y * b + barycentric.z * c);
}

f16vec3 transform_normal(f16mat4 transform, f16vec3 normal) {
    return (transform * f16vec4(normal, 0.0hf)).xyz;
}

RayHit get_ray_hit(rayQueryEXT query) {
    RayHit hit;
    hit.instance = rayQueryGetIntersectionInstanceIdEXT(query, true);
    Instance instance = instances[hit.instance];
    vec3 positions[3];
    rayQueryGetIntersectionTriangleVertexPositionsEXT(query, true, positions);
    int base_index = rayQueryGetIntersectionPrimitiveIndexEXT(query, true) * 3;
    uint triangle_indices[3]
        = uint[3](indices[base_index], indices[base_index + 1], indices[base_index + 2]);
    Vertex triangle_vertices[3]
        = Vertex[3](vertices[triangle_indices[0]], vertices[triangle_indices[1]], vertices[triangle_indices[2]]);
    TangentFrame tangent_frames[3] = TangentFrame[3](
        decode_tangent_frame(triangle_vertices[0].tangent_frame),
        decode_tangent_frame(triangle_vertices[1].tangent_frame),
        decode_tangent_frame(triangle_vertices[2].tangent_frame)
    );
    f16vec3 barycentric = f16vec3(0.0hf, f16vec2(rayQueryGetIntersectionBarycentricsEXT(query, true)));
    barycentric.x = 1.0hf - barycentric.y - barycentric.z;
    hit.normal = interpolate_vec3(barycentric,
        transform_normal(instance.normal_transform, tangent_frames[0].normal),
        transform_normal(instance.normal_transform, tangent_frames[1].normal),
        transform_normal(instance.normal_transform, tangent_frames[2].normal)
    );
    hit.tangent = interpolate_vec3(barycentric,
        transform_normal(instance.normal_transform, tangent_frames[0].tangent),
        transform_normal(instance.normal_transform, tangent_frames[1].tangent),
        transform_normal(instance.normal_transform, tangent_frames[2].tangent)
    );
    float16_t bitangent_sign = sign(dot(barycentric, f16vec3(
        tangent_frames[0].bitangent_sign,
        tangent_frames[1].bitangent_sign,
        tangent_frames[2].bitangent_sign
    )));
    hit.bitangent = bitangent_sign * cross(hit.normal, hit.tangent);
    hit.texcoord = barycentric.x * triangle_vertices[0].texcoord
        + barycentric.y * triangle_vertices[1].texcoord
        + barycentric.z * triangle_vertices[2].texcoord;
    hit.world_position = float(barycentric.x) * positions[0]
        + float(barycentric.y) * positions[1]
        + float(barycentric.z) * positions[2];
    return hit;
}

void main() {
    uvec2 pixel_index = gl_GlobalInvocationID.xy;
    if (any(greaterThanEqual(pixel_index, constants.screen_size))) {
        return;
    }

    vec2 ndc = (vec2(pixel_index) / vec2(constants.screen_size)) * 2.0 - 1.0;
    vec3 ray_direction =  create_camera_ray(ndc, constants.inverse_proj, constants.inverse_view);
    
    rayQueryEXT ray_query;
    rayQueryInitializeEXT(
        ray_query,
        acceleration_structure,
        gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsOpaqueEXT | gl_RayFlagsSkipClosestHitShaderEXT,
        0xff,
        constants.camera_position.xyz,
        1.0e-3,
        ray_direction,
        1000.0
    );

    rayQueryProceedEXT(ray_query);

    if (rayQueryGetIntersectionTypeEXT(ray_query, true) == gl_RayQueryCommittedIntersectionNoneEXT) {
        imageStore(target, ivec2(pixel_index), vec4(abs(ray_direction), 1.0));
        return;
    }

    RayHit hit = get_ray_hit(ray_query);
    Instance instance = instances[hit.instance];
    Material material = materials[instance.material];

    vec2 texcoord = vec2(hit.texcoord);

    // TODO: Use ray differentials.
    f16vec3 tangent_normal = octahedron_decode(
        f16vec2(textureLod(textures[material.normal_texture], texcoord, 0.0).xy)
    );
    f16vec3 normal = normalize(
        tangent_normal.x * hit.tangent
            + tangent_normal.y * hit.bitangent
            + tangent_normal.z * hit.normal
    );

    f16vec3 albedo = f16vec3(textureLod(textures[material.albedo_texture], texcoord, 0.0).rgb);
    f16vec2 specular = f16vec2(textureLod(textures[material.specular_texture], texcoord, 0.0).rg);
    f16vec3 emissive = f16vec3(textureLod(textures[material.emissive_texture], texcoord, 0.0).rgb);

    float16_t metallic = specular.r * material.metallic;
    float16_t roughness = specular.g * material.roughness;
    roughness *= roughness;

    imageStore(target, ivec2(pixel_index), vec4(f16vec4(albedo, 1.0hf)));
}
