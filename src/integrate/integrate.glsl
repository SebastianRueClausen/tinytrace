

float16_t random(inout uint seed) {
    seed = seed * 747796405 + 1;
    uint word = ((seed >> ((seed >> 28) + 4)) ^ seed) * 277803737;
    word = (word >> 22) ^ word;
    return float16_t(float(word) / 4294967295.0f);
}

struct RayHit {
    vec3 world_position;
    f16vec3 normal;
    f16vec3 tangent;
    f16vec3 bitangent;
    f16vec2 texcoord;
    f16vec4 color;
    int instance;
};

f16vec4 get_vertex_color(uint base_index) {
    return f16vec4(
        colors[base_index + 0],
        colors[base_index + 1],
        colors[base_index + 2],
        colors[base_index + 3]
    );
}

RayHit get_ray_hit(rayQueryEXT query) {
    RayHit hit;

    hit.instance = rayQueryGetIntersectionInstanceCustomIndexEXT(query, true);
    Instance instance = instances[hit.instance];

    vec3 positions[3];
    rayQueryGetIntersectionTriangleVertexPositionsEXT(query, true, positions);

    uint base_index = 3 * rayQueryGetIntersectionPrimitiveIndexEXT(query, true);
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

    f16vec3 barycentric
        = f16vec3(0.0hf, f16vec2(rayQueryGetIntersectionBarycentricsEXT(query, true)));
    barycentric.x = 1.0hf - barycentric.y - barycentric.z;

    TangentFrame tangent_frames[3] = TangentFrame[3](
        decode_tangent_frame(triangle_vertices[0].tangent_frame),
        decode_tangent_frame(triangle_vertices[1].tangent_frame),
        decode_tangent_frame(triangle_vertices[2].tangent_frame)
    );

    f16mat3 normal_transform = f16mat3(instance.normal_transform);
    hit.normal = barycentric.x * (normal_transform * tangent_frames[0].normal)
        + barycentric.y * (normal_transform * tangent_frames[1].normal)
        + barycentric.z * (normal_transform * tangent_frames[2].normal);
    hit.tangent = barycentric.x * (normal_transform * tangent_frames[0].tangent)
        + barycentric.y * (normal_transform * tangent_frames[1].tangent)
        + barycentric.z * (normal_transform * tangent_frames[2].tangent);
    float16_t bitangent_sign = sign(dot(barycentric, f16vec3(
        tangent_frames[0].bitangent_sign,
        tangent_frames[1].bitangent_sign,
        tangent_frames[2].bitangent_sign
    )));
    hit.bitangent = bitangent_sign * cross(hit.normal, hit.tangent);

    hit.texcoord = barycentric.x * triangle_vertices[0].texcoord
        + barycentric.y * triangle_vertices[1].texcoord
        + barycentric.z * triangle_vertices[2].texcoord;

    hit.world_position = barycentric.x * (instance.transform * vec4(positions[0], 1.0)).xyz
        + barycentric.y * (instance.transform * vec4(positions[1], 1.0)).xyz
        + barycentric.z * (instance.transform * vec4(positions[2], 1.0)).xyz;

    if (instance.color_offset != INVALID_INDEX) {
        hit.color = barycentric.x * get_vertex_color(instance.color_offset + (triangle_indices[0] - instance.vertex_offset) * 4)
            + barycentric.y * get_vertex_color(instance.color_offset + (triangle_indices[1] - instance.vertex_offset) * 4)
            + barycentric.z * get_vertex_color(instance.color_offset + (triangle_indices[2] - instance.vertex_offset) * 4);
    } else {
        hit.color = f16vec4(1.0hf);
    }

    return hit;
}

struct Ray {
    vec3 direction;
    vec3 origin;
};

bool trace_ray(Ray ray, out RayHit hit) {
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
    hit = get_ray_hit(ray_query);
    return true;
}

const uint SAMPLE_COUNT = 16;
const uint BOUNCE_COUNT = 4;

void main() {
    uvec2 pixel_index = gl_GlobalInvocationID.xy;
    if (any(greaterThanEqual(pixel_index, constants.screen_size))) {
        return;
    }

    uint seed = pixel_index.x + pixel_index.y * constants.screen_size.x;
    vec2 ndc = (vec2(pixel_index) / vec2(constants.screen_size)) * 2.0 - 1.0;

    f16vec3 accumulated = f16vec3(0.0hf);

    for (uint sample_index = 0; sample_index < SAMPLE_COUNT; sample_index++) {
        f16vec3 color = f16vec3(1.0hf);

        Ray ray;
        RayHit hit;

        ray.direction = create_camera_ray(ndc, constants.inverse_proj, constants.inverse_view);
        ray.origin = constants.camera_position.xyz;

        for (uint bounce = 0; bounce < BOUNCE_COUNT; bounce++) {
            if (!trace_ray(ray, hit)) {
                color *= f16vec3(1.0);
                accumulated += color;
                break;
            } else {
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

                f16vec4 albedo = f16vec4(textureLod(textures[material.albedo_texture], texcoord, 0.0));
                albedo *= f16vec4(material.base_color[0], material.base_color[1], material.base_color[2], material.base_color[3]);
                albedo *= hit.color;

                f16vec2 specular = f16vec2(textureLod(textures[material.specular_texture], texcoord, 0.0).rg);
                f16vec3 emissive = f16vec3(textureLod(textures[material.emissive_texture], texcoord, 0.0).rgb);
                emissive *= f16vec3(material.emissive[0], material.emissive[1], material.emissive[2]);

                float16_t metallic = specular.r * material.metallic;
                float16_t roughness = specular.g * material.roughness;
                roughness *= roughness;

                accumulated += emissive * color; 

                color *= albedo.rgb;

                float16_t theta = PI * 2.0hf * random(seed);
                float16_t u = 2.0hf * random(seed) - 1.0hf;
                float16_t r = sqrt(1.0hf - u * u);
                ray.direction = normalize(normal + f16vec3(r * cos(theta), r * sin(theta), u));
                ray.origin = hit.world_position + 0.0001 * normal;
            }
        }
    }

    accumulated /= float16_t(SAMPLE_COUNT);
    imageStore(target, ivec2(pixel_index), vec4(f16vec4(accumulated, 1.0hf)));
}
