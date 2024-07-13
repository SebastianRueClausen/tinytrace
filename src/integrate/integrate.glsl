

float16_t random(inout uint seed) {
    seed = seed * 747796405 + 1;
    uint word = ((seed >> ((seed >> 28) + 4)) ^ seed) * 277803737;
    word = (word >> 22) ^ word;
    return float16_t(float(word) / 4294967295.0f);
}

uint initial_seed(uint x, uint y, uint frame) {
    uint seed = (x * 2654435769u) ^ (y * 2654435769u) ^ (frame * 2654435769u);
    seed = (seed ^ (seed >> 16u)) * 2654435769u;
    return seed;
}

struct RayHit {
    vec3 world_position;
    f16vec3 normal;
    f16vec3 tangent;
    f16vec3 bitangent;
    f16vec2 texcoord;
    f16vec2 texcoord_ddx, texcoord_ddy;
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

struct Ray {
    vec3 direction;
    vec3 origin;
};

// Returns the barycentric coordinates of ray triangle intersection.
f16vec3 triangle_intersection(vec3 triangle[3], Ray ray) {
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
    return f16vec3(vec3(1.0 - b - c, b, c));
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

    hit.world_position = barycentric.x * positions[0]
        + barycentric.y * positions[1]
        + barycentric.z * positions[2];

    // Calculate the differentials of the texture coordinates using ray
    // differentials if it is the first bounce.
    if (bounce == 0) {
        vec2 texel_size = 2.0 / vec2(constants.screen_size);

        Ray ray;
        ray.origin = rayQueryGetWorldRayOriginEXT(query);

        ray.direction = camera_ray_direction(vec2(ndc.x + texel_size.x, ndc.y));
        f16vec3 hx = triangle_intersection(positions, ray);

        ray.direction = camera_ray_direction(vec2(ndc.x, ndc.y + texel_size.y));
        f16vec3 hy = triangle_intersection(positions, ray);

        f16vec3 ddx = barycentric - hx;
        f16vec3 ddy = barycentric - hy;

        hit.texcoord_ddx = ddx.x * triangle_vertices[0].texcoord
            + ddx.y * triangle_vertices[1].texcoord
            + ddx.z * triangle_vertices[2].texcoord;
        hit.texcoord_ddy = ddy.x * triangle_vertices[0].texcoord
            + ddy.y * triangle_vertices[1].texcoord
            + ddy.z * triangle_vertices[2].texcoord;
    }

    if (instance.color_offset != INVALID_INDEX) {
        hit.color = barycentric.x * get_vertex_color(instance.color_offset + (triangle_indices[0] - instance.vertex_offset) * 4)
            + barycentric.y * get_vertex_color(instance.color_offset + (triangle_indices[1] - instance.vertex_offset) * 4)
            + barycentric.z * get_vertex_color(instance.color_offset + (triangle_indices[2] - instance.vertex_offset) * 4);
    } else {
        hit.color = f16vec4(1.0hf);
    }

    return hit;
}

// Use Gram-Schmidt to find a vector orthonormal to the normal most like the surface tangent.
f16vec3 world_space_tangent(f16vec3 normal, f16vec3 surface_tangent) {
    return normalize(surface_tangent - normal * dot(normal, surface_tangent));
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

const uint SAMPLE_COUNT = 4;
const uint BOUNCE_COUNT = 4;

void main() {
    uvec2 pixel_index = gl_GlobalInvocationID.xy;
    if (any(greaterThanEqual(pixel_index, constants.screen_size))) {
        return;
    }

    uint seed = (pixel_index.x + pixel_index.y * constants.screen_size.y) * 100;
    vec2 ndc = (vec2(pixel_index) / vec2(constants.screen_size)) * 2.0 - 1.0;

    f16vec3 accumulated = f16vec3(0.0hf);

    for (uint sample_index = 0; sample_index < SAMPLE_COUNT; sample_index++) {
        f16vec3 attenuation = f16vec3(1.0hf);

        Ray ray;
        RayHit hit;

        ray.direction = camera_ray_direction(ndc);
        ray.origin = constants.camera_position.xyz;

        float mip_level;

        for (uint bounce = 0; bounce < BOUNCE_COUNT; bounce++) {
            if (!trace_ray(ray, bounce, ndc, hit)) {
                accumulated += attenuation;
                break;
            } else {
                Instance instance = instances[hit.instance];
                Material material = materials[instance.material];

                vec2 texcoord = vec2(hit.texcoord);

                // Calculate the mip level using standard GLSL method.
                if (bounce == 0) {
                    hit.texcoord_ddx *= f16vec2(vec2(constants.screen_size));
                    hit.texcoord_ddy *= f16vec2(vec2(constants.screen_size));
                    float16_t max_length_sqr =
                        max(dot(hit.texcoord_ddx, hit.texcoord_ddx), dot(hit.texcoord_ddy, hit.texcoord_ddy));
                    mip_level = float(0.5hf * log2(max_length_sqr));
                }

                f16vec3 tangent_normal = octahedron_decode(
                    f16vec2(textureLod(textures[material.normal_texture], texcoord, mip_level).xy)
                );
                f16vec3 normal = normalize(
                    tangent_normal.x * hit.tangent
                        + tangent_normal.y * hit.bitangent
                        + tangent_normal.z * hit.normal
                );
                f16vec3 tangent = world_space_tangent(normal, hit.tangent);

                // Surface Properties.
                SurfaceProperties surface;

                f16vec4 albedo = f16vec4(textureLod(textures[material.albedo_texture], texcoord, mip_level));
                albedo *= f16vec4(material.base_color[0], material.base_color[1], material.base_color[2], material.base_color[3]);
                albedo *= hit.color;

                surface.albedo = albedo.rgb;

                f16vec2 specular = f16vec2(textureLod(textures[material.specular_texture], texcoord, mip_level).rg);
                f16vec3 emissive = f16vec3(textureLod(textures[material.emissive_texture], texcoord, mip_level).rgb);
                emissive *= f16vec3(material.emissive[0], material.emissive[1], material.emissive[2]);

                surface.metallic = specular.r * material.metallic;
                surface.roughness = specular.g * material.roughness;
                surface.roughness *= surface.roughness;

                // Heuristic for mip level.
                mip_level += float(surface.roughness);

                float16_t dielectric_specular = (material.ior - 1.0hf) / (material.ior + 1.0hf);
                dielectric_specular *= dielectric_specular;
                surface.fresnel_min = mix(f16vec3(dielectric_specular), surface.albedo, surface.metallic);
                surface.fresnel_max = clamp(dot(surface.fresnel_min, f16vec3(50.0hf * 0.33hf)), 0.0hf, 1.0hf);

                surface.view_direction = f16vec3(normalize(ray.origin - hit.world_position));
                surface.normal_dot_view = clamp(dot(normal, surface.view_direction), 0.0001hf, 1.0hf);

                // Find bounce direction (Uniform hemisphere sampling).
                float16_t theta = PI * 2.0hf * random(seed);
                float16_t u = 2.0hf * random(seed) - 1.0hf;
                float16_t r = sqrt(1.0hf - u * u);
                ray.direction = normalize(normal + f16vec3(r * cos(theta), r * sin(theta), u));
                ray.origin = hit.world_position + 0.0001 * normal;

                float16_t pdf = 1.0hf / (2.0hf * PI);

                // Scatter Properties.
                ScatterProperties scatter;
                scatter.direction = f16vec3(ray.direction);
                scatter.half_vector = normalize(surface.view_direction + scatter.direction);
                scatter.normal_dot_half = normalize(dot(normal, scatter.half_vector));
                scatter.normal_dot_scatter = normalize(dot(normal, scatter.direction));
                scatter.view_dot_half = normalize(dot(surface.view_direction, scatter.half_vector));

                f16vec3 brdf = ggx_specular(surface, scatter) + lambert_diffuse(surface);

                accumulated += emissive * attenuation; 
                attenuation *= (brdf * abs(scatter.normal_dot_scatter)) / pdf;
            }
        }
    }

    accumulated /= float16_t(SAMPLE_COUNT);
    imageStore(target, ivec2(pixel_index), vec4(f16vec4(accumulated, 1.0hf)));
}
