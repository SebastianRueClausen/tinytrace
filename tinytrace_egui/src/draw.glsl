layout (push_constant) uniform DrawParameters {
    vec2 screen_size_in_points;
    uvec2 area_offset, area_size;
    uint texture_index;
    int vertex_offset;
    uint index_start, index_count;
};

vec3 barycentric(vec2 position, vec2 triangle[3]) {
    vec2 v0 = triangle[1] - triangle[0], v1 = triangle[2] - triangle[0], v2 = position - triangle[0];
    float d00 = dot(v0, v0), d01 = dot(v0, v1), d11 = dot(v1, v1), d20 = dot(v2, v0), d21 = dot(v2, v1);
    float denom = d00 * d11 - d01 * d01;
    float v = (d11 * d20 - d01 * d21) / denom, w = (d00 * d21 - d01 * d20) / denom;
    return vec3(1.0 - v - w, v, w);
}

vec2 get_position(float x, float y) {
    return vec2(x, y) / screen_size_in_points;// * 2.0 - 1.0;
}

vec4 get_color(uint color) {
    return vec4(float(color & 255), float((color >> 8) & 255), float((color >> 16) & 255), float((color >> 24) & 255)) / 255.0;
}

void main() {
    if (any(greaterThanEqual(gl_GlobalInvocationID.xy, area_size))) return;
    uvec2 pixel_index = gl_GlobalInvocationID.xy + area_offset;
    vec2 position = get_position(pixel_index.x + 0.5, pixel_index.y + 0.5);
    for (uint index_offset = index_start; index_offset < index_count + index_start; index_offset += 3) {
        Vertex v1 = vertices[indices[index_offset + 0] + vertex_offset];
        Vertex v2 = vertices[indices[index_offset + 1] + vertex_offset];
        Vertex v3 = vertices[indices[index_offset + 2] + vertex_offset];
        vec3 lambda = barycentric(position, vec2[3](get_position(v1.x, v1.y), get_position(v2.x, v2.y), get_position(v3.x, v3.y)));
        if (!all(greaterThanEqual(lambda, vec3(-0.001)))) continue;
        vec4 color = get_color(v1.color) * lambda.x + get_color(v2.color) * lambda.y + get_color(v3.color) * lambda.z;
        vec2 uv = vec2(v1.u, v1.v) * lambda.x + vec2(v2.u, v2.v) * lambda.y + vec2(v3.u, v3.v) * lambda.z;
        vec4 src = color * textureLod(textures[texture_index], uv, 0.0);
        vec4 background = imageLoad(render_target, ivec2(pixel_index));
        imageStore(render_target, ivec2(pixel_index), background * (1.0 - src.a) + src);
    }
}
