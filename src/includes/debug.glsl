#ifndef DEBUG
#define DEBUG

vec3 debug_color_from_hash(uint hash) {
    return vec3(uvec3(hash & 0x3ff, (hash >> 11) & 0x7ff, (hash >> 22) & 0x7ff)) / vec3(1023.0, 2047.0, 2047.0);
}

#endif
