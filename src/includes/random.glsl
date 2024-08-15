#ifndef RANDOM
#define RANDOM

uint jenkins_hash(uint x) {
    x += x << 10;
    x ^= x >> 6;
    x += x << 3;
    x ^= x >> 11;
    x += x << 15;
    return x;
}

struct Generator {
    uint state;
};

Generator init_generator_from_index(uint index, uint frame) {
    uint seed = index ^ jenkins_hash(frame);
    return Generator(jenkins_hash(seed));
}

Generator init_generator_from_pixel(uvec2 pixel, uvec2 resolution, uint frame) {
    return init_generator_from_index(pixel.x + pixel.y * resolution.x, frame);
}

float uint_to_unit_float(uint value) {
    return uintBitsToFloat(0x3f800000 | (value >> 9)) - 1.0;
}

uint xor_shift(inout uint state) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}

float random_float(inout Generator generator) {
    return uint_to_unit_float(xor_shift(generator.state));
}

uint random_uint(inout Generator generator, uint end) {
    return xor_shift(generator.state) % end;
}

vec3 random_vec3(inout Generator generator) {
    return vec3(random_float(generator), random_float(generator), random_float(generator));
}

#endif
