#ifndef RANDOM
#define RANDOM

float random(inout uint seed) {
    seed = seed * 747796405 + 1;
    uint word = ((seed >> ((seed >> 28) + 4)) ^ seed) * 277803737;
    word = (word >> 22) ^ word;
    return float(word) / 4294967295.0f;
}

uint hash(uint x) {
    x = ((x >> 16) ^ x) * 0x45d9f3b;
    x = ((x >> 16) ^ x) * 0x45d9f3b;
    x = (x >> 16) ^ x;
    return x;
}

struct Generator {
    uint state;
};

uint random_uint(inout Generator generator) {
    uint x = generator.state;
	x ^= x << 13;
	x ^= x >> 17;
	x ^= x << 5;
    generator.state = x;
	return x;
}

float random_float(inout Generator generator) {
    return float(random_uint(generator)) / 4294967295.0f;
}

#endif