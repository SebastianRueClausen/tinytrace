#include "<bindings>"

void main() {
    uint index = gl_GlobalInvocationID.x;
    if (index < src.length()) {
        dst[index] = src[index];
    }
}
