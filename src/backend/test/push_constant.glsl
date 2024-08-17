layout (push_constant) uniform PushConstant {
    uint value;
};

void main() {
    uint index = gl_GlobalInvocationID.x;
    if (index < dst.length()) {
        dst[index] = value;
    }
}
