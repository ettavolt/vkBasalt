#version 450
#extension GL_EXT_scalar_block_layout : require

layout(local_size_x_id = 1, local_size_y_id = 2, local_size_z = 1) in;
layout(constant_id = 2) const uint CHANNELS = 1;
layout(push_constant) uniform PushConsts {
    uint width;
    uint height;
};
layout(std430, set = 0, binding = 0) buffer restrict Tensor {
    float tensor[];
};
layout(std430, set = 1, binding = 0) buffer restrict readonly Stats {
    //These are -(mean/std)+bias and 1/std.
    float stats[CHANNELS * 2];
};

void main() {
    if (gl_GlobalInvocationID.x >= width * height) return;
    const uint c = gl_GlobalInvocationID.y;
    const uint pxPos = gl_GlobalInvocationID.x * CHANNELS + c;
    tensor[pxPos] = max(fma(tensor[pxPos], stats[CHANNELS + c], stats[c]), 0.0);
}
