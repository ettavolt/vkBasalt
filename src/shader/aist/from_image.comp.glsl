#version 450
#extension GL_EXT_scalar_block_layout : require

layout(local_size_x_id = 1, local_size_y_id = 2) in;
layout(push_constant) uniform PushConsts {
    uint width;
    uint height;
};

layout(rgba8, set = 0, binding = 0) uniform restrict readonly image2D inImage;
layout(std430, set = 1, binding = 0) buffer restrict writeonly OutTensor {
    float outTensor[][3];
};

void main() {
    if (gl_GlobalInvocationID.x >= width || gl_GlobalInvocationID.y >= height) return;
    vec4 imagePixel = imageLoad(inImage, ivec2(gl_GlobalInvocationID.x, gl_GlobalInvocationID.y));
    uint outPos = gl_GlobalInvocationID.x * height + gl_GlobalInvocationID.y;
    outTensor[outPos][0] = imagePixel.r;
    outTensor[outPos][1] = imagePixel.g;
    outTensor[outPos][2] = imagePixel.b;
}
