#version 450
#extension GL_EXT_scalar_block_layout : require

layout(local_size_x_id = 1, local_size_y_id = 2) in;
layout(push_constant) uniform PushConsts {
    uint width;
    uint height;
};
const uint OUT_CHANNELS = 3;

layout(rgba8, set = 0, binding = 0) uniform restrict writeonly image2D outImage;
layout(std430, set = 1, binding = 0) buffer restrict readonly InTensor {
    float inTensor[][OUT_CHANNELS];
};

void main() {
    if (gl_GlobalInvocationID.x >= width || gl_GlobalInvocationID.y >= height) return;
    const int sx = int(gl_GlobalInvocationID.x);
    const int sy = int(gl_GlobalInvocationID.y);
    float res[OUT_CHANNELS] = inTensor[sx * height + sy];
    vec3 rgbValues = vec3(res[0], res[1], res[2]);
    rgbValues = fma(tanh(rgbValues), vec3(0.5), vec3(0.5));
    imageStore(outImage, ivec2(sx, sy), vec4(rgbValues, 1.0));
}
