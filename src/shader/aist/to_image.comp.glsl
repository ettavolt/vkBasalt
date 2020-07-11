#version 450
#extension GL_EXT_scalar_block_layout : require

layout(local_size_x_id = 1, local_size_y_id = 2) in;
layout(push_constant) uniform PushConsts {
    uint width;
    uint height;
};
const uint IN_CHANNELS = 3;

layout(set = 0, binding = 0, rgba8) uniform restrict writeonly image2D outImage;
layout(set = 1, binding = 0, std430) buffer restrict readonly InTensor {
    float inTensor[][IN_CHANNELS];
};

void main() {
    const int sx = int(gl_GlobalInvocationID.x);
    const int sy = int(gl_GlobalInvocationID.y);
    if (sx >= width || sy >= height) return;
    float res[IN_CHANNELS] = inTensor[sx * height + sy];
    vec3 rgbValues = vec3(res[0], res[1], res[2]);
    //rgbValues = fma(tanh(rgbValues), vec3(0.5), vec3(0.5));
    imageStore(outImage, ivec2(sx, sy), vec4(rgbValues, 1.0));
}
