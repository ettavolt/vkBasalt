#version 450
#extension GL_GOOGLE_include_directive : require
#include "consts.comp.glsl"
layout(local_size_x = 16, local_size_y = 16) in;

const uint IN_CHANNELS = 3;
layout(set = 1, binding = 0, rgba8) uniform restrict writeonly image2D outImage;
layout(std430, set = 1, binding = 1) buffer restrict readonly InTensor {
    float inTensor[WIDTH * HEIGHT][IN_CHANNELS];
};

float activate(in float res) {
    return tanh(res) / 2 + 0.5;
}

void main() {
    const int sx = int(gl_GlobalInvocationID.x);
    const int sy = int(gl_GlobalInvocationID.y);
    if (sx >= WIDTH || sy >= HEIGHT) return;
    float res[IN_CHANNELS] = inTensor[sx * HEIGHT + sy];
    imageStore(outImage, ivec2(sx, sy), vec4(activate(res[0]), activate(res[1]), activate(res[2]), 1.0));
}
