#version 450
#extension GL_GOOGLE_include_directive : require
#include "consts.comp.glsl"
layout(local_size_x = 16, local_size_y = 16) in;

const int IN_CHANNELS = 32;
layout(set = 1, binding = 0, rgba8) uniform restrict writeonly image2D outImage;
layout(std430, set = 1, binding = 1) buffer restrict readonly InTensor {
    float inTensor[WIDTH * HEIGHT][IN_CHANNELS];
};

void main() {
    const int cx = int(gl_GlobalInvocationID.x);
    const int cy = int(gl_GlobalInvocationID.y);
    if (cx >= WIDTH || cy >= HEIGHT) return;
    const int offset = (cx % 2 * 2 + cy % 2) * 3;
    float res[IN_CHANNELS] = inTensor[cx / 2 * (HEIGHT / 2) + cy / 2];
    imageStore(outImage, ivec2(cx, cy), vec4(res[offset+0], res[offset+1], res[offset+2], 1.0));
}
