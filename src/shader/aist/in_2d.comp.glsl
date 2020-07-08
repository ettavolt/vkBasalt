#version 450
#extension GL_GOOGLE_include_directive : require
#include "consts.comp.glsl"
layout(constant_id = 2) const uint CHANNELS = 1;
const uint SIZE_SQRT = 32;
layout(local_size_x = SIZE_SQRT * SIZE_SQRT, local_size_y = 1, local_size_z = 1) in;

layout(push_constant) uniform PushConsts {
    uint substage;
};
const uint imageSize = WIDTH * HEIGHT;

layout(std430, set = 0, binding = 0) uniform restrict readonly Weights {
    //This is effectively scale + bias, but offsets are not calculated with spec constants
    //thus we can't have two fields in the struct.
    float weights[CHANNELS * 2];
};
layout(std430, set = 1, binding = 0) buffer restrict Tensor {
    float inTensor[imageSize * CHANNELS];
};
layout(std430, set = 1, binding = 1) buffer restrict Stats {
    float stats[CHANNELS * 2];
};

shared float[gl_WorkGroupSize.x] aggregate;

void main() {
    const uint c = gl_GlobalInvocationID.z;
    switch (substage) {
        case 0:
        float mean = 0.0;
        for (uint x = gl_LocalInvocationID.x; x < WIDTH; x += gl_WorkGroupSize.x) {
            for (uint y = 0; y < HEIGHT; y++) {
                mean += inTensor[(x * HEIGHT + y) * CHANNELS + c];
            }
        }
        aggregate[gl_LocalInvocationID.x] = mean;
        if (gl_LocalInvocationID.x >= SIZE_SQRT) return;
        barrier();
        mean = 0.0;
        for (uint x = gl_LocalInvocationID.x; x < SIZE_SQRT; x += SIZE_SQRT) {
            mean += aggregate[x];
        }
        aggregate[gl_LocalInvocationID.x] = mean;
        if (gl_LocalInvocationID.x > 0) return;
        barrier();
        mean = 0.0;
        for (uint x = 0; x < SIZE_SQRT; x++) {
            mean += aggregate[x];
        }
        stats[c] = mean / imageSize;
//        deviation = weights[c] / (sqrt(deviation / imageSize) + 0.00001);
//        stats[c] = - mean * deviation + weights[CHANNELS + c];
//        stats[CHANNELS + c] = deviation;
        break;
//        case 1:
//        return;
//        uint pos = gl_GlobalInvocationID.x * (HEIGHT / SPATIAL_DIVISOR) * CHANNELS
//        + gl_LocalInvocationID.y * CHANNELS
//        + c;
//        float x = image[pos];
//        image[pos] = min(fma(x, stats[CHANNELS + c], stats[c]), 0.0);
//        break;
    }
}
