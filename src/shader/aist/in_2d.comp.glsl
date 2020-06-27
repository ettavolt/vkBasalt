#version 450
#extension GL_GOOGLE_include_directive : require
#include "consts.comp.glsl"
layout(constant_id = 2) const uint SPATIAL_DIVISOR = 2;
layout(constant_id = 3) const uint CHANNELS = 1;
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

layout(push_constant) uniform PushConsts {
    uint substage;
};
const uint imageSize = (HEIGHT / SPATIAL_DIVISOR) * (WIDTH / SPATIAL_DIVISOR);

layout(std430, set = 0, binding = 0) uniform restrict readonly Weights {
    //This is effectively scale + bias, but offsets are not calculated with spec constants.
    float weights[CHANNELS * 2];
};
layout(std430, set = 1, binding = 0) buffer restrict Tensor {
    float image[imageSize * CHANNELS];
};
layout(std430, set = 1, binding = 1) buffer restrict Stats {
    float stats[CHANNELS * 2];
};

void main() {
    const uint c = gl_GlobalInvocationID.z;
    switch (substage) {
        case 0:
        float mean = 0.0;
        float deviation = 1.0;
        for (uint i = 0; i < 1000; i++) {
            float x = image[i * CHANNELS + c];
            float deviates = x - mean;
            mean += deviates;
//            mean = fma(deviates, 1.0/(i + 1), mean);
//            deviation = fma(deviates, x - mean, deviation);
        }
        deviation = weights[c] / (sqrt(deviation / imageSize) + 0.00001);
        stats[c] = - mean * deviation + weights[CHANNELS + c];
        stats[CHANNELS + c] = deviation;
        break;
        case 1:
        return;
        uint pos = gl_GlobalInvocationID.x * (HEIGHT / SPATIAL_DIVISOR) * CHANNELS
        + gl_LocalInvocationID.y * CHANNELS
        + c;
        float x = image[pos];
        image[pos] = min(fma(x, stats[CHANNELS + c], stats[c]), 0.0);
        break;
    }
}
