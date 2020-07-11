#version 450
#extension GL_GOOGLE_include_directive : require
#include "consts.comp.glsl"
layout(constant_id = 2) const uint CHANNELS = 1;
//TODO: push max local x here.
layout(local_size_x = 1024, local_size_y = 1, local_size_z = 1) in;

layout(push_constant) uniform PushConsts {
    uint substage;
};
const uint imageSize = WIDTH * HEIGHT;

layout(std430, set = 0, binding = 0) uniform restrict readonly Weights {
    //This is effectively scale & bias, but offsets are not calculated with spec constants
    //thus we can't have two fields in the struct.
    float weights[CHANNELS * 2];
};
//Making this channels outer reduces two substages time by 410 μs for vulkanscene sample.
layout(std430, set = 1, binding = 0) buffer readonly restrict Tensor {
    float inTensor[imageSize * CHANNELS];
};
layout(std430, set = 1, binding = 1) buffer restrict Stats {
    float stats[CHANNELS * 2];
};

shared float[gl_WorkGroupSize.x] aggregate;

void main() {
    const uint c = gl_GlobalInvocationID.z;
    //const uint featureMapOffset = c * imageSize;
    const uint featureMapOffset = c;
    const uint sizeSqrt = uint(ceil(sqrt(gl_WorkGroupSize.x)));
    float buf = 0.0;
    switch (substage) {
        case 0:
        for (uint i = gl_LocalInvocationID.x; i < imageSize; i += gl_WorkGroupSize.x) {
            buf += inTensor[i * CHANNELS + featureMapOffset];
        }
        aggregate[gl_LocalInvocationID.x] = buf;
        if (gl_LocalInvocationID.x >= sizeSqrt) return;
        barrier();
        buf = 0.0;
        for (uint i = gl_LocalInvocationID.x; i < gl_WorkGroupSize.x; i += sizeSqrt) {
            buf += aggregate[i];
        }
        aggregate[gl_LocalInvocationID.x] = buf;
        if (gl_LocalInvocationID.x > 0) return;
        barrier();
        buf = 0.0;
        for (uint i = 0; i < sizeSqrt; i++) {
            buf += aggregate[i];
        }
        stats[c] = buf / imageSize;
        break;

        case 1:
        const float mean = stats[c];
        for (uint i = gl_LocalInvocationID.x; i < imageSize; i += gl_WorkGroupSize.x) {
            float diff = inTensor[i * CHANNELS + featureMapOffset];
            buf += diff * diff;
        }
        aggregate[gl_LocalInvocationID.x] = buf;
        if (gl_LocalInvocationID.x >= sizeSqrt) return;
        barrier();
        buf = 0.0;
        for (uint i = gl_LocalInvocationID.x; i < gl_WorkGroupSize.x; i += sizeSqrt) {
            buf += aggregate[i];
        }
        aggregate[gl_LocalInvocationID.x] = buf;
        if (gl_LocalInvocationID.x > 0) return;
        barrier();
        buf = 0.0;
        for (uint i = 0; i < sizeSqrt; i++) {
            buf += aggregate[i];
        }
        buf = weights[c] / sqrt(buf / (imageSize - 1) + 1e-5);
        stats[CHANNELS + c] = buf;
        //Predivide by standard deviation and merge with bias.
        stats[c] = weights[CHANNELS + c] - mean * buf;
        break;
    }
}
