#version 450
#extension GL_EXT_scalar_block_layout : require

layout(local_size_x_id = 3, local_size_y = 1, local_size_z_id = 5) in;
layout(constant_id = 0) const uint CHANNELS = 1;
layout(constant_id = 3) const uint THREADS = 1;
layout(push_constant) uniform PushConsts {
    uint width;
    uint height;
};
//Making this channels outer reduces two substages time by 410 μs for vulkanscene sample.
layout(std430, set = 0, binding = 0) buffer restrict readonly InTensor {
    float inTensor[];
};
layout(std430, set = 1, binding = 0) buffer restrict writeonly Stats {
    //Only push CHANNELS means here.
    float stats[CHANNELS * 2];
};

shared float[THREADS] aggregate;

void main() {
    const uint c = gl_GlobalInvocationID.z;
    const uint imageSize = width * height;
    //const uint featureMapOffset = c * imageSize;
    const uint featureMapOffset = c;
    const uint sizeSqrt = uint(ceil(sqrt(THREADS)));
    float buf = 0.0;
    for (uint i = gl_LocalInvocationID.x; i < imageSize; i += THREADS) {
        buf += inTensor[i * CHANNELS + featureMapOffset];
    }
    aggregate[gl_LocalInvocationID.x] = buf;
    if (gl_LocalInvocationID.x >= sizeSqrt) return;
    barrier();
    buf = 0.0;
    for (uint i = gl_LocalInvocationID.x; i < THREADS; i += sizeSqrt) {
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
}
