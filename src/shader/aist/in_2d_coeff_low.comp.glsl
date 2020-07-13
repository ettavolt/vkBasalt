#version 450
#extension GL_EXT_scalar_block_layout : require

layout(local_size_x_id = 1, local_size_y = 1, local_size_z_id = 3) in;
layout(constant_id = 1) const uint THREADS = 1;
layout(constant_id = 2) const uint CHANNELS = 1;
layout(push_constant) uniform PushConsts {
    uint width;
    uint height;
};
layout(std430, set = 0, binding = 0) buffer restrict readonly InTensor {
    float inTensor[];
};
layout(std430, set = 1, binding = 0) buffer restrict Stats {
    //Appends CHANNELS count of std¯¹.
    //Also replaces means with -(mean/std)+bias
    float stats[CHANNELS * 2];
};
layout(std430, set = 2, binding = 0) uniform restrict readonly Weights {
    //This is effectively scale & bias, but offsets are not calculated with spec constants
    //thus we can't have two fields in the struct.
    float weights[CHANNELS * 2];
};

shared float[THREADS] aggregate;

void main() {
    const uint c = gl_GlobalInvocationID.z;
    const uint imageSize = width * height;
    //const uint featureMapOffset = c * imageSize;
    const uint featureMapOffset = c;
    const uint sizeSqrt = uint(ceil(sqrt(THREADS)));
    float buf = 0.0;
    const float neg_mean = -stats[c];
    for (uint i = gl_LocalInvocationID.x; i < imageSize; i += THREADS) {
        float diff = inTensor[i * CHANNELS + featureMapOffset] + neg_mean;
        buf += diff * diff;
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
    buf = weights[c] / sqrt(buf / (imageSize - 1) + 1e-5);
    stats[CHANNELS + c] = buf;
    //Predivide by standard deviation and merge with bias.
    stats[c] = fma(neg_mean, buf, weights[CHANNELS + c]);
}
