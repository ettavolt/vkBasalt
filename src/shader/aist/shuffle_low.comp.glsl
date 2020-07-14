#version 450
#extension GL_EXT_scalar_block_layout : require

layout(local_size_x_id = 3, local_size_y = 1, local_size_z_id = 5) in;
layout(constant_id = 0) const uint LOW_CHANNELS = 1;
layout(constant_id = 1) const uint HIGH_CHANNELS = 1;
layout(constant_id = 2) const bool TO_LOW = false;
layout(constant_id = 5) const uint OUT_CHANNELS = 1;
layout(push_constant) uniform PushConsts {
    uint width;
    uint height;
};
layout(std430, set = 0, binding = 0) buffer restrict readonly InTensor {
    float inTensor[];
};
layout(std430, set = 1, binding = 0) buffer restrict writeonly OutTensor {
    float outTensor[];
};
layout(std430, set = 2, binding = 0) uniform restrict readonly Weights {
    float weights[HIGH_CHANNELS * LOW_CHANNELS + HIGH_CHANNELS];
};

void main() {
    const uint pxPos = gl_GlobalInvocationID.x;
    if (pxPos >= width * height) return;
    const uint inMaxCh = TO_LOW ? HIGH_CHANNELS : LOW_CHANNELS;
    const uint outCh = gl_GlobalInvocationID.z;

    float buf = weights[HIGH_CHANNELS * LOW_CHANNELS + outCh];
    const uint inOffset = pxPos * inMaxCh;
    for (uint inCh = 0; inCh < inMaxCh; inCh++) {
        const uint wIdx = (TO_LOW ? inCh : outCh) * LOW_CHANNELS + (TO_LOW ? outCh : inCh);
        buf = fma(inTensor[inOffset + inCh], weights[wIdx], buf);
    }

    outTensor[pxPos * OUT_CHANNELS + outCh] = buf;
}
