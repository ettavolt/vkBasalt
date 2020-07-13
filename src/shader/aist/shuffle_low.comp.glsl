#version 450
#extension GL_EXT_scalar_block_layout : require

layout(local_size_x_id = 1, local_size_y_id = 2, local_size_z_id = 3) in;
layout(constant_id = 3) const uint outMaxCh = 16;
layout(push_constant) uniform PushConsts {
    uint width;
    uint height;
};
const uint LOW_CHANNELS = 15;
const uint HIGH_CHANNELS = 16;
layout(std430, set = 0, binding = 0) buffer restrict readonly InTensor {
    float inTensor[];
};
layout(std430, set = 1, binding = 0) buffer restrict writeonly OutTensor {
    float outTensor[];
};
layout(std430, set = 2, binding = 0) uniform restrict readonly Weights {
    float convs[HIGH_CHANNELS][LOW_CHANNELS];
    float biases[HIGH_CHANNELS];
};

const bool toOut = outMaxCh == HIGH_CHANNELS;

void main() {
    const uint pxPos = gl_GlobalInvocationID.x;
    if (pxPos >= width * height) return;
    const uint inMaxCh = toOut ? LOW_CHANNELS : HIGH_CHANNELS;
    const uint outCh = gl_GlobalInvocationID.z;

    float buf = biases[outCh];
    for (uint inCh = 0; inCh < inMaxCh; inCh++) {
        buf = fma(inTensor[pxPos * inMaxCh + inCh], convs[toOut ? outCh : inCh][toOut ? inCh : outCh], buf);
    }

    outTensor[pxPos * outMaxCh + outCh] = buf;
}
