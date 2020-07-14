#version 450
#extension GL_EXT_scalar_block_layout : require

layout(local_size_x_id = 3, local_size_y_id = 4, local_size_z_id = 5) in;
layout(constant_id = 0) const uint OUT_CHANNELS = 1;
layout(constant_id = 1) const uint IN_CHANNELS = 1;
layout(constant_id = 2) const bool RE_LU = false;
layout(push_constant) uniform PushConsts {
    uint outWidth;
    uint outHeight;
};
const uint IN_CHANNELS_GROUP = IN_CHANNELS / OUT_CHANNELS;
//Channels dimension is intentionally unspecified to test different layouts.
layout(std430, set = 0, binding = 0) buffer restrict writeonly OutTensor {
    float outTensor[];
};
//Can't have two SpecConstant-sized fields in one struct, because their offsets are calculated for just one element.
layout(std430, set = 1, binding = 0) buffer restrict readonly InTensor {
    float inTensor[];
};
//Dynamically indexed uniform buffer must have definite sizes.
layout(std430, set = 2, binding = 0) uniform restrict readonly Convs {
    float weights[IN_CHANNELS * 10];
};

uint outCh = gl_GlobalInvocationID.z;

void addFromSource(const in uint inOffset, const in int dx, const in int dy, inout float buf) {
    const int convIndex = (dx + 1) * 3 + dy + 1;
    for (uint inChG = 0; inChG < IN_CHANNELS_GROUP; inChG++) {
        const uint inCh = (outCh * IN_CHANNELS_GROUP + inChG);
        buf = fma(inTensor[inOffset + inCh], weights[inCh * 9 + convIndex], buf);
    }
}

void main() {
    if (gl_GlobalInvocationID.x >= outWidth || gl_GlobalInvocationID.y >= outHeight) return;
    const uint inWidth = (outWidth + 1) / 2;
    const uint inHeight = (outHeight + 1) / 2;
    const uint sx = gl_GlobalInvocationID.x / 2;
    const uint sy = gl_GlobalInvocationID.y / 2;
    const int dx = int(gl_GlobalInvocationID.x % 2);
    const int dy = int(gl_GlobalInvocationID.y % 2);
    float buf = weights[IN_CHANNELS * 9 + outCh];
    uint inOffset = (sx * inHeight + sy) * IN_CHANNELS;
    addFromSource(inOffset, dx, dy, buf);
    if (dx > 0) {
        inOffset = ((sx + 1) * inHeight + sy) * IN_CHANNELS;
        addFromSource(inOffset, -1, dy, buf);
    }
    if (dy > 0) {
        inOffset = (sx * inHeight + sy + 1) * IN_CHANNELS;
        addFromSource(inOffset, dx, -1, buf);
    }
    if (dx > 0 && dy > 0) {
        inOffset = ((sx + 1) * inHeight + sy + 1) * IN_CHANNELS;
        addFromSource(inOffset, -1, -1, buf);
    }
    const uint outPos = (gl_GlobalInvocationID.x * outHeight + gl_GlobalInvocationID.y) * OUT_CHANNELS + outCh;
    if (RE_LU) {
        outTensor[outPos] = max(buf, 0.0);
    } else {
        outTensor[outPos] = buf;
    }
}
