#version 450
#extension GL_EXT_scalar_block_layout : require

layout(local_size_x_id = 1, local_size_y_id = 2, local_size_z_id = 3) in;
layout(push_constant) uniform PushConsts {
    uint inWidth;
    uint inHeight;
};
const uint IN_CHANNELS = 3;
const uint OUT_CHANNELS = 15;
layout(std430, set = 0, binding = 0) buffer restrict readonly InTensor {
    float inTensor[];
};
layout(std430, set = 1, binding = 0) buffer restrict writeonly OutTensor {
    float outTensor[];
};
layout(std430, set = 2, binding = 0) uniform restrict readonly Weights {
    float convs[OUT_CHANNELS][3 * 3];
    float biases[OUT_CHANNELS];
};

void main() {
    const uint outWidth = (inWidth + 1) / 2;
    const uint outHeight = (inHeight + 1) / 2;
    if (gl_GlobalInvocationID.x >= outWidth || gl_GlobalInvocationID.y >= outHeight) return;
    const uint cx = gl_GlobalInvocationID.x * 2;
    uint cy = gl_GlobalInvocationID.y * 2;
    const uint outCh = gl_GlobalInvocationID.z;
    const uint inCh = outCh * IN_CHANNELS / OUT_CHANNELS;
    const uint by = max(cy - 1, 0) * IN_CHANNELS + inCh;
    const uint dy = min(cy + 1, inHeight - 1) * IN_CHANNELS + inCh;
    cy = cy * IN_CHANNELS + inCh;

    float buf = biases[outCh];

    uint x = max(cx - 1, 0) * inHeight * IN_CHANNELS;
    buf = fma(convs[outCh][0], inTensor[x + by], buf);
    buf = fma(convs[outCh][1], inTensor[x + cy], buf);
    buf = fma(convs[outCh][2], inTensor[x + dy], buf);
    x = cx * inHeight * IN_CHANNELS;
    buf = fma(convs[outCh][3], inTensor[x + by], buf);
    buf = fma(convs[outCh][4], inTensor[x + cy], buf);
    buf = fma(convs[outCh][5], inTensor[x + dy], buf);
    x = min(cx + 1, inWidth - 1) * inHeight * IN_CHANNELS;
    buf = fma(convs[outCh][6], inTensor[x + by], buf);
    buf = fma(convs[outCh][7], inTensor[x + cy], buf);
    buf = fma(convs[outCh][8], inTensor[x + dy], buf);
    uint outPos = (gl_GlobalInvocationID.x * outHeight + gl_GlobalInvocationID.y) * OUT_CHANNELS + outCh;
    outTensor[outPos] = buf;
}
