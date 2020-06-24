#version 450
#extension GL_GOOGLE_include_directive : require
#include "consts.comp.glsl"
layout(local_size_x = 4, local_size_y = 4) in;

const uint IN_CHANNELS = 32;
const uint OUT_CHANNELS = 3;
layout(std430, set = 0, binding = 0) uniform restrict readonly Convs {
    float convs[IN_CHANNELS][OUT_CHANNELS][3 * 3];
    float biases[OUT_CHANNELS];
};
layout(set = 1, binding = 0, rgba8) uniform restrict writeonly image2D outImage;
layout(std430, set = 1, binding = 1) buffer restrict coherent InTensor {
    float inTensor[WIDTH / 2 * HEIGHT / 2][IN_CHANNELS];
    float outTensor[WIDTH * HEIGHT][OUT_CHANNELS];
};

void calcPixel(const in uint inPos, const in int dx, const in int dy, const in bool add) {
    const int cx = int(gl_GlobalInvocationID.x) * 2 + dx;
    const int cy = int(gl_GlobalInvocationID.y) * 2 + dy;
    if (cx >= WIDTH || cy >= HEIGHT) return;
    if (cx < 0 || cy < 0) return;
    const int outPos = cx * HEIGHT + cy;
    float[OUT_CHANNELS] buf;
    if (add) {
        buf = outTensor[outPos];
    } else {
        buf = biases;
    }
    const int convIndex = (dx + 1) * 3 + dy + 1;
    for (uint ic = 0; ic < IN_CHANNELS; ic++) {
        for (uint oc = 0; oc < OUT_CHANNELS; oc++) {
            buf[oc] = fma(inTensor[inPos][ic], convs[ic][oc][convIndex], buf[oc]);
        }
    }
//    imageStore(outImage, ivec2(cx, cy), vec4(res[0], res[1], res[2], 1.0));
    outTensor[outPos] = buf;
}

void pushPixel(in int dx, in int dy) {
    const int cx = int(gl_GlobalInvocationID.x) * 2 + dx;
    const int cy = int(gl_GlobalInvocationID.y) * 2 + dy;
    if (cx >= WIDTH || cy >= HEIGHT) return;
    float res[OUT_CHANNELS] = outTensor[cx * HEIGHT + cy];
    imageStore(outImage, ivec2(cx, cy), vec4(res[0], res[1], res[2], 1.0));
}

void main() {
    const uint sx = gl_GlobalInvocationID.x;
    const uint sy = gl_GlobalInvocationID.y;
    if (sx >= WIDTH / 2 || sy >= HEIGHT / 2) return;
    const uint inPos = sx * HEIGHT / 2 + sy;
    calcPixel(inPos, 0, 0, false);
    calcPixel(inPos, 0, 1, false);
    calcPixel(inPos, 1, 1, false);
    calcPixel(inPos, 1, 0, false);
    memoryBarrierBuffer();
    barrier();
    calcPixel(inPos, 1, -1, true);
    calcPixel(inPos, 0, -1, true);
    memoryBarrierBuffer();
    barrier();
    calcPixel(inPos, -1, -1, true);
    memoryBarrierBuffer();
    barrier();
    calcPixel(inPos, -1, 0, true);
    calcPixel(inPos, -1, 1, true);
    memoryBarrierBuffer();
    barrier();
    pushPixel(0, 0);
    pushPixel(0, 1);
    pushPixel(1, 1);
    pushPixel(1, 0);
}
