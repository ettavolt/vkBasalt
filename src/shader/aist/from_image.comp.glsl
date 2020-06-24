#version 450
#extension GL_GOOGLE_include_directive : require
#include "consts.comp.glsl"

const int IN_CHANNELS = 3;
//const int OUT_CHANNELS = 32;
layout(set = 0, binding = 0) uniform Convs {
    //These are for multiplying rgb directly
    vec3 convs[10];
    //vec3 sels[OUT_CHANNELS];
};
layout(set = 1, binding = 0, rgba8) uniform restrict readonly image2D inImage;
layout(set = 1, binding = 1) buffer restrict writeonly OutTensor {
    float outTensor[WIDTH * HEIGHT][IN_CHANNELS];
};

void main() {
    const int cx = int(gl_GlobalInvocationID.x);
    const int cy = int(gl_GlobalInvocationID.y);
    if (cx >= WIDTH || cy >= HEIGHT) return;
    const int bx = max(cx - 1, 0);
    const int dx = min(cx + 1, WIDTH - 1);
    const int by = max(cy - 1, 0);
    const int dy = min(cy + 1, HEIGHT - 1);
    vec3 buf = convs[0];
    buf = fma(convs[1], imageLoad(inImage, ivec2(bx, by)).rgb, buf);
    buf = fma(convs[2], imageLoad(inImage, ivec2(bx, cy)).rgb, buf);
    buf = fma(convs[3], imageLoad(inImage, ivec2(bx, dy)).rgb, buf);
    buf = fma(convs[4], imageLoad(inImage, ivec2(cx, by)).rgb, buf);
    buf = fma(convs[5], imageLoad(inImage, ivec2(cx, cy)).rgb, buf);
    buf = fma(convs[6], imageLoad(inImage, ivec2(cx, dy)).rgb, buf);
    buf = fma(convs[7], imageLoad(inImage, ivec2(dx, by)).rgb, buf);
    buf = fma(convs[8], imageLoad(inImage, ivec2(dx, cy)).rgb, buf);
    buf = fma(convs[9], imageLoad(inImage, ivec2(dx, dy)).rgb, buf);
    //TODO: fill OUT_CHANNELS now
    const int outPos = cx * WIDTH + cy;
    for (int c = 0; c < IN_CHANNELS; c++) {
        outTensor[outPos][c] = buf[c];
    }
}
