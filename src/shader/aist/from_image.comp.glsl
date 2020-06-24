#version 450
#extension GL_GOOGLE_include_directive : require
#include "consts.comp.glsl"

const int IN_CHANNELS = 3;
//const int OUT_CHANNELS = 32;
layout(std430, set = 0, binding = 0) uniform Convs {
    //These are for multiplying rgb directly
    float convs[10][3];
    //vec3 sels[OUT_CHANNELS];
};
layout(set = 1, binding = 0, rgba8) uniform restrict readonly image2D inImage;
layout(std430, set = 1, binding = 1) buffer restrict writeonly OutTensor {
    float outTensor[WIDTH * HEIGHT][IN_CHANNELS];
};

vec3 load_fma(int x, int y, int c, vec3 buf) {
    return fma(
        vec3(convs[c][0], convs[c][1], convs[c][2]),
        imageLoad(inImage, ivec2(x, y)).rgb,
        buf
    );
}

void main() {
    const int cx = int(gl_GlobalInvocationID.x);
    const int cy = int(gl_GlobalInvocationID.y);
    if (cx >= WIDTH || cy >= HEIGHT) return;
    const int bx = max(cx - 1, 0);
    const int dx = min(cx + 1, WIDTH - 1);
    const int by = max(cy - 1, 0);
    const int dy = min(cy + 1, HEIGHT - 1);
    vec3 buf = vec3(convs[0][0], convs[0][1], convs[0][2]);
    buf = load_fma(bx, by, 1, buf);
    buf = load_fma(bx, cy, 2, buf);
    buf = load_fma(bx, dy, 3, buf);
    buf = load_fma(cx, by, 4, buf);
    buf = load_fma(cx, cy, 5, buf);
    buf = load_fma(cx, dy, 6, buf);
    buf = load_fma(dx, by, 7, buf);
    buf = load_fma(dx, cy, 8, buf);
    buf = load_fma(dx, dy, 9, buf);
    //TODO: fill OUT_CHANNELS now
    const int outPos = cx * WIDTH + cy;
    for (int c = 0; c < IN_CHANNELS; c++) {
        outTensor[outPos][c] = buf[c];
    }
}
