#version 450
#extension GL_EXT_scalar_block_layout : require

layout(local_size_x_id = 2, local_size_y_id = 3, local_size_z_id = 4) in;
layout(constant_id = 0) const int IN_CHANNELS = 3;
layout(constant_id = 1) const int OUT_CHANNELS = 24;
layout(std430, set = 0, binding = 0) uniform restrict readonly Convs {
    float convs[OUT_CHANNELS][IN_CHANNELS][3 * 3];
};
layout(set = 1, binding = 0, rgba8) uniform restrict readonly image2D inImage;
layout(std430, set = 1, binding = 1) buffer restrict writeonly OutTensor {
    float outTensor[WIDTH / 2 * (HEIGHT / 2)][OUT_CHANNELS];
};

void main() {
    const int cx = int(gl_GlobalInvocationID.x) * 2;
    const int cy = int(gl_GlobalInvocationID.y) * 2;
    if (cx >= WIDTH || cy >= HEIGHT) return;
    const uint c = gl_GlobalInvocationID.z;
    const int by = max(cy - 1, 0);
    const int dy = min(cy + 1, HEIGHT - 1);
    float buf = 0.0;
    float conv[IN_CHANNELS][9] = convs[c];
    int x = max(cx - 1, 0);
    buf += dot(
    vec3(conv[0][0], conv[1][0], conv[2][0]),
    imageLoad(inImage, ivec2(x, by)).rgb
    );
    buf += dot(
    vec3(conv[0][1], conv[1][1], conv[2][1]),
    imageLoad(inImage, ivec2(x, cy)).rgb
    );
    buf += dot(
    vec3(conv[0][2], conv[1][2], conv[2][2]),
    imageLoad(inImage, ivec2(x, dy)).rgb
    );
    x = cx;
    buf += dot(
    vec3(conv[0][3], conv[1][3], conv[2][3]),
    imageLoad(inImage, ivec2(cx, by)).rgb
    );
    buf += dot(
    vec3(conv[0][4], conv[1][4], conv[2][4]),
    imageLoad(inImage, ivec2(cx, cy)).rgb
    );
    buf += dot(
    vec3(conv[0][5], conv[1][5], conv[2][5]),
    imageLoad(inImage, ivec2(cx, dy)).rgb
    );
    x = min(cx + 1, WIDTH - 1);
    buf += dot(
    vec3(conv[0][6], conv[1][6], conv[2][6]),
    imageLoad(inImage, ivec2(x, by)).rgb
    );
    buf += dot(
    vec3(conv[0][7], conv[1][7], conv[2][7]),
    imageLoad(inImage, ivec2(x, cy)).rgb
    );
    buf += dot(
    vec3(conv[0][8], conv[1][8], conv[2][8]),
    imageLoad(inImage, ivec2(x, dy)).rgb
    );
    outTensor[gl_GlobalInvocationID.x * HEIGHT / 2 + gl_GlobalInvocationID.y][c] = buf;
}
