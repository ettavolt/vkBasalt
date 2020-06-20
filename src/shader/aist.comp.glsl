#version 450

layout(local_size_x = 16, local_size_y = 16) in;
layout(binding = 0, rgba8) uniform restrict readonly image2D inImage;
layout(binding = 1, rgba8) uniform restrict writeonly image2D outImage;
layout(push_constant) uniform PushConsts {
    uint width;
    uint height;
};

void main()
{
    uvec2 tile = uvec2(ceil(vec2(width, height) / 16.0));
    uint startX = tile.x * gl_LocalInvocationID.x;
    uint endX = min(startX + tile.x, width);
    uint startY = tile.y * gl_LocalInvocationID.y;
    uint endY = min(startY + tile.y, height);
    for (uint x = startX; x < endX; x++) {
        for (uint y = startY; y < endY; y++) {
            ivec2 pos = ivec2(x, y);
            vec4 src = imageLoad(inImage, pos);
            src.rgb = 1.0 - src.rgb;
            imageStore(outImage, pos, src);
        }
    }
}
