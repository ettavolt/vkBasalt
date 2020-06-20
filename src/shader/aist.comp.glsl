#version 450

layout(local_size_x = 16, local_size_y = 16) in;
layout(push_constant) uniform PushConsts {
    uint width;
    uint height;
};
/*layout(binding = 0) buffer coherent restrict IntermediateMemory {
    float memory[];
};*/
/*layout(binding = 1) buffer restrict readonly Weights {
    float weights[];
};*/
layout(binding = 2, rgba8) uniform restrict readonly image2D inImage;
layout(binding = 3, rgba8) uniform restrict writeonly image2D outImage;

void main()
{
    const uvec2 tile = uvec2(ceil(vec2(width, height) / 16.0));
    const uint startX = tile.x * gl_LocalInvocationID.x;
    const uint endX = min(startX + tile.x, width);
    const uint startY = tile.y * gl_LocalInvocationID.y;
    const uint endY = min(startY + tile.y, height);
    for (uint x = startX; x < endX; x++) {
        for (uint y = startY; y < endY; y++) {
            const ivec2 pos = ivec2(x, y);
            vec4 src = imageLoad(inImage, pos);
            src.rgb = 1.0 - src.rgb;
            imageStore(outImage, pos, src);
        }
    }
    memoryBarrierBuffer();
    barrier();
}
