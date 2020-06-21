#version 450

layout(constant_id = 0) const int WIDTH = 1;
layout(constant_id = 1) const int HEIGHT = 1;
layout(local_size_x = 16, local_size_y = 16) in;
layout(binding = 2, rgba8) uniform restrict readonly image2D inImage;
layout(binding = 3, rgba8) uniform restrict writeonly image2D outImage;

void main()
{
    const ivec2 inPos = ivec2(gl_GlobalInvocationID.xy);
    const ivec2 outPos = ivec2(WIDTH-1, HEIGHT-1) - inPos;
    if (outPos.x < 0 || outPos.y < 0) return;
    vec4 src = imageLoad(inImage, inPos);
    src.rgb = 1.0 - src.rgb;
    imageStore(outImage, outPos, src);
}
