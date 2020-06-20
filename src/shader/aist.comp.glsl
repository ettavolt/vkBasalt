#version 450

layout(local_size_x = 16, local_size_y = 16) in;
layout(binding = 0, rgba8) uniform restrict readonly image2D inImage;
layout(binding = 1, rgba8) uniform restrict writeonly image2D outImage;

void main()
{
    ivec2 size = imageSize(inImage);
    ivec2 field = ivec2(ceil(vec2(size) / 16.0));
    int startX = field.x * int(gl_LocalInvocationID.x);
    int endX = min(startX + field.x, size.x);
    int startY = field.y * int(gl_LocalInvocationID.y);
    int endY = min(startY + field.y, size.y);
    for (int x = startX; x < endX; x++) {
        for (int y = startY; y < endY; y++) {
            ivec2 pos = ivec2(x, y);
            vec4 src = imageLoad(inImage, pos);
            src.rgb = 1.0 - src.rgb;
            //vec4 res = vec4(1.0 - src.rgb, src.a);
            imageStore(outImage, pos, src);
        }
    }
}
