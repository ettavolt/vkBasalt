shader_src = [
    'aist/from_image.comp.glsl',
    'aist/in_2d.comp.glsl',
    'aist/up_conv_32_3.comp.glsl',
    'aist/to_image.comp.glsl',
    'cas.frag.glsl',
    'deband.frag.glsl',
    'dls.frag.glsl',
    'full_screen_triangle.vert.glsl',
    'fxaa.frag.glsl',
    'lut.frag.glsl',
    'smaa_blend.frag.glsl',
    'smaa_blend.vert.glsl',
    'smaa_edge_color.frag.glsl',
    'smaa_edge_luma.frag.glsl',
    'smaa_edge.vert.glsl',
    'smaa_neighbor.frag.glsl',
    'smaa_neighbor.vert.glsl',
]

glsl_compiler = find_program('glslangValidator')
glsl_generator = generator(glsl_compiler,
    output    : [ '@BASENAME@.h' ],
    arguments : [ '-V', '-x', '@INPUT@', '-o', '@OUTPUT@' ])

shader_include = glsl_generator.process(shader_src, preserve_path_from: meson.current_source_dir())
