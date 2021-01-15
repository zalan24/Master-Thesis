add_subdirectory(compiler)
add_subdirectory(shadermanager)

file(GLOB SHADERS_SOURCES *.vert *.frag *.comp)
set(SHADERCODE_H "${CMAKE_CURRENT_BINARY_DIR}/shadermanager/shadercodes.h")

add_custom_target(ShaderCodes ALL DEPENDS ${SHADERCODE_H})

add_custom_command(
    OUTPUT   ${SHADERCODE_H}
    COMMAND  AnimationsShaderCompiler ${SHADERCODE_H} ${SHADERS_SOURCES}
    DEPENDS  ${SHADERS_SOURCES} AnimationsShaderCompiler
)

# TODO add clang proprecosser to shaders using -E