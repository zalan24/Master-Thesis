# add_subdirectory(shadermanager)

file(GLOB SHADERS_SOURCES *.sd *.sh)
set(SHADERBIN "${CMAKE_PROJECT_BINARY_DIR}/shaders/shaders.bin")

add_custom_target(ShaderCodes ALL DEPENDS ${SHADERBIN})

add_custom_command(
    OUTPUT   ${SHADERBIN}
    COMMAND  AnimationsShaderCompiler -r${CMAKE_CURRENT_SOURCE_DIR} -o${SHADERBIN} ${SHADERS_SOURCES}
    DEPENDS  ${SHADERS_SOURCES} AnimationsShaderCompiler
)

# TODO add clang proprecosser to shaders using -E