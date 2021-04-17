file(GLOB SHADERS_HEADERS *.sh)
file(GLOB SHADERS_SOURCES *.sd)
set(SHADERBIN "${CMAKE_BINARY_DIR}/shaders/shaders.bin")
set(SHADER_DEBUG "${CMAKE_BINARY_DIR}/shaders/debug/")

set(GENERATED_FILES "${CMAKE_CURRENT_BINARY_DIR}/shaderregistry.h")

foreach(H IN LISTS SHADERS_HEADERS SHADERS_SOURCES)
    get_filename_component(NAME ${H} NAME_WE)
    list(APPEND GENERATED_FILES "${CMAKE_CURRENT_BINARY_DIR}/shader_header_${NAME}.h" "${CMAKE_CURRENT_BINARY_DIR}/shader_header_${NAME}.cpp")
endforeach()
foreach(S IN LISTS SHADERS_SOURCES)
    get_filename_component(NAME ${S} NAME_WE)
    list(APPEND GENERATED_FILES "${CMAKE_CURRENT_BINARY_DIR}/shader_${NAME}.h" "${CMAKE_CURRENT_BINARY_DIR}/shader_${NAME}.cpp")
endforeach()

add_custom_target(ShaderCodes ALL DEPENDS ${SHADERBIN})

add_custom_command(
    OUTPUT   ${SHADERBIN} ${GENERATED_FILES}
    COMMAND  AnimationsShaderCompiler -r${CMAKE_CURRENT_SOURCE_DIR} -c${CMAKE_CURRENT_BINARY_DIR}/shadercache --headers ${CMAKE_CURRENT_BINARY_DIR} -o${SHADERBIN} -d${SHADER_DEBUG} ${SHADERS_HEADERS} ${SHADERS_SOURCES}
    DEPENDS  ${SHADERS_HEADERS} ${SHADERS_SOURCES} AnimationsShaderCompiler
)

add_library(AnimationShaders ${GENERATED_FILES})
add_dependencies(AnimationShaders ShaderCodes)

target_include_directories(AnimationShaders PUBLIC ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(AnimationShaders PUBLIC AnimationsShaderManager)
